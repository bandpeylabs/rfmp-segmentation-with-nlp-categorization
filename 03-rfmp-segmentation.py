# Databricks notebook source
# MAGIC %md
# MAGIC # RFMP Segmentation
# MAGIC
# MAGIC ## Load “gold” RFMP‐scored data from Unity Catalog into Spark

# COMMAND ----------

gold_df = spark.read.table("demos.rfmp_segmentation.gold")

display(gold_df.limit(10))

# COMMAND ----------

selected_rows_df = gold_df.filter(gold_df.CustomerID == 14700)
display(selected_rows_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## For each CustomerID, pick the single most‐frequently bought ProductCategoryID from gold_df

# COMMAND ----------

# Cell 2: For each CustomerID, pick the single (category, cluster) row with highest Frequency

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 2.1  Define a window partitioned by CustomerID, ordered by Frequency DESC
window_by_customer = Window.partitionBy("CustomerID") \
                           .orderBy(F.desc("Frequency"))

# 2.2  Add a row_number() so that, per customer, the top‐Frequency row gets rn = 1
ranked = gold_df.withColumn(
    "rn",
    F.row_number().over(window_by_customer)
)

# 2.3  Filter to keep only rn = 1 (the favorite category+cluster per customer)
favorite_per_customer = (
    ranked
      .filter(F.col("rn") == 1)
      .select(
        F.col("CustomerID"),
        F.col("ProductCategoryID").alias("FavoriteCategoryID"),
        F.col("ClusterName").alias("FavoriteClusterName"),
        F.col("Recency"),
        F.col("Frequency"),
        F.col("MonetaryValue"),
        F.col("r_bin"),
        F.col("f_bin"),
        F.col("m_bin")
      )
)

# 2.4  Display and cache
favorite_per_customer.cache()
display(favorite_per_customer.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the relevant columns to Pandas for t-SNE
# MAGIC
# MAGIC ### Analyze Cluster Suitability Using t-SNE
# MAGIC
# MAGIC Before assigning cluster labels, it’s useful to first understand the distribution of our scored data to evaluate if clustering makes sense. A practical way to do this is by applying t-Distributed Stochastic Neighbor Embedding (t-SNE) — a dimensionality reduction technique that projects high-dimensional data into 2D or 3D space.
# MAGIC
# MAGIC While t-SNE isn’t suitable for downstream ML models, it’s a valuable tool for visualizing the internal structure of complex datasets. It helps us assess whether distinct groupings or patterns exist that could justify the use of clustering algorithms.

# COMMAND ----------

# Select only the columns we need for t-SNE (and any identifiers for later)
tsne_source_df = favorite_per_customer.select(
    "CustomerID", 
    "FavoriteCategoryID", 
    "r_bin", 
    "f_bin", 
    "m_bin"
)

# Convert to Pandas (this assumes the result fits comfortably in memory).
# If your data is very large, consider sampling or using Spark-ML t-SNE instead.
inputs_pd = tsne_source_df.toPandas()

# Quick sanity check:
print("Pandas shape:", inputs_pd.shape)
print(inputs_pd.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run 2-D t-SNE over the (r_bin, f_bin, m_bin) columns

# COMMAND ----------

from sklearn.manifold import TSNE
import numpy as np

# We only care about the numeric bins for t-SNE:
bin_cols = ["r_bin", "f_bin", "m_bin"]
X_bins   = inputs_pd[bin_cols].values   # shape = (N_customers, 3)

# Configure and run t-SNE:
tsne = TSNE(
    n_components=2,
    perplexity=80,
    n_iter=1000,
    init="pca",
    learning_rate="auto",
    random_state=42
)

tsne_results = tsne.fit_transform(X_bins)
# tsne_results has shape (N_customers, 2)

# Append them back into inputs_pd:
inputs_pd["tsne_one"] = tsne_results[:, 0]
inputs_pd["tsne_two"] = tsne_results[:, 1]

# Display first few rows to confirm:
inputs_pd[["CustomerID", "FavoriteCategoryID", "r_bin", "f_bin", "m_bin", "tsne_one", "tsne_two"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize t-SNE colored by each bin (r_bin, f_bin, m_bin)
# MAGIC
# MAGIC Now we visualize the data points using their RFM score bins to better understand the distribution and structure within the customer base. Each point represents a customer-category pair, and color intensity reflects the relative value of the metric — with warmer tones indicating higher scores. This allows us to visually detect patterns, density regions, and separability that inform clusterability.

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Configure a 1×3 grid for (r_bin, f_bin, m_bin) scatterplots
f, axes = plt.subplots(nrows=1, ncols=3, figsize=(32, 10), squeeze=True)

for i, metric in enumerate(["r_bin", "f_bin", "m_bin"]):
    # Determine how many unique levels this metric has
    n_levels = inputs_pd[metric].nunique()

    axes[i].set_title(metric)
    sns.scatterplot(
        x="tsne_one",
        y="tsne_two",
        hue=metric,
        palette=sns.color_palette("coolwarm", n_levels),
        data=inputs_pd,
        legend=False,
        alpha=0.4,
        s=100,  # Adjust this value to make the dots bigger
        ax=axes[i],
    )

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the visualizations, we can see some clear divisions between users along these three metrics. We can also see how different regions align with one another with regards to the strength of their recency, frequency and monetary value metrics. We will return to these visualizations after cluster assignment to help us understand what each cluster tells us about our customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Customers Using RFM Score Bins
# MAGIC
# MAGIC Based on the t-SNE visualizations, we observe significant overlap across the RFM dimensions, indicating that a full 5×5×5 (125) segmentation would be excessive and not particularly actionable for marketing. Instead, we can focus on identifying a smaller, more practical number of clusters — typically in the range of 2 to 20. From the density and separability in the plots, it’s likely that we’ll need far fewer than 20.
# MAGIC
# MAGIC Note: Clustering performance scales with the number of virtual cores available in your Spark cluster, especially when running multiple k values in parallel.

# COMMAND ----------

# DBTITLE 1,Evaluate Cluster Size
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pyspark.sql.functions as fn

# define max value of k to explore
max_k = 20
 
# copy binned_pd to each worker node to facilitate parallel evaluation
inputs_pd_broadcast = sc.broadcast(inputs_pd[['r_bin','f_bin','m_bin']])
 
 
# function to train and score clusters based on k cluster count
@fn.udf('float')
def get_silhouette(k):
 
  # train a model on k
  km = KMeans(
    n_clusters=k, 
    init='random',
    n_init=10000
    )
  kmeans = km.fit( inputs_pd_broadcast.value )
 
  # get silhouette score for model 
  silhouette = silhouette_score( 
      inputs_pd_broadcast.value,  # x values
      kmeans.predict(inputs_pd_broadcast.value) # cluster assignments 
      )
  
  # return score
  return float(silhouette)
 
 
# assemble an dataframe containing each k value
iterations = (
  spark
    .range(2, max_k + 1, step=1, numPartitions=sc.defaultParallelism) # get values for k
    .withColumnRenamed('id','k') # rename to k
    .repartition( max_k-1, 'k' ) # ensure data are well distributed
    .withColumn('silhouette', get_silhouette('k'))
  )
  
# release the distributed dataset
inputs_pd_broadcast.unpersist()
 
# display the results of our analysis
display( 
  iterations
      )


# COMMAND ----------

# MAGIC %md
# MAGIC From our chart, it appears 8 clusters might be a good target number of clusters. Yes, there are higher silhouette scores we could achieve, but it appears from the curve that at 8 clusters, the incremental gains with added clusters begin to decline.
# MAGIC
# MAGIC With that in mind, we can define our model to support 8 clusters and finalize our pipeline.

# COMMAND ----------

from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer
from sklearn.compose import ColumnTransformer

# defining binning transformation
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
 
# apply binner to each column
col_trans = ColumnTransformer(
  [
    ('r_bin', binner, ['Recency']),
    ('f_bin', binner, ['Frequency']),
    ('m_bin', binner, ['MonetaryValue'])
    ],
  remainder='drop'
  )

# COMMAND ----------

# DBTITLE 1,Assemble & Train Pipeline
from sklearn.pipeline import Pipeline

# define model
model = KMeans(
  n_clusters=8, 
  init='random',
  n_init=10000
  )
 
# couple model with transformations
pipe = Pipeline(steps=[
  ('binnerize', col_trans),
  ('cluster', model)
  ])
 
# train pipeline
fitted_pipe = pipe.fit( inputs_pd )
 
# assign clusters
inputs_pd['cluster'] = pipe.predict( inputs_pd )
 
# display cluster assignments
display(inputs_pd)

# COMMAND ----------


