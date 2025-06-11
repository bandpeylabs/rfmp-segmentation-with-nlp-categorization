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

# Select all columns from favorite_per_customer
tsne_source_df = favorite_per_customer.select(
    "CustomerID", 
    "FavoriteCategoryID", 
    "FavoriteClusterName",
    "Recency", 
    "Frequency", 
    "MonetaryValue", 
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

# We only care about the numeric bins for t-SNE:
bin_cols = ["r_bin", "f_bin", "m_bin"]
X_bins = inputs_pd[bin_cols].values  # shape = (N_customers, 3)

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
display(inputs_pd)

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

# DBTITLE 1,Visualize Cluster Assignments
f, axes = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(42, 10))
 
axes[0].set_title('cluster')
sns.scatterplot(
  x='tsne_one',
  y='tsne_two',
  hue='cluster',
  palette=sns.color_palette('husl', inputs_pd[['cluster']].nunique()[0]),
  data=inputs_pd,
  alpha=0.4,
  ax = axes[0]
  )
axes[0].legend(loc='lower left', ncol=2, fancybox=True)
 
# chart the RFM scores
for i, metric in enumerate(['r_bin', 'f_bin', 'm_bin']):
  
  # unique values for this metric
  n = inputs_pd[['{0}'.format(metric)]].nunique()[0]
  
  # use metric name as chart title
  axes[i+1].set_title(metric)
  
  # define chart
  sns.scatterplot(
    x='tsne_one',
    y='tsne_two',
    hue='{0}'.format(metric),
    palette=sns.color_palette('coolwarm', n),
    data=inputs_pd,
    legend=False,
    alpha=0.4,
    ax = axes[i+1]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC The visualization of clusters relative to the RFM metrics helps us understand how clusters related to the metric values and the range of values associated with each cluster. We could perform more detailed analysis of the clusters to understand the distance between members and between the various clusters but a quick visual inspection is often sufficient for this kind of work.
# MAGIC
# MAGIC In addition, we can extract the centroids of each cluster to more precisely understand how each relates to the RFM metrics. Please note that these centroids have exact positions that are captured in fractional values. But to help simplify the comparison of the clusters, we've rounded these up to the nearest integer value:

# COMMAND ----------

import numpy as np
import pandas as pd

clusters = []

# for each cluster
for c in range(0, pipe[-1].n_clusters):
  # get integer values for metrics associated with each centroid
  centroids = np.abs(pipe[-1].cluster_centers_[c].round(0).astype('int')).tolist()
  # capture cluster and centroid values
  clusters += [[c] + centroids]

# convert details to dataframe
clusters_pd = pd.DataFrame(clusters, columns=['cluster', 'r_bin', 'f_bin', 'm_bin'])

display(clusters_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist Cluster Assignments
# MAGIC The previous steps in this notebook are intended to demonstrate how an RFM segmentation might be performed, but how might we operationalize the model for on-going work? Every time we retrain our model, the centroids attached to a cluster id will vary. If we wish to re-score customers periodically but keep cluster centroids the same between those runs, we need to persist and re-use our model. This is made easy using the MLFlow model registry:

# COMMAND ----------

model_name = 'rfmp_segmentation'

# COMMAND ----------

# to ensure this notebook runs in jobs
import mlflow

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, model_name))

# COMMAND ----------

with mlflow.start_run(run_name='deployment ready'):
 
  mlflow.sklearn.log_model(
    fitted_pipe,
    'model',
    registered_model_name=model_name
    )


# COMMAND ----------

# MAGIC %md
# MAGIC We can then elevate our model to production status to indicate it is ready for use in an on-going ETL pipeline:

# COMMAND ----------

# connect to mlflow
client = mlflow.tracking.MlflowClient()
 
# identify model version in registry
latest_model_info = client.search_model_versions(f"name='{model_name}'")[0]
model_version = latest_model_info.version
model_status = latest_model_info.current_stage
 
# move model to production status
if model_status.lower() != 'production':
  client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage='production',
    archive_existing_versions=True
    ) 

# COMMAND ----------

# MAGIC %md
# MAGIC With our model persisted and elevated to production status, applying it to data is relatively easy:

# COMMAND ----------

# define user-defined function for rfm segment assignment
rfmp_segment = mlflow.pyfunc.spark_udf(spark, model_uri=f'models:/{model_name}/production')
 
# apply function to summary customer metrics
display(
  gold_df
    .withColumn('Cluster', 
        rfmp_segment(fn.struct('Recency','Frequency','MonetaryValue'))
          )
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Of course, to make use of these clusters, we'll want access to descriptive information about what each cluster represents. Typically, the marketing team will assign friendly labels to each cluster that explain what they represent in easy to understand terms. For our purposes, we'll just persist the centroid information extracted in the last step. These data could then be joined with the output of the previous cell to provide friendly labels for each cluster assignment:

# COMMAND ----------

_ = (
  spark
    .createDataFrame(clusters_pd)
    .withColumn( # more typically, a friendly name would be assigned by marketing
      'label', 
      fn.expr("concat('Cluster ', cluster, ': r=', r_bin, ', f=', f_bin, ', m=', m_bin )")
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('demos.rfmp_segmentation.rfmp_clusters')
  )
 
display(spark.table('demos.rfmp_segmentation.rfmp_clusters'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🧠 RFMP Segment Names
# MAGIC
# MAGIC **Cluster 0 – “High-Value Loyalists of Everyday Decor”**
# MAGIC
# MAGIC * `[r=4, f=3, m=3]`
# MAGIC * 🔥 High engagement across **Tealight Essentials**, **Heart Motif Decor**, and **Novelty Bathroomware**
# MAGIC * ✅ These are repeat champions, consistently buying in popular categories. They're ideal for loyalty programs, exclusive drops, and premium bundling.
# MAGIC
# MAGIC **Cluster 1 – “Engaged Spenders of Seasonal Favorites”**
# MAGIC
# MAGIC * `[r=1, f=2, m=3]`
# MAGIC * 🕯️ Strong traction in **Candleware**, **Bakeware**, and **Heart Motif Decor**
# MAGIC * ⚡ Show strong intent but may have paused recently. Ideal for seasonal or holiday promotions and curated product highlights.
# MAGIC
# MAGIC **Cluster 2 – “Fresh Browsers of Giftware & Decor”**
# MAGIC
# MAGIC * `[r=4, f=2, m=1]`
# MAGIC * 🎁 Touchpoints with **Decorative Stationery**, **Photoframes**, and **Heart Motif Decor**, but lower spend
# MAGIC * 📈 These are newer, budget-conscious customers with cross-category interest. Nurture with tiered promotions, product education, and cart builders.
# MAGIC
# MAGIC **Cluster 3 – “Infrequent Big Spenders in Tealights”**
# MAGIC
# MAGIC * `[r=3, f=0, m=3]`
# MAGIC * 💡 Spikes in **Tealight Essentials**, but low frequency
# MAGIC * 💸 Luxury or impulse buyers who go big when they shop. Good candidates for personalized seasonal reminders and high-end recommendations.
# MAGIC
# MAGIC **Cluster 4 – “Dormant Whales of Candleware & Wall Decor”**
# MAGIC
# MAGIC * `[r=0, f=0, m=3]`
# MAGIC * 🕯️ Used to buy from **Candleware**, **Wallpiece Selection**, but went cold
# MAGIC * 💤 Once valuable — now inactive. Bring them back with emotional brand storytelling, “we miss you” bundles, or nostalgic reminders.
# MAGIC
# MAGIC **Cluster 5 – “Silent One-Timers”**
# MAGIC
# MAGIC * `[r=0, f=0, m=0]`
# MAGIC * ⚪️ No strong signal across categories — mostly scattered low-intent entries
# MAGIC * 🎯 Use broad campaigns to test reactivation (e.g., flash sale, win-back email flows), but deprioritize for high-cost channels.
# MAGIC
# MAGIC **Cluster 6 – “Low-Value Occasionals in Gifting”**
# MAGIC
# MAGIC * `[r=3, f=0, m=0]`
# MAGIC * 🎀 Minor activity in **Photoframe Picks** and **Gift Decor**
# MAGIC * 📬 Engage with “gift ideas under €20” or mini-bundle campaigns. This group is price-sensitive and may respond to urgency triggers.
# MAGIC
# MAGIC **Cluster 7 – “Engaged Low Spenders of Bathroom & Decor Basics”**
# MAGIC
# MAGIC * `[r=1, f=2, m=1]`
# MAGIC * 🧼 Moderate interest in **Novelty Bathroomware**, **Tealights**, and **Decorative Stationery**
# MAGIC * 📌 Solid foundation for cross-sell, upsell and loyalty-building — ideal for nurturing into Cluster 0 via targeted nudges.
# MAGIC

# COMMAND ----------

# Read orders_silver from delta table
orders_silver_df = spark.table('demos.rfmp_segmentation.orders_silver')

# Convert inputs_pd to Spark DataFrame
inputs_pd_df = spark.createDataFrame(inputs_pd)

# Select necessary columns from inputs_pd
inputs_pd_selected = inputs_pd_df.select('CustomerID', 'Cluster', 'Recency', 'Frequency', 'MonetaryValue')

# Ensure CustomerID columns are of the same type
orders_silver_df = orders_silver_df.withColumn('CustomerID', orders_silver_df['CustomerID'].cast('string'))
inputs_pd_selected = inputs_pd_selected.withColumn('CustomerID', inputs_pd_selected['CustomerID'].cast('string'))

# Join orders_silver with inputs_pd on CustomerID
enriched_orders_df = orders_silver_df.join(inputs_pd_selected, on='CustomerID', how='left')

# Display the enriched DataFrame
display(enriched_orders_df)

# COMMAND ----------

from pyspark.sql.functions import create_map, lit
from itertools import chain

# Define the cluster mapping
cluster_mapping = {
    "0": "High-Value Loyalists of Everyday Decor",
    "1": "Engaged Spenders of Seasonal Favorites",
    "2": "Fresh Browsers of Giftware & Decor",
    "3": "Infrequent Big Spenders in Tealights",
    "4": "Dormant Whales of Candleware & Wall Decor",
    "5": "Silent One-Timers",
    "6": "Low-Value Occasionals in Gifting",
    "7": "Engaged Low Spenders of Bathroom & Decor Basics"
}

# Create a mapping expression
mapping_expr = create_map([lit(x) for x in chain(*cluster_mapping.items())])

# Add the cluster names column
enriched_orders_df = enriched_orders_df.withColumn('ClusterNames', mapping_expr[enriched_orders_df['Cluster']])

# Save the results to the delta table
enriched_orders_df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').saveAsTable('demos.rfmp_segmentation.orders_gold')

# Display the enriched DataFrame
display(enriched_orders_df)

# COMMAND ----------

display(spark.table('demos.rfmp_segmentation.orders_gold'))
