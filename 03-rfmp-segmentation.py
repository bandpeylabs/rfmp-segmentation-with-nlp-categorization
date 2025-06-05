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
# MAGIC ## Convert the relevant columns to Pandas for t-SNE

# COMMAND ----------

# Select only the columns we need for t-SNE (and any identifiers for later)
tsne_source_df = gold_df.select(
    "CustomerID", 
    "ProductCategoryID", 
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
inputs_pd[["CustomerID", "ProductCategoryID", "r_bin", "f_bin", "m_bin", "tsne_one", "tsne_two"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize t-SNE colored by each bin (r_bin, f_bin, m_bin)

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
        ax=axes[i],
    )

plt.tight_layout()
plt.show()

# COMMAND ----------


