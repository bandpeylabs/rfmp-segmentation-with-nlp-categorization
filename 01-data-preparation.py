# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Introduction
# MAGIC
# MAGIC Not all customers contribute equally to a retailer’s success. Recognizing this variation enables smarter, more profitable customer engagement. But how can we identify which customers hold the highest potential, and how their behavior, frequency of purchases, and product preferences contribute to long-term value?
# MAGIC
# MAGIC While many organizations turn to Customer Lifetime Value (CLV) models to answer these questions, such methods can be complex, data-hungry, and often overkill for tactical segmentation. Instead, a simpler and more interpretable method, RFMP segmentation, can offer actionable insights by analyzing four key behavioral signals: Recency, Frequency, Monetary value, and Product affinity.
# MAGIC
# MAGIC RFMP extends the classic RFM framework by adding an understanding of what customers purchase, not just how often or how much. This allows retailers to create customer groups that reflect not only value, but also product category preferences, paving the way for more targeted marketing, better inventory alignment, and personalized experiences.
# MAGIC
# MAGIC In this project, we demonstrate how to implement RFMP segmentation by first clustering product descriptions using modern NLP techniques to infer categories, and then calculating RFMP scores for each customer. The result is a practical and powerful segmentation approach that marketing and analytics teams can operationalize quickly.

# COMMAND ----------

# MAGIC %pip install openpyxl==3.1.5

# COMMAND ----------

# MAGIC %sh 
# MAGIC  
# MAGIC rm -rf /dbfs/tmp/clv/online_retail  # drop any old copies of data
# MAGIC mkdir -p /dbfs/tmp/clv/online_retail # ensure destination folder exists
# MAGIC  
# MAGIC # download data to destination folder
# MAGIC wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx -P /dbfs/tmp/clv/online_retail

# COMMAND ----------

import numpy as np
import pandas as pd

xlsx_filename = "/dbfs/tmp/clv/online_retail/Online Retail.xlsx"
 
# schema of the excel spreadsheet data range
orders_schema = {
  'InvoiceNo':str,
  'StockCode':str,
  'Description':str,
  'Quantity':np.int64,
  'InvoiceDate':np.datetime64,
  'UnitPrice':np.float64,
  'CustomerID':str,
  'Country':str  
  }
 
# read spreadsheet to pandas dataframe
# the openpyxl library must be installed for this step to work 
orders_pd = pd.read_excel(
  xlsx_filename, 
  sheet_name='Online Retail',
  header=0, # first row is header
  dtype=orders_schema
  )
 
# calculate sales amount as quantity * unit price
orders_pd['SalesAmount'] = orders_pd['Quantity'] * orders_pd['UnitPrice']
 
# display dataset
display(orders_pd)

# COMMAND ----------

# Convert pandas dataframe to spark dataframe
orders_spark_df = spark.createDataFrame(orders_pd)

# COMMAND ----------

# Create schema if not exists
spark.sql("CREATE SCHEMA IF NOT EXISTS demos.rfmp_segmentation")

# COMMAND ----------

# Write the dataframe to a Delta table
orders_spark_df.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("demos.rfmp_segmentation.orders_bronze")

# COMMAND ----------

# DBTITLE 1,Load the Bronze Table
orders_bronze_df = spark.read.table("demos.rfmp_segmentation.orders_bronze")
display(orders_bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #  Build a TF-IDF pipeline in Spark
# MAGIC - Loads the bronze table (orders_bronze)
# MAGIC - ...

# COMMAND ----------

# DBTITLE 1,Read bronze and extract unique descriptions
# We’ll cluster _distinct_ product descriptions. If you only want to 
# cluster each unique product once, do .distinct(). Otherwise cluster all rows.
distinct_desc_df = orders_bronze_df.select("Description").where("Description IS NOT NULL").distinct()

display(distinct_desc_df)

# COMMAND ----------

# MAGIC %md
# MAGIC - embed is now a Spark UDF.
# MAGIC - If you we the above “emb_df” snippet, you should see one row with a 768‐dimensional float array → that confirms BERT loaded correctly.
# MAGIC - This is too much for KMeans, so let's store and the later reduce dimention with PCA

# COMMAND ----------

# DBTITLE 1,Tokenize + remove English stop words → raw “words” column
from pyspark.ml.feature import Tokenizer, StopWordsRemover

tokenizer = Tokenizer(inputCol="Description", outputCol="words_raw")
stop_remover = StopWordsRemover(inputCol="words_raw", outputCol="words")

tokenized_df = tokenizer.transform(distinct_desc_df)
clean_df     = stop_remover.transform(tokenized_df)

# Show token lists
display(clean_df.select("Description", "words").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC At this point...

# COMMAND ----------

# DBTITLE 1,HashingTF → “tf_features”  (sparse term frequencies)
# followed by IDF → “tfidf_features”
from pyspark.ml.feature import HashingTF, IDF

hashing_tf = HashingTF(
    inputCol="words",
    outputCol="tf_features",
    numFeatures=4096    # you can adjust (e.g. 2048, 4096, 8192). 4096 is a decent trade-off.
)

# First apply HashingTF to get raw counts
tf_df = hashing_tf.transform(clean_df)

idf = IDF(inputCol="tf_features", outputCol="tfidf_features", minDocFreq=2)
idf_model = idf.fit(tf_df)          # fit on all distinct descriptions
tfidf_df = idf_model.transform(tf_df)

# tfidf_df now contains “tfidf_features” column of type Vector (dim = 4096)
display(tfidf_df.select("Description", "tfidf_features").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Dimensionality Reduction
# MAGIC
# MAGIC Clustering directly on a 4096-dim TF-IDF can work, but if you want to speed things up further, reduce to ~50 or 100 dims first using PCA or TruncatedSVD.

# COMMAND ----------

# DBTITLE 1,Using Spark PCA (distributed)
# SparkPCA to cut down 4096 → 50
from pyspark.ml.feature import PCA as SparkPCA

pca = SparkPCA(
    k=50,
    inputCol="tfidf_features",
    outputCol="pca_features"
)
pca_model = pca.fit(tfidf_df)
pca_df    = pca_model.transform(tfidf_df)

# “pca_features” is now a Vector of length 50
display(pca_df.select("Description", "pca_features").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # KMeans on TF-IDF or PCA output
# MAGIC
# MAGIC Cluster on the 50-dim pca_features.

# COMMAND ----------

# DBTITLE 1,KMeans (k=10) on the 50-dim PCA vectors
# Cell 5: KMeans (k=10) on the 50-dim PCA vectors
from pyspark.ml.clustering import KMeans

kmeans = KMeans(
    k=10,
    seed=42,
    featuresCol="pca_features",
    predictionCol="ProductCategoryID"
)
kmeans_model = kmeans.fit(pca_df)
clusters_df  = kmeans_model.transform(pca_df)

# clusters_df schema: (Description: string, words_raw, words, tf_features, 
#                      tfidf_features, pca_features, ProductCategoryID: int)
display(clusters_df.select("Description", "ProductCategoryID").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Inspect cluster contents

# COMMAND ----------

# DBTITLE 1,Manually Map Cluster IDs → Category Names
# Cell 6: show a few sample descriptions per cluster
for cid in range(10):
    print(f"\nCluster {cid} samples:")
    clusters_df.filter(f"ProductCategoryID = {cid}") \
               .select("Description") \
               .limit(10) \
               .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Join cluster IDs back to orders
# MAGIC
# MAGIC Now that each unique Description has a ProductCategoryID, join back onto your full orders table and write out a Silver (or Gold) version:

# COMMAND ----------

# DBTITLE 1,Attach cluster names and save lookup to Silver
from pyspark.sql.functions import col

# 1) Read the original bronze table
orders_bronze_df = spark.read.table("demos.rfmp_segmentation.orders_bronze")

# 2) Define the mapping from ProductCategoryID → ProductCategoryName
cluster_name_list = [
    (0, "Candleware Collection"),
    (1, "Decorative Stationery"),
    (2, "Tealight Essentials"),
    (3, "Gardenware Assortment"),
    (4, "Novelty Bathroomware"),
    (5, "Heart Motif Decor"),
    (6, "Vintage Bagline"),
    (7, "Wallpiece Selection"),
    (8, "Photoframe Picks"),
    (9, "Bakeware Series")
]
cluster_names_df = spark.createDataFrame(
    cluster_name_list,
    schema=["ProductCategoryID", "ProductCategoryName"]
)

# 3) clusters_df (from the KMeans step) must contain: Description, ProductCategoryID
#    Join orders_bronze to clusters_df to attach ProductCategoryID
orders_with_id = orders_bronze_df.join(
    clusters_df.select("Description", "ProductCategoryID"),
    on="Description",
    how="left"
)

# 4) Now join cluster_names_df to attach ProductCategoryName
orders_with_cat = orders_with_id.join(
    cluster_names_df,
    on="ProductCategoryID",
    how="left"
)

# 5) Write out the enriched DataFrame as a new Silver table
orders_with_cat \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .saveAsTable("demos.rfmp_segmentation.orders_silver")
