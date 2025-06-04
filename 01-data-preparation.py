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
# MAGIC - Loads the bronze table (orders_bronze)
# MAGIC - Uses bert_base_uncased as a Spark UDF to compute an embedding vector for each distinct “Description”
# MAGIC - Joins those embeddings back into the original DataFrame
# MAGIC - Writes that augmented DataFrame (with one extra embedding column) out as our silver table (orders_silver)
# MAGIC 	
# MAGIC ⸻
# MAGIC
# MAGIC 📒 Prerequisites
# MAGIC - You have already installed gte_small (Marketplace model) into your workspace under the Catalog databricks_gte_models.
# MAGIC - You are running on a Databricks cluster with at least ML Runtime 14.x (or a runtime that supports mlflow.pyfunc.spark_udf).
# MAGIC - Your cluster already has scikit-learn and pandas available (if not, you can install them in a separate %pip install ... cell).

# COMMAND ----------

# DBTITLE 1,Define & Register the BERT UDF
# Cell 2 (fixed): skip null/empty Descriptions before tokenizing

from transformers import AutoTokenizer, AutoModel
import torch

# 1) Load tokenizer & model once
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

# 2) Define a safe embedding function that returns a 768‐dim zero vector if input is null/empty
ZERO_VEC = [0.0] * 768

def embed_text(text: str) -> list[float]:
    if not text:  # catches None or empty string
        return ZERO_VEC
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb.tolist()

# 3) Register as Spark UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

embed_udf = udf(embed_text, ArrayType(FloatType()))
spark.udf.register("embed_text", embed_text, ArrayType(FloatType()))

# 4) Test on a DataFrame instead of a single string
test_df = spark.createDataFrame(
    [("WHITE HANGING HEART T-LIGHT HOLDER",), ("RED WOOLLY HOTTIE WHITE HEART",)],
    ["Description"]
)

emb_df = test_df.withColumn("embedding", embed_udf("Description"))
display(emb_df)

# COMMAND ----------

# MAGIC %md
# MAGIC - embed is now a Spark UDF.
# MAGIC - If you we the above “emb_df” snippet, you should see one row with a 768‐dimensional float array → that confirms BERT loaded correctly.
# MAGIC - This is too much for KMeans, so let's store and the later reduce dimention with PCA

# COMMAND ----------

# DBTITLE 1,Generate & Persist 768-dim Embeddings (Silver)
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType


embed_udf = udf(embed_text, ArrayType(FloatType()))

orders_bronze_df = spark.read.table("demos.rfmp_segmentation.orders_bronze")

orders_silver = (
    orders_bronze_df
      .withColumn("embedding", embed_udf("Description"))
      .write
      .mode("overwrite")
      .format("delta")
      .saveAsTable("demos.rfmp_segmentation.orders_silver")
)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point, orders_silver has schema:
# MAGIC ```
# MAGIC root
# MAGIC  |-- Description: string (nullable = true)
# MAGIC  |-- embedding: array<float> (nullable = true, length 768)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Run SparkPCA on the 768-dim Column
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType

# 2.1) Read your silver table
silver_df = spark.read.table("demos.rfmp_segmentation.orders_silver") \
                .select("Description", "embedding") \
                .distinct()  # one row per distinct Description

# 2.2) Convert the Python array<float> → Spark Dense Vector
to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())

vector_df = silver_df.withColumn("features", to_vector_udf(col("embedding")))

# 2.3) Run PCA (reduce 768 → e.g. 50 dimensions)
pca = SparkPCA(k=50, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(vector_df)
pca_df = pca_model.transform(vector_df) \
                  .select("Description", "pca_features")

display(pca_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Write Out the Silver Table (with Embeddings)
# 5.1 Write the augmented DataFrame to a Delta table as "orders_silver"
(
    orders_silver_intermediate
      .write
      .mode("overwrite")
      .format("delta")
      .saveAsTable("demos.rfmp_segmentation.orders_silver")
)

print("✅ Silver table created: demos.rfmp_segmentation.orders_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we have all embeddings stored in orders_silver, the usual workflow is:
# MAGIC - Collect only (Description, embedding) from orders_silver into a Pandas DataFrame
# MAGIC - Run KMeans (10 clusters) locally, assign each distinct Description → ProductCategoryID
# MAGIC - Manually name each cluster (e.g. “Lighting,” “Home Decor,” etc.)
# MAGIC - Create a small Spark lookup of (Description → ProductCategoryID, ProductCategoryName)
# MAGIC - Join that lookup back onto orders_silver → you get an enriched DataFrame with two new columns
# MAGIC   - ProductCategoryID (Int 0..9)
# MAGIC   - ProductCategoryName (string)
# MAGIC - Write that final DataFrame out as the gold table orders_gold

# COMMAND ----------

# DBTITLE 1,Collect Embeddings Locally & Cluster in Pandas/Scikit-Learn
# Cell 6: Collect (Description, embedding) to Pandas for clustering
import pandas as pd

# 6.1 Read only the two columns from silver
silver_df = spark.read.table("demos.rfmp_segmentation.orders_silver") \
                    .select("Description", "embedding") \
                    .distinct() \
                    .na.drop(subset=["embedding"])

# 6.2 Convert to Pandas (embedding is array<float>, Pandas will see it as list[float])
silver_pd = silver_df.toPandas()

# 6.3 Split out embeddings into a 2D array for clustering
import numpy as np

emb_matrix = np.vstack(silver_pd["embedding"].values)  
print("emb_matrix shape:", emb_matrix.shape)  # should be (num_distinct_desc, 384)

# 6.4 Run KMeans for 10 clusters
from sklearn.cluster import KMeans

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
silver_pd["ProductCategoryID"] = kmeans.fit_predict(emb_matrix)

# 6.5 Inspect a few examples per cluster, so you can name them manually
for cid in range(n_clusters):
    sample_texts = silver_pd[silver_pd.ProductCategoryID == cid]["Description"] \
                                .sample(5, random_state=42).tolist()
    print(f"\nCluster {cid} samples:\n", sample_texts)

# COMMAND ----------

# DBTITLE 1,Manually Map Cluster IDs → Category Names
# Cell 7: after inspecting the cluster samples, fill in your own names:
category_name_map = {
    0: "Lighting",
    1: "Home Decor",
    2: "Kitchen & Tableware",
    3: "Toys & Crafts",
    4: "Seasonal Goods",
    5: "Textiles & Linens",
    6: "Office & Stationery",
    7: "Garden & Outdoors",
    8: "Bathroom & Wellness",
    9: "Miscellaneous"
}

# Convert that mapping into a tiny Pandas DataFrame
mapping_pd = pd.DataFrame({
    "ProductCategoryID": list(category_name_map.keys()),
    "ProductCategoryName": list(category_name_map.values())
})

display(mapping_pd)

# COMMAND ----------

# DBTITLE 1,Create a Spark Lookup for Description → (CategoryID, CategoryName)
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 8.1 Build a DataFrame: [ Description, ProductCategoryID ] from silver_pd
schema_desc_map = StructType([
    StructField("Description", StringType(), nullable=False),
    StructField("ProductCategoryID", IntegerType(), nullable=False)
])

desc_rows = [
    Row(Description=row["Description"], ProductCategoryID=int(row["ProductCategoryID"]))
    for _, row in silver_pd.iterrows()
]
desc_map_spark = spark.createDataFrame(desc_rows, schema_desc_map)

# 8.2 Build a DataFrame: [ ProductCategoryID, ProductCategoryName ] from mapping_pd
schema_cat_name = StructType([
    StructField("ProductCategoryID", IntegerType(), nullable=False),
    StructField("ProductCategoryName", StringType(), nullable=False)
])

catname_rows = [
    Row(ProductCategoryID=int(r.ProductCategoryID), ProductCategoryName=r.ProductCategoryName)
    for r in mapping_pd.itertuples()
]
catname_map_spark = spark.createDataFrame(catname_rows, schema=schema_cat_name)

# 8.3 Join them so we have [ Description, ProductCategoryID, ProductCategoryName ]
description_to_cat_spark = desc_map_spark.join(
    catname_map_spark,
    on="ProductCategoryID",
    how="left"
)

display(description_to_cat_spark.limit(10))

# COMMAND ----------

# DBTITLE 1,Join Categories Back onto Silver → Enrich & Write Gold
# Cell 9: read silver table, then join on Description → add two new columns
orders_silver_df = spark.read.table("demos.rfmp_segmentation.orders_silver")

# 9.1 Join to bring in both ID and Name
orders_with_final_cat = orders_silver_df.join(
    description_to_cat_spark,
    on="Description",
    how="left"
)

display(orders_with_final_cat.limit(10))

# COMMAND ----------

# 9.2 Write out the “gold” table (final RFMP-ready table)
(
    orders_with_final_cat
      .write
      .mode("overwrite")
      .format("delta")
      .saveAsTable("demos.rfmp_segmentation.orders_gold")
)

print("✅ Gold table created: demos.rfmp_segmentation.orders_gold")
