# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis
# MAGIC ## Load Silver orders (with category IDs and names)

# COMMAND ----------

# DBTITLE 1,Load Silver orders (with category IDs and names)
orders_silver_df = spark.read.table("demos.rfmp_segmentation.orders_silver")
orders_silver_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consolidate positive‐sales transactions by customer, category, and date

# COMMAND ----------

from pyspark.sql import functions as fn

orders_consolidated = (
    orders_silver_df
      .filter("SalesAmount > 0.0")                                # exclude returns
      .withColumn("InvoiceDate", fn.to_date("InvoiceDate"))       # drop time component
      .groupBy("CustomerID", "ProductCategoryID", "ProductCategoryName", "InvoiceDate")
        .agg(fn.sum("SalesAmount").alias("DailySales"))           # sum per customer‐category‐date
)

display(orders_consolidated)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute R, F, M for each (CustomerID, ProductCategoryID, ClusterName)

# COMMAND ----------

# 3.1 Find the global “last date” in this consolidated dataset
last_date = (
    orders_consolidated
      .agg(fn.max("InvoiceDate").alias("GlobalLastDate"))
)

# 3.2 Cross‐join so each row knows the GlobalLastDate, then compute per‐group metrics
rfmp_metrics = (
    orders_consolidated
      .crossJoin(last_date)
      .groupBy("CustomerID", "ProductCategoryID", "ProductCategoryName")
        .agg(
          # Recency: days since the most recent purchase date in this (cust, cat) group
          fn.min(fn.datediff("GlobalLastDate", "InvoiceDate")).alias("Recency"),
          # Frequency: count of unique purchase dates for this (cust, cat) group
          fn.countDistinct("InvoiceDate").alias("Frequency"),
          # Monetary: average daily sales (since we've already summed per‐date)
          fn.avg("DailySales").alias("MonetaryValue")
        )
)

display(rfmp_metrics)

# COMMAND ----------

import matplotlib.pyplot as plt

# send metrics to pandas for visualizations
df_pd = rfmp_metrics.toPandas()
 
# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))
 
# generate one chart per metric
for i, metric in enumerate(['Recency', 'Frequency', 'MonetaryValue']):
   
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define histogram chart
  axes[i].hist(df_pd[metric], bins=10)

# COMMAND ----------

# MAGIC %md
# MAGIC From these visualizations, we can see that outliers are affecting the distributions of our frequency and monetary value metrics. If we were to dig into the dataset a bit, we'd see that a few high-frequency purchasers and a few (incredibly) high spend customers are creating this. With more context, we might determine how best to handle these outliers, but without this, we might simply put a cap on values for these metric as follows:

# COMMAND ----------

# Cell 4: Cap outlier values (optional, analogous to RFM cleansing)
rfmp_metrics_cleansed = (
    rfmp_metrics
      .withColumn(
        "Frequency",
        fn.expr("CASE WHEN Frequency > 30 THEN 30 ELSE Frequency END")
      )
      .withColumn(
        "MonetaryValue",
        fn.expr("CASE WHEN MonetaryValue > 2500 THEN 2500 ELSE MonetaryValue END")
      )
)

display(rfmp_metrics_cleansed)

# COMMAND ----------

# extract metrics to pandas for visualization
df_pd = rfmp_metrics_cleansed.toPandas()
 
# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))
 
# generate one chart per metric
for i, metric in enumerate(['Recency', 'Frequency', 'MonetaryValue']):
   
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define chart
  axes[i].hist(df_pd[metric], bins=10)


# COMMAND ----------

# MAGIC %md
# MAGIC Having finalized our metrics, we'll extract these data to a pandas dataframe as this will better align with the remaining work we are to perform. If you have too much data for a pandas dataframe, everything we are doing in the remainder of this notebook can be recreated using capabilities in Spark MLLib. However, we believe it is easier to perform this work using sklearn and that you will find more examples online implemented using that library. Having consolidated your data to a few metrics per customer identifier, a pandas dataframe should be more than sufficient for many 10s of millions of customers on a good sized cluster. Should you experience memory pressures with a pandas dataframe, consider taking a random sample to support model training:

# COMMAND ----------

inputs_pd = rfmp_metrics_cleansed.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Organize Customers into Quantiles
# MAGIC With our metrics properly cleansed, we can assign our different recency, frequency and monetary values to bins. In doing this, we need to consider both the number of bins to employ and the number of values to assign to each.
# MAGIC
# MAGIC In RFM segmentation, we typically select either 5 or 10 bins per metric. Our goal with this exercise is not to optimize our cluster design from a statistical point view but instead to find a workable number of segments that our marketing team can effectively employ. With 5 bins, we have 125 potential groupings and with 10 bins, we have 1,000 potential groupings. In practice, we will often find a smaller number of meaningful clusters from these possible combinations, but already the smaller of these values is excessive for our needs. So, we'll start with 5 bins per metric. If we don't find good clustering results later, we could return to this step and bump the quantiles up to 10.
# MAGIC
# MAGIC To divide our metrics into bins, we can use a number of strategies. With a uniform binning strategy, each bin has a consistent width, much like within the histograms above. With a quantile strategy, bins have a variable width so that values are distributed relatively evenly between each bin. As our goal is to differentiate between users based on relative value, the quantile approach seems to be more appropriate:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Binning Logic

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

# MAGIC %md
# MAGIC One quick note before implementing the binning is that our frequency and monetary value metrics reflect better customer performance as their values increase. Separately, our recency values indicate lower performance as its values increase. To help our marketing team make sense of these values, we'll reverse the bin ordinals we calculate on recency so that as the recency bin increases it too reflects better customer performance:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Binning Logic

# COMMAND ----------

# invert the recency values so that higher is better
inputs_pd['Recency'] = inputs_pd['Recency'] * -1
 
# bin the data
bins = col_trans.fit_transform(inputs_pd)
 
# add bins to input data
inputs_pd['r_bin'] = bins[:,0]
inputs_pd['f_bin'] = bins[:,1]
inputs_pd['m_bin'] = bins[:,2]
 
# display dataset
display(inputs_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Bin Distributions

# COMMAND ----------

# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))
 
for i, metric in enumerate(['r_bin','f_bin','m_bin']):
   
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define chart
  axes[i].hist(inputs_pd[metric], bins=5)

# COMMAND ----------

# MAGIC %md
# MAGIC From the visualizations, we can see that our bins are not perfectly even and, in the case of frequency, we even have a gap in values. This is to be expected with metrics such as RFM that demonstrate high degrees of skew.

# COMMAND ----------

# Convert pandas DataFrame to Spark DataFrame
inputs_spark_df = spark.createDataFrame(inputs_pd)

# Write the DataFrame to a Gold table
inputs_spark_df.write.format("delta").mode("overwrite").saveAsTable("demos.rfmp_segmentation.gold")
