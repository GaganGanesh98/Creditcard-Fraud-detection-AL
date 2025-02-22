# Databricks notebook source
# MAGIC %md
# MAGIC Load and Explore the Data

# COMMAND ----------

# Load the dataset
file_path = "/FileStore/tables/creditcard.csv"  # Update with your file path
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Show the first 5 rows
display(df.limit(5))

# COMMAND ----------

# Print the schema
df.printSchema()

# Count the total number of rows
print("Total rows:", df.count())

# Check the class distribution (fraud vs. non-fraud)
display(df.groupBy("Class").count())

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Data Preprocessing

# COMMAND ----------

df = df.drop("Time")

# COMMAND ----------

# MAGIC %md
# MAGIC Handle Imbalanced Data

# COMMAND ----------

from pyspark.sql.functions import col

# Separate fraud and non-fraud data
fraud_df = df.filter(col("Class") == 1)
non_fraud_df = df.filter(col("Class") == 0)

# Undersample non-fraud data
sampled_non_fraud_df = non_fraud_df.sample(fraction=0.01, seed=42)

# Combine the datasets
balanced_df = fraud_df.union(sampled_non_fraud_df)

# Check the new class distribution
display(balanced_df.groupBy("Class").count())

# COMMAND ----------

# MAGIC %md
# MAGIC scale the features

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=["Amount"] + [f"V{i}" for i in range(1, 29)], outputCol="features")
df = assembler.transform(df)

# Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Show the transformed DataFrame
display(df.select("scaledFeatures", "Class").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Train a Machine Learning Model

# COMMAND ----------

# MAGIC %md
# MAGIC Split the Data

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Train a Logistic Regression Model

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Initialize the model
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="Class")

# Train the model
lr_model = lr.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the Model

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Make predictions
predictions = lr_model.transform(test_df)

# Evaluate using AUC-ROC
evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc}")

# COMMAND ----------

# MAGIC %md
# MAGIC getting an AUC-ROC score of 0.9687 (96.87%) accuracy.
