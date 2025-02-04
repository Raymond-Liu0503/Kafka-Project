from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import StructType, StringType, FloatType, LongType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("KafkaFinancialConsumer") \
    .getOrCreate()

# Kafka Config
KAFKA_TOPIC = "financial_data"
KAFKA_BROKER = "localhost:9092"

# Read Kafka Stream
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# Convert Kafka value from binary to string
df = df.selectExpr("CAST(value AS STRING) as json_data")

# Define Schema for Financial Data (based on yfinance output)
schema = StructType() \
    .add("Open", FloatType()) \
    .add("High", FloatType()) \
    .add("Low", FloatType()) \
    .add("Close", FloatType()) \
    .add("Volume", FloatType()) \
    .add("Dividends", FloatType()) \
    .add("Stock Splits", FloatType()) \
    .add("symbol", StringType())

# Parse JSON Data
parsed_df = df.withColumn("data", from_json(col("json_data"), schema))

# Flatten the Data by Extracting Columns
flattened_df = parsed_df.select(
    col("data.Open").alias("Open"),
    col("data.High").alias("High"),
    col("data.Low").alias("Low"),
    col("data.Close").alias("Close"),
    col("data.Volume").alias("Volume"),
    col("data.Dividends").alias("Dividends"),
    col("data.Stock Splits").alias("Stock_Splits"),
    col("data.symbol").alias("Symbol")
)

# Write Stream Output to Console
query = flattened_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()