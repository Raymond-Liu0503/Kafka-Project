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

# Define Schema for Financial Data
schema = StructType() \
    .add("c", FloatType()) \
    .add("h", FloatType()) \
    .add("l", FloatType()) \
    .add("n", LongType()) \
    .add("o", FloatType()) \
    .add("t", LongType()) \
    .add("v", LongType()) \
    .add("vw", FloatType()) \
    .add("symbol", StringType())

# Parse JSON Data
parsed_df = df.withColumn("data", from_json(col("json_data"), schema))

# Flatten the Data by Extracting Columns
flattened_df = parsed_df.select(
    col("data.c").alias("Close"),
    col("data.h").alias("High"),
    col("data.l").alias("Low"),
    col("data.n").alias("Num_Trades"),
    col("data.o").alias("Open"),
    col("data.t").alias("Timestamp"),
    col("data.v").alias("Volume"),
    col("data.vw").alias("VWAP"),
    col("data.symbol").alias("Symbol")
)

# Convert 'Timestamp' from milliseconds to a readable format
flattened_df = flattened_df.withColumn("Formatted_Timestamp", to_timestamp((col("Timestamp") / 1000)))

# Write Stream Output to Console
query = flattened_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
