from pyspark.sql import SparkSession

# Start a Spark session with the Kafka connector
spark = SparkSession.builder \
    .appName("KafkaSparkIntegration") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka_2.13:3.9.0") \
    .getOrCreate()

# Read data from Kafka (change topic and broker accordingly)
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "test-topic") \
    .load()

# Select only the "value" column (Kafka message)
messages = kafka_stream.selectExpr("CAST(value AS STRING)")

# Write data to the console for testing
query = messages.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
