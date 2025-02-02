# Kafka-Project
sudo apt install scala

Zookeeper Kafka environment:
cd kafka_2.13-3.9.0
bin/zookeeper-server-start.sh config/zookeeper.properties

To start Kafka server:
cd kafka_2.13-3.9.0
bin/kafka-server-start.sh config/server.properties

To end Kafka Environment:
rm -rf /tmp/kafka-logs /tmp/zookeeper /tmp/kraft-combined-logs

To create topic:
cd kafka_2.13-3.9.0
bin/kafka-topics.sh --create --topic demo-messages --bootstrap-server localhost:9092

To list topics:
bin/kafka-topics.sh --list --bootstrap-server localhost:9092

Produce events:
cd Kafka-Project/src
python producer.py
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092

Python Environment
source /mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/activate
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip show pyspark
kafka-topics.sh --bootstrap-server localhost:9092 --list

pip install kafka-python-ng
pip show kafka-python-ng
pip install pyspark
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install kafka-python-ng requests
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install influxdb-client
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install 'influxdb-client[ciso]'


export SPARK_HOME=/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/spark-3.5.4-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.4 spark_kafka_consumer.py

