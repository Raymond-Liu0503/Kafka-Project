# Kafka-Project

Zookeeper Kafka environment:
bin/zookeeper-server-start.sh config/zookeeper.properties

To start Kafka server:
bin/kafka-server-start.sh config/server.properties

To end Kafka Environment:
rm -rf /tmp/kafka-logs /tmp/zookeeper /tmp/kraft-combined-logs

To create topic:
cd kafka_2.13-3.9.0
bin/kafka-topics.sh --create --topic demo-messages --bootstrap-server localhost:9092

To list topics:
bin/kafka-topics.sh --list --bootstrap-server localhost:9092

Produce events:
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092

Python Environment
source /mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/activate
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install pyspark

pip install kafka-python-ng
pip show kafka-python-ng
pip install pyspark