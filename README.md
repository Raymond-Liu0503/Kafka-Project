# Kafka-Project

Zookeeper Kafka environment:
bin/zookeeper-server-start.sh config/zookeeper.properties

To start Kafka server:
bin/kafka-server-start.sh config/kraft/server.properties

To end Kafka Environment:
rm -rf /tmp/kafka-logs /tmp/zookeeper /tmp/kraft-combined-logs

To create topic:
cd kafka_2.13-3.9.0
bin/kafka-topics.sh --create --topic demo-messages --bootstrap-server localhost:9092

Produce events:
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092