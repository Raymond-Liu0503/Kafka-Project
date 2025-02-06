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
source /mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/activate
python producer.py
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092

To consume using spark:
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.4 spark_kafka_consumer.py

Python Environment
source /mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/activate
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip show pyspark
kafka-topics.sh --bootstrap-server localhost:9092 --list

pip install kafka-python-ng
pip show kafka-python-ng
pip install pyspark
sudo apt install postgresql
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install tensorflow
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install tensorflowonspark
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install scikit-learn

/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install yfinance
pip list | grep pyspark
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install pyspark
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install alpha_vantage
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install pmdarima
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install matplotlib


/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install kafka-python-ng requests
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install influxdb-client
/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/kafka-proj/bin/pip install 'influxdb-client[ciso]'


echo 'export SPARK_HOME=/usr/local/spark' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc
source ~/.bashrc

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.4 spark_kafka_consumer.py


PostgreSQL

User: Raymond
Password: Ciena
DB: bitcoin_data

Setup:
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -i -u postgres
psql


CREATE DATABASE mydb;
CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;

To connect:
psql -U myuser -d mydb -h localhost -p 5432

To test connection:
sudo systemctl status postgresql

docker exec -it postgres bash
psql -U Raymond -d bitcoin_data

Alpha Vantage: 7HX5TK1UEV1V9TT5