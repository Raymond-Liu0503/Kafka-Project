import json
import time
import yfinance as yf
from kafka import KafkaProducer

# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "financial_data"

# Cryptocurrency Symbol (e.g., BTC-USD, ETH-USD)
CRYPTO_SYMBOL = "BTC-USD"

# Create Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")  # Serialize messages to JSON
)

# Function to fetch cryptocurrency data using yfinance
def fetch_crypto_data(symbol):
    try:
        # Fetch data for the given symbol
        crypto = yf.Ticker(symbol)
        # Get historical market data for the last minute
        data = crypto.history(period="1mo")
        if not data.empty:
            # Convert the latest data point to a dictionary
            latest_data = data.iloc[-1].to_dict()
            latest_data["symbol"] = symbol  # Add symbol to the data
            return latest_data
        else:
            print(f"No data found for symbol: {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")
        return None

# Function to send data to Kafka
def send_to_kafka(data):
    if data:
        try:
            # Send data to Kafka topic
            producer.send(KAFKA_TOPIC, value=data)
            producer.flush()  # Ensure the message is sent
            print(f"Sent to Kafka: {data}")
        except Exception as e:
            print(f"Error sending data to Kafka: {e}")

# Main loop to fetch and send data periodically
def main():
    while True:
        data = fetch_crypto_data(CRYPTO_SYMBOL)
        if data:
            send_to_kafka(data)
        time.sleep(10)  # Fetch data every 60 seconds

if __name__ == "__main__":
    main()