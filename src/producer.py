from kafka import KafkaProducer
import requests
import json
import time

# Kafka configuration
KAFKA_BROKER = "localhost:9092"  # Change to your broker address
TOPIC_NAME = "financial_data"

# Polygon API configuration
POLYGON_API_KEY = "5yJIkZulShVcF4tAPIhcXwB0Mp5dr6za"  # Replace with your API key
SYMBOL = str(input("Enter stock symbol: "))  # Stock symbol to track
POLYGON_URL = f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/prev?apiKey={POLYGON_API_KEY}"

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def fetch_stock_data():
    """Fetch stock data from Polygon API"""
    response = requests.get(POLYGON_URL)
    if response.status_code == 200:
        data = response.json()
        return data["results"][0] if "results" in data else None
    else:
        print(f"Error fetching data: {response.text}")
        return None

def send_to_kafka():
    """Fetch data from Polygon and send it to Kafka"""
    while True:
        stock_data = fetch_stock_data()
        if stock_data:
            stock_data["symbol"] = SYMBOL
            producer.send(TOPIC_NAME, stock_data)
            print(f"Sent to Kafka: {stock_data}")
        time.sleep(12)  # Fetch data every 10 seconds

if __name__ == "__main__":
    send_to_kafka()
