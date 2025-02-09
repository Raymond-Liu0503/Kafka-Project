from kafka import KafkaConsumer
import geocoder
import requests
import json
import pandas as pd
from llama_cpp import Llama
from functools import lru_cache

# Mapbox API Key (Replace with your own key)
MAPBOX_ACCESS_TOKEN = "MAPBOX_API"

@lru_cache(maxsize=1)
def load_llm_model():
    """Loads the LLM model once and caches it for faster inference."""
    print(load_llm_model.cache_info())

    return Llama(model_path="/mnt/c/Users/raymo/Documents/Coding-Programs/KafkaProject/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", 
                 n_gpu_layers=-1, n_ctx=2048)

def get_nearby_restaurants(lat, lng, limit=10):
    """Fetches nearby restaurants using the correct Mapbox API."""
    url = f"https://api.mapbox.com/search/v1/category/restaurant"
    params = {
        "access_token": MAPBOX_ACCESS_TOKEN,
        "proximity": f"{lng},{lat}",  # Mapbox uses longitude first
        "limit": limit,
        "language": "en"  # English results
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "features" in data:
        return data["features"]
    else:
        return None

def process_restaurant_data(restaurants):
    """Extracts relevant restaurant information into a DataFrame."""
    processed_data = []
    for restaurant in restaurants:
        properties = restaurant.get("properties", {})
        processed_data.append({
            "name": properties.get("feature_name", "Unknown"),
            "category": properties.get("poi_category", "Unknown")
        })
    return pd.DataFrame(processed_data)

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "restaurant_data"

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='earliest',
    group_id='restaurant-group'
)

for message in consumer:
    print(f"Received: {message.value.decode('utf-8')}")
    data = json.loads(message.value)
    coordinates = data.get("coordinates")
    taste = data.get("taste")
    llm = load_llm_model()  # Load model only once

    if coordinates:
        latitude, longitude = coordinates
        print(f"Your current GPS coordinates:\nLatitude: {latitude}\nLongitude: {longitude}\n")

        restaurants = get_nearby_restaurants(latitude, longitude)
        if restaurants:
            df = process_restaurant_data(restaurants)
            print(df.head())

            # Prepare LLM prompt
            user_reqs = f"Based on this data, list the best {taste} restaurants."
            messages = [
                {"role": "system", "content": "You are a concise and helpful AI assistant."},
                {"role": "user", "content": f"{user_reqs} \n {df.to_dict()}"}
            ]

            input_tokens = llm.tokenize(messages[0]['content'].encode('utf-8'))
            print(f"Input tokens: {len(input_tokens)}")

            response = llm.create_chat_completion(messages=messages, max_tokens=2048, stop=[])
            print(response['choices'][0]['message']['content'])
        else:
            print("No restaurants found.")
    else:
        print("Unable to retrieve GPS coordinates.")
