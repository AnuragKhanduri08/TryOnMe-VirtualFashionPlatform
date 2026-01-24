import requests
import time
import random

BASE_URL = "http://localhost:8000"

ENDPOINTS = [
    "/products",
    "/search?q=dress",
    "/search?q=shirt",
    "/recommend/1",
    "/recommend/2",
    "/health"
]

def generate_traffic():
    print("Generating traffic for dashboard...")
    for i in range(20):
        endpoint = random.choice(ENDPOINTS)
        try:
            url = f"{BASE_URL}{endpoint}"
            print(f"Requesting {url}")
            requests.get(url)
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(0.5)
    print("Traffic generation complete.")

if __name__ == "__main__":
    generate_traffic()
