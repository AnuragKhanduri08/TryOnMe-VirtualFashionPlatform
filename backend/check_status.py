import requests
import json

BASE_URL = "http://localhost:8000"

def check_backend():
    print("Checking Backend Health...")
    try:
        res = requests.get(f"{BASE_URL}/health")
        print(f"Health: {res.status_code} - {res.json()}")
    except Exception as e:
        print(f"Backend down: {e}")
        return

    print("\nChecking Products...")
    try:
        res = requests.get(f"{BASE_URL}/products")
        products = res.json()
        print(f"Product Count: {len(products)}")
        if len(products) > 0:
            p = products[0]
            print(f"Sample Product ID: {p['id']}")
            print(f"Name: {p['name']}")
            print(f"Category: {p['category']}")
            print(f"Description: {p['description']}")
            print(f"Image: {p['image_url']}")
    except Exception as e:
        print(f"Products failed: {e}")

    print("\nChecking Search...")
    try:
        res = requests.get(f"{BASE_URL}/search?q=shirt&limit=3")
        print(f"Search Results: {len(res.json()['results'])}")
        
        print("Checking Autocomplete...")
        res = requests.get(f"{BASE_URL}/search/suggestions?q=shi&limit=5")
        sug = res.json().get('suggestions', [])
        print(f"Suggestions for 'shi': {sug}")
    except Exception as e:
        print(f"Search failed: {e}")
        
    print("\nChecking Recommendations...")
    try:
        # Use first product ID
        if len(products) > 0:
            pid = products[0]['id']
            res = requests.get(f"{BASE_URL}/recommend/{pid}")
            print(f"Recommendations for {pid}: {len(res.json()['recommendations'])}")
    except Exception as e:
        print(f"Recommendations failed: {e}")

if __name__ == "__main__":
    check_backend()
