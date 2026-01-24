import requests
import sys

BASE_URL = "http://localhost:8000"

def test_search():
    print("Testing /search?q=shirt...")
    try:
        r = requests.get(f"{BASE_URL}/search?q=shirt&limit=5")
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            print(f"Success. Found {len(results)} results.")
            if results:
                print(f"Sample: {results[0]['name']} (ID: {results[0]['id']})")
                return results[0]['id']
        else:
            print(f"Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error: {e}")
    return None

def test_recommend(product_id):
    if not product_id:
        print("Skipping recommendation test (no product ID).")
        return

    print(f"Testing /recommend/{product_id}...")
    try:
        r = requests.get(f"{BASE_URL}/recommend/{product_id}")
        if r.status_code == 200:
            data = r.json()
            recs = data.get("recommendations", [])
            print(f"Success. Found {len(recs)} recommendations.")
        else:
            print(f"Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pid = test_search()
    test_recommend(pid)
