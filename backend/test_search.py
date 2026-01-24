import requests
import json

def test_text_search():
    url = "http://127.0.0.1:8000/search"
    query = "something for summer party"
    params = {"q": query, "limit": 3}
    
    try:
        print(f"Searching for: '{query}'...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            results = response.json()
            print("\nSearch Results:")
            print(json.dumps(results, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_text_search()
