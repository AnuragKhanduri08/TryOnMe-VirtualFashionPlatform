import requests
import json
import sys

def test_search(query):
    url = "http://localhost:8000/search"
    payload = {"query": query, "limit": 5}
    try:
        # Use params for GET request
        response = requests.get(url, params={"q": query, "limit": 5, "use_ai": True})
        response.raise_for_status()
        results = response.json()
        print(f"Query: '{query}'")
        print(f"Status Code: {response.status_code}")
        print(f"Results Found: {len(results.get('results', []))}")
        print("First result sample:")
        if results.get('results'):
             print(json.dumps(results['results'][0], indent=2))
        return results
    except Exception as e:
        print(f"Error testing search: {e}")
        if hasattr(e, 'response') and e.response:
             print(e.response.text)

if __name__ == "__main__":
    query = "red dress"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    test_search(query)
