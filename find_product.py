import json

with open('backend/products.json', 'r') as f:
    products = json.load(f)

for p in products:
    if "Inkfruit Mens Chain Reaction T-shirt" in p['name']:
        print(json.dumps(p, indent=2))
        break
