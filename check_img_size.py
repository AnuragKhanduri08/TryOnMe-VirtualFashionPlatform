import requests
from PIL import Image
from io import BytesIO

url = "http://assets.myntassets.com/v1/images/style/properties/9c1b19682ecf926c296f520d5d6e0972_images.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
print(f"Image Size: {img.size}")
