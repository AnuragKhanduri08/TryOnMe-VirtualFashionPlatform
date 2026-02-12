from sqlalchemy import Column, Integer, String, Float, Text
from database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    category = Column(String, index=True)
    image_url = Column(String)
    gender = Column(String, index=True)
    masterCategory = Column(String, index=True)
    subCategory = Column(String, index=True)
    articleType = Column(String, index=True)
    baseColour = Column(String, nullable=True)
    season = Column(String, nullable=True)
    usage = Column(String, nullable=True)
    price = Column(Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "image_url": self.image_url,
            "gender": self.gender,
            "masterCategory": self.masterCategory,
            "subCategory": self.subCategory,
            "articleType": self.articleType,
            "baseColour": self.baseColour,
            "season": self.season,
            "usage": self.usage,
            "price": self.price
        }
