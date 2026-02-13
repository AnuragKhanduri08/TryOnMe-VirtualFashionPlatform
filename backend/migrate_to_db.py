import os
import sys
import json
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import Product

# Add parent directory to path to import backend modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def migrate_to_db():
    print("========================================")
    print("   DATABASE MIGRATION (JSON -> DB)")
    print("========================================")
    
    # 1. Create Tables
    print("\n[1/3] Creating Database Tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully.")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return

    # 2. Load JSON Data
    print("\n[2/3] Loading products.json...")
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products.json")
    if not os.path.exists(json_path):
        print(f"❌ products.json not found at {json_path}")
        return
        
    try:
        with open(json_path, "r") as f:
            products_data = json.load(f)
        print(f"✅ Loaded {len(products_data)} products from JSON.")
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return

    # 3. Insert into Database
    print("\n[3/3] Inserting Data into Database...")
    db = SessionLocal()
    try:
        # Check if DB is already populated
        existing_count = db.query(Product).count()
        if existing_count > 0:
            print(f"⚠️ Database already contains {existing_count} products.")
            
            # Check for FORCE_MIGRATE env var
            force_migrate = os.getenv("FORCE_MIGRATE", "false").lower() == "true"
            
            if force_migrate:
                print("FORCE_MIGRATE=true detected. Clearing existing data...")
                db.query(Product).delete()
                db.commit()
                print("Existing data cleared.")
            else:
                print("Skipping import to avoid duplicates. (Set FORCE_MIGRATE=true to overwrite)")
                return

        # Bulk Insert
        print("Inserting products...", end=" ")
        products_to_add = []
        for p_data in products_data:
            product = Product(
                id=p_data.get("id"),
                name=p_data.get("name"),
                description=p_data.get("description"),
                category=p_data.get("category"),
                image_url=p_data.get("image_url"),
                gender=p_data.get("gender"),
                masterCategory=p_data.get("masterCategory"),
                subCategory=p_data.get("subCategory"),
                articleType=p_data.get("articleType"),
                baseColour=p_data.get("baseColour"),
                season=p_data.get("season"),
                usage=p_data.get("usage"),
                price=p_data.get("price")
            )
            products_to_add.append(product)
            
        db.add_all(products_to_add)
        db.commit()
        print(f"\n✅ Successfully inserted {len(products_to_add)} products into the database.")
        
    except Exception as e:
        print(f"\n❌ Database Error: {e}")
        db.rollback()
    finally:
        db.close()

    print("\n✅ Migration Complete!")

if __name__ == "__main__":
    migrate_to_db()
