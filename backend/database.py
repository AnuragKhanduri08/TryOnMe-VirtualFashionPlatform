import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load .env file
load_dotenv()

# Default to SQLite for easy setup, but allow overriding with DATABASE_URL env var
# Example Postgres URL: postgresql://user:password@localhost/dbname
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

# Ensure we use psycopg2 if postgres is specified but driver not explicit
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Add SSL requirement for Render Postgres if it's a remote connection
if "render.com" in SQLALCHEMY_DATABASE_URL or "postgresql" in SQLALCHEMY_DATABASE_URL:
    if "?" not in SQLALCHEMY_DATABASE_URL:
        SQLALCHEMY_DATABASE_URL += "?sslmode=require"
    elif "sslmode" not in SQLALCHEMY_DATABASE_URL:
        SQLALCHEMY_DATABASE_URL += "&sslmode=require"

print(f"Connecting to database: {SQLALCHEMY_DATABASE_URL}")

# SQLite specific args
connect_args = {"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}

# For Postgres on Render, sometimes we need extra pool settings
engine_args = {"connect_args": connect_args}
if "postgresql" in SQLALCHEMY_DATABASE_URL:
    engine_args["pool_pre_ping"] = True
    engine_args["pool_recycle"] = 3600

engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
