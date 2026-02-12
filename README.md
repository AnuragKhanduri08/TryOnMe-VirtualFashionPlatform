# AI-Driven Virtual Fashion & E-Commerce Platform

## Overview
This project aims to build a comprehensive e-commerce platform integrated with advanced AI features to enhance the user shopping experience.

## Core Modules

### 1. Smart Search (Text + Image)
- **Goal:** Allow users to search for products using natural language descriptions or by uploading images.
- **Tech:** CLIP (Contrastive Language-Image Pre-training), Color Histograms (Fallback), Vector Database.

### 2. Body Measurement & Fit Prediction
- **Goal:** Estimate user body measurements from photos and predict the best fit for clothing items.
- **Tech:** Computer Vision (YOLOv8 Pose), OpenCV.

### 3. Fashion Recommendation System
- **Goal:** Personalized product suggestions based on user behavior and style preferences.
- **Tech:** Content-based Filtering (Embeddings/Histograms).

### 4. GAN-Based Virtual Try-On
- **Goal:** Realistic simulation of how clothes look on the user.
- **Tech:** Heuristic Warping (Initial), GANs (Future).

### 5. Dashboard
- **Goal:** Monitor system metrics, user activity, and AI model performance.
- **Tech:** React/Next.js (Frontend), FastAPI (Backend), Recharts.

## Architecture
- **Backend:** Python (FastAPI) - Handling API requests and AI inference.
- **Frontend:** React.js / Next.js - User Interface.
- **Database:** JSON (Prototyping), PostgreSQL (Planned).

## How to Run (Quick Start)

### 1. Environment Setup
**Before running the project**, you must set up your environment variables:
1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
   *(Or manually rename `.env.example` to `.env` in your file explorer)*
2. (Optional) Edit `.env` to configure your database (defaults to SQLite, so no change needed for quick testing).

### 2. Generate AI Models (Critical Step)
Since large AI model files are not included in the repository, you **must** generate them locally for search and recommendations to work.

1. Run `run.bat`
2. Select **Option [6] Setup AI Models**
3. Wait for the process to complete (it will download the CLIP model and generate embeddings).

### 3. Run the Application
#### Option A: One-Click Run (Windows)
Simply double-click the `run.bat` file in the project root folder.
1. Select **Option [2]** to install dependencies (First time only).
2. Select **Option [1]** to start the project.

### 4. (Optional) Using PostgreSQL
By default, the app uses a simple SQLite database. If you want to use **PostgreSQL** for better performance:

1. **Install PostgreSQL** and create a new database (e.g., `virtual_fashion`).
2. **Configure Environment**:
   - Rename `.env.example` to `.env`.
   - Uncomment and update the `DATABASE_URL` line:
     ```bash
     DATABASE_URL=postgresql://postgres:password@localhost/virtual_fashion
     ```
3. **Migrate Data**:
   - Run `run.bat`.
   - Select **Option [7] Migrate Data to Database**.
   - This will create the tables and import products from `products.json` into your PostgreSQL database.
4. **Setup AI Models**:
   - Don't forget to run **Option [6]** again if you switched databases, to ensure embeddings are aligned with the new DB IDs.

#### Option B: Manual Setup (Linux/Mac/Windows)

### Prerequisites
- Python 3.8+
- Node.js 18+

### 1. Start the Backend
Open a terminal and run:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Note: The first run might take a moment to download AI models.*

### 2. Start the Frontend
Open a **new** terminal and run:
```bash
cd frontend
npm install
npm run dev
```

### 3. Access the Application
Open your browser and go to: [http://localhost:3000](http://localhost:3000)

## Directory Structure
- `/backend`: API server and business logic.
- `/frontend`: Web application.
- `/ai_modules`: Isolated AI components.
