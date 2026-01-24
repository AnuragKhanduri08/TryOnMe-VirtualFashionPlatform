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

## How to Run

### Option A: One-Click Run (Windows)
Simply double-click the `run.bat` file in the project root folder.
This will automatically open two terminal windows (one for backend, one for frontend).

### Option B: Using Make (Linux/Mac/WSL)
If you have `make` installed:
```bash
make install       # Install dependencies
make run-backend   # Run backend
make run-frontend  # Run frontend (in a new tab)
```

### Option C: Manual Setup

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
