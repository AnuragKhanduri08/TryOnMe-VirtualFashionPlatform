.PHONY: install run-backend run-frontend

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

run-backend:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev
