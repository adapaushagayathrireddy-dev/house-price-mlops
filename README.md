# 🏠 House Price MLOps Capstone

End-to-end MLOps pipeline for California house price prediction.

## Tech Stack

- **Model:** California Housing regression with scikit-learn
- **Tracking:** MLflow
- **API:** FastAPI
- **UI:** Streamlit
- **Containerization:** Docker & Docker Compose
- **CI/CD:** GitHub Actions

## Architecture

Browser → Streamlit → FastAPI → ML Model

## Run Locally

### FastAPI

```bash
uvicorn api.main:app --reload --port 8000


API documentation:

http://localhost:8000/docs

### Streamlit

```bash
streamlit run app/streamlit_app.py