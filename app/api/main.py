import sys
import os
from fastapi import FastAPI
from app.api.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Stock Prediction API is up and running!"}