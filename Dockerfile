# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV MODEL_PATH="/models/saved_models/"
ENV DATA_PATH="/data/processed/"

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port where the FastAPI app will run
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]