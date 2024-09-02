FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file to /app in the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code into the container
COPY . .

# Create necessary directories
RUN mkdir -p /app/models/saved_models /app/models/logs/prediction_logs /app/data/processed

# List contents of /app for debugging
RUN ls -R /app

# Set the PYTHONPATH environment variable to /app
ENV PYTHONPATH=/app

# Set the default command to run the script
CMD ["bash", "src/scripts/eval_predict.sh"]