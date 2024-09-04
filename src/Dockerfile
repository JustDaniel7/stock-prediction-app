# Dockerfile
FROM python:3.11-slim

# Set the working directory to /app/src
WORKDIR /app/src

# Copy the requirements.txt file to /app/src in the container
COPY src/requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY src/ .

# Create necessary directories
RUN mkdir -p /app/src/models/saved_models /app/src/models/logs/prediction_logs /app/src/data/processed

# List contents of /app/src for debugging
RUN ls -R /app/src

# Set the PYTHONPATH environment variable to include /app/src
ENV PYTHONPATH=/app:$PYTHONPATH

# Set the default command to run the script
CMD ["bash", "full_cycle.sh"]