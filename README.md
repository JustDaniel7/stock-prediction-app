# Stock Prediction App on AWS with ECS Fargate

This project is a cloud-based machine learning application to predict stock prices for top companies using AWS infrastructure. It leverages an LSTM neural network model implemented in PyTorch, with data retrieved from Yahoo Finance and features engineered using pandas. The application is containerized with Docker and deployed on AWS ECS Fargate using three main services: `model`, `API`, and `frontend`.

## Table of Contents
1. [Architecture](#architecture)
2. [AWS Setup](#aws-setup)
3. [Docker Configuration](#docker-configuration)
4. [Deployment](#deployment)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)

## Architecture

### Overview
The project is structured with three primary components:
1. **Model Service**: Runs the machine learning model for stock predictions.
2. **API Service**: A REST API built with FastAPI that handles requests from the frontend.
3. **Frontend Service**: A Next.js application that serves the user interface.

### AWS Services
The following AWS services are used:
- **Amazon ECS Fargate**: For containerized application deployment.
- **Amazon S3**: For storing raw and processed data files.
- **Amazon ECR**: To host Docker images for the `model`, `API`, and `frontend`.
- **Application Load Balancer (ALB)**: To route traffic to the `frontend` service.

## AWS Setup

### Step 1: Create S3 Buckets
Create an S3 bucket for storing the raw and processed data files:
- **Folder structure**:
  - `data/raw/`: Store raw data files.
  - `data/processed/`: Store processed data files.

### Step 2: Configure ECR Repositories
Create three ECR repositories to host Docker images:
1. `model`: Stores the model container.
2. `api`: Stores the API container.
3. `frontend`: Stores the frontend container.

### Step 3: Set up ECS Fargate Cluster
1. Create an ECS Fargate cluster.
2. Define separate task definitions for `model`, `API`, and `frontend` services with the appropriate configurations:
   - **Model Service**: 1 vCPU, 2GB memory
   - **API Service**: 0.5 vCPU, 1GB memory
   - **Frontend Service**: 0.5 vCPU, 1GB memory, port 3000 exposed
3. Configure an **Application Load Balancer** for the `frontend` service.

## Docker Configuration

Each component has its own Docker image, which can be tagged and pushed to ECR for deployment.

## Deployment

### Step 1: Set Up IAM Roles
Create an IAM role with permissions for ECS and Lambda access to ECR and other necessary resources.

### Step 2: Deploy ECS Services
Deploy each service to ECS, ensuring:
1. The `model` and `API` services have **private access** since they don't require direct public access.
2. The `frontend` service is accessible via the Application Load Balancer.

### Step 3: Configure Target Groups and Health Checks
1. Set up target groups for the `frontend` service, with health checks configured for path `/` on port 3000.
2. Verify that targets are registered and healthy after deployment.

## Usage

### Frontend Access
- Once the frontend service is up and running with the ALB, access it via the ALB’s DNS name in a browser.

### API Endpoints
- The API endpoints handle requests for prediction data and other application features.

### Data Processing
- Raw data is downloaded from Yahoo Finance and stored in S3. The model container pulls data from `data/raw/`, processes it, and stores processed files in `data/processed/`.

## Troubleshooting

- **ALB DNS Not Accessible**: Ensure the frontend service is correctly registered in the ALB target group and that health checks are passing.
- **Container Architecture Mismatch**: Ensure images are built for `linux/arm64` to be compatible with ECS Fargate.
- **Permissions Issues**: Confirm IAM roles attached to ECS tasks have permissions to access ECR and other required services.

## License
This project is licensed under the MIT License.
