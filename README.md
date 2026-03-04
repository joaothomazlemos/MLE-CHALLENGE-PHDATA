
# End-to-End Machine Learning System: Real Estate Price Prediction

## Executive Summary

This project demonstrates a production-oriented Machine Learning Engineering workflow, covering the full lifecycle of an ML system — from model training and evaluation to API deployment, versioning, and scalability design.

Rather than focusing solely on model accuracy, this project emphasizes system reliability, maintainability, reproducibility, and architectural decisions required to operate ML systems in real-world environments.

The objective is to simulate the transition from a proof-of-concept model to a production-ready inference service capable of supporting business operations.

---

## Problem Context

A real estate organization aims to operationalize an existing property valuation model trained on historical housing data.

The baseline model:
- Uses a limited subset of features
- Excludes potentially informative attributes
- Exists only as a training script and serialized artifact
- Has not been deployed for real-world consumption

The engineering challenge is to productionize this model in a way that:

- Ensures reproducible inference
- Supports structured API access
- Handles feature enrichment internally
- Enables versioned model updates
- Is designed with horizontal scalability in mind

---

## System Design Principles

The system was designed with the following principles:

- **Separation of concerns** (training vs. inference)
- **Stateless API architecture**
- **Reproducible environments**
- **Explicit feature contracts**
- **Backward-compatible model updates**
- **Infrastructure-aware design decisions**

While the implementation runs locally, architectural decisions reflect production deployment patterns.

---

## Architecture Overview

The project includes:

- Model training pipeline
- Serialized model artifact (`model.pkl`)
- Feature schema definition (`model_features.json`)
- REST API for inference
- Automated test script
- Containerization setup
- Model evaluation workflow

The inference service:

- Accepts structured JSON input
- Performs internal feature enrichment (zipcode demographic joins)
- Validates and orders features according to the training schema
- Returns predictions alongside metadata
- Is structured to allow model versioning without downtime

---

## Model Evaluation

The evaluation process includes:

- Holdout validation performance analysis
- Generalization behavior assessment
- Bias/variance considerations
- Review of feature limitations
- Opportunities for incremental model improvements using traditional ML techniques

The goal is not to pursue marginal accuracy gains, but to ensure predictable and stable performance in a production setting.

---

## Deployment Strategy

Although executed locally, the system is designed to be:

- Containerized
- Horizontally scalable
- Compatible with load-balanced environments
- Adaptable to rolling or blue-green deployment strategies
- Capable of serving multiple model versions concurrently

Key production considerations discussed include:

- Autoscaling patterns
- Model artifact management
- Backward compatibility of feature contracts
- Observability and monitoring hooks
- Safe model replacement strategies

---

## Testing and Validation

A lightweight test client validates:

- Endpoint responsiveness
- Feature preprocessing correctness
- Prediction output structure
- Reproducibility of inference behavior

This ensures deterministic model serving behavior across environments.

---

## Skills Demonstrated

- Production ML system design
- Model artifact lifecycle management
- RESTful inference services (Python-based)
- Feature contract enforcement
- Data enrichment pipelines
- Containerized deployment workflows
- Scalability-oriented architectural thinking
- Technical communication for mixed audiences

---

## Engineering Perspective

Machine Learning Engineering extends beyond experimentation. It requires building reliable, observable, and maintainable systems around models.

This project reflects a systems-oriented mindset:
- Prioritizing operational stability over novelty
- Designing for change
- Reducing coupling between components
- Making trade-offs explicit
- Treating models as versioned production artifacts

---

## Repository Structure

.
├── data/
│   ├── kc_house_data.csv
│   ├── zipcode_demographics.csv
│   └── future_unseen_examples.csv
├── model/
│   ├── model.pkl
│   └── model_features.json
├── create_model.py
├── conda_environment.yml
├── api/
├── tests/
└── README.md

---
## Run the API

```bash

# ============================================================
# End-to-End ML System — Setup & Run Script
# ============================================================
# This script walks you through:
# 1. Cloning the repository
# 2. Creating the Conda environment
# 3. Training the model
# 4. Starting the API
# 5. Testing the endpoint
# ============================================================


# ------------------------------------------------------------
# 1️⃣ Clone the repository
# ------------------------------------------------------------
# Downloads the project locally

git clone https://github.com/joaothomazlemos/MLE-CHALLENGE-PHDATA.git
cd MLE-CHALLENGE-PHDATA


# ------------------------------------------------------------
# 2️⃣ Create the Conda environment
# ------------------------------------------------------------
# Creates a reproducible Python environment
# defined in conda_environment.yml

conda env create -f conda_environment.yml


# ------------------------------------------------------------
# 3️⃣ Activate the environment
# ------------------------------------------------------------
# Activates the environment so dependencies are available

conda activate housing


# ------------------------------------------------------------
# 4️⃣ Train the model
# ------------------------------------------------------------
# Generates:
# - model/model.pkl
# - model/model_features.json

python create_model.py


# ------------------------------------------------------------
# 5️⃣ Start the API server
# ------------------------------------------------------------
# Launches the FastAPI app using Uvicorn
# The API will be available at:
# http://127.0.0.1:8000

cd api
uvicorn main:app --host 127.0.0.1 --port 8000 --reload


# ------------------------------------------------------------
# 6️⃣ Test the API (run this in a NEW terminal)
# ------------------------------------------------------------
# Keep the server running.
# Open another terminal and activate the environment again:
#
#   conda activate housing
#
# Then run:

# curl -X POST http://127.0.0.1:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{
#         "bedrooms": 3,
#         "bathrooms": 2,
#         "sqft_living": 1800,
#         "sqft_lot": 5000,
#         "floors": 1,
#         "waterfront": 0,
#         "view": 0,
#         "condition": 3,
#         "grade": 7,
#         "sqft_above": 1800,
#         "sqft_basement": 0,
#         "yr_built": 1995,
#         "yr_renovated": 0,
#         "zipcode": 98103,
#         "lat": 47.65,
#         "long": -122.34,
#         "sqft_living15": 1700,
#         "sqft_lot15": 4800
#       }'


# ------------------------------------------------------------
# Run automated test script
# ------------------------------------------------------------

# python tests/test_api.py

```

## Closing Note

This repository represents a production-minded approach to deploying and operating ML systems — balancing pragmatism, architectural clarity, and maintainability.

The focus is not on complexity, but on building systems that can evolve safely.
