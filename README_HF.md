---
title: Chicago Crime Prediction
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Chicago Crime Prediction Dashboard

Interactive ML-powered crime prediction dashboard for Chicago.

## Features

- **Single Location Prediction**: Get crime predictions for any Chicago location
- **Hotspots Analysis**: View top crime hotspots with risk levels
- **Crime Type Distribution**: See breakdown of predicted crime types
- **Interactive Maps**: Folium-powered visualization

## Model

- Random Forest Regressor with 14 features
- 95.9% R² score, 5.5% MAPE
- Trained on Chicago crime data (2024-2025)

## Tech Stack

- FastAPI backend
- Streamlit dashboard
- scikit-learn ML model
- Folium maps + Plotly charts
