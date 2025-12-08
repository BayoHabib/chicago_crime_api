# Chicago Crime Prediction API

[![CI/CD](https://github.com/BayoHabib/chicago-crime-api/actions/workflows/ci.yml/badge.svg)](https://github.com/BayoHabib/chicago-crime-api/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade **Crime Prediction API** built with FastAPI, combining:

- **[EventFlow](https://github.com/BayoHabib/eventflow)** - ML adapter library for crime prediction modeling
- **[Chicago Crime Data CLI](https://github.com/BayoHabib/chicago_crime_data_cli)** - Real Chicago crime data pipeline

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│  EventFlow ML   │────▶│  MLflow Server  │
│   (Prediction)  │     │   Adapters      │     │  (Model Store)  │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Prometheus    │────▶│    Grafana      │
│   (Metrics)     │     │  (Dashboards)   │
└─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/BayoHabib/chicago-crime-api.git
cd chicago-crime-api

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start development server
uvicorn chicago_crime_api.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Start all services (API, MLflow, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 📡 API Endpoints

### Health Checks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/ready` | GET | Readiness probe (checks model) |
| `/health/live` | GET | Liveness probe |

### Predictions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Single location prediction |
| `/api/v1/predict/grid` | POST | Grid-based predictions |

### Example Request

```bash
# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 41.8781,
    "longitude": -87.6298,
    "hour": 14,
    "day_of_week": 2,
    "month": 6
  }'

# Grid prediction
curl -X POST "http://localhost:8000/api/v1/predict/grid" \
  -H "Content-Type: application/json" \
  -d '{
    "center_latitude": 41.8781,
    "center_longitude": -87.6298,
    "radius_km": 1.0,
    "grid_size": 5,
    "hour": 14,
    "day_of_week": 2,
    "month": 6
  }'
```

### Example Response

```json
{
  "predictions": [
    {
      "latitude": 41.8781,
      "longitude": -87.6298,
      "risk_score": 0.73,
      "risk_level": "medium"
    }
  ],
  "model_version": "1.0.0",
  "prediction_timestamp": "2024-01-15T14:30:00Z"
}
```

## 🛠️ Configuration

Environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `chicago-crime-api` | Application name |
| `DEBUG` | `false` | Debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MODEL_PATH` | `models/latest` | Path to model artifacts |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

## 📊 Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`:

- Request latency histogram
- Request count by endpoint
- Model prediction latency
- Error rates

### Grafana Dashboards

Access Grafana at `http://localhost:3000`:

- **Default credentials**: admin/admin
- Pre-configured dashboards for API monitoring

### MLflow Model Registry

Access MLflow at `http://localhost:5000`:

- Model versioning
- Experiment tracking
- Model deployment management

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chicago_crime_api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run integration tests only
pytest tests/ -m integration
```

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI prediction service |
| `mlflow` | 5000 | MLflow tracking server |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Monitoring dashboards |

## 📁 Project Structure

```
chicago-crime-api/
├── src/chicago_crime_api/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings
│   ├── main.py            # FastAPI app factory
│   ├── schemas.py         # Request/response models
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py      # Health endpoints
│   │   └── predictions.py # Prediction endpoints
│   └── services/
│       ├── __init__.py
│       └── prediction.py  # ML prediction service
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   └── test_api.py        # API tests
├── models/                 # Model artifacts (gitignored)
├── .github/workflows/
│   └── ci.yml             # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow includes:

1. **Lint** - Code quality checks (ruff, mypy)
2. **Test** - Unit and integration tests with coverage
3. **Build** - Docker image build and push
4. **Deploy** - Automatic deployment to staging/production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Chicago Data Portal](https://data.cityofchicago.org/) for crime data
- [EventFlow](https://github.com/BayoHabib/eventflow) for ML adapters
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
