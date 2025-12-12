# Chicago Crime Prediction UI

Interactive dashboard for exploring crime predictions in Chicago.

## Quick Start

### 1. Start the API (in one terminal)
```bash
cd chicago-crime-api
.\.venv\Scripts\python.exe -m uvicorn chicago_crime_api.main:app --reload --port 8000
```

### 2. Start the Dashboard (in another terminal)
```bash
cd chicago-crime-api
.\.venv\Scripts\streamlit run ui/app.py
```

The dashboard will open at http://localhost:8501

## Features

- **Single Location Prediction**: Click or enter coordinates to get crime forecast
- **Hotspots View**: See top crime hotspots with heatmap visualization
- **Full Grid View**: City-wide prediction heatmap
- **Interactive Map**: Pan, zoom, click for details

## Architecture

```
ui/
├── app.py              # Main Streamlit dashboard
├── api_client.py       # API client (reusable for React)
├── map_utils.py        # Map visualization utilities
└── react-migration-config.json  # Config for React migration
```

## React + Leaflet Migration

The codebase is structured for easy migration to React:

| Streamlit | React Equivalent |
|-----------|-----------------|
| `api_client.py` | `src/api/crimeApiClient.ts` |
| `map_utils.py` | `src/components/CrimeMap.tsx` |
| `app.py` views | `src/pages/*.tsx` |

See `react-migration-config.json` for:
- API endpoint definitions
- Map configuration
- Heatmap gradients
- Recommended React dependencies

## Screenshots

### Hotspots View
- Heatmap overlay showing crime intensity
- Ranked markers for top hotspots
- Risk level indicators (Critical/High/Medium/Low)

### Single Prediction
- Point-specific forecasts
- Confidence intervals
- Grid cell information
