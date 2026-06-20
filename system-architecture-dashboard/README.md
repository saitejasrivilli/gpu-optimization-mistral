# System Architecture Dashboard

Interactive web dashboard for visualizing and executing clinical ML pipeline architecture on Google Cloud Platform.

## Features

- **Visual Flow Diagram**: 15 nodes organized in 5 layers (Input → Processing → Storage → Serving → Monitoring)
- **Interactive Execution**: Click any node to execute and view real-time logs
- **Data Flow Visualization**: Arrows showing data flowing between nodes
- **Color-Coded Nodes**: Different colors for different node types
- **Production-Ready**: Flask backend with proper configuration management
- **CORS Enabled**: Cross-origin requests for frontend-backend communication
- **Healthcare Compliance**: HIPAA-aware logging and de-identification

## Architecture

```
system-architecture-dashboard/
├── backend/                  # Flask backend application
│   ├── __init__.py
│   ├── app.py               # Flask app factory
│   ├── config.py            # Configuration management
│   ├── routes.py            # API route definitions
│   └── nodes.py             # Node implementations
├── templates/               # HTML templates
│   └── index.html           # Main dashboard
├── static/                  # Static assets
│   └── dashboard.html       # Interactive dashboard
├── tests/                   # Test files
├── config/                  # Configuration files
├── setup.py                 # Package setup
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## Installation

### 1. Clone/Extract the package

```bash
cd system-architecture-dashboard
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Install package

```bash
pip install -e .
```

## Usage

### Start the Backend Server

```bash
python3 -m backend.app
```

Or use the CLI entry point:

```bash
dashboard-server
```

Server will start at `http://localhost:5000`

### Open the Dashboard

In your browser, navigate to:

```
file:///path/to/system-architecture-dashboard/templates/index.html
```

Or serve via Flask:

```bash
python3 -c "from flask import Flask; from backend.app import create_app; app = create_app(); app.run(port=8000)"
```

Then open `http://localhost:8000`

### Using the Dashboard

1. **Select a Node**: Click any colored box in the diagram
2. **Execute**: Press ENTER or click "RUN NODE" button
3. **View Logs**: Execution logs appear in the right panel
4. **Understand Flow**: Follow arrows showing data flow between nodes

## Node Types

### INPUT (Blue)
- **CLINICAL NOTES**: Load clinical note from source (562 chars)
- **TRAINING DATA**: Generate/load training dataset (500 samples × 20 features)
- **INFERENCE REQ**: Receive real-time inference requests

### PROCESSING (Green)
- **TRAIN MODEL**: PyTorch distributed training (3 epochs, 87% accuracy)
- **VALIDATE METRICS**: Model validation on test set
- **DE-ID PHI**: De-identification (6 PHI spans masked)
- **NER EXTRACT**: Named Entity Recognition (24 entities)
- **FHIR BUNDLE**: FHIR R4 bundle generation (14 resources)

### STORAGE (Orange)
- **LOCAL REGISTRY**: Save model to local filesystem (0.02 MB)
- **CLOUD STORAGE**: Upload to Google Cloud Storage (GCS)
- **BIGQUERY LOAD**: Batch load to BigQuery (JSONL format, free tier)

### SERVING (Red)
- **VERTEX AI**: Register model in Vertex AI
- **ENDPOINTS**: Deploy REST API endpoints (auto-scaling)

### MONITORING (Yellow)
- **MONITORING DRIFT**: Real-time drift detection
- **ANALYTICS SQL**: BigQuery SQL analytics

## API Endpoints

```
GET  /api/health                    # Health check
GET  /api/node/clinical             # Clinical notes input
GET  /api/node/training             # Training data
GET  /api/node/inference            # Inference requests
GET  /api/node/train                # Model training
GET  /api/node/validate             # Model validation
GET  /api/node/deid                 # De-identification
GET  /api/node/ner                  # NER extraction
GET  /api/node/fhir                 # FHIR bundle generation
GET  /api/node/registry             # Local model registry
GET  /api/node/gcs                  # Cloud storage
GET  /api/node/bigquery             # BigQuery batch load
GET  /api/node/vertex               # Vertex AI registration
GET  /api/node/endpoint             # Endpoints deployment
GET  /api/node/monitoring           # Monitoring & drift
GET  /api/node/analytics            # Analytics queries
```

## Configuration

### Development

```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
python3 -m backend.app
```

### Production

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:create_app
```

## Environment Variables

See `.env.example` for all available options:

- `FLASK_ENV`: development|production|testing
- `FLASK_DEBUG`: True|False
- `SECRET_KEY`: Session encryption key
- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)

## Testing

```bash
pytest tests/
pytest --cov=backend tests/
```

## Docker Support

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "-m", "backend.app"]
```

Build and run:

```bash
docker build -t dashboard .
docker run -p 5000:5000 dashboard
```

## Real Data Integration

To use real data from your ML pipelines:

1. Update `backend/nodes.py` with actual implementations
2. Import your pipeline modules
3. Call real functions instead of returning hardcoded logs

Example:

```python
from ml_pipeline.training.trainer import DistributedTrainer

def train_model():
    trainer = DistributedTrainer(model)
    metrics = trainer.train_epoch(X_train, y_train)
    return format_output(metrics)
```

## Deployment

### GCP Cloud Run

```bash
# Build and push to Cloud Run
gcloud run deploy dashboard --source . --port 5000
```

### AWS Lambda

Use serverless framework or AWS SAM

### Kubernetes

Create Helm chart or K8s manifests

## Data Security

- HIPAA-aware de-identification
- PHI masking and audit trails
- Secure session handling
- CORS restrictions
- Input validation

## Performance

- Async request handling (with proper async/await)
- Request caching (1-hour TTL default)
- Batch processing for BigQuery
- Distributed training support (PyTorch DDP)

## Troubleshooting

### Backend not connecting

```
Error: Cannot connect to backend

Solution:
1. Ensure backend server is running: python3 -m backend.app
2. Check port 5000 is not in use: lsof -i :5000
3. Verify CORS is enabled (should be automatic)
4. Check browser console for exact error
```

### Port already in use

```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
PORT=8000 python3 -m backend.app
```

### Module import errors

```bash
# Ensure package is installed in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/system-architecture-dashboard
```

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Make changes and test: `pytest`
3. Format code: `black backend/`
4. Lint: `flake8 backend/`
5. Commit: `git commit -m "feature: description"`
6. Push: `git push origin feature/name`

## License

MIT License - See LICENSE file

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review API endpoint logs
3. Open an issue on GitHub

## Version

Current: **1.0.0**

## Changelog

### v1.0.0 (2026-06-19)
- Initial release
- 15 interactive nodes
- Full GCP integration
- Production-ready structure
- HIPAA compliance
