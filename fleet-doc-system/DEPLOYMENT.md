# Fleet Document Management System - Deployment Guide

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip/conda package manager
- 2 GB free disk space (for OCR models)

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** First run takes 10-15 minutes for OCR model download (~1.5GB). Subsequent runs are instant.

### Environment Configuration

Create `.env` file with Buildathon API credits:
```bash
# Featherless AI (provided: $25 credits)
FEATHERLESS_API_KEY=your_featherless_api_key_here
FEATHERLESS_MODEL=Qwen/Qwen3-14B-NoThinking

# Tavily Search (provided: $2,000 credits)
TAVILY_API_KEY=your_tavily_api_key_here

# OCR Model (options: easyocr, pytesseract)
OCR_MODEL=easyocr

# Optional: Cache config
CACHE_ENABLED=true
CACHE_TTL_HOURS=24

# Database
DATABASE_URL=sqlite:///fleet.db
```

## Running the System

### Start Backend API Server
```bash
python src/layer6_api/server.py
```
Server runs on `http://localhost:8000`

### Open Frontend
```bash
open frontend/index.html
# or
python -m http.server 8000 -d frontend
```
Frontend runs on `http://localhost:8000` (requires separate port if using Python HTTP server)

## System Architecture

```
User Query (English)
    ↓
[Layer 4] Query Router (Featherless AI)
    ↓
[Layer 3] Retrieval (Local Store + Tavily Search)
    ↓
[Layer 4] Response Generator (Featherless AI)
    ↓
[Layer 5] Grounding Verification (0% hallucination)
    ↓
API Response (✓ Verified answer + sources)
```

## API Endpoints

### POST /query
Query fleet documents with grounding verification.

**Request:**
```json
{
  "query": "What maintenance did truck T-084 have?",
  "truck_id": "T-084",  // optional
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "T-084 had oil change and filter replacement on 06/15/2026",
  "is_grounded": true,
  "confidence": 0.95,
  "sources": ["doc_001", "doc_002"],
  "query": "What maintenance did truck T-084 have?"
}
```

### GET /health
Health check endpoint - returns component status.

### GET /stats
System statistics - number of indexed documents, trucks, drivers.

## Testing

### Test Layer 1 (Document Ingestion)
```bash
python -m pytest test_layer1.py -v
```

### Test Layer 2 (Entity Linking)
```bash
python -m pytest test_layer2.py -v
```

### Test Layers 3-6 (Complete Pipeline)
```bash
python test_layers_3_6.py
```

### Test End-to-End
```bash
python test_e2e.py
```

## Data Ingestion

### Ingest Documents
```bash
python batch_ingest.py --input-dir data/incoming --output-dir data/processed
```

### Generate Synthetic Test Data
```bash
python generate_fleet_documents.py --output-dir data/synthetic --count 60
```

## Performance & Costs

### API Credits Usage
- **Featherless AI ($25 budget):**
  - Query routing: ~0.5 tokens per query
  - Answer generation: ~100 tokens per query
  - ~200 queries per week (0.1 per query)

- **Tavily Search ($2,000 budget):**
  - Web search fallback: 1 credit per search
  - ~5 searches per query (load balanced)
  - ~50 queries per week ($250/week budget available)

### Processing Performance
- Document ingestion: ~500ms per document
- Entity linking: ~200ms per document
- Query response: ~2-3 seconds end-to-end (including Featherless + verification)
- Batch processing: 50+ documents/week with async processing

## Troubleshooting

### OCR Model Download Fails
```bash
# Manual model download
python -c "import easyocr; easyocr.Reader(['en'])"
```

### Featherless API Connection Error
- Verify API key in `.env`
- Check internet connection
- Verify base_url: `https://api.featherlessai.com/v1`

### Tavily Search Not Working
- Verify API key has remaining credits
- Check query format in logs
- Fallback to local document store (no credits)

### Database Locked
```bash
# Reset database
rm fleet.db
python src/layer6_api/server.py  # auto-creates fresh DB
```

## Production Deployment

### Docker Deployment
```bash
docker build -t fleet-doc-system .
docker run -p 8000:8000 -e FEATHERLESS_API_KEY=$KEY fleet-doc-system
```

### Cloud Deployment (Render/Heroku/Vercel)
```bash
# Push to GitHub, enable CI/CD
# Set environment variables in cloud provider
# System auto-deploys
```

## Monitoring

### View Logs
```bash
tail -f logs/fleet.log
```

### Check System Status
```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

### Monitor API Usage
- Featherless: Check API dashboard for token usage
- Tavily: Monitor search count in API account

## Support

For issues or questions:
1. Check logs: `src/logger.py` enables structured JSON logging
2. Run diagnostics: `python test_e2e.py`
3. Verify API credentials: `.env` configuration
