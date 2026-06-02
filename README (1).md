# MolecularML Platform

> End-to-end scientific ML platform for **molecular property prediction** and **protein similarity search** — production-deployed with full CI/CD, observability, and a React dashboard.

**[Live Demo — Frontend](https://molecularml-frontend.vercel.app)** &nbsp;|&nbsp; **[Live API](https://saitejasrivilli-molecularml-platform.hf.space/docs)** &nbsp;|&nbsp; **[API Docs](https://saitejasrivilli-molecularml-platform.hf.space/docs)**

---

## What It Does

A full-stack ML platform that mirrors the kind of tooling built at companies like Schrödinger, Recursion, and Insilico Medicine — enabling scientists to run property predictions and protein searches through a clean web interface backed by production ML infrastructure.

| Feature | Details |
|---|---|
| Molecular property prediction | PyTorch MLP on RDKit Morgan fingerprints — solubility, lipophilicity, toxicity + Lipinski Rule of Five |
| Protein similarity search | ESM-2 protein embeddings + FAISS vector index — top-k similar proteins from UniProt DB |
| Analytics dashboard | Real-time P50/P95/P99 latency, request counts, model health — auto-refreshes every 15s |
| Drug-likeness scoring | Lipinski Rule of Five with violation count and drug-like classification |

---

## Architecture

![Architecture Diagram](architecture.svg)

---

## Results

| Metric | Value |
|---|---|
| API response latency (P50) | < 100ms |
| Protein search (FAISS cosine) | sub-second across full index |
| Frontend build | Vite · < 2s cold build |
| CI/CD pipeline | test → deploy on every push to master |
| Deployment uptime | tracked via `/health` + `/metrics` endpoints |

---

## Tech Stack

### ML & Backend
- **PyTorch** — MLP model for molecular property regression
- **RDKit** — SMILES parsing, Morgan fingerprint generation, molecular descriptors
- **HuggingFace Transformers** — ESM-2 protein language model for sequence embeddings
- **FAISS** — vector similarity search over protein embedding index
- **FastAPI** — async REST API with Pydantic validation
- **Python 3.11**

### Frontend
- **React** — three-tab SPA: molecule predictor, protein search, analytics dashboard
- **Vite** — fast build tooling
- **JavaScript**

### Infrastructure & DevOps
- **Docker** — containerised backend deployed to HuggingFace Spaces
- **Kubernetes + Helm** — Helm chart included for K8s deployment
- **GitHub Actions** — full CI/CD: test → build → deploy on every push
- **Vercel** — frontend CDN deployment
- **Observability** — P50/P95/P99 latency tracking, request logs, model health endpoints

---

## Project Structure

```
molecularml-platform/
├── backend/
│   ├── main.py                        # FastAPI app — all endpoints
│   ├── models/
│   │   ├── property_predictor.py      # PyTorch MLP + RDKit fingerprints
│   │   └── protein_search.py          # ESM-2 embeddings + FAISS index
│   ├── monitoring.py                  # P50/P95/P99 latency tracking
│   ├── tests/test_api.py              # API test suite (9 tests)
│   ├── Dockerfile                     # HuggingFace Spaces deployment
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx                    # Tab routing
│       ├── api.js                     # Centralised API client
│       └── components/
│           ├── MoleculePredictor.jsx  # SMILES input + property results
│           ├── ProteinSearch.jsx      # Sequence input + similarity results
│           └── Dashboard.jsx          # Real-time analytics
├── helm/values.yaml                   # Kubernetes Helm chart
├── scripts/deploy_hf.py               # HuggingFace deploy script
├── architecture.svg                   # Architecture diagram
└── .github/workflows/ci.yml           # GitHub Actions CI/CD
```

---

## API Reference

```bash
GET  /health              # system health, model status, uptime
GET  /metrics             # P50/P95/P99 latency, request counts
POST /predict             # molecular property prediction
POST /search              # protein similarity search
GET  /molecules/examples  # example SMILES strings
GET  /proteins/examples   # example protein sequences
```

### Predict molecular properties
```bash
curl -X POST https://saitejasrivilli-molecularml-platform.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "properties": ["solubility", "lipophilicity", "toxicity"]
  }'
```

**Response:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": {
    "properties": {
      "solubility":    { "value": -2.14, "unit": "log(mol/L)",     "interpretation": "moderate" },
      "lipophilicity": { "value":  1.82, "unit": "log D (pH 7.4)", "interpretation": "moderate" },
      "toxicity":      { "value":  0.18, "unit": "probability",    "interpretation": "low risk" }
    },
    "descriptors": { "molecular_weight": 180.16, "qed_score": 0.553, "logp": 1.31 },
    "lipinski_rule_of_five": { "passes": true, "violations": 0, "drug_like": true }
  },
  "latency_ms": 43.2
}
```

### Search similar proteins
```bash
curl -X POST https://saitejasrivilli-molecularml-platform.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT", "top_k": 3}'
```

---

## CI/CD Pipeline

Every push to `master` runs the full pipeline:

```
push to master
    │
    ├── test-backend   (pytest · Python 3.11)
    ├── test-frontend  (npm install · vite build)
    │
    ├── deploy-backend  ──→  HuggingFace Spaces (Docker)
    └── deploy-frontend ──→  Vercel (CDN)
```

Secrets managed via GitHub Actions: `HF_TOKEN`, `HF_USERNAME`, `VERCEL_TOKEN`, `VITE_API_URL`.

---

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
# Swagger UI: http://localhost:7860/docs
```

### Frontend
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:7860" > .env.local
npm run dev
# App: http://localhost:3000
```

### Docker
```bash
cd backend
docker build -t molecularml-backend .
docker run -p 7860:7860 molecularml-backend
```

### Kubernetes
```bash
helm install molecularml ./helm --values helm/values.yaml
```

---

## Why This Project

Schrödinger, Recursion, Insilico Medicine, and similar companies need ML engineers who can build *platform infrastructure* — not just train models. This project demonstrates:

- **End-to-end production ML** — raw SMILES/sequence input → served predictions with latency tracking
- **Scientific ML domain** — molecular fingerprints, protein language models, drug-likeness scoring
- **Infrastructure breadth** — Docker, Kubernetes, CI/CD, observability, REST APIs, React frontend
- **Full-stack delivery** — deployed, live, accessible via public URL

---

## License

MIT

---

*Built by [Sai Teja Srivillibhutturu](https://linkedin.com/in/saitejasrivillibhutturu) · [saitejasrivilli.github.io](https://saitejasrivilli.github.io)*
