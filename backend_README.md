---
title: MolecularML Platform API
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# MolecularML Platform — Backend API

FastAPI backend for molecular property prediction and protein similarity search.

- `POST /predict` — molecular property prediction (solubility, lipophilicity, toxicity)
- `POST /search` — protein similarity search via ESM-2 + FAISS
- `GET /health` — system health and model status
- `GET /metrics` — P50/P95/P99 latency, request counts

**Frontend:** deployed separately on Vercel.

**Docs:** `/docs` (Swagger UI available after deployment)
