# Quick Start Guide - 5 Minutes to Running

## 1. Install (2 min)

```bash
cd fleet-doc-system
pip install -r requirements.txt
```

**Note:** First run downloads OCR models (~1.5GB). Subsequent runs instant.

## 2. Configure (1 min)

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```
FEATHERLESS_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## 3. Run (2 min)

**Terminal 1 - Backend:**
```bash
python src/layer6_api/server.py
# Output: INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend:**
```bash
open frontend/index.html
# or: python -m http.server 8000 -d frontend
```

## 4. Query

Visit `http://localhost:8000` and try:
- "What maintenance did truck T-084 have?"
- "How much fuel did T-127 use?"
- "What is the VIN for truck 42?"

## Next Steps

- **Test everything**: `python test_e2e.py`
- **Ingest documents**: `python batch_ingest.py --input-dir data/incoming`
- **View logs**: `tail -f logs/fleet.log`
- **Deploy**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## Troubleshooting

**"ModuleNotFoundError: No module named 'easyocr'"**
- Run: `pip install easyocr`
- First import downloads model (~1.5GB)

**"FEATHERLESS_API_KEY not set"**
- Check `.env` file exists with valid key
- Restart backend after changing `.env`

**"Connection refused" when querying**
- Ensure backend is running on Terminal 1
- Check `http://localhost:8000/health`

**Tavily search returning empty?**
- API key might be invalid or out of credits
- System falls back to local store automatically

## Features

✅ 50+ documents/week processing
✅ Zero hallucination guarantee (grounded responses)
✅ Beautiful minimalistic UI
✅ Powered by Featherless AI + Tavily
✅ Production-ready

For detailed docs, see [README.md](README.md) and [DEPLOYMENT.md](DEPLOYMENT.md).
