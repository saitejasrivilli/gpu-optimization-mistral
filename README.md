# 🎙️ Multi-Agent Voice AI — Google ADK

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Google ADK](https://img.shields.io/badge/Google%20ADK-1.0-orange.svg)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A multi-agent voice AI system built with Google ADK, featuring domain-specialized agents for Healthcare, Finance, and Retail — with parallel tool execution, BM25 RAG, per-user rate limiting, safety guardrails, and a live metrics dashboard.

**[Live Demo](https://huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk)** — no setup required.

---

## What This Is

Most agent demos route every query to a single general-purpose LLM and call it multi-agent. This system does something different: a **Planner agent** classifies intent and selects tools, then a domain-specific **Executor agent** runs those tools in parallel and synthesizes the result. Each domain (Healthcare, Finance, Retail, General) has its own system prompt, toolset, and constraints.

The practical consequence: a healthcare query routes to `drug_interaction_check` + RAG over uploaded clinical docs, while a finance query routes to `financial_risk_score` + `stock_price` + `calculator` — running concurrently, not sequentially. The Planner never touches domain logic; the Executor never touches routing logic. Each agent has one job.

This separation matters for enterprise deployment: when a domain's tools or compliance requirements change, you update one agent without touching the others.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
│              (Voice / Text / Uploaded Doc)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  SAFETY GUARDRAILS                           │
│         (regex patterns · length limits · PII)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              GOOGLE ADK — PLANNER AGENT                      │
│                  (LLaMA 3.3 70B / Groq)                     │
│  Identifies: INTENT · TOOLS_NEEDED · QUERIES · TASK         │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐       ┌──────────────────┐
  │ PARALLEL TOOL │       │  DOMAIN CONTEXT  │
  │  EXECUTION    │       │  (Healthcare /   │
  │               │       │  Finance/Retail) │
  │ web_search    │       └──────────────────┘
  │ get_weather   │
  │ calculate     │
  │ stock_price   │
  │ BM25 RAG      │
  │ drug_check    │
  │ risk_score    │
  │ sentiment     │
  └───────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│            GOOGLE ADK — EXECUTOR AGENT                       │
│                  (LLaMA 3.3 70B / Groq)                     │
│     Synthesizes tool results + conversation context          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                REASONING TRACE GENERATOR                     │
│          Step 1 → Step 2 → Step 3 → Conclusion              │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐          ┌──────────────┐
   │  TEXT OUTPUT │          │ VOICE OUTPUT │
   │  (Gradio UI) │          │  (pyttsx3 /  │
   └─────────────┘          │    gTTS)     │
                             └──────────────┘
```

---

## Domain Agents

| Domain      | Tools                                              | Example Query                              |
|-------------|----------------------------------------------------|--------------------------------------------|
| General     | web_search, weather, calculator, stocks            | "What's the weather in Tokyo?"             |
| Healthcare  | drug_interaction_check, web_search, RAG            | "Check interaction between aspirin and ibuprofen" |
| Finance     | financial_risk_score, stock_price, calculator      | "Risk score for AAPL, TSLA, GOOGL"        |
| Retail      | product_sentiment, web_search, calculator          | "Product sentiment for Nike shoes"         |

Each domain has an isolated system prompt and toolset. Switching domains does not share context or tools across sessions.

---

## Key Implementation Details

**Parallel tool execution**

When a query needs multiple tools, they run concurrently via `ThreadPoolExecutor`. A finance query needing both stock price and web search fires both simultaneously — not sequentially.

```python
futures = {
    executor_pool.submit(run_one, tool, query): tool
    for tool, query in zip(tools_needed, queries)
}
```

**BM25 RAG (custom implementation)**

Upload `.pdf`, `.txt`, or `.md` files and query them in natural language. Uses full BM25 scoring with TF normalization and document length penalty — not keyword overlap.

```
Score = Σ IDF(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

**Per-user rate limiting (token bucket)**

Each user gets an isolated token bucket: 10 token capacity, refill at 1 token/6 seconds (~10 req/min). Burst traffic is absorbed; sustained overuse is throttled with an HTTP-style wait-time response. Rate limit events are tracked in the metrics dashboard.

**Safety guardrails**

Input validation blocks injection patterns (`sql injection`, `hack`, `exploit`), harmful content, PII (SSN, credit card formats), and oversized inputs (>2000 chars) before any LLM call.

**Session isolation**

Each user gets a UUID-keyed ADK `InMemoryRunner`. State — including multi-turn conversation context — is isolated per session. No cross-user state leakage.

**Reasoning traces**

Every query produces an explicit step-by-step reasoning chain before the final answer:

```
Step 1: User is asking about AAPL stock volatility
Step 2: Retrieved 30-day price history, computed daily returns
Step 3: Annualized std dev = 22.3% → Medium risk tier
```

---

## Production Patterns

| Pattern               | Implementation                                      |
|-----------------------|-----------------------------------------------------|
| Rate limiting         | Token bucket, per-user isolated (10 req/min)        |
| Safety                | Pre-LLM input validation, PII blocking              |
| Observability         | P95 latency, token usage, tool call distribution    |
| Session management    | UUID-keyed InMemoryRunner, multi-user isolation     |
| Fault tolerance       | pyttsx3 offline TTS with gTTS fallback              |
| Parallel execution    | ThreadPoolExecutor across tool calls                |

---

## Live Metrics Dashboard

Real-time tracking visible in the UI:
- Average and P95 query latency
- Token usage per query
- Tool call distribution across domains
- Guardrail block count and error rates

---

## Quick Start

**Run on Hugging Face Spaces — no setup:**
👉 [huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk](https://huggingface.co/spaces/SaiTejaSrivilli/voice-agent-adk)

**Run locally:**
```bash
git clone https://github.com/saitejasrivilli/voice-agent-adk
cd voice-agent-adk
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"  # Free at console.groq.com
python app.py
# Open http://localhost:7860
```

---

## Tech Stack

| Component        | Technology                            |
|-----------------|---------------------------------------|
| Agent Framework  | Google ADK 1.0                        |
| LLM              | LLaMA 3.3 70B via Groq               |
| Speech-to-Text   | Groq Whisper large-v3-turbo           |
| Text-to-Speech   | pyttsx3 (offline) + gTTS fallback     |
| RAG              | BM25 scoring (custom implementation)  |
| Parallel Execution | Python ThreadPoolExecutor           |
| UI               | Gradio 6.x                            |
| Deployment       | Hugging Face Spaces (Python 3.11)     |
| Weather API      | Open-Meteo (free, no key)             |
| Finance API      | Yahoo Finance (free, no key)          |
| Search           | DuckDuckGo HTML scraping              |

---

## Project Structure

```
voice-agent-adk/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (espeak for TTS)
├── .python-version     # Python 3.11 pin for HF Spaces
└── README.md
```

---

## Environment Variables

| Variable       | Required | Description                    |
|---------------|----------|--------------------------------|
| GROQ_API_KEY   | ✅       | Free at console.groq.com       |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Sai Teja Srivillibhutturu · [GitHub](https://github.com/saitejasrivilli) · [LinkedIn](https://linkedin.com/in/saitejasrivilli)*
