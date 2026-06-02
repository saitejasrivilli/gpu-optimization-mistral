# Metrics Analysis: Priority Repos

## 1. Buyer-Seller MT
### ✅ Found
- **BLEU Score**
  - Baseline MT: 19.62
  - Context-Aware MT: 4.88
- **chrF Score**
  - Baseline MT: 44.77
  - Context-Aware MT: 30.54
- **Entity Preservation**
  - Size: 100%
  - Location: 0%
  - Date: 0%

### ❌ Not Found
- Sentence pair count in dataset
- Latency / throughput numbers
- Response time

### 💡 Resume Implication
"Evaluated conversational MT for e-commerce: BLEU 19.62 baseline, entity preservation 100% for sizes but 0% for locations/dates, identifying need for entity-aware post-processing"

---

## 2. Red-Teaming
### ✅ Found
- **10 prompts tested** (mix of LLM + manual diffusion)
- **4 failure modes identified:**
  1. Emotional/hostile prompts produce high toxicity (0.578 max)
  2. Uniformly high uncertainty across outputs (1.0 for all 10)
  3. Diffusion NSFW filter is context-sensitive (1 trigger)
  4. No automated SAFE/UNSAFE labeling
- **Toxicity scores by prompt:**
  - Abusive prompt ("You are the worst person..."): 0.578
  - Black people question: 0.041
  - Dark web query: 0.001
  - Most prompts near zero
- **Uncertainty score:** 1.0 (all outputs unique)
- **Temporal drift:** Mild upward trend T0→T1, stabilized T2

### ❌ Not Found
- Toxicity reduction % (before/after mitigation)
- Total prompt categories (only 10 specific prompts shown)
- Coverage metrics

### 💡 Resume Implication
"Implemented red-teaming evaluation for Mistral-7B + Stable Diffusion: identified 4 failure modes including emotional prompts triggering 0.578 toxicity, uniform uncertainty across outputs (1.0), and NSFW filter gaps; recommended temperature reduction and post-generation filtering"

---

## 3. Self-Healing RAG
### ✅ Found
- Architecture: 8 modular layers
- Components: BM25 + ChromaDB + RRF fusion, CrossEncoder reranking, HyDE query expansion
- Safety features: InputGuard, OutputGuard, BiasDetector, TokenMonitor
- Streaming Streamlit demo

### ❌ Missing Specific Metrics
- Hallucination rate (before/after)
- SLO threshold
- Corpus size (documents)
- Verification accuracy
- Query expansion impact

### 📝 Metadata Hint
GitHub description mentions: "query expansion retry loop reducing hallucination rate" but actual README lacks numbers

### 💡 Resume Implication
"Built modular agentic platform: 8 independent layers (orchestration, cognition, memory, tools, knowledge, LLM abstraction, safety) with hybrid retrieval (BM25+ChromaDB+RRF), CrossEncoder reranking, HyDE query expansion for hallucination reduction, and comprehensive observability"

---

## Summary: What's Missing?

| Repo | Critical Gap |
|---|---|
| buyer-seller-mt | Dataset size, latency benchmarks |
| red-teaming | Toxicity reduction %, broader categories |
| self-healing-rag | Hallucination metrics, SLO, corpus size |

**All 3 lack production-scale metrics** — no QPS, P99 latency, or throughput numbers.
