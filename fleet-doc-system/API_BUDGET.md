# API Budget Tracking - Buildathon Dallas 2026

## Provided Credits

| Service | Budget | Link |
|---------|--------|------|
| **Featherless AI** | $25 | https://featherlessai.com |
| **Tavily Search** | $2,000 | https://tavily.com |

## Integration Status

### ✅ Featherless AI (Fully Integrated)

**What it does:**
- Intent classification (Layer 4: Query Routing)
- Answer generation (Layer 4: Response Generator)

**Implementation:**
- `src/utils/llm_client.py`: FeatherlessAIClient class
- Base URL: `https://api.featherlessai.com/v1/chat/completions`
- Model: `Qwen/Qwen3-14B-NoThinking`
- Auth: Bearer token via `FEATHERLESS_API_KEY`

**Token Usage Per Query:**
- Classification: ~20 tokens (input) + ~5 tokens (output) = ~25 tokens
- Answer generation: ~150 tokens (input) + ~50 tokens (output) = ~200 tokens
- **Total per query: ~225 tokens**

**Cost Estimate:**
- Token pricing: ~$0.00044 per 1K tokens (Qwen3 14B)
- Per query: 225 tokens × $0.00044 / 1000 = **~$0.0001 per query**
- Weekly capacity: $25 / $0.0001 = **250,000 queries/week** ✅

### ✅ Tavily Search (Fully Integrated)

**What it does:**
- Web search fallback (Layer 3: Retrieval)
- Called when local document store returns no results

**Implementation:**
- `src/layer3_retrieval/tavily_rag.py`: TavilyRetriever class
- Base URL: `https://api.tavily.com/search`
- Auth: API key via `TAVILY_API_KEY`

**Credit Usage Per Query:**
- 1 credit per search
- Used as fallback only (not for every query)
- Estimated: 5 searches per day in production (10% of 50+ queries/week)

**Cost Estimate:**
- Per search: 1 credit
- Weekly (5 searches): 5 credits
- Weekly capacity: 2,000 / 5 = **400 weeks of operation** ✅

## Actual Usage in System

### Request Flow

```
1. User submits query
   ↓
2. Layer 4: QueryRouter.route_query() 
   → Calls Featherless AI for intent classification
   → Token usage: ~25 tokens
   ↓
3. Layer 3: Retrieval
   a) Try LocalDocumentStore.search_by_truck()
      → No API cost
   b) If empty, call TavilyRetriever.retrieve()
      → Calls Tavily API (1 credit per search)
   ↓
4. Layer 4: ResponseGenerator.generate_answer()
   → Calls Featherless AI to generate answer
   → Token usage: ~200 tokens
   ↓
5. Layer 5: GroundingVerifier.verify_answer()
   → No API calls (local verification)
   ↓
6. Return response to user
```

## Budget Tracking

### Daily Log Format

```json
{
  "date": "2026-06-18",
  "featherless": {
    "queries": 42,
    "tokens_used": 9450,
    "estimated_cost": "$0.00415",
    "intent_classifications": 42,
    "answer_generations": 42
  },
  "tavily": {
    "searches": 3,
    "credits_used": 3,
    "fallback_rate": "7.1%"
  }
}
```

### Monitoring Commands

**Check Featherless AI Usage:**
```bash
# Visit: https://console.featherlessai.com/usage
# Or via API (after implementation)
curl -H "Authorization: Bearer $FEATHERLESS_API_KEY" \
  https://api.featherlessai.com/v1/usage
```

**Check Tavily Usage:**
```bash
# Visit: https://tavily.com/dashboard
# API endpoint available in Tavily dashboard
```

## Cost Optimization Strategies

### 1. Reduce Token Usage (Featherless)
- ✅ Implemented: Minimal prompts (50-300 tokens total)
- ✅ Implemented: Deterministic classification (temp=0.0)
- ✅ Implemented: Context window optimization

### 2. Reduce Tavily Calls
- ✅ Implemented: LocalDocumentStore as primary (instant, free)
- ✅ Implemented: Tavily as fallback only (~10% of queries)
- ✅ Implemented: Batch web searches for multiple queries

### 3. Response Caching
- ✅ Implemented: In-memory cache for identical queries
- TTL: 300 seconds (configurable in `.env`)
- Expected hit rate: 20-30% of queries

## Weekly Budget Example (Production Scale)

**Scenario: 50 queries/week**

### Featherless AI
- Queries: 50
- Tokens: 50 × 225 = 11,250 tokens
- Cost: 11,250 × $0.00044 / 1000 = **$0.0049/week**
- Budget usage: $0.0049 / $25 = **0.02%** ✅

### Tavily Search
- Queries: 50
- Fallback rate: 10% = 5 searches
- Cost: 5 × 1 credit = **5 credits/week**
- Budget usage: 5 / 2,000 = **0.25%** ✅

**Total weekly cost: ~$0.005 (0.27% of combined budget)**
**Runway: 370+ weeks** ✅✅✅

## Production Deployment Checklist

- [ ] Set `FEATHERLESS_API_KEY` environment variable
- [ ] Set `TAVILY_API_KEY` environment variable
- [ ] Configure logging to track API usage
- [ ] Set up monitoring dashboard (optional)
- [ ] Implement daily usage reports (optional)
- [ ] Set usage alerts at 80% budget (optional)

## API Endpoints Reference

### Featherless AI

**Endpoint:** `https://api.featherlessai.com/v1/chat/completions`

**Request:**
```bash
curl -X POST https://api.featherlessai.com/v1/chat/completions \
  -H "Authorization: Bearer $FEATHERLESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-14B-NoThinking",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 300,
    "temperature": 0.0
  }'
```

### Tavily Search

**Endpoint:** `https://api.tavily.com/search`

**Request:**
```bash
curl -X POST https://api.tavily.com/search \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "$TAVILY_API_KEY",
    "query": "truck maintenance",
    "include_answer": true,
    "max_results": 5
  }'
```

## Support

**Questions about usage?**
- Featherless: https://featherlessai.com/docs
- Tavily: https://docs.tavily.com

**Buildathon issues?**
- Check QUICKSTART.md for setup
- Run `python verify_buildathon_apis.py` to verify integration
- Review logs: `tail -f logs/fleet.log`
