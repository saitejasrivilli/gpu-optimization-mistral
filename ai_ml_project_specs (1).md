# AI/ML Portfolio — Full Project Specifications

**Hardware:** 4x NVIDIA A30 (24GB VRAM each, 96GB total) + Google Colab Pro  
**Goal:** Maximum depth across 4 projects covering the key gaps in cutting-edge AI/ML

---

## Project 1 — Reasoning Fine-Tuning Lab
### DPO + GRPO + Process Reward Model on a 7B base

**One-line pitch:** A rigorous comparison of post-training alignment methods with a working process reward model and test-time compute scaling — on your own hardware.

**Gaps covered:** DPO/GRPO, reward modeling, test-time compute, synthetic data generation, RLAIF

---

### Phase 1 — Environment & Baseline (Week 1)

**Goal:** Reproducible training environment and a clean baseline to measure against.

**Steps:**
1. Set up DeepSpeed ZeRO-2 across 2x A30s. Verify inter-GPU communication with a small test run.
2. Download Qwen2.5-7B-Instruct as your base model (strong reasoning baseline, permissive license).
3. Run baseline evaluation on GSM8K, MATH-500, and HumanEval. Record pass@1 for each. These are your floor numbers — every subsequent experiment is measured against them.
4. Profile baseline throughput: tokens/sec during training, GPU memory usage, time per gradient step.
5. Set up Weights & Biases logging. Every run from here logs: loss curve, reward mean/variance, eval accuracy per checkpoint.

**What to record:**
- Baseline GSM8K pass@1, MATH-500 accuracy, HumanEval pass@1
- Training throughput (tokens/sec) with 2x A30 DeepSpeed ZeRO-2
- Peak VRAM usage per GPU

---

### Phase 2 — Synthetic Data Pipeline (Week 1–2)

**Goal:** Generate preference pairs and reasoning traces without paying for large API calls.

**Steps:**
1. Use Gemini Flash (free tier, generous limits) or GPT-4o-mini to generate 5,000–10,000 math and code problem/solution pairs.
2. For DPO: generate 2 responses per prompt, use a rule-based verifier (answer correctness for math, unit tests for code) to label chosen/rejected automatically. No human labeling needed.
3. For GRPO: you need a scalar reward signal only. Use the same verifier — correct answer = +1, wrong = −1, format violation = −0.5.
4. For PRM training data: use the "process supervision" approach — generate step-by-step solutions, then use the verifier to label each intermediate step as correct/incorrect by checking if a correct solution is reachable from that step (Monte Carlo rollouts).
5. Log dataset statistics: distribution of problem difficulty, chosen/rejected quality gap, step-level label distribution.

**What to record:**
- Dataset size and composition
- Label agreement rate between your verifier and ground truth on a held-out sample (100 problems)
- Cost of data generation (should be near-zero)

---

### Phase 3 — SFT → DPO → GRPO Ladder (Week 2–3)

**Goal:** The core ablation. Same base, same data, three methods. Quantify each step's contribution.

**Steps:**

**SFT (2 days):**
1. Fine-tune Qwen2.5-7B on your generated correct solutions using QLoRA (r=32, alpha=64, 4-bit base).
2. Train for 2–3 epochs, early stop on validation loss.
3. Evaluate on all three benchmarks. Record delta vs baseline.

**DPO (2 days):**
1. Using TRL's DPOTrainer, run DPO on top of your SFT model with beta=0.1.
2. Sweep beta in {0.05, 0.1, 0.2} — record how KL divergence from reference changes.
3. Evaluate on all three benchmarks.

**GRPO (3 days):**
1. Using TRL's GRPOTrainer (or implement from scratch for depth), run GRPO with your rule-based verifier as the reward function.
2. Key hyperparameter: group size G (number of completions per prompt). Run G=4, G=8, G=16.
3. Evaluate on all three benchmarks per group size.

**What to record:**
- GSM8K pass@1, MATH-500 accuracy, HumanEval pass@1 at each stage: Base → SFT → DPO → GRPO
- Training compute per stage (GPU-hours)
- GRPO: reward mean and variance per training step for each G value
- DPO: KL divergence from reference model at convergence
- Plot: accuracy vs training steps for all three methods on the same axes

**The non-obvious finding to aim for:** GRPO's advantage over DPO is not uniform — it's largest on harder problems (MATH level 4–5) and nearly zero on easy ones (GSM8K). Show this difficulty-stratified breakdown.

---

### Phase 4 — Process Reward Model (Week 3–4)

**Goal:** A working PRM that scores individual reasoning steps, not just final answers.

**Steps:**
1. Fine-tune a smaller model (Qwen2.5-1.5B or 3B) as your PRM using your step-labeled data from Phase 2. This is a binary classifier per step.
2. Evaluate PRM accuracy: on a held-out set of step-labeled problems, what fraction of steps does it correctly label as correct/incorrect? Report this as your PRM quality metric.
3. Measure correlation: does a higher average PRM score on a solution correlate with final answer correctness? Report Pearson r.

**What to record:**
- PRM step-level accuracy (binary classification accuracy on held-out labeled steps)
- Correlation between PRM score and answer correctness (Pearson r)
- PRM inference latency (ms per step) — this matters for best-of-N

---

### Phase 5 — Test-Time Compute Scaling (Week 4)

**Goal:** Show that more inference compute improves accuracy, and find the optimal N.

**Steps:**
1. Use your best GRPO model + PRM to implement Best-of-N: generate N solutions, score each with PRM, pick the highest-scoring one.
2. Sweep N ∈ {1, 4, 8, 16, 32, 64}. For each N, record accuracy on MATH-500 and compute cost (total tokens generated).
3. Plot the scaling curve: accuracy vs N. It should rise then plateau.
4. Calculate compute-optimal N: the point where marginal accuracy gain per additional sample drops below 1%.
5. Compare Best-of-N with PRM vs Best-of-N with majority voting (no PRM). Quantify how much the PRM adds.

**What to record:**
- Accuracy at N = 1, 4, 8, 16, 32, 64 on MATH-500
- Scaling curve plot (accuracy vs log N)
- Compute-optimal N value with justification
- PRM-guided vs majority-vote accuracy gap at each N

---

### Phase 6 — Failure Analysis (Week 4)

**Goal:** Honest documentation of where and why each method fails.

**Steps:**
1. **Reward hacking:** Manually inspect 50 GRPO solutions the verifier rewarded but that are mathematically wrong. Categorize: format tricks, partial solutions, edge case exploitation. Quantify the rate.
2. **GRPO instability:** Plot reward variance across training. Identify runs where variance explodes. Show what group size G most correlates with instability.
3. **PRM failure modes:** Find 20 examples where PRM scores a wrong solution higher than a correct one. Categorize: hallucinated intermediate steps that look plausible, early divergence not caught until late.
4. **Distribution shift:** Evaluate your GRPO model on MMLU and TruthfulQA (general benchmarks). Quantify how much general capability degraded after reasoning-focused fine-tuning.

**What to record:**
- Reward hacking rate (% of rewarded solutions that are actually wrong)
- GRPO instability rate by group size G
- PRM false positive rate on wrong solutions
- MMLU accuracy before and after GRPO fine-tuning (capability regression)

---

### README Must Answer
- Why GRPO over PPO? (Empirical answer from your runs, not just theory)
- What is the minimum dataset size where DPO training is stable?
- What G value is the practical sweet spot and why?
- Where does your PRM fail and how would you fix it?
- Is the compute cost of Best-of-N worth it vs just training longer?

---

## Project 2 — GraphRAG + Agentic Retrieval System
### Knowledge graph meets multi-hop RAG with a reusable eval framework

**One-line pitch:** A systematic comparison of retrieval strategies — BM25, dense, graph, hybrid, agentic — on multi-hop questions, with a calibrated LLM-as-judge eval harness you can reuse across projects.

**Gaps covered:** GraphRAG, agentic RAG, hybrid search, LLM-as-judge evals, RAG eval framework

---

### Phase 1 — Data & Benchmark Setup (Week 1)

**Goal:** A fixed evaluation set so every retrieval experiment is comparable.

**Steps:**
1. Choose your corpus: Wikipedia subsets work well (science, history). Target 50,000–200,000 passages. This is enough to make retrieval non-trivial.
2. Choose your benchmark: **MuSiQue** (requires 2–4 hop reasoning, has gold supporting facts) or **HotpotQA** (2-hop, widely used). Use the dev set — don't touch test until the end.
3. Split questions by hop count: 1-hop, 2-hop, 3+-hop. You'll report metrics per category — this is where the interesting differences appear.
4. Set up your retrieval evaluation harness: given a question, each retrieval method returns top-K passages. Compute Recall@5 (are the gold supporting passages in the top 5?) and Precision@5 for every method uniformly.

**What to record:**
- Dataset statistics: corpus size, question count per hop category
- Gold passage distribution: are supporting passages clustered or spread?

---

### Phase 2 — Baseline Retrievers (Week 1–2)

**Goal:** Establish the BM25 and dense retrieval baselines.

**Steps:**

**BM25:**
1. Index your corpus with `rank_bm25` or Elasticsearch (local).
2. Run retrieval on all dev questions. Record Recall@5 and Precision@5 per hop category.
3. Note: BM25 should do well on 1-hop (keyword match) and poorly on 3+-hop.

**Dense retrieval:**
1. Embed corpus with `bge-m3` or `e5-mistral-7b-instruct` (run on A30, these fit easily).
2. Index with FAISS or Qdrant (local). Use HNSW for approximate nearest neighbor.
3. Run retrieval on all dev questions. Record same metrics.
4. Report embedding throughput (passages/sec) and index memory size.

**Hybrid (BM25 + dense):**
1. Combine scores with Reciprocal Rank Fusion (RRF) — simple, no training needed.
2. Tune the RRF k parameter on a small validation subset.
3. Record metrics.

**What to record:**
- Recall@5, Precision@5 for BM25 / Dense / Hybrid — broken down by hop count
- Embedding time for full corpus
- Index size on disk

---

### Phase 3 — Graph Construction (Week 2)

**Goal:** Build a knowledge graph from your corpus. Measure its quality.

**Steps:**
1. Use a local LLM (Qwen2.5-7B on A30) to extract entity-relation-entity triples from each passage. Prompt: "Extract all factual relationships as (entity, relation, entity) triples."
2. Deduplicate and normalize entity names (simple string matching + embedding similarity).
3. Load into Neo4j (free local install) or NetworkX for small corpora.
4. **Measure graph quality:** Sample 200 passages and manually verify: what fraction of the important entities were extracted? What fraction of extracted relations are correct? Report Entity Recall and Relation Precision.
5. **Graph statistics:** number of nodes, edges, average degree, diameter, density. Plot degree distribution.

**What to record:**
- Entity recall (fraction of important entities captured)
- Relation precision (fraction of extracted relations that are correct)
- Graph scale: nodes, edges, average degree
- Construction time and cost (GPU-hours for LLM extraction)

---

### Phase 4 — Graph Retrieval (Week 2–3)

**Goal:** Use the graph to improve multi-hop retrieval.

**Steps:**
1. **Entity linking:** Given a question, identify the entities mentioned, then find them in the graph.
2. **Graph traversal retrieval:** Starting from question entities, traverse up to K hops and collect all connected passages. Return those as candidates.
3. **Combined graph + dense:** Use graph traversal to get candidates, then re-rank with dense embeddings. This is the key GraphRAG approach.
4. Run on all dev questions. Record Recall@5 and Precision@5.
5. **Critical experiment:** Vary graph completeness by randomly dropping edges (90%, 70%, 50% of edges). Show how retrieval quality degrades. This answers "how complete does the graph need to be?"

**What to record:**
- Recall@5, Precision@5 for graph retrieval and graph+dense — by hop count
- Graph completeness sensitivity curve (retrieval quality vs % edges retained)
- Average retrieval latency per query (ms)

---

### Phase 5 — Agentic Multi-Hop Retrieval (Week 3)

**Goal:** An agent that decides what to retrieve next based on what it has found so far.

**Steps:**
1. Implement a simple retrieval agent loop using LangGraph:
   - State: {question, retrieved passages so far, reasoning trace}
   - Action space: {retrieve(query), answer(response), give_up}
   - At each step, the LLM decides the next query based on what's been retrieved
2. Run with your best retriever (hybrid) as the underlying retrieval function.
3. Cap at 4 retrieval steps to control latency.
4. **Ablation:** single-shot retrieval (1 call) vs 2-step agentic vs 4-step agentic. Record accuracy AND latency for each.
5. Find the crossover: on what question types does 4-step agentic beat single-shot? On what types is it not worth the latency?

**What to record:**
- Answer F1 for: single-shot / 2-step / 4-step agentic — by hop count
- Average latency per query for each (ms)
- Crossover analysis: which question types benefit most from agentic retrieval
- Failure cases: where does the agent loop in circles or retrieve irrelevant passages?

---

### Phase 6 — LLM-as-Judge Eval Harness (Week 3–4)

**Goal:** A calibrated, reusable judge that you'll use across Projects 1–4.

**Steps:**
1. Build a judge using local Qwen2.5-7B that scores RAG answers on three dimensions:
   - **Faithfulness:** Is the answer supported by the retrieved passages? (1–5)
   - **Relevance:** Does the answer address the question? (1–5)
   - **Completeness:** Does the answer cover all required facts? (1–5)
2. **Calibrate the judge:** Collect 100 human ratings on the same answers (you or a collaborator). Compute Spearman rank correlation between judge scores and human scores per dimension.
3. **Find failure modes:** Systematically find question types where the judge is poorly calibrated. Common failures: long answers (judge rewards length), technical jargon (judge can't verify), ambiguous questions (judge inconsistent).
4. **Prompt sensitivity test:** Run the same judge with 3 slightly different prompts. Report variance in scores. High variance = unreliable judge.

**What to record:**
- Spearman r between judge and human ratings per dimension (faithfulness, relevance, completeness)
- Judge variance across 3 prompt variants
- Identified failure mode categories and their frequency
- Judge latency (ms per answer) — important for running it at scale

---

### Phase 7 — Full Comparison & Failure Analysis (Week 4)

**Goal:** The definitive table and the honest failure story.

**Steps:**
1. Run all methods end-to-end (BM25 → Dense → Hybrid → GraphRAG → Agentic GraphRAG) and report Answer F1 by hop count using your LLM judge.
2. **Graph sparsity failure:** Show 10 concrete examples where GraphRAG fails because the relevant entity connection is missing from the graph.
3. **Latency vs accuracy Pareto:** Plot each method as a point (latency, accuracy). The Pareto frontier shows which methods are efficient.
4. **Decision guide:** Based on your results, write clear rules: "Use GraphRAG when X, use hybrid when Y, use agentic when Z."

**What to record:**
- Full comparison table: method × hop count × metric (Recall@5, Precision@5, Answer F1, latency)
- Pareto plot (latency vs accuracy)
- Graph sparsity failure rate as a function of graph density
- Concrete decision rules backed by your numbers

---

### README Must Answer
- When is GraphRAG worth the construction cost vs just using hybrid retrieval?
- What graph density is needed for GraphRAG to beat hybrid?
- How many agentic steps is the practical sweet spot?
- How sensitive is the LLM judge to the prompt and the underlying model?

---

## Project 3 — VLM Fine-Tuning + Multimodal Eval Suite
### Vision-language fine-tuning with KV cache optimization and hallucination analysis

**One-line pitch:** Domain-specific VLM fine-tuning with rigorous ablations on LoRA rank, a KV cache throughput benchmark, and a hallucination breakdown that quantifies what fine-tuning actually fixes.

**Gaps covered:** VLMs, multimodal RAG, KV cache optimization, hallucination detection

---

### Phase 1 — Base Model Evaluation (Week 1)

**Goal:** Comprehensive baseline before any fine-tuning touches the model.

**Steps:**
1. Download Qwen2-VL-7B-Instruct (strong open VLM, fits on 2x A30 at 4-bit).
2. Run on standard benchmarks: MMMU (general multimodal), ScienceQA (science reasoning), TextVQA (text in images), MMBench (broad capability).
3. Run CHAIR hallucination evaluation on COCO captions: generate captions for 500 COCO images, compute CHAIR_s (sentence-level) and CHAIR_i (instance-level). This is your hallucination baseline.
4. Manually categorize 50 hallucinated captions into: object hallucination (things not present), attribute errors (wrong color/size/etc), relation errors (wrong spatial relationships). Record the distribution.

**What to record:**
- MMMU accuracy, ScienceQA accuracy, TextVQA accuracy, MMBench score
- CHAIR_s and CHAIR_i on COCO
- Hallucination type distribution (object / attribute / relation %)
- Inference latency and throughput (tokens/sec) on A30

---

### Phase 2 — Domain Dataset Preparation (Week 1)

**Goal:** A domain-specific image-text dataset to fine-tune on.

**Recommended domains (pick one — specificity beats breadth):**
- Medical imaging: use public chest X-ray datasets (NIH ChestX-ray14 + radiology reports)
- Scientific figures: arXiv papers with figure captions
- Document understanding: DocVQA dataset

**Steps:**
1. Download and preprocess your chosen dataset. Target 5,000–20,000 image-text pairs.
2. Split: 80% train, 10% validation, 10% test. Freeze the test set — don't look at it until final evaluation.
3. Format into instruction-following pairs: {"image": ..., "question": "Describe this image", "answer": "..."} — the standard VLM chat format.
4. Compute dataset statistics: image resolution distribution, answer length distribution, vocabulary overlap with base model training data.

**What to record:**
- Dataset size and split counts
- Image resolution statistics
- Domain coverage relative to COCO (how different is your domain?)

---

### Phase 3 — QLoRA Fine-Tuning (Week 2)

**Goal:** The fine-tuning ablation — find the optimal LoRA rank for this task.

**Steps:**
1. Fine-tune using LLaMA-Factory or ms-swift (both support Qwen2-VL natively) with QLoRA (4-bit base, bf16 adapters).
2. **LoRA rank sweep:** Train separate runs at r = 8, 16, 32, 64. For each:
   - Record: validation loss at convergence, domain benchmark score, trainable parameter count, training time, peak VRAM
3. Apply LoRA to both vision encoder and language model (not just LM) — this is important for VLMs and often skipped.
4. Use cosine LR schedule, warmup 3% of steps.
5. Evaluate each checkpoint on your domain test set AND on MMMU/ScienceQA (general benchmarks).

**What to record:**
- For each rank r: {trainable params, VRAM, train time, domain score, MMMU score}
- Plot: domain accuracy vs trainable parameter count (the efficiency curve)
- The rank where 90% of peak domain gain is achieved — this is your "practical sweet spot"
- Forgetting: MMMU score before vs after fine-tuning at each rank

**The non-obvious finding to aim for:** Higher LoRA rank hurts general benchmarks more than it helps domain benchmarks beyond a threshold. The rank-vs-forgetting tradeoff is the key insight.

---

### Phase 4 — Hallucination Analysis (Week 2–3)

**Goal:** Quantify exactly what fine-tuning fixes and what it makes worse.

**Steps:**
1. Run CHAIR on your best fine-tuned model (same 500 COCO images as baseline). Record CHAIR_s, CHAIR_i.
2. Re-categorize hallucinations: object / attribute / relation. Compare distribution before and after fine-tuning.
3. **The key experiment:** Fine-tuning on domain images should reduce hallucinations for domain-specific objects but may increase them for out-of-domain objects (the model trades general calibration for domain specificity). Test this by:
   - Run on 200 domain images (your fine-tuning domain): record hallucination rate
   - Run on 200 out-of-domain images (different domain): record hallucination rate
   - Compare before/after for both categories
4. **Calibration analysis:** Use temperature scaling. Does your fine-tuned model produce well-calibrated confidence scores? (Are high-confidence answers more likely to be correct?) Plot calibration curves.

**What to record:**
- CHAIR_s and CHAIR_i: base vs fine-tuned
- Hallucination rate on in-domain vs out-of-domain images: base vs fine-tuned
- Hallucination type distribution change after fine-tuning
- Calibration curves (reliability diagrams)

---

### Phase 5 — KV Cache & Inference Optimization (Week 3)

**Goal:** Concrete throughput numbers for KV cache and prefix caching. This is the inference engineering depth signal.

**Steps:**

**Setup:**
1. Serve your fine-tuned model with vLLM (supports Qwen2-VL).
2. Baseline: measure throughput (tokens/sec) and TTFT (time to first token) with no caching, batch size = 1, 4, 8, 16.

**KV Cache experiments:**
1. Enable and vary KV cache size (25%, 50%, 75%, 100% of available VRAM allocated to cache).
2. For each cache size: measure throughput and TTFT at batch size = 8.
3. Record GPU memory usage. Plot: cache size vs throughput vs memory.

**Prefix caching:**
1. Enable vLLM prefix caching.
2. Benchmark with a shared system prompt (common in VLM applications): run 100 requests with the same image prefix, measure cache hit rate, TTFT speedup vs no caching.
3. Vary prefix length (128, 256, 512 tokens). Report TTFT at each prefix length with and without caching.
4. **The key metric:** TTFT reduction ratio as a function of prefix length. Longer prefixes = bigger speedup.

**Grouped Query Attention (GQA):**
1. If Qwen2-VL uses GQA (it does), profile the KV cache memory footprint vs equivalent MHA.
2. Report: KV cache size (GB) for sequence lengths 512, 1024, 2048, 4096. Compare to theoretical MHA size.

**What to record:**
- Throughput (tokens/sec) at batch sizes 1, 4, 8, 16 — with and without prefix caching
- TTFT (ms) vs prefix length — with and without prefix caching
- KV cache size (GB) vs sequence length
- Memory-throughput Pareto: which configuration maximizes throughput within 20GB VRAM budget?

---

### Phase 6 — Multimodal RAG (Week 3–4)

**Goal:** Retrieve relevant images + text together, use VLM to answer.

**Steps:**
1. Build a multimodal index: embed both images (using CLIP or your VLM's vision encoder) and text passages into a shared FAISS index.
2. Given a query, retrieve top-5 image+text pairs.
3. Feed retrieved images + text as context to your fine-tuned VLM to answer.
4. Benchmark on a multimodal QA task: WebQA or your domain-specific QA set.
5. **Ablation:** text-only retrieval + VLM vs multimodal retrieval + VLM. Quantify how much the image retrieval adds.

**What to record:**
- QA accuracy: text-only retrieval vs multimodal retrieval
- Retrieval Recall@5 for text vs image modalities separately
- Latency breakdown: embedding + retrieval + VLM inference (ms each)

---

### README Must Answer
- What LoRA rank is optimal and why does higher rank hurt?
- Does fine-tuning reduce hallucinations or just shift them to different categories?
- When does prefix caching meaningfully help — what's the minimum prefix length?
- How much does adding image retrieval improve over text-only in multimodal RAG?

---

## Project 4 — Long-Horizon Computer-Use Agent
### Sandboxed browser/code agent with memory architecture and systematic tool-use evals

**One-line pitch:** A browser agent powered by your local VLM that tackles multi-step tasks, with a memory system ablation and a tool-use eval harness that quantifies exactly where agents fail at scale.

**Gaps covered:** Computer use agents, agent memory architectures, tool-use evals, sandboxed code execution

---

### Phase 1 — Core Agent Infrastructure (Week 1)

**Goal:** A working agent loop with proper sandboxing before any experiments.

**Steps:**
1. Set up Playwright (browser control) and E2B or Docker (sandboxed code execution). Test both independently.
2. Define your agent's action space: {click(element), type(text), scroll(direction), navigate(url), execute_code(code), search_web(query), done(answer), fail(reason)}.
3. Implement the basic agent loop with LangGraph:
   - Observe: screenshot → Qwen2-VL-7B describes the current state
   - Think: LLM decides next action given state + history
   - Act: execute the action via Playwright or E2B
   - Repeat until done or max steps reached (cap at 15 steps)
4. Instrument everything: log every action, observation, and decision with timestamps. You'll analyze these logs for the failure analysis.
5. Run 10 manual test tasks to verify the loop works before any systematic evaluation.

**What to record:**
- Action success rate per action type in your 10 test tasks
- Average latency per agent step (ms): screenshot capture + VLM inference + action execution
- Any systematic errors in the basic loop

---

### Phase 2 — Task Benchmark Setup (Week 1–2)

**Goal:** A fixed eval set that fairly measures agent capability.

**Steps:**
1. Use **WebArena** (50 tasks across web apps) or build your own task set of 50 tasks spanning:
   - Navigation tasks (5–10 steps): find specific information on a website
   - Form-filling tasks: fill multi-field forms with given data
   - Multi-app tasks: transfer information between two apps/tools
   - Code tasks: write and run code to solve a problem, verify with tests
2. Define your success metric carefully:
   - Binary: did the agent complete the final goal? (strict)
   - Partial credit: what fraction of required steps completed? (lenient)
   - Report both
3. Categorize tasks by step count: short (1–4 steps), medium (5–9 steps), long (10+ steps).

**What to record:**
- Task distribution by category and step count
- Ground truth step count for each task (you'll compare to agent's actual steps)

---

### Phase 3 — Memory Architecture Ablation (Week 2)

**Goal:** Quantify how much memory helps and where it stops helping.

**Three conditions:**
1. **No memory:** agent only sees current screenshot + last 3 actions (minimal context)
2. **Episodic memory only:** agent stores and retrieves summaries of past steps using ChromaDB (semantic search over action history)
3. **Episodic + semantic memory:** also maintains a structured "world model" — key facts discovered, entities seen, goals accomplished — updated after each step

**Steps:**
1. Implement all three conditions in LangGraph. The memory module should be swappable.
2. Run each condition on your full 50-task benchmark. Record:
   - Task success rate (binary and partial credit) — by task category
   - Number of steps taken per task
   - Memory retrieval latency (ms per lookup)
3. **Memory scaling experiment:** for episodic memory, vary history length (last 5, 10, 20 episodes). Record how retrieval accuracy and agent performance change.

**What to record:**
- Success rate: no memory vs episodic vs episodic+semantic — by task category and step count
- Partial credit score distribution for each condition
- Memory retrieval latency vs history size
- Specific task types where memory helps most and least

---

### Phase 4 — Tool-Use Eval Harness (Week 2–3)

**Goal:** A reusable eval framework that measures tool reliability, not just task success.

**Steps:**
1. For each tool in your action space, define:
   - **Call success rate:** did the tool execute without error?
   - **Semantic correctness rate:** did it do the right thing? (requires checking the outcome)
   - **Error recovery rate:** when a tool fails, does the agent recover in the next 2 steps?
   - **Average retries:** how many attempts before success or giving up?
2. Run your 50-task benchmark and collect per-tool metrics.
3. **Error taxonomy:** Manually categorize all tool failures:
   - Wrong element selected (click on wrong thing)
   - Stale state (page changed, element no longer exists)
   - Action out of order (tried to submit form before filling fields)
   - Infinite loop (agent repeats the same action)
   - Hallucinated action (described action that isn't in the action space)
4. For each error type, record frequency and whether the agent recovered.

**What to record:**
- Per-tool: call success rate, semantic correctness rate, error recovery rate, avg retries
- Error type distribution across all failed tool calls
- Recovery rate by error type
- The most common unrecoverable error pattern

---

### Phase 5 — Error Compounding Analysis (Week 3)

**Goal:** Show the phase transition where long tasks fail catastrophically. This is the key finding.

**Steps:**
1. Plot task success rate vs task step count (1–15 steps). Use your 50-task benchmark results.
2. **The hypothesis:** success rate doesn't degrade linearly — it falls sharply after a threshold (typically 5–8 steps) because errors compound. Verify or refute this with your data.
3. **Error propagation:** for each failed task, find the first error in the action log. Then count: how many steps later did the task become unrecoverable? This is your "error propagation delay."
4. Show 5 concrete examples of error compounding with the action trace: Step 3 error → Step 4 confused → Step 5–7 trying to fix → Step 8 unrecoverable.
5. **Memory's role:** does the episodic+semantic memory condition shift the failure threshold (e.g., from step 6 to step 8)? Report this explicitly.

**What to record:**
- Success rate by step count bin: 1–3, 4–6, 7–10, 11+ steps
- Average error propagation delay (steps from first error to task failure)
- Memory's effect on the failure threshold
- 5 annotated failure traces showing error compounding

---

### Phase 6 — Local VLM vs API Model Comparison (Week 3–4)

**Goal:** Honest quantification of the gap between your local 7B VLM and frontier models.

**Steps:**
1. Run your 50-task benchmark with your local Qwen2-VL-7B agent.
2. Run the same benchmark with GPT-4o via API — but minimize API calls to control cost. Use 10–20 representative tasks.
3. Compare on the 10–20 overlapping tasks: success rate, step efficiency (fewer steps = better), error rate per step.
4. **Where does the gap close?** Categorize tasks where local 7B matches GPT-4o (likely: simple navigation, form-filling) and where it fails badly (likely: complex reasoning, ambiguous UI states).
5. **Latency comparison:** local inference vs API round-trip. Show the real cost of using API vs local in terms of both dollars and latency per task.

**What to record:**
- Success rate: local 7B vs GPT-4o on overlapping tasks
- Task categories where gap is small vs large
- Latency per step: local (A30 inference) vs API (network + inference)
- Estimated cost per task for API model vs local (amortized hardware cost)

---

### Phase 7 — Failure Analysis & Decision Guide (Week 4)

**Goal:** The honest documentation that turns this from a demo into engineering.

**Steps:**
1. **Unrecoverable error taxonomy:** From all failed tasks across all conditions, what are the top 5 failure patterns? Give each a name and count.
2. **Sandboxing overhead:** Measure E2B/Docker latency overhead vs running code directly. At what execution frequency does sandboxing become the bottleneck?
3. **VLM screenshot understanding accuracy:** Sample 100 screenshots from your logs. For each, check if the VLM's state description was accurate. Report: fraction of steps with incorrect state understanding, and whether incorrect understanding led to task failure.
4. **Memory false retrieval:** For episodic memory, how often does the agent retrieve an irrelevant memory (semantic search returns wrong episode)? Report the false retrieval rate and its downstream effect on task success.

**What to record:**
- Top 5 unrecoverable failure patterns with counts
- Sandboxing latency overhead (ms per code execution)
- VLM state understanding accuracy and its correlation with task success
- Memory false retrieval rate

---

### README Must Answer
- What is the practical step limit for your local 7B VLM before success rate collapses?
- Which task types close the gap between local 7B and frontier API models?
- Does episodic memory help meaningfully or just add latency?
- What are the unrecoverable errors and how would you fix them?
- How much does sandboxing cost in latency and is it worth it?

---

## Cross-Project Infrastructure

### Shared setup (do this once, reuse across all projects)

**Training infrastructure:**
- DeepSpeed config: ZeRO-2 for 2-GPU runs, ZeRO-3 for 4-GPU runs
- Standardized logging: W&B project with consistent metric names across experiments
- Checkpoint management: save every 500 steps, keep top-3 by validation metric

**Evaluation infrastructure:**
- Wrap all benchmark evaluations in a single `evaluate.py` script with standard output format
- Store all results in JSON with: run_id, timestamp, model_path, hyperparams, metrics
- This makes the final comparison tables trivial to generate

**The lab notebook:**
- Keep a `EXPERIMENTS.md` in each repo
- Format: date, hypothesis, what you ran, what you found, what it changed about your approach
- The "what didn't work" entries are as important as successes

---

## Recommended Build Order

**Week 1–2:** Project 1 Phase 1–2 (setup + data) AND Project 3 Phase 1 (VLM baseline) in parallel  
**Week 2–4:** Project 1 Phase 3–6 (main training runs) — use spare GPU for Project 3 fine-tuning  
**Week 4–6:** Project 3 Phase 3–6 (VLM ablations + KV cache)  
**Week 6–8:** Project 2 (GraphRAG — mostly CPU/memory, can run alongside GPU work)  
**Week 8–10:** Project 4 (Agent — builds on Project 3's VLM)

**Total estimated GPU-hours:** ~200–300 hours across all 4 projects  
**Total estimated cost:** $0 (all local A30s) + Colab Pro for parallelization where needed
