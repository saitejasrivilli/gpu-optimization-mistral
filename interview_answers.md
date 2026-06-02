# Interview Q&A - Short & Sharp (Based on Your CV)

---

## ORACLE - REST APIs & Cloud Run

### Q1: You built REST APIs supporting internal dashboards with sub-200ms response times. How?

**Answer:**
"Parallelized downstream service calls with `Promise.all()` instead of sequential execution. Added Redis caching (5-minute TTL) for reports that don't change frequently. Used Cloud Run autoscaling with minimum instances=1 to avoid cold starts. FastAPI's async support meant I could handle 8K+ concurrent requests efficiently. Profiled with Datadog APM to identify slow queries — added database indexes, which cut latency from 400ms to 180ms."

**Keywords:** Promise.all, Redis caching, Cloud Run, FastAPI async, Datadog, database indexing, p99 latency

---

### Q2: 99%+ uptime over 4 years across 25 services. What caused major outages?

**Answer:**
"Connection pool exhaustion — a bug in a downstream service left connections open. Fixed with PgBouncer connection pooler and circuit breaker logic to fail fast instead of hanging. Cache stampede when Redis died — 1000 requests hit the database simultaneously for the same key, cascading failure. Solved with probabilistic early expiration (expire cache at TTL - jitter) so not all keys expire at once. Added alerting at 80% pool capacity."

**Keywords:** connection pooling, circuit breaker, PgBouncer, cache stampede, jitter, failover, observability

---

### Q3: You mention CI/CD deployment to Cloud Run. What's your pipeline?

**Answer:**
"Code push triggers GitHub Actions. Build Docker image, run unit tests in container, push to Artifact Registry. Cloud Run auto-deploys to staging first, runs integration tests. If pass, deploy to production with gradual traffic shift (10% → 50% → 100%) using Cloud Load Balancer. Rollback if error rate spikes (monitored with Datadog). End-to-end: 5 minutes from push to production."

**Keywords:** GitHub Actions, Docker, Artifact Registry, Cloud Run, gradual rollout, integration testing, error rate monitoring

---

## NOMURA - Financial APIs & Data Pipelines

### Q4: You designed API contracts enforcing data ordering for regulatory compliance. Why does ordering matter?

**Answer:**
"Financial transactions must be auditable. If transaction 1 happens before transaction 2, the audit log must reflect that order universally. We returned results sorted by database sequence number, not insertion time. For consistency across services, we used Kafka with sequence IDs — downstream services replayed events in order. This guaranteed that any service checking transaction history saw the same order."

**Keywords:** audit trail, sequence number, Kafka, eventual consistency, database ordering, regulatory requirements

---

### Q5: You rebuilt nightly aggregation from 8 minutes to 90 seconds on 2TB data. How?

**Answer:**
"Identified bottleneck with query execution plan analysis — full table scan of 2TB. First, partitioned by date (reduced scan to ~100GB for last 30 days). Then clustered by account_id (the GROUP BY key) so the aggregation became local, not a shuffle. Added materialized view for frequently-queried aggregations. Query time dropped 8 mins → 90 seconds. Scan cost went from $10 to $0.50 per run (BigQuery billing)."

**Keywords:** partitioning, clustering, query optimization, materialized views, BigQuery cost, EXPLAIN plan

---

### Q6: You built asynchronous event-driven pipelines with transactional correctness. No data loss in 4 years?

**Answer:**
"Used the outbox pattern. When a service processes an event, it writes both the business state change AND an outbox event in a single database transaction. If it crashes between consuming and committing, both rollback — no inconsistency. A separate poller reads unpublished outbox events and publishes to Kafka. Downstream services use idempotency keys — if they receive the same event twice (due to retry), they ignore duplicates."

**Keywords:** outbox pattern, exactly-once semantics, idempotency keys, transactional correctness, event sourcing

---

## UNIVERSITY OF TEXAS - Python Tooling & Distributed Jobs

### Q7: You reduced job processing wait times from 4+ hours to 30 minutes (8x). How?

**Answer:**
"Built a scheduling system with three changes: (1) Job prioritization — short jobs run first, freeing resources for longer jobs. (2) Job bundling — recognized patterns and bundled similar simulations to run together instead of competing for cores. (3) Real-time dashboard showing position in queue. The bottleneck wasn't compute — it was the workflow. Most time was waiting in queue, not running."

**Keywords:** job scheduling, priority queue, resource pooling, workflow optimization, queue management

---

## ANGULAR FRONTEND

### Q8: You built Angular components and reactive forms. Show me complex form state management.

**Answer:**
"Use a form state service with BehaviorSubject. Each step component updates a shared service on value changes (with debounceTime to avoid spamming). For multi-step forms, I save state to the service, not component. Async validators like email uniqueness are handled by Angular automatically — returns Observable that the form waits for. On submit, retrieve complete state from service and POST to backend."

**Code snippet:**
```typescript
// form-state.service.ts
export class FormStateService {
  private formData$ = new BehaviorSubject({step1: null, step2: null});
  
  updateStep(stepNum: number, data: any) {
    const current = this.formData$.value;
    this.formData$.next({...current, [`step${stepNum}`]: data});
  }
  
  getFormData$() { return this.formData$.asObservable(); }
}

// In component:
this.form.valueChanges
  .pipe(debounceTime(300), distinctUntilChanged())
  .subscribe(value => this.stateService.updateStep(1, value));
```

**Keywords:** BehaviorSubject, RxJS operators (debounceTime, distinctUntilChanged), async validators, form state separation

---

### Q9: What's the difference between Default vs OnPush change detection?

**Answer:**
"Default: Angular re-checks *entire component tree* after any event (click, HTTP response, timer). OnPush: Component only re-renders if @Input properties change (by reference) or event fired within it. For dumb components displaying data, use OnPush — cuts change detection time by 50-80% in large apps. Trade-off: if you mutate objects in place, OnPush won't detect changes. Always pass new object references."

**Code:**
```typescript
@Component({
  selector: 'app-user-card',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class UserCardComponent {
  @Input() user: User; // OnPush watches this
}
```

**Keywords:** ChangeDetectionStrategy.OnPush, performance optimization, immutability, @Input properties

---

### Q10: You mention route-level guards. Write one for admin-only routes.

**Answer:**
```typescript
@Injectable({providedIn: 'root'})
export class AdminGuard implements CanActivate {
  constructor(private authService: AuthService, private router: Router) {}

  canActivate(): Observable<boolean> {
    return this.authService.getCurrentUser().pipe(
      map(user => user?.role === 'admin'),
      tap(isAdmin => {
        if (!isAdmin) {
          this.router.navigate(['/login']);
        }
      })
    );
  }
}

// In routing module:
{ path: 'admin', component: AdminDashboard, canActivate: [AdminGuard] }
```

**Keywords:** CanActivate, Observable guards, async auth checks, role-based access, tap operator

---

## GCP & DEVOPS

### Q11: Cloud Run vs traditional VMs — when do you use each?

**Answer (Perspective 1 - Cost):**
"Cloud Run for bursty traffic (pay-per-request, scale to 0). VMs for steady baseline (always running but cheaper at consistent load). Hybrid: use VMs for baseline, Cloud Run for spikes."

**Answer (Perspective 2 - Cold Starts):**
"Cloud Run has 1-2 second cold starts. If latency-critical (APIs requiring <100ms), use VMs or keep minimum instances. We set minimum instances=1 during business hours."

**Answer (Perspective 3 - Statefulness):**
"Cloud Run is stateless. Can't cache large objects in process memory. Use Redis instead. VMs allow in-process caching."

**Combined:**
"We use Cloud Run for internal APIs (Nomura) because they're I/O-heavy and scale unpredictably. Set minimum instances=1 to avoid cold starts. Moved caching to Redis because Cloud Run instances are ephemeral. For customer-facing APIs, we'd use VMs for consistency."

**Keywords:** cold starts, auto-scaling, pay-per-request, stateless, ephemeral instances, minimum instances

---

### Q12: BigQuery optimization — how do you handle 2TB efficiently?

**Answer (Perspective 1 - Partitioning):**
"Partitioned by date. Instead of scanning 2TB, only scan data for the time range needed. Reduced from 2TB → ~100GB scans."

**Answer (Perspective 2 - Clustering):**
"Clustered by the GROUP BY key (account_id). Rows with same account_id stored together on disk. Aggregation becomes local, not a shuffle. Cut execution time from 3 mins → 90 seconds."

**Answer (Perspective 3 - Cost):**
"BigQuery bills $5/TB scanned. Before: $10/run. After: $0.50/run."

**Combined:**
"For the nightly job on 2TB: (1) Partitioned by date to skip irrelevant data. (2) Clustered by account_id (the aggregation key) to avoid expensive shuffles. (3) Used materialized views for frequently-accessed aggregations. Result: 8 mins → 90 seconds, cost from $10 → $0.50 per run."

**Keywords:** partitioning, clustering, materialized views, query cost, full table scan, shuffle operations

---

### Q13: Cloud SQL for production databases — how ensure 99.9% uptime?

**Answer (Perspective 1 - HA):**
"Multi-zone high availability: primary in zone A, automatic standby in zone B with synchronous replication. If primary fails, failover automatic (~2-3 mins). Zero data loss because synchronous."

**Answer (Perspective 2 - Read Scaling):**
"Read replicas in other regions for read-heavy queries (dashboards). Async replication, so slightly stale. Primary handles transactions."

**Answer (Perspective 3 - Backup):**
"Automated daily backups + point-in-time recovery (7-day retention). Can restore to any point in last 7 days."

**Combined:**
"For 99.9% uptime (43 mins downtime/month): Multi-zone HA with synchronous standby. Read replicas for scaling. Automated backups with PITR. Monitor replication lag (alert if >1s). This handles zone failure, instance failure, even regional failure (replicas in other region can be promoted)."

**Keywords:** multi-zone HA, synchronous replication, read replicas, point-in-time recovery, failover, replication lag

---

## PROJECTS

### Q14: Watch-your-LLM — how compute costs per API call?

**Answer (Perspective 1 - Token Counting):**
"Captured input_tokens and output_tokens from API response."

**Answer (Perspective 2 - Pricing Model):**
"OpenAI charges different rates for input vs output. Claude charges per token. Stored pricing per model in config."

**Answer (Perspective 3 - Computation):**
"cost = (input_tokens × input_rate) + (output_tokens × output_rate). Persisted to PostgreSQL with timestamp and model name for analytics."

**Combined:**
"Captured request-level metrics (input_tokens, output_tokens) from each API call. Stored pricing per model/provider in config (e.g., Claude Opus: $0.000015/input, $0.000075/output). Computed cost = (input_tokens × rate) + (output_tokens × rate). Stored in PostgreSQL. Dashboard shows per-model cost breakdown and daily trends."

**Keywords:** token counting, variable pricing, request metrics, per-model rates, cost analytics

---

### Q15: LangFetch — how handle multi-step SQL decomposition?

**Answer:**
"Parse natural language query into multiple steps. Step 1: identify tables and joins from schema. Step 2: generate SELECT for relevant columns. Step 3: execute and carry forward intermediate results. Step 4: synthesize final answer. Example: 'Show top customers by revenue' → (1) find customers table + orders table, (2) JOIN on customer_id, (3) SUM(order_amount) GROUP BY customer, (4) ORDER BY DESC LIMIT 10."

**Keywords:** query decomposition, schema navigation, multi-step execution, intermediate results, SQL synthesis

---

### Q16: Distributed Semantic Search — why consistent hashing vs random?

**Answer (Perspective 1 - Determinism):**
"Consistent hashing: same document always goes to same node. Random: document could be on any node. With random, you'd query all nodes for every search."

**Answer (Perspective 2 - Scaling):**
"Add new node: consistent hashing redistributes ~1/n documents. Random requires rebalancing everything."

**Answer (Perspective 3 - Implementation):**
"Hash the doc_id, find which node's range it falls in using bisect. FAISS indexed chunks distributed across 3-node cluster with consistent hash ring."

**Combined:**
"Used consistent hash ring instead of random assignment. Each document deterministically maps to a node based on hash(doc_id). Benefits: (1) Same query always hits the same node, no redundant searching. (2) Adding nodes only redistributes 1/n documents, not all. (3) Node failure causes automatic failover to next node in ring. With replicas on next 2 nodes, single-node loss has zero downtime."

**Keywords:** consistent hashing, deterministic routing, shard distribution, node failover, FAISS, p99 latency

---

## BEHAVIORAL

### Q17: Production incident — most significant one you caught early?

**Answer:**
"Memory leak in transaction aggregation service at Nomura. Datadog showed memory increasing 50MB/hour. After 48 hours, service hit memory limit and crashed. Profiled with pprof, found global request cache was never cleared — persisted across requests. Fixed by replacing with @lru_cache(maxsize=10000), which auto-evicts old entries. Reduced restarts from 1 per 2 days to ~1 per quarter. Impact: eliminated on-call restarts, ~$50K/year in avoided incident costs."

**Keywords:** Datadog profiling, pprof, memory leak, lru_cache, monitoring, incident cost

---

### Q18: Biggest engineering mistake you made?

**Answer:**
"At Nomura, I initially didn't add circuit breakers to downstream service calls. When one service was slow, my service would timeout waiting. Cascaded to all dependent services. Fixed by adding circuit breaker (Hystrix pattern) — fail fast after N failures, return fallback response. Learned: defensive programming matters. Always assume dependencies can fail."

**Keywords:** circuit breaker, cascading failure, defensive programming, fallback response, timeout handling

---

## TECHNICAL DEEP DIVES

### Q19: TypeScript vs Python — when use each?

**Answer:**
"Python for backend (FastAPI, async support, data science libraries). TypeScript for frontend (type safety catches bugs early, integrates with Angular). Python slower than compiled languages, but for I/O-bound APIs (network, DB queries), speed is latency-bound, not CPU-bound. For CPU-intensive work, use compiled language. My choice: Python backend + TypeScript frontend for full-stack type safety."

**Keywords:** type safety, I/O-bound vs CPU-bound, FastAPI, async, compiled vs interpreted

---

### Q20: How approach database optimization?

**Answer:**
"(1) Measure first — use EXPLAIN to see query plan. (2) Identify bottleneck — is it full table scan, shuffle, or slow join? (3) Apply fix — index, partition, cluster, or query rewrite. Example: BigQuery query scanning 2TB → partitioned by date → scans 100GB. Measure impact before/after. Avoid premature optimization; profile production, not local dev."

**Keywords:** EXPLAIN, query profiling, indexing, partitioning, bottleneck analysis, measurement

---

## SHORT ANSWERS FOR QUICK FOLLOW-UPS

### "What's your biggest strength?"
"End-to-end ownership. I've shipped features from database schema to frontend UI. But strongest area is backend optimization — profiling, caching, async patterns."

### "What's your biggest weakness?"
"JavaScript fundamentals could be deeper. I use TypeScript/Angular comfortably, but I'd benefit from deeper knowledge of closures, event loop, and async/await internals."

### "Why leave Nomura?"
"Learned a lot in 4 years, but wanted broader exposure. UTA grad program let me study distributed systems, NLP, and deep learning. Oracle project gave me cloud infrastructure depth."

### "Why Oracle next?"
"Wanted to deepen AI platform work. Building features end-to-end (backend APIs, frontend dashboards, data pipelines) that directly impact AI cost/latency tracking."

---

## QUICK KEYWORD CHECKLIST

**Backend:** FastAPI, async, REST APIs, PostgreSQL, BigQuery, Redis, Kafka, event-driven, transactional correctness, circuit breaker, connection pooling

**Frontend:** Angular, reactive forms, RxJS, async validators, change detection, route guards, immutability

**Cloud:** Cloud Run, cold starts, auto-scaling, Cloud SQL HA, multi-zone failover, read replicas, Artifact Registry

**System Design:** consistent hashing, distributed systems, eventual consistency, outbox pattern, idempotency, circuit breaker, cache stampede

**Optimization:** EXPLAIN plans, partitioning, clustering, indexing, materialized views, query profiling, caching strategies

**Tools:** Datadog, Jaeger, pprof, GitHub Actions, Docker, BigQuery, Kafka, Redis, PgBouncer

---

End.
