# Architecture Decisions Record (ADR)

## Context

Goal: Create 2 portfolio projects demonstrating all Cisco JD requirements for ML data infrastructure role.

**Cisco JD Minimum Requirements:**
- Python + data structures
- TensorFlow/PyTorch/scikit-learn
- Spark/Hadoop/Flink (distributed data processing)
- Docker/Kubernetes

**Cisco JD Preferred:**
- AI/ML hands-on
- Cloud platforms (AWS/Azure/GCP)
- Distributed systems (scalability, reliability, fault tolerance, data consistency, load balancing, consensus, inter-service communication)

---

## Decision 1: Two-Project Approach

**Decision:** Create 2 separate projects covering different aspects of the JD.

**Rationale:**
- **Mini Spark** addresses Spark/Hadoop/Flink gap (hard requirement)
- **ML Pipeline** demonstrates AI/ML + distributed systems in practice
- Together = complete coverage of all JD requirements
- More impressive portfolio than single generic project

**Alternatives Considered:**
1. Single monolithic project combining everything
   - ❌ Too complex, less clear separation of concerns
   - ❌ Harder to explain distinct concepts
2. Three separate projects
   - ❌ Too much code to maintain
   - ❌ Diminishing returns on portfolio value

**Trade-offs:**
- ✅ More comprehensive coverage
- ✅ Clearer architectural separation
- ✅ Each project demonstrates specific concepts
- ❌ More code to write and maintain
- ❌ Longer time to complete

**Status:** ✅ ACCEPTED

---

## Decision 2: Mini Spark as Distributed Data Processing Engine

**Decision:** Implement custom "mini Spark" distributed batch processing framework instead of using actual Spark.

**Rationale:**
- **Shows deep understanding**: Building from scratch demonstrates architectural knowledge, not just library usage
- **Addresses Spark gap**: User has no visible Spark projects on GitHub; this is the largest gap
- **All 7 distributed systems concepts**: Task DAG, load balancing, fault tolerance, state management, inter-service communication, scalability, data consistency
- **Interview discussion points**: Can explain design decisions, trade-offs, implementation details
- **Resume differentiator**: "Implemented distributed batch processing engine" > "Used Apache Spark"

**Alternatives Considered:**
1. Use actual Apache Spark
   - ❌ Doesn't demonstrate systems knowledge
   - ❌ Less impressive for portfolio
   - ✅ Could show practical usage
2. Use PySpark on Kubernetes
   - ❌ Might require JVM/Scala knowledge
   - ❌ Still doesn't show architecture understanding
3. Implement mini MapReduce instead
   - ⚠️ Simpler but less feature-complete
   - ⚠️ Doesn't demonstrate modern patterns (DAG scheduling)

**Components Chosen:**
- **Task Graph**: DAG abstraction for job definition (like Spark)
- **Master**: Centralized state management via Redis (fault-tolerant)
- **Workers**: Distributed task execution (scalable)
- **Driver**: Orchestrator with load balancing (reliability)

**Trade-offs:**
- ✅ Demonstrates architectural knowledge
- ✅ All systems concepts covered
- ✅ Interview-friendly (explain every design)
- ❌ Not production-ready (actual Spark is battle-tested)
- ❌ Requires more implementation effort

**Status:** ✅ ACCEPTED

---

## Decision 3: Redis for Distributed State

**Decision:** Use Redis as the distributed state store for Mini Spark.

**Rationale:**
- **Reliable**: Persistent KV store for job/task state
- **Simple**: Easy to understand, no complex setup
- **Fast**: In-memory access for state queries
- **Demonstrates consistency**: Shows data consistency patterns (versioning, atomic updates)
- **Cloud-friendly**: Available on AWS/GCP/Azure
- **Learning value**: Shows distributed state management concepts

**Alternatives Considered:**
1. In-memory dict (no persistence)
   - ❌ Loses state on crash
   - ❌ Doesn't demonstrate fault tolerance
2. PostgreSQL/MySQL
   - ⚠️ Overkill for this use case
   - ⚠️ Slower than Redis
   - ✅ More production-like
3. etcd (Kubernetes-native)
   - ⚠️ Adds complexity
   - ⚠️ Requires K8s cluster
   - ✅ More enterprise-like

**Trade-offs:**
- ✅ Simple, understandable
- ✅ Fast for state access
- ✅ Shows fault tolerance patterns
- ❌ Not optimal for massive scale
- ❌ Requires Redis dependency

**Status:** ✅ ACCEPTED

---

## Decision 4: Python for Implementation

**Decision:** Implement both projects in Python.

**Rationale:**
- **Cisco JD requires:** "Proficiency in Python"
- **User strength:** All 84 GitHub projects are Python
- **Accessibility:** Easy to run, test, deploy
- **ML ecosystem:** PyTorch, scikit-learn, FastAPI all Python
- **Startup speed:** Faster to implement than Go/Java
- **Interview alignment:** Can speak fluently about code

**Alternatives Considered:**
1. Go for mini-spark (systems language)
   - ✅ Better performance
   - ⚠️ User has no Go portfolio (except 1 project)
   - ❌ Doesn't match Cisco "Python proficiency" requirement
2. Mixed languages (Go + Python)
   - ❌ More complex to coordinate
   - ❌ Deployment complexity
   - ⚠️ Less cohesive portfolio

**Trade-offs:**
- ✅ Matches JD requirement explicitly
- ✅ Leverages user expertise
- ✅ Simpler to understand and modify
- ❌ Slower than systems languages
- ❌ Less suitable for true high-scale systems

**Status:** ✅ ACCEPTED

---

## Decision 5: PyTorch for ML Framework

**Decision:** Use PyTorch (not TensorFlow) for ML Pipeline training.

**Rationale:**
- **User familiarity**: PyTorch appears in multiple GitHub projects (flash-attn, quantization, optimization)
- **Easier debugging**: PyTorch's eager execution better for learning
- **Modern standard**: Industry standard for research/production
- **Cisco requirement**: JD lists "TensorFlow, PyTorch, or scikit-learn" - PyTorch is explicitly mentioned

**Alternatives Considered:**
1. TensorFlow/Keras
   - ⚠️ Also acceptable per JD
   - ❌ Less aligned with user's GitHub history
   - ⚠️ More verbose for simple models
2. scikit-learn only
   - ❌ Limited to shallow models
   - ❌ Not suitable for distributed training demo
3. Multiple frameworks
   - ❌ Adds complexity
   - ❌ Dilutes focus

**Trade-offs:**
- ✅ Aligns with user expertise
- ✅ Easier to implement distributed patterns
- ✅ Better for simple examples
- ⚠️ TensorFlow might be more "enterprise" feeling

**Status:** ✅ ACCEPTED

---

## Decision 6: FastAPI for ML Pipeline API

**Decision:** Use FastAPI for serving ML models via HTTP.

**Rationale:**
- **Modern**: Fastest Python web framework (benchmarks prove it)
- **Type-safe**: Pydantic validation out of box
- **Kubernetes-ready**: Built for container deployment
- **OpenAPI**: Auto-generated docs (swagger UI)
- **Async support**: For high-concurrency inference
- **Demonstrates systems thinking**: Shows API design, health checks, monitoring

**Alternatives Considered:**
1. Flask
   - ⚠️ Simpler but older
   - ❌ Less built for distributed systems
   - ❌ Manual validation
2. Django
   - ❌ Overkill for simple API
   - ❌ Heavier framework
3. gRPC
   - ✅ Better performance
   - ❌ More complex for beginners
   - ❌ Harder to test manually

**Trade-offs:**
- ✅ Modern, well-maintained
- ✅ Great for Kubernetes deployment
- ✅ Production patterns built-in
- ⚠️ Less "proven" than Flask (but widely used now)

**Status:** ✅ ACCEPTED

---

## Decision 7: Production-Grade Directory Structure

**Decision:** Use enterprise-standard `src/` layout with separation of concerns.

**Rationale:**
- **Importability**: `pip install -e .` makes package importable
- **Testability**: Tests in separate directory (standard practice)
- **Scalability**: Easy to add new modules without flat clutter
- **Professionalism**: Shows understanding of Python packaging best practices
- **CI/CD ready**: Standard structure works with GitHub Actions, GitLab CI, etc.

**Structure:**
```
src/package/        → Importable code
tests/              → Test suite (unit + integration)
config/             → Configuration management
docker/             → Container images
kubernetes/         → K8s manifests
docs/               → Documentation
examples/           → Runnable examples
scripts/            → Utility scripts
setup.py            → Package definition
requirements.txt    → Dependencies
```

**Alternatives Considered:**
1. Flat structure (code in root)
   - ❌ Messy at scale
   - ❌ Namespace conflicts
   - ❌ Hard to package
2. `lib/` instead of `src/`
   - ⚠️ Works but less standard
   - ❌ Less explicit about purpose

**Trade-offs:**
- ✅ Professional appearance
- ✅ Follows Python best practices
- ✅ Easier to maintain/extend
- ✅ CI/CD friendly
- ❌ Slightly more setup initially

**Status:** ✅ ACCEPTED

---

## Decision 8: Include Kubernetes Manifests

**Decision:** Provide K8s deployment files for both projects.

**Rationale:**
- **Cisco JD requirement**: "Experience with containerization and orchestration tools, such as Docker and Kubernetes"
- **Cloud platforms**: Shows AWS/GCP/Azure readiness
- **Production pattern**: Demonstrates deployment thinking
- **Interview value**: Can discuss auto-scaling, health checks, resource limits

**Contents:**
- **Deployment**: Pod definitions with replicas, health checks
- **Service**: Load balancer configuration
- **ConfigMap**: Environment configuration
- **Base + Overlays**: Development vs Production separation

**Alternatives Considered:**
1. Docker Compose only
   - ✅ Simpler for local dev
   - ❌ Doesn't address K8s requirement
   - ❌ Less production-like
2. Helm charts
   - ✅ More professional
   - ❌ Adds complexity
   - ❌ Harder for beginners to understand
3. No infrastructure as code
   - ❌ Misses JD requirement entirely

**Trade-offs:**
- ✅ Addresses K8s requirement
- ✅ Shows production thinking
- ✅ Cloud-deployment ready
- ❌ Adds files to maintain
- ❌ Requires K8s knowledge

**Status:** ✅ ACCEPTED

---

## Decision 9: Distributed Systems Concepts Coverage

**Decision:** Map each project to all 7 distributed systems concepts.

**Concepts Required (Cisco JD):**
1. Scalability - horizontal scaling, partition
2. Reliability - state tracking, persistence
3. Fault tolerance - retry, recovery, monitoring
4. Data consistency - versioning, atomic updates
5. Load balancing - task/request distribution
6. Consensus algorithms - coordination patterns
7. Inter-service communication - RPC, API calls

**Coverage:**

**Mini Spark:**
1. ✅ **Scalability**: Workers scale horizontally, task partitioning
2. ✅ **Reliability**: Job/task state in Redis, persistent storage
3. ✅ **Fault tolerance**: Failed task detection, retry logic
4. ✅ **Data consistency**: Partitioned state, versioned results
5. ✅ **Load balancing**: Round-robin task scheduling across workers
6. ✅ **Consensus**: Master coordinates task execution, ordering
7. ✅ **Inter-service communication**: Worker ↔ Master state sync

**ML Pipeline:**
1. ✅ **Scalability**: Distributed training batches, multi-replica inference
2. ✅ **Reliability**: Feature versioning, model checkpointing
3. ✅ **Fault tolerance**: Model rollback, replica monitoring
4. ✅ **Data consistency**: Feature store versioning with hashing
5. ✅ **Load balancing**: Round-robin inference replicas
6. ✅ **Canary deployment**: Gradual traffic shift (consensus on model quality)
7. ✅ **Inter-service communication**: API endpoints, feature fetch, model loading

**Trade-offs:**
- ✅ Complete coverage of JD requirements
- ✅ Interview-ready explanations for each
- ✅ Demonstrates systems maturity
- ❌ Slightly verbose code (more patterns)
- ❌ Could have simplified further

**Status:** ✅ ACCEPTED

---

## Decision 10: Testing Structure

**Decision:** Include unit tests + integration tests, with separate directories.

**Rationale:**
- **Professional standard**: Shows quality engineering practices
- **CI/CD ready**: Tests can run in pipeline
- **Maintenance**: Easier to refactor with test coverage
- **Interview value**: "How do you ensure reliability?"

**Structure:**
```
tests/
├── unit/                # Fast, isolated tests
│   ├── test_feature_store.py
│   ├── test_task.py
└── integration/         # End-to-end tests
    └── test_pipeline.py
```

**Alternatives Considered:**
1. No tests
   - ❌ Doesn't demonstrate quality
   - ❌ Breaks in interviews
2. Single test file
   - ⚠️ Works but less organized
   - ❌ Hard to scale

**Trade-offs:**
- ✅ Demonstrates software engineering discipline
- ✅ Easier to catch regressions
- ✅ Professional appearance
- ❌ Adds implementation time

**Status:** ✅ ACCEPTED

---

## Decision 11: Documentation Approach

**Decision:** Include ARCHITECTURE.md for each project + this DECISIONS.md.

**Rationale:**
- **Clarity**: Explains design intent, not just implementation
- **Maintainability**: Future contributors understand reasoning
- **Interview prep**: Can reference documentation during discussion
- **Professionalism**: Shows thinking beyond code

**Content:**
- ARCHITECTURE.md: Component overview, data flow, deployment
- DECISIONS.md: This file - decisions with rationale & trade-offs

**Alternatives Considered:**
1. README only
   - ✅ Minimal
   - ❌ Doesn't explain design
2. Extensive docstrings
   - ✅ Inline documentation
   - ❌ Harder to see big picture
3. Wiki or external docs
   - ❌ Not in repo
   - ❌ Diverges from code

**Trade-offs:**
- ✅ Clear architectural understanding
- ✅ Easier onboarding
- ✅ Interview discussion points
- ❌ Docs must stay in sync with code

**Status:** ✅ ACCEPTED

---

## Decision 12: Local Development with Docker Compose

**Decision:** Provide docker-compose.yml for easy local testing without K8s.

**Rationale:**
- **Accessibility**: Works on any machine with Docker
- **Parity**: Services match K8s deployment
- **Learning**: Understand components before K8s complexity
- **CI/CD**: Can run tests in containers
- **Reproducibility**: "Works on my machine" → "Works everywhere"

**Approach:**
```yaml
services:
  redis:        # (Mini Spark only)
    image: redis:7-alpine
  api:          # (ML Pipeline)
    build: .
  example:      # Run example inline
    depends_on: [api]
```

**Alternatives Considered:**
1. Requires manual local setup
   - ❌ Error-prone
   - ❌ Hard to reproduce
2. K8s only (no Docker Compose)
   - ❌ High barrier to entry
   - ❌ Requires minikube/cluster

**Trade-offs:**
- ✅ Easy to run locally
- ✅ Low setup friction
- ✅ Parity with K8s
- ⚠️ Still requires Docker

**Status:** ✅ ACCEPTED

---

## Decision 13: Cloud Deployment Templates

**Decision:** Include templates for AWS/GCP deployment (in docs + k8s manifests).

**Rationale:**
- **Cisco JD preferred**: "Familiarity with major cloud platforms"
- **Portfolio strength**: Shows beyond local development thinking
- **Real-world pattern**: K8s manifests work on any cloud
- **Templates vs actual**: Provide patterns, not real credentials

**Approach:**
```yaml
# kubernetes/base/deployment.yaml works on:
# - AWS EKS
# - Google GKE
# - Azure AKS
# - Any Kubernetes cluster
```

**Alternatives Considered:**
1. Terraform files for cloud infrastructure
   - ✅ More complete
   - ❌ Requires cloud account
   - ❌ Adds complexity
2. No cloud references
   - ❌ Misses "cloud platform" JD requirement

**Trade-offs:**
- ✅ Shows cloud understanding
- ✅ Reusable on any cloud
- ✅ Doesn't require credentials
- ⚠️ Requires K8s knowledge

**Status:** ✅ ACCEPTED

---

## Decision 14: Modular Architecture Within Projects

**Decision:** Break each project into logical modules (not monolithic files).

**Mini Spark modules:**
- `core/` - Task graph abstraction
- `master/` - Job/state management
- `worker/` - Task execution
- `driver/` - Orchestration
- `utils/` - Shared utilities

**ML Pipeline modules:**
- `feature_store/` - Data versioning
- `model_registry/` - Model versioning
- `training/` - Distributed training
- `inference/` - Model serving
- `api/` - HTTP interface
- `monitoring/` - Metrics

**Rationale:**
- **Separation of concerns**: Each module has single responsibility
- **Testability**: Easy to unit test individual modules
- **Maintainability**: Changes isolated to relevant module
- **Scalability**: Easy to add features (e.g., new training strategy)
- **Interview value**: Can explain module interactions

**Alternatives Considered:**
1. Single monolithic file
   - ❌ Hard to navigate
   - ❌ Hard to test
   - ❌ Difficult to maintain
2. Flat structure (all in one dir)
   - ⚠️ Works for small projects
   - ❌ Doesn't scale

**Trade-offs:**
- ✅ Clear responsibilities
- ✅ Easier to test
- ✅ Professional structure
- ❌ More files to manage

**Status:** ✅ ACCEPTED

---

## Decision 15: Explicit vs Implicit Trade-offs

**Decision:** Code demonstrates trade-offs explicitly in comments/documentation.

**Examples:**
```python
# Mini Spark uses round-robin load balancing (simple)
# vs. workload-aware scheduling (complex)

# Feature store uses in-memory dict (simple)
# vs. Redis (distributed, fault-tolerant)

# Inference uses synchronous API (simple)
# vs. async/streaming (complex but scalable)
```

**Rationale:**
- **Engineering maturity**: Shows understanding that all designs involve trade-offs
- **Interview discussion**: Demonstrates thoughtfulness
- **Future improvements**: Clear path for optimization

**Trade-offs:**
- ✅ Shows systems thinking
- ✅ Enables future optimization
- ⚠️ Could feel over-engineered for simple examples

**Status:** ✅ ACCEPTED

---

## Summary of Decisions

| Decision | Status | Justification |
|----------|--------|---|
| Two separate projects | ✅ | Complete JD coverage, clear separation |
| Mini Spark engine | ✅ | Fills Spark gap, demonstrates architecture |
| Redis for state | ✅ | Simple, reliable, demonstrates patterns |
| Python | ✅ | Matches JD + user expertise |
| PyTorch | ✅ | User familiar, modern ML standard |
| FastAPI | ✅ | Production-ready, K8s friendly |
| Production structure | ✅ | Professional, CI/CD ready |
| Kubernetes manifests | ✅ | Addresses K8s requirement |
| 7 distributed concepts | ✅ | Complete JD coverage |
| Testing + docs | ✅ | Professional quality |
| Docker Compose | ✅ | Easy local development |
| Cloud templates | ✅ | Shows cloud thinking |
| Modular architecture | ✅ | Maintainable, testable |
| Explicit trade-offs | ✅ | Engineering maturity |

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Mini Spark not "real Spark" | Interview concern | Document design rationale, explain architecture |
| Too much code | Maintenance burden | Keep modules focused, document well |
| Redis dependency | Setup friction | Docker Compose handles setup |
| K8s too complex | Intimidating | Provide working manifests, explain step-by-step |
| Python slower than systems languages | Performance perception | Acknowledge trade-off, explain why Python chosen |

---

## How to Use This Document

1. **Before interview**: Reference to explain architecture decisions
2. **If challenged**: "Why did you choose X?" - refer to this document
3. **For portfolio**: Link this in README as "Architecture Decision Record"
4. **For improvements**: Use as template for future decisions

---

## Validation

**Cisco JD Coverage:**
- ✅ Python proficiency (both projects)
- ✅ ML frameworks (PyTorch in Pipeline)
- ✅ Distributed data processing (Mini Spark)
- ✅ Docker/Kubernetes (both projects)
- ✅ AI/ML hands-on (ML Pipeline)
- ✅ Cloud platforms (K8s templates for AWS/GCP/Azure)
- ✅ Distributed systems (all 7 concepts in both projects)

**Amazon JD Coverage:**
- ✅ Platform services design (Mini Spark + ML Pipeline)
- ✅ Automation (Spark engine + model deployment)
- ✅ AWS-ready (K8s manifests work on EKS)
- ✅ On-call reliability (health checks, monitoring, fault tolerance)
- ✅ Cross-functional tools (both projects are platforms for others)

**Portfolio Strength:**
- ✅ Demonstrates systems thinking
- ✅ Shows engineering discipline
- ✅ Production-grade structure
- ✅ Interview-friendly explanations
- ✅ Extension points for future improvements

---

**Document Version:** 1.0  
**Last Updated:** 2026-06-02  
**Status:** FINAL - All decisions justified and documented
