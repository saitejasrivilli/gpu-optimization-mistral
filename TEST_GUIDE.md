# End-to-End Test Guide

Complete system testing with modular visibility at each step.

## Quick Start

Run all tests:
```bash
bash RUN_TESTS.sh
```

## Individual Tests

### Test 1: ML-Pipeline (Training → Save → Load → Predict)

Tests full ML training workflow without GCP.

```bash
python test_ml_pipeline.py
```

**What it tests:**
- [1/6] Data generation (500 samples, 20 features)
- [2/6] Training loop (3 epochs with metrics)
- [3/6] Model registry save
- [4/6] Model registry load
- [5/6] Predictions on test data
- [6/6] Promote to production

**Expected output:**
```
ML-PIPELINE END-TO-END TEST
[1/6] GENERATE DATA
  Train: (400, 20), Test: (100, 20)
[2/6] TRAIN MODEL (3 epochs)
  Epoch 1: loss=0.6123, acc=0.6234, f1=0.5891
  Epoch 2: loss=0.5234, acc=0.6891, f1=0.6543
  Epoch 3: loss=0.4891, acc=0.7234, f1=0.7123
[3/6] SAVE TO REGISTRY
  Saved: prod-classifier_v1.0.0
[4/6] LOAD FROM REGISTRY
  Loaded: SimpleNN
[5/6] MAKE PREDICTIONS
  Sample 1: 0.4231
  Sample 2: 0.6123
  Sample 3: 0.5891
[6/6] PROMOTE TO PRODUCTION
  Promoted: prod-classifier_v1.0.0
✓ ML-PIPELINE TEST COMPLETE
```

---

### Test 2: Clinical-NLP (Note → De-ID → NER → FHIR)

Tests clinical NLP pipeline without GCP.

```bash
python test_clinical_nlp.py
```

**What it tests:**
- [1/7] Raw clinical note input
- [2/7] De-identification (PHI masking)
- [3/7] Named entity recognition (diseases, meds)
- [4/7] Medication extraction
- [5/7] FHIR R4 bundle generation
- [6/7] Export to BigQuery schema
- [7/7] Audit readiness

**Expected output:**
```
CLINICAL-NLP PIPELINE END-TO-END TEST
[1/7] RAW CLINICAL NOTE INPUT
  Length: 412 chars
[2/7] PROCESS NOTE (de-ID → NER → FHIR)
  ✓ Processed
[3/7] DE-IDENTIFICATION RESULTS
  PHI spans detected: 6
  De-ID sample: Patient: [PATIENT], DOB: [DATE], MRN: [ID]...
[4/7] CLINICAL NER EXTRACTION
  Total entities: 15
  Diagnoses: 3
    - Type 2 Diabetes
    - Hypertension
[5/7] MEDICATION EXTRACTION
  Total medications: 4
    - Metformin 500 mg twice daily
    - Lisinopril 10 mg once daily
[6/7] FHIR R4 BUNDLE GENERATION
  Bundle ID: urn:uuid:12a34b56-78c9-...
  Total resources: 8
    - Patient: Patient/pt-001
    - Condition: Condition/cond-001
    - Medication: Medication/med-001
[7/7] BIGQUERY EXPORT READY
  FHIR bundles table: 1 row
  PHI tracking table: 6 rows
  Entities table: 15 rows
✓ CLINICAL-NLP PIPELINE TEST COMPLETE
```

---

### Test 3: GCP Integration (Batch Load - Free Tier)

Tests GCP components with batch load (no streaming inserts).

```bash
# First set credentials (optional if using local testing)
export GOOGLE_CLOUD_PROJECT=ml-pipeline-sa
export GOOGLE_APPLICATION_CREDENTIALS=./sa-key.json

python test_gcp_integration.py
```

**What it tests:**
- [1/5] GCP credentials validation
- [2/5] Clinical NLP processing
- [3/5] BigQuery exporter initialization
- [4/5] Batch load to BigQuery (JSONL format)
- [5/5] Summary verification

**Expected output (without credentials):**
```
GCP INTEGRATION TEST (Batch Load - Free Tier)
[1/5] CHECK GCP CREDENTIALS
  Project: None
  Credentials: None
  ⚠ Credentials not configured
  Set: export GOOGLE_CLOUD_PROJECT=ml-pipeline-sa
  Set: export GOOGLE_APPLICATION_CREDENTIALS=./sa-key.json
```

**Expected output (with credentials):**
```
GCP INTEGRATION TEST (Batch Load - Free Tier)
[1/5] CHECK GCP CREDENTIALS
  Project: ml-pipeline-sa
  Credentials: ./sa-key.json
  ✓ Credentials OK
[2/5] PROCESS CLINICAL NOTE
  Entities: 12
  PHI: 4
[3/5] INITIALIZE BIGQUERY EXPORTER
  ✓ Exporter ready
[4/5] BATCH LOAD TO BIGQUERY
  ✓ Batch load job: bq-job-abc123def456
[5/5] SUMMARY
  ✓ GCP credentials OK
  ✓ Clinical NLP pipeline OK
  ✓ BigQuery integration OK
  ✓ Batch load method OK (free tier compatible)
✓ GCP INTEGRATION TEST COMPLETE
```

---

## Architecture

```
test_ml_pipeline.py
  └─ SimpleNN (neural network model)
  └─ DistributedTrainer (training loop)
  └─ ModelRegistry (save/load/promote)

test_clinical_nlp.py
  └─ process_note() (pipeline entry)
  ├─ De-identification (PHI masking)
  ├─ Named Entity Recognition (diseases, meds)
  ├─ FHIR R4 Bundle generation
  └─ BigQuery schema validation

test_gcp_integration.py
  └─ FHIRBigQueryExporter
  ├─ GCP credentials check
  ├─ Batch load via JSONL
  └─ Free tier validation

RUN_TESTS.sh
  └─ Orchestrates all three tests sequentially
  └─ Summarizes results
```

---

## Testing Phases

### Phase 1: Local (No GCP)
Run `test_ml_pipeline.py` and `test_clinical_nlp.py` first. These work without any credentials.

### Phase 2: GCP Setup (Optional)
Set credentials for `test_gcp_integration.py`:
```bash
# Create service account (one-time)
gcloud iam service-accounts create ml-pipeline-sa
gcloud iam service-accounts keys create sa-key.json --iam-account=ml-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com

# Set environment variables
export GOOGLE_CLOUD_PROJECT=ml-pipeline-sa
export GOOGLE_APPLICATION_CREDENTIALS=./sa-key.json
```

### Phase 3: Full System (All Tests)
Run entire suite:
```bash
bash RUN_TESTS.sh
```

---

## Troubleshooting

### ImportError: No module named X
Missing dependency. Install:
```bash
pip install -r ml-pipeline/requirements.txt
pip install -r clinical-nlp-pipeline/requirements.txt
```

### GCP credentials error in test_gcp_integration.py
Expected if credentials not set. Test still validates batch-load method exists.

### BigQuery free tier error
Expected. Batch load method is the workaround.

---

## Test Output Interpretation

Each test prints numbered steps showing:
- Input data (size, samples)
- Processing (durations, metrics)
- Output (generated artifacts, counts)
- Success marker (✓ or ✗)

View intermediate results to debug flow:
```bash
python test_ml_pipeline.py 2>&1 | head -20
```

---

## Next Steps

All tests passing? Ready for:
1. Deploy to GCP with full credentials
2. Configure CI/CD pipeline (.github/workflows/ci.yaml)
3. Set up monitoring (Vertex AI, BigQuery dashboards)
4. Configure pre-commit hooks: `pre-commit install`
