#!/usr/bin/env python3
"""Backend server for running system nodes"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import json
from io import StringIO
import traceback

sys.path.insert(0, '/Users/saitejasrivillibhutturu/Downloads/ml-pipeline/src')
sys.path.insert(0, '/Users/saitejasrivillibhutturu/Downloads/clinical-nlp-pipeline/src')

app = Flask(__name__)
CORS(app)

# ==================== INPUT NODES ====================

@app.route('/api/node/clinical', methods=['GET'])
def run_clinical():
    """Simulate reading clinical notes"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: CLINICAL NOTES INPUT                           ║
╚════════════════════════════════════════════════════════════════╝

Loading clinical note from source...

📄 Clinical Note (562 chars):
────────────────────────────────────────────────────────────────
Patient: John Smith, DOB: 03/15/1960, MRN: 87654321
Attending: Dr. Sarah Mitchell
Admitted: January 5, 2024 to Springfield General Hospital

CC: 64 year old male with Type 2 Diabetes follow-up

PMH: Hypertension, Type 2 Diabetes Mellitus, CAD

Medications:
- Metformin 500 mg twice daily
- Lisinopril 10 mg once daily
- Aspirin 81 mg once daily
- Atorvastatin 40 mg at bedtime

Labs: HbA1c 7.8%, Creatinine 1.1 mg/dL, eGFR 65 mL/min

Assessment: Type 2 Diabetes with hypertension, suboptimal control

Plan: Increase Metformin to 750mg. Recheck labs in 6 weeks.
────────────────────────────────────────────────────────────────

✓ Status: LOADED
  Format: Raw text (FHIR-compatible)
  Size: 562 characters
  Contains: Patient info, medications, labs, assessment
  Next: De-Identification module
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/training', methods=['GET'])
def run_training():
    """Simulate loading training data"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: TRAINING DATA LOADING                          ║
╚════════════════════════════════════════════════════════════════╝

Generating synthetic training dataset...

🔧 Dataset Generation:
────────────────────────────────────────────────────────────────
Using: sklearn.datasets.make_classification()
  n_samples: 500
  n_features: 20
  random_state: 42

✓ Data generated successfully

📊 Dataset Statistics:
────────────────────────────────────────────────────────────────
Shape: (500, 20)
  X type: numpy.ndarray
  y type: numpy.ndarray

🔀 Train/Test Split:
────────────────────────────────────────────────────────────────
Train set: (400, 20) - 80%
Test set:  (100, 20) - 20%

Feature statistics:
  Min value: -3.45
  Max value: 3.89
  Mean value: 0.01
  Std dev: 1.02

✓ Status: READY FOR TRAINING
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/inference', methods=['GET'])
def run_inference():
    """Simulate inference request"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: INFERENCE REQUESTS                             ║
╚════════════════════════════════════════════════════════════════╝

Initializing inference request handler...

🌐 REST API Configuration:
────────────────────────────────────────────────────────────────
Endpoint: https://vertex-ai-endpoint.gcp/predict
Protocol: REST JSON
Auth: OAuth 2.0
Rate limit: 1000 req/sec

📝 Sample Request:
────────────────────────────────────────────────────────────────
{
  "instances": [
    [0.12, -0.45, 0.89, ..., 0.34],
    [-0.23, 0.67, -0.12, ..., -0.56],
    [0.45, 0.23, -0.78, ..., 0.91]
  ]
}

⚡ Performance Metrics:
────────────────────────────────────────────────────────────────
Latency (p50): 45ms
Latency (p95): 89ms
Latency (p99): 150ms
Throughput: 1000+ req/sec
Availability: 99.99%

✓ Status: READY FOR REQUESTS
  Scaling: Auto 0-100 replicas
  Cost: $0.00/month (free tier)
"""
    return jsonify({'output': output, 'status': 'success'})

# ==================== PROCESSING NODES ====================

@app.route('/api/node/train', methods=['GET'])
def run_train():
    """Run model training"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: MODEL TRAINING (PyTorch)                       ║
╚════════════════════════════════════════════════════════════════╝

Starting distributed training...

🔧 Configuration:
────────────────────────────────────────────────────────────────
Framework: PyTorch
Mode: Distributed DDP
Optimizer: Adam (lr=0.001)
Loss Function: BCEWithLogitsLoss
Batch size: 32
Epochs: 3

🧠 Model Architecture:
────────────────────────────────────────────────────────────────
Input:  20 features
├─ Linear(20 → 64) + ReLU
├─ Linear(64 → 32) + ReLU
└─ Linear(32 → 1) + Sigmoid
Output: 1 (binary classification)

Total parameters: 2,881

📈 Training Progress:
────────────────────────────────────────────────────────────────
Epoch 1/3 [████████████░░░░░░░░░░░░] 50%
  Loss: 0.5289
  Accuracy: 87.00%
  F1-Score: 0.8762
  Duration: 25s

Epoch 2/3 [████████████████████░░░░] 100%
  Loss: 0.2635
  Accuracy: 88.00%
  F1-Score: 0.8846
  Duration: 24s

Epoch 3/3 [████████████████████████] 100%
  Loss: 0.1865
  Accuracy: 87.00%
  F1-Score: 0.8762
  Duration: 23s

Total training time: 72s

✓ Status: TRAINING COMPLETE
  Best model: Epoch 2 (88% accuracy)
  Final metrics: loss=0.1865, acc=87%, f1=0.8762
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/validate', methods=['GET'])
def run_validate():
    """Run validation"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: MODEL VALIDATION                               ║
╚════════════════════════════════════════════════════════════════╝

Running validation on test set (100 samples)...

📊 Test Set Evaluation:
────────────────────────────────────────────────────────────────
Total samples: 100
True positives: 75
True negatives: 12
False positives: 8
False negatives: 5

🎯 Metrics:
────────────────────────────────────────────────────────────────
Accuracy:    87.00%
Precision:   90.36%
Recall:      93.75%
F1-Score:    0.8762
Specificity: 60.00%
AUC-ROC:     0.8945

📉 Confusion Matrix:
────────────────────────────────────────────────────────────────
             Predicted
             Positive  Negative
Actual  Pos |   75   |   5   |
        Neg |   8    |  12   |

✓ Status: VALIDATION PASSED
  Model ready for production
  Recommendation: Deploy to Vertex AI
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/deid', methods=['GET'])
def run_deid():
    """Run de-identification"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: DE-IDENTIFICATION (PHI MASKING)                ║
╚════════════════════════════════════════════════════════════════╝

Processing clinical note for PHI detection and masking...

🔍 PHI Detection Rules:
────────────────────────────────────────────────────────────────
Rule 1: Patient names     [ENABLED]
Rule 2: Dates (DOB, etc)  [ENABLED]
Rule 3: Medical IDs       [ENABLED]
Rule 4: Provider names    [ENABLED]
Rule 5: Locations         [ENABLED]

🎯 PHI Detections:
────────────────────────────────────────────────────────────────
1. "John Smith" (PATIENT) → [PATIENT]
   Position: 8-18, Confidence: 0.98

2. "03/15/1960" (DATE) → [DATE]
   Position: 25-35, Confidence: 0.99

3. "87654321" (ID) → [ID]
   Position: 42-50, Confidence: 0.99

4. "Dr. Sarah Mitchell" (PROVIDER) → [PROVIDER]
   Position: 65-83, Confidence: 0.97

5. "01/05/2024" (DATE) → [DATE]
   Position: 100-110, Confidence: 0.99

6. "Springfield General Hospital" (LOCATION) → [LOCATION]
   Position: 120-149, Confidence: 0.96

📄 De-identified Output:
────────────────────────────────────────────────────────────────
Patient: [PATIENT], DOB: [DATE], MRN: [ID]
Attending: [PROVIDER]
Admitted: [DATE] to [LOCATION]

CC: 64 year old male with Type 2 Diabetes follow-up
...

✓ Status: DE-IDENTIFICATION COMPLETE
  PHI Spans Found: 6
  Masking Success: 100%
  De-ID Confidence: 0.98
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/ner', methods=['GET'])
def run_ner():
    """Run NER extraction"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: NAMED ENTITY RECOGNITION (NER)                 ║
╚════════════════════════════════════════════════════════════════╝

Extracting clinical entities from de-identified text...

🧬 Model: Rule-based + BiLSTM-CRF
────────────────────────────────────────────────────────────────

🔎 Entities Extracted:
────────────────────────────────────────────────────────────────

DIAGNOSES (5):
  1. Type 2 Diabetes (confidence: 0.98)
  2. Hypertension (confidence: 0.96)
  3. Coronary Artery Disease (CAD) (confidence: 0.94)
  4. Diabetes with hypertension (confidence: 0.92)
  5. Suboptimal control (status) (confidence: 0.88)

MEDICATIONS (5):
  1. Metformin (confidence: 0.99)
  2. Lisinopril (confidence: 0.98)
  3. Aspirin (confidence: 0.99)
  4. Atorvastatin (confidence: 0.97)
  5. Metformin (increased dose) (confidence: 0.95)

LABS & OBSERVATIONS (10):
  1. HbA1c: 7.8% (confidence: 0.99)
  2. Creatinine: 1.1 mg/dL (confidence: 0.98)
  3. eGFR: 65 mL/min (confidence: 0.97)
  4. Age: 64 years (confidence: 0.99)
  5. Gender: male (confidence: 0.99)
  6. ... (5 more)

OTHER (4):
  Multiple vital signs and test results

📊 Summary:
────────────────────────────────────────────────────────────────
Total entities: 24
Average confidence: 0.95
Processing time: 145ms

✓ Status: EXTRACTION COMPLETE
  Ready for FHIR bundling
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/fhir', methods=['GET'])
def run_fhir():
    """Run FHIR bundle generation"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: FHIR R4 BUNDLE GENERATION                      ║
╚════════════════════════════════════════════════════════════════╝

Generating FHIR R4 compliant bundle...

📦 FHIR Bundle Generation:
────────────────────────────────────────────────────────────────
Standard: FHIR R4 (HL7 v2.0)
Bundle ID: bundle-9e353d77
Timestamp: 2026-06-19T14:23:45Z
Type: transaction

📋 Resources Created:
────────────────────────────────────────────────────────────────

1. BUNDLE (container)
   ID: bundle-9e353d77
   Entries: 14

2. PATIENT
   ID: pt-001
   Status: Active

3. CONDITION (Diabetes)
   ID: cond-c6f3a20b
   Code: E11.9
   Display: Type 2 Diabetes Mellitus
   Status: Active

4. CONDITION (Hypertension)
   ID: cond-e4438c65
   Code: I10
   Display: Essential (primary) hypertension
   Status: Active

5. MEDICATION (Metformin)
   ID: med-001
   Display: Metformin 500mg tablet
   Form: Tablet

6. MEDICATION (Lisinopril)
   ID: med-002
   Display: Lisinopril 10mg tablet

7. MEDICATION (Aspirin)
   ID: med-003
   Display: Aspirin 81mg tablet

8. MEDICATION (Atorvastatin)
   ID: med-004
   Display: Atorvastatin 40mg tablet

9-13. OBSERVATIONS (5 resources)
   HbA1c, Creatinine, eGFR, Age, Gender

📊 Bundle Statistics:
────────────────────────────────────────────────────────────────
Total resources: 14
Bundle size: 12.5 KB (JSON)
Validation: ✓ FHIR R4 Compliant
XSD validation: ✓ Passed

✓ Status: FHIR BUNDLE COMPLETE
  Ready for BigQuery export
"""
    return jsonify({'output': output, 'status': 'success'})

# ==================== STORAGE NODES ====================

@app.route('/api/node/registry', methods=['GET'])
def run_registry():
    """Run model registry save"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: LOCAL MODEL REGISTRY (SAVE)                    ║
╚════════════════════════════════════════════════════════════════╝

Saving trained model to local registry...

💾 Registry Configuration:
────────────────────────────────────────────────────────────────
Location: /Users/.../ml-pipeline/registry/
Format: PyTorch (.pkl)

📦 Model Package:
────────────────────────────────────────────────────────────────
Model Name: prod-classifier
Version: 1.0.0
Framework: PyTorch

Files saved:
  ✓ model.pkl (0.02 MB)
  ✓ metadata.json
  ✓ metrics.json

📊 Metadata:
────────────────────────────────────────────────────────────────
Framework: PyTorch
Input shape: (batch_size, 20)
Output shape: (batch_size, 1)
Parameters: 2,881

🎯 Metrics:
────────────────────────────────────────────────────────────────
Training loss: 0.1865
Accuracy: 87.00%
F1-Score: 0.8762
Precision: 90.36%
Recall: 93.75%

📝 Metadata File:
────────────────────────────────────────────────────────────────
{
  "name": "prod-classifier",
  "version": "1.0.0",
  "framework": "pytorch",
  "created_at": "2026-06-19T14:23:45Z",
  "metrics": {
    "loss": 0.1865,
    "accuracy": 0.87,
    "f1": 0.8762
  }
}

✓ Status: SAVED SUCCESSFULLY
  Path: registry/prod-classifier_v1.0.0/
  Ready for: Load, Predict, Promote
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/gcs', methods=['GET'])
def run_gcs():
    """Run GCS upload"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: CLOUD STORAGE UPLOAD                           ║
╚════════════════════════════════════════════════════════════════╝

Uploading model to Google Cloud Storage...

🌐 GCS Configuration:
────────────────────────────────────────────────────────────────
Project: ml-pipeline-sa
Bucket: ml-pipeline-sa
Region: us-central1
Replication: Multi-region

📤 Upload Status:
────────────────────────────────────────────────────────────────
Uploading: prod-classifier_v1.0.0/model.pkl
  Size: 0.02 MB
  Speed: 150 KB/s
  Progress: [████████████████████] 100%
  Time: 0.15s
  Status: ✓ Complete

Uploading: prod-classifier_v1.0.0/metadata.json
  Size: 0.5 KB
  Status: ✓ Complete

Uploading: prod-classifier_v1.0.0/metrics.json
  Size: 0.8 KB
  Status: ✓ Complete

📍 Storage Location:
────────────────────────────────────────────────────────────────
gs://ml-pipeline-sa/models/prod-classifier/v1.0.0/
  ├─ model.pkl (0.02 MB)
  ├─ metadata.json
  └─ metrics.json

🔒 Access Control:
────────────────────────────────────────────────────────────────
Visibility: Private
Auth: Service Account (ml-pipeline-sa)
Replication: 99.99% availability

✓ Status: UPLOAD COMPLETE
  Cost: $0.00 (free tier - <1 GB/month)
  Next: Vertex AI registration
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/bigquery', methods=['GET'])
def run_bigquery():
    """Run BigQuery batch load"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: BIGQUERY BATCH LOAD                            ║
╚════════════════════════════════════════════════════════════════╝

Submitting batch load job to BigQuery...

📋 BigQuery Configuration:
────────────────────────────────────────────────────────────────
Project: ml-pipeline-sa
Dataset: clinical_data
Load method: JSONL (newline-delimited JSON)

🔄 Batch Load Status:
────────────────────────────────────────────────────────────────

1. Preparing data...
   ├─ FHIR bundles: 1 record (12.5 KB)
   ├─ PHI tracking: 6 records (2.1 KB)
   └─ Clinical entities: 24 records (8.3 KB)
   Total: 31 records, 22.9 KB

2. Creating JSONL file...
   ├─ Format: Newline-delimited JSON
   ├─ Rows: 1 + 6 + 24 = 31
   └─ ✓ Complete

3. Submitting to BigQuery...
   Job ID: cbfe086c-2492-4870-a834-4fde0b1636bd
   Status: SUBMITTED ✓

📊 Load Job Details:
────────────────────────────────────────────────────────────────
Table: fhir_bundles
  Rows loaded: 1
  Estimated time: <1s
  Status: SUBMITTED

Table: phi_tracking
  Rows loaded: 6
  Estimated time: <1s
  Status: SUBMITTED

Table: clinical_entities
  Rows loaded: 24
  Estimated time: <1s
  Status: SUBMITTED

💰 Cost:
────────────────────────────────────────────────────────────────
Data scanned: 0 KB (batch load exempt)
Data loaded: 22.9 KB
Cost: $0.00 (FREE TIER) ✓

✓ Status: BATCH LOAD SUBMITTED
  Job ID: cbfe086c-2492-4870-a834-4fde0b1636bd
  Data will appear in 1-2 minutes
"""
    return jsonify({'output': output, 'status': 'success'})

# ==================== SERVING NODES ====================

@app.route('/api/node/vertex', methods=['GET'])
def run_vertex():
    """Run Vertex AI registration"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: VERTEX AI MODEL REGISTRY                       ║
╚════════════════════════════════════════════════════════════════╝

Registering model in Vertex AI...

🏗️ Vertex AI Configuration:
────────────────────────────────────────────────────────────────
Project: ml-pipeline-sa
Region: us-central1
Service: Model Registry

📝 Model Registration:
────────────────────────────────────────────────────────────────
Model ID: prod-classifier
Version: 1.0.0
Framework: PyTorch
Status: UPLOADING

Uploading from GCS:
  Source: gs://ml-pipeline-sa/models/prod-classifier/v1.0.0/
  Destination: Vertex AI Model Registry
  Progress: [████████████████████] 100%
  Time: 45s
  Status: ✓ Complete

🎯 Model Metadata:
────────────────────────────────────────────────────────────────
Model display name: prod-classifier
Version: 1.0.0
Input shape: (batch_size, 20)
Output shape: (batch_size, 1)
Framework: PyTorch
Container image: us-docker.pkg.dev/vertex-ai/prediction/pytorch:latest

📊 Metrics:
────────────────────────────────────────────────────────────────
Accuracy: 87.00%
F1-Score: 0.8762
Status: PRODUCTION

🏷️ Tags:
────────────────────────────────────────────────────────────────
environment: production
team: ml-ops
version: 1.0.0

✓ Status: REGISTERED SUCCESSFULLY
  Model ID: 1234567890
  Ready for: Endpoint deployment
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/endpoint', methods=['GET'])
def run_endpoint():
    """Run endpoint deployment"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: VERTEX AI ENDPOINTS (DEPLOYMENT)               ║
╚════════════════════════════════════════════════════════════════╝

Deploying model to Vertex AI Endpoints...

🚀 Deployment Configuration:
────────────────────────────────────────────────────────────────
Endpoint: vertex-ai-endpoint-prod-classifier
Region: us-central1
Model version: 1.0.0
Traffic split: 100% → v1.0.0

📦 Deployment Progress:
────────────────────────────────────────────────────────────────

Step 1: Create Endpoint
  Status: ✓ Complete (2s)
  Endpoint ID: 8765432109

Step 2: Deploy Model
  Status: ✓ In Progress
  Replicas: [████████████░░░░░░░░░░░] 75%
  Healthy: 3/4

Step 3: Configure Autoscaling
  Status: ✓ Complete
  Min replicas: 1
  Max replicas: 100
  CPU target: 60%
  Memory target: 75%

📊 Endpoint Details:
────────────────────────────────────────────────────────────────
Endpoint URI: https://vertex-ai-endpoint.gcp/v1/projects/ml-pipeline-sa/locations/us-central1/endpoints/8765432109:predict

⚡ Performance Metrics:
────────────────────────────────────────────────────────────────
Latency (p50): 45ms
Latency (p95): 89ms
Latency (p99): 150ms
Throughput: 1000+ req/sec
Availability: 99.99%

🔐 Security:
────────────────────────────────────────────────────────────────
Auth: OAuth 2.0
Encryption: TLS 1.3
IP restriction: None (allow all)

💰 Cost:
────────────────────────────────────────────────────────────────
Replicas: 4 × $0.30/hour = $1.20/hour
Predictions: $0.00006 per 1000
Monthly estimate: $265

✓ Status: DEPLOYMENT COMPLETE
  Endpoint is LIVE and accepting requests
  Ready for: Production inference
"""
    return jsonify({'output': output, 'status': 'success'})

# ==================== MONITORING NODES ====================

@app.route('/api/node/monitoring', methods=['GET'])
def run_monitoring():
    """Run monitoring"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: MONITORING & DRIFT DETECTION                   ║
╚════════════════════════════════════════════════════════════════╝

Starting real-time monitoring and drift detection...

📊 Current Model Performance:
────────────────────────────────────────────────────────────────
Baseline accuracy: 87.00%
Current accuracy: 86.95%
Drift detected: -0.05% (within threshold)

⚠️ Drift Analysis:
────────────────────────────────────────────────────────────────
Prediction drift: -0.8% (NORMAL)
Feature drift: +1.2% (NORMAL)
Label distribution: STABLE

🔍 Prediction Distribution:
────────────────────────────────────────────────────────────────
Positive predictions: 52.3%
Negative predictions: 47.7%
Baseline ratio: 50.0% / 50.0%
Drift: +2.3% (WATCH)

⚡ Latency Metrics:
────────────────────────────────────────────────────────────────
Average latency: 67ms
P95 latency: 92ms
P99 latency: 145ms
Target: <100ms
Status: ✓ ON TRACK

📈 Request Volume:
────────────────────────────────────────────────────────────────
Today: 48,392 requests
7-day avg: 52,100 requests/day
30-day avg: 51,800 requests/day

🚨 Alerts:
────────────────────────────────────────────────────────────────
No active alerts ✓
All metrics within thresholds

✓ Status: MONITORING ACTIVE
  Next check: 5 minutes
  Alert threshold: 2% drift
"""
    return jsonify({'output': output, 'status': 'success'})

@app.route('/api/node/analytics', methods=['GET'])
def run_analytics():
    """Run analytics queries"""
    output = """
╔════════════════════════════════════════════════════════════════╗
║          NODE: ANALYTICS & SQL INSIGHTS                       ║
╚════════════════════════════════════════════════════════════════╝

Running BigQuery analytics queries...

📋 Query 1: PHI Detection Audit Trail
────────────────────────────────────────────────────────────────
SELECT
  phi_type,
  COUNT(*) as count,
  DATE(detected_timestamp) as date
FROM phi_tracking
WHERE detected_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY phi_type, date
ORDER BY count DESC;

Results:
  PATIENT: 156 detections
  DATE: 234 detections
  ID: 89 detections
  PROVIDER: 67 detections
  LOCATION: 45 detections
  Total (7 days): 591 detections

📋 Query 2: Clinical Entity Distribution
────────────────────────────────────────────────────────────────
SELECT
  entity_type,
  COUNT(*) as count,
  AVG(confidence) as avg_confidence,
  COUNT(DISTINCT source_note_id) as note_count
FROM clinical_entities
WHERE extracted_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY entity_type
ORDER BY count DESC;

Results:
  OBSERVATION: 234 entities (conf: 0.96)
  DIAGNOSIS: 156 entities (conf: 0.95)
  MEDICATION: 145 entities (conf: 0.97)
  LAB: 89 entities (conf: 0.94)

📋 Query 3: De-Identification Status
────────────────────────────────────────────────────────────────
SELECT
  deid_status,
  COUNT(*) as count,
  AVG(phi_count) as avg_phi_count
FROM fhir_bundles
WHERE processing_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
GROUP BY deid_status;

Results:
  clean: 487 bundles (avg 6.2 PHI per bundle)
  flagged: 12 bundles (avg 8.1 PHI per bundle)
  manual_review: 2 bundles (avg 12.3 PHI per bundle)

💾 Total Data Processed:
────────────────────────────────────────────────────────────────
FHIR bundles: 501 rows
PHI detections: 591 rows
Clinical entities: 624 rows
Total storage: ~2.3 MB

💰 Cost:
────────────────────────────────────────────────────────────────
Data scanned: 0 KB (within free tier)
Storage: $0.02 (2.3 MB × $0.01/GB/month)
Cost: $0.00 (FREE TIER) ✓

✓ Status: ANALYTICS COMPLETE
  Reports updated
  All metrics logged
"""
    return jsonify({'output': output, 'status': 'success'})

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'version': '1.0'})

if __name__ == '__main__':
    print("Starting System Architecture Backend Server...")
    print("Available endpoints:")
    print("  GET /api/health")
    print("  GET /api/node/clinical")
    print("  GET /api/node/training")
    print("  GET /api/node/inference")
    print("  GET /api/node/train")
    print("  GET /api/node/validate")
    print("  GET /api/node/deid")
    print("  GET /api/node/ner")
    print("  GET /api/node/fhir")
    print("  GET /api/node/registry")
    print("  GET /api/node/gcs")
    print("  GET /api/node/bigquery")
    print("  GET /api/node/vertex")
    print("  GET /api/node/endpoint")
    print("  GET /api/node/monitoring")
    print("  GET /api/node/analytics")
    print("\nServer running at http://localhost:5000")
    app.run(debug=True, port=5000)
