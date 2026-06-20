#!/usr/bin/env python3
"""Test clinical-nlp-pipeline end-to-end"""

import sys
sys.path.insert(0, '/Users/saitejasrivillibhutturu/Downloads/clinical-nlp-pipeline/src')

from pipeline import process_note

print("=" * 70)
print("CLINICAL-NLP PIPELINE END-TO-END TEST")
print("=" * 70)

# Step 1: Raw note
print("\n[1/7] RAW CLINICAL NOTE INPUT")
raw_note = """
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
"""
print(f"  Length: {len(raw_note)} chars")

# Step 2: Process
print("\n[2/7] PROCESS NOTE (de-ID → NER → FHIR)")
result = process_note(raw_note, export_to_bigquery=False, patient_id="pt-001", note_id="note-001")
print("  ✓ Processed")

# Step 3: De-ID
print("\n[3/7] DE-IDENTIFICATION RESULTS")
print(f"  PHI spans detected: {result['phi_spans_found']}")
print(f"  De-ID sample: {result['deid_text'][:80]}...")

# Step 4: NER
print("\n[4/7] CLINICAL NER EXTRACTION")
print(f"  Total entities: {len(result['entities'])}")
print(f"  Diagnoses: {len(result['diagnoses'])}")
for d in result['diagnoses'][:2]:
    print(f"    - {d['diagnosis']}")

# Step 5: Medications
print("\n[5/7] MEDICATION EXTRACTION")
print(f"  Total medications: {len(result['medications'])}")
for m in result['medications'][:2]:
    print(f"    - {m['name']} {m.get('dose', '')}")

# Step 6: FHIR
print("\n[6/7] FHIR R4 BUNDLE GENERATION")
bundle = result['fhir_bundle']
print(f"  Bundle ID: {bundle['id']}")
print(f"  Total resources: {bundle['total']}")
for entry in bundle['entry'][:3]:
    res_type = entry['resource']['resourceType']
    res_id = entry['resource']['id']
    print(f"    - {res_type}: {res_id}")

# Step 7: Export ready
print("\n[7/7] BIGQUERY EXPORT READY")
print(f"  FHIR bundles table: 1 row")
print(f"  PHI tracking table: {result['phi_spans_found']} rows")
print(f"  Entities table: {len(result['entities'])} rows")

print("\n" + "=" * 70)
print("✓ CLINICAL-NLP PIPELINE TEST COMPLETE")
print("=" * 70)
