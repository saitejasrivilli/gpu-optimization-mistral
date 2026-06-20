#!/bin/bash
# Master test runner - shows each step clearly

cd /Users/saitejasrivillibhutturu/Downloads

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              END-TO-END SYSTEM TEST SUITE                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Test 1: ML-Pipeline
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: ML-PIPELINE (Training → Save → Load → Predict)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python test_ml_pipeline.py
TEST1=$?

# Test 2: Clinical-NLP
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: CLINICAL-NLP (Note → De-ID → NER → FHIR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python test_clinical_nlp.py
TEST2=$?

# Test 3: GCP Integration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: GCP INTEGRATION (Batch Load)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export GOOGLE_CLOUD_PROJECT=ml-pipeline-sa
export GOOGLE_APPLICATION_CREDENTIALS="/Users/saitejasrivillibhutturu/sa-key.json"
python test_gcp_integration.py
TEST3=$?

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                       TEST SUMMARY                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Test 1 (ML-Pipeline):        $([ $TEST1 -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo "Test 2 (Clinical-NLP):       $([ $TEST2 -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo "Test 3 (GCP Integration):    $([ $TEST3 -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo ""

if [ $TEST1 -eq 0 ] && [ $TEST2 -eq 0 ] && [ $TEST3 -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              ✓ ALL TESTS PASSED - SYSTEM READY                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              ✗ SOME TESTS FAILED - CHECK OUTPUT               ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 1
fi
