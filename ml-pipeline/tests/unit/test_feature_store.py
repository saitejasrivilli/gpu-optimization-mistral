"""Unit tests for Feature Store"""
import sys
sys.path.insert(0, '../../src')

from ml_pipeline.feature_store.store import FeatureStore

def test_write_features():
    store = FeatureStore()
    features = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    store.write_features("test_dataset", features)
    assert "test_dataset" in store.list_datasets()

def test_read_features():
    store = FeatureStore()
    features = {"col1": [1, 2, 3]}
    store.write_features("test_dataset", features)
    retrieved = store.read_features("test_dataset")
    assert retrieved == features

def test_versioning():
    store = FeatureStore()
    store.write_features("test_dataset", {"col1": [1, 2, 3]}, version="v1")
    store.write_features("test_dataset", {"col1": [4, 5, 6]}, version="v2")
    history = store.get_version_history("test_dataset")
    assert len(history) == 2

if __name__ == "__main__":
    test_write_features()
    test_read_features()
    test_versioning()
    print("✓ All tests passed")
