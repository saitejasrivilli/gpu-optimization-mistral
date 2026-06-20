"""Tests for backend application"""

import pytest
from backend.app import create_app


@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'


def test_node_clinical(client):
    """Test clinical notes node"""
    response = client.get('/api/node/clinical')
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert 'output' in response.json


def test_node_training(client):
    """Test training data node"""
    response = client.get('/api/node/training')
    assert response.status_code == 200
    assert response.json['status'] == 'success'


def test_node_train(client):
    """Test model training node"""
    response = client.get('/api/node/train')
    assert response.status_code == 200
    assert response.json['status'] == 'success'


def test_node_validate(client):
    """Test model validation node"""
    response = client.get('/api/node/validate')
    assert response.status_code == 200
    assert response.json['status'] == 'success'


def test_node_bigquery(client):
    """Test BigQuery node"""
    response = client.get('/api/node/bigquery')
    assert response.status_code == 200
    assert response.json['status'] == 'success'


def test_404_error(client):
    """Test 404 error handling"""
    response = client.get('/api/nonexistent')
    assert response.status_code == 404


def test_all_nodes_available(client):
    """Test all nodes are available"""
    nodes = [
        'clinical', 'training', 'inference',
        'train', 'validate', 'deid', 'ner', 'fhir',
        'registry', 'gcs', 'bigquery',
        'vertex', 'endpoint', 'monitoring', 'analytics'
    ]

    for node in nodes:
        response = client.get(f'/api/node/{node}')
        assert response.status_code == 200, f"Node {node} failed"
        assert response.json['status'] == 'success'
        assert 'output' in response.json
