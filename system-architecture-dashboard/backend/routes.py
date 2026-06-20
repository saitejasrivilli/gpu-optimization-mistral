"""API routes for System Architecture Dashboard"""

from flask import jsonify
from .nodes import (
    clinical_notes, training_data, inference_requests,
    train_model, validate_metrics, deid_module, ner_extraction,
    fhir_bundle, local_registry, cloud_storage, bigquery_load,
    vertex_ai, endpoints, monitoring_drift, analytics_sql
)

def register_routes(app):
    """Register all API routes"""

    # INPUT NODES
    @app.route('/api/node/clinical', methods=['GET'])
    def node_clinical():
        return jsonify({'output': clinical_notes(), 'status': 'success'})

    @app.route('/api/node/training', methods=['GET'])
    def node_training():
        return jsonify({'output': training_data(), 'status': 'success'})

    @app.route('/api/node/inference', methods=['GET'])
    def node_inference():
        return jsonify({'output': inference_requests(), 'status': 'success'})

    # PROCESSING NODES
    @app.route('/api/node/train', methods=['GET'])
    def node_train():
        return jsonify({'output': train_model(), 'status': 'success'})

    @app.route('/api/node/validate', methods=['GET'])
    def node_validate():
        return jsonify({'output': validate_metrics(), 'status': 'success'})

    @app.route('/api/node/deid', methods=['GET'])
    def node_deid():
        return jsonify({'output': deid_module(), 'status': 'success'})

    @app.route('/api/node/ner', methods=['GET'])
    def node_ner():
        return jsonify({'output': ner_extraction(), 'status': 'success'})

    @app.route('/api/node/fhir', methods=['GET'])
    def node_fhir():
        return jsonify({'output': fhir_bundle(), 'status': 'success'})

    # STORAGE NODES
    @app.route('/api/node/registry', methods=['GET'])
    def node_registry():
        return jsonify({'output': local_registry(), 'status': 'success'})

    @app.route('/api/node/gcs', methods=['GET'])
    def node_gcs():
        return jsonify({'output': cloud_storage(), 'status': 'success'})

    @app.route('/api/node/bigquery', methods=['GET'])
    def node_bigquery():
        return jsonify({'output': bigquery_load(), 'status': 'success'})

    # SERVING NODES
    @app.route('/api/node/vertex', methods=['GET'])
    def node_vertex():
        return jsonify({'output': vertex_ai(), 'status': 'success'})

    @app.route('/api/node/endpoint', methods=['GET'])
    def node_endpoint():
        return jsonify({'output': endpoints(), 'status': 'success'})

    # MONITORING NODES
    @app.route('/api/node/monitoring', methods=['GET'])
    def node_monitoring():
        return jsonify({'output': monitoring_drift(), 'status': 'success'})

    @app.route('/api/node/analytics', methods=['GET'])
    def node_analytics():
        return jsonify({'output': analytics_sql(), 'status': 'success'})
