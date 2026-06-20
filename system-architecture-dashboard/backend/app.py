#!/usr/bin/env python3
"""Flask backend server for System Architecture Dashboard"""

from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

from .routes import register_routes
from .config import config

# Load environment variables
load_dotenv()

def create_app(config_name='development'):
    """Application factory"""
    app = Flask(__name__,
                static_folder='../static',
                template_folder='../templates')

    # Load configuration
    app.config.from_object(config[config_name])

    # Enable CORS
    CORS(app)

    # Register routes
    register_routes(app)

    # Health check
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'environment': config_name
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    return app

def main():
    """Entry point"""
    app = create_app(os.getenv('FLASK_ENV', 'development'))
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False') == 'True'

    print("=" * 70)
    print("System Architecture Dashboard - Backend Server")
    print("=" * 70)
    print(f"Environment: {os.getenv('FLASK_ENV', 'development')}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print("\nAvailable endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/node/<node_id>")
    print("\nNode IDs: clinical, training, inference, train, validate,")
    print("          deid, ner, fhir, registry, gcs, bigquery,")
    print("          vertex, endpoint, monitoring, analytics")
    print("\nServer running at http://localhost:{port}")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    main()
