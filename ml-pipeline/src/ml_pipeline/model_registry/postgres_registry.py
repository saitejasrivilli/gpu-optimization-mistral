"""Production Model Registry - PostgreSQL backend for version control"""

import pickle
import hashlib
import json
from typing import Any, Dict, List
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("psycopg2 not installed. Install: pip install psycopg2-binary")

class PostgresModelRegistry:
    """Production model registry with PostgreSQL backend"""

    def __init__(self, host: str = 'localhost', port: int = 5432,
                 database: str = 'ml_pipeline', user: str = 'postgres', password: str = 'postgres'):
        try:
            self.conn = psycopg2.connect(
                host=host, port=port, database=database, user=user, password=password
            )
            self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
            self._init_db()
            print(f"✓ Connected to PostgreSQL at {host}:{port}/{database}")
        except psycopg2.OperationalError as e:
            print(f"✗ PostgreSQL connection failed: {e}")
            print("  Start PostgreSQL: brew services start postgresql")
            print("  Create DB: createdb ml_pipeline")
            raise

    def _init_db(self):
        """Initialize database tables"""
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                version VARCHAR(50) NOT NULL,
                model_bytes BYTEA NOT NULL,
                metrics JSONB,
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_hash VARCHAR(64),
                model_size_bytes INTEGER,
                UNIQUE(model_name, version)
            )
        """)

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS production_models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) UNIQUE NOT NULL,
                version VARCHAR(50) NOT NULL,
                promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                promoted_by VARCHAR(255) DEFAULT 'system',
                FOREIGN KEY (model_name, version) REFERENCES models(model_name, version)
            )
        """)

        self.conn.commit()

    def save_model(self, model: Any, model_name: str, metrics: Dict = None, version: str = None) -> str:
        """Save model with version and metrics"""
        if version is None:
            self.cur.execute("SELECT COUNT(*) FROM models WHERE model_name = %s", (model_name,))
            count = self.cur.fetchone()[0]
            version = f"v{count}"

        version_key = f"{model_name}:{version}"

        # Serialize model
        model_bytes = pickle.dumps(model)
        model_hash = hashlib.md5(model_bytes).hexdigest()
        model_size_bytes = len(model_bytes)

        # Store in PostgreSQL
        self.cur.execute("""
            INSERT INTO models (model_name, version, model_bytes, metrics, model_hash, model_size_bytes)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_name, version) DO UPDATE
            SET metrics = EXCLUDED.metrics, model_hash = EXCLUDED.model_hash
        """, (
            model_name,
            version,
            model_bytes,
            json.dumps(metrics or {}),
            model_hash,
            model_size_bytes
        ))

        self.conn.commit()

        print(f"✓ Saved model {version_key} ({model_size_bytes/1024/1024:.2f}MB)")
        return version_key

    def load_model(self, version_key: str) -> Any:
        """Load model by version"""
        model_name, version = version_key.split(":")

        self.cur.execute("""
            SELECT model_bytes FROM models WHERE model_name = %s AND version = %s
        """, (model_name, version))

        result = self.cur.fetchone()
        if not result:
            raise ValueError(f"Model {version_key} not found")

        return pickle.loads(result['model_bytes'])

    def promote_to_production(self, version_key: str):
        """Promote model to production with automatic demotion of previous"""
        model_name, version = version_key.split(":")

        # Demote previous production model
        self.cur.execute("""
            UPDATE models SET status = 'staging'
            WHERE model_name = %s AND status = 'production'
        """, (model_name,))

        # Promote new model to production
        self.cur.execute("""
            UPDATE models SET status = 'production'
            WHERE model_name = %s AND version = %s
        """, (model_name, version))

        # Track promotion
        self.cur.execute("""
            INSERT INTO production_models (model_name, version)
            VALUES (%s, %s)
            ON CONFLICT (model_name) DO UPDATE
            SET version = EXCLUDED.version, promoted_at = CURRENT_TIMESTAMP
        """, (model_name, version))

        self.conn.commit()
        print(f"✓ Promoted {version_key} to production")

    def get_production_model(self, model_name: str) -> tuple:
        """Get current production model"""
        self.cur.execute("""
            SELECT m.model_bytes, m.metrics, m.version
            FROM models m
            WHERE m.model_name = %s AND m.status = 'production'
        """, (model_name,))

        result = self.cur.fetchone()
        if not result:
            raise ValueError(f"No production model found for {model_name}")

        model = pickle.loads(result['model_bytes'])
        metadata = {
            'version': result['version'],
            'metrics': json.loads(result['metrics']) if result['metrics'] else {}
        }
        return model, metadata

    def get_model_history(self, model_name: str) -> List[Dict]:
        """Get all versions of a model"""
        self.cur.execute("""
            SELECT model_name, version, metrics, status, created_at
            FROM models
            WHERE model_name = %s
            ORDER BY created_at DESC
        """, (model_name,))

        rows = self.cur.fetchall()
        return [dict(row) for row in rows]

    def compare_models(self, v1: str, v2: str) -> Dict:
        """Compare metrics between two model versions"""
        m1_name, m1_version = v1.split(":")
        m2_name, m2_version = v2.split(":")

        self.cur.execute("""
            SELECT metrics FROM models WHERE model_name = %s AND version = %s
        """, (m1_name, m1_version))
        m1_metrics = json.loads(self.cur.fetchone()['metrics'] or '{}')

        self.cur.execute("""
            SELECT metrics FROM models WHERE model_name = %s AND version = %s
        """, (m2_name, m2_version))
        m2_metrics = json.loads(self.cur.fetchone()['metrics'] or '{}')

        return {
            'model_1': v1,
            'model_2': v2,
            'metrics_1': m1_metrics,
            'metrics_2': m2_metrics,
            'improvement': {
                k: m2_metrics.get(k, 0) - m1_metrics.get(k, 0)
                for k in m1_metrics.keys()
            }
        }

    def close(self):
        """Close database connection"""
        self.cur.close()
        self.conn.close()
