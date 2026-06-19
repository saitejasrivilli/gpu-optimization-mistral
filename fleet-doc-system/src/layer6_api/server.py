"""Layer 6: FastAPI server - fleet document query interface."""

import time
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from src.logger import get_logger
from src.config import get_settings
from src.layer3_retrieval.tavily_rag import TavilyRetriever, LocalDocumentStore
from src.layer4_routing.query_router import QueryRouter, ResponseGenerator
from src.layer5_verification.grounding import GroundingVerifier
from src.models import VerifiedResponse
from src.layer1_ingestion.batch_processor import BatchIngestionProcessor
from src.metrics import metrics
from src.layer6_api.query_cache import QueryCache

logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Fleet Document System", version="1.0")
app.state.limiter = limiter

# Query caching (1 hour TTL)
query_cache = QueryCache(ttl_seconds=3600)

# Initialize components
tavily_retriever = TavilyRetriever()
query_router = QueryRouter()
response_generator = ResponseGenerator()
grounding_verifier = GroundingVerifier()
doc_store = LocalDocumentStore()
batch_processor = BatchIngestionProcessor(output_dir="data/processed")

# Text-to-SQL pipeline (Phase 1: MVP)
from src.layer4_routing.query_router import SQLGenerator
sql_generator = SQLGenerator()

# Phase 2: SQL Execution + Verification
from src.layer6_api.sql_executor import SQLExecutor, SQLProber
from src.layer6_api.intent_grounding import IntentGrounder, MultiAgentVerifier
import sqlite3

# Wire real database
try:
    db_conn = sqlite3.connect("fleet_data.db", check_same_thread=False)
    db_conn.row_factory = sqlite3.Row  # Return rows as dicts
    sql_executor = SQLExecutor(db_connection=db_conn)
    logger.info("database_connected", db_path="fleet_data.db")
except Exception as e:
    logger.warning("database_connection_failed", error=str(e))
    sql_executor = SQLExecutor(db_connection=None)

sql_prober = SQLProber(sql_executor)
intent_grounder = IntentGrounder(response_generator.llm_client)
multi_agent_verifier = MultiAgentVerifier(response_generator.llm_client)

def _format_results_as_table(results: list) -> str:
    """Format query results as ASCII table."""
    if not results:
        return "No results found"

    # Get column names from first row
    if isinstance(results[0], dict):
        columns = list(results[0].keys())
    else:
        return str(results)

    # Calculate column widths
    col_widths = {col: len(str(col)) for col in columns}
    for row in results:
        for col in columns:
            col_widths[col] = max(col_widths[col], len(str(row.get(col, ""))))

    # Build table
    lines = []

    # Header
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for row in results:
        row_str = " | ".join(
            str(row.get(col, "")).ljust(col_widths[col]) for col in columns
        )
        lines.append(row_str)

    return "\n".join(lines)


# Serve frontend index.html at root
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("frontend/index.html")


def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Verify Bearer token from Authorization header."""
    api_key = os.getenv("API_KEY", "demo-key-12345")

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = parts[1]
    if token != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return token


class QueryRequest(BaseModel):
    """User query request."""

    query: str
    truck_id: str = None
    include_sources: bool = True
    ab_threshold: Optional[float] = None
    text_to_sql: bool = True  # Phase 1: Text-to-SQL generation (enabled by default)
    execute_sql: bool = True  # Phase 2: Execute generated SQL (enabled by default)
    use_probes: bool = True  # Phase 2: SQL probing for error recovery
    verify_intent: bool = False  # Phase 2: Intent grounding verification
    multi_agent: bool = False  # Phase 2: Multi-agent verification


class QueryResponse(BaseModel):
    """Query response with grounding."""

    answer: str
    is_grounded: bool
    confidence: float
    sources: list = []
    query: str
    response_time_ms: float = 0
    sql_query: Optional[str] = None  # Generated SQL
    sql_results: Optional[list] = None  # Raw SQL results


class BatchUploadResponse(BaseModel):
    """Batch upload response."""

    processed: int
    failed: int
    total: int
    documents: list = []


@app.post("/query")
@limiter.limit("100/minute")
async def query_documents(request: Request, query: QueryRequest, token: str = Depends(verify_api_key)) -> QueryResponse:
    """
    Query fleet documents in natural English.

    Requires: Authorization: Bearer {api_key}
    """
    start_time = time.time()
    query_id = None

    try:
        # Check cache first (skip expensive computation if hit)
        cached = query_cache.get(query.query)
        if cached:
            response_time_ms = (time.time() - start_time) * 1000
            logger.info("query_cache_hit", response_time_ms=response_time_ms)
            cached["response_time_ms"] = response_time_ms
            return QueryResponse(**cached)

        # Step 1: Route query
        routed = await query_router.route_query(query.query)
        logger.info("query_routed", intent=routed.intent)

        # Step 1b & 2: Run SQL generation and document retrieval in parallel
        sql = None
        sql_results = None
        sql_success = False
        retrieved = []

        async def generate_sql_task():
            nonlocal sql, sql_success
            if not query.text_to_sql:
                return None
            sql = await sql_generator.generate_sql(query.query)
            if sql:
                logger.info("text_to_sql_generated", sql=sql[:100])
                if query.execute_sql:
                    sql_success, results, _ = await sql_executor.execute_with_recovery(
                        sql,
                        max_probes=3 if query.use_probes else 0,
                        llm_client=response_generator.llm_client,
                    )
                    if sql_success:
                        nonlocal sql_results
                        sql_results = results
                        logger.info("sql_execution_done", success=True, rows=len(results) if results else 0)
            return sql

        async def retrieve_documents_task():
            nonlocal retrieved
            docs = []
            if routed.truck_id:
                docs = doc_store.search_by_truck(routed.truck_id)
                if docs:
                    logger.info("documents_retrieved_from_store", truck_id=routed.truck_id, count=len(docs))
            if not docs:
                tavily_results = await tavily_retriever.retrieve(
                    query.query,
                    truck_id=routed.truck_id,
                    top_k=5,
                )
                docs.extend(tavily_results)
            return docs

        # Run both in parallel
        sql, retrieved = await asyncio.gather(
            generate_sql_task(),
            retrieve_documents_task(),
            return_exceptions=False
        )

        if not retrieved and not sql_success:
            raise HTTPException(status_code=404, detail="No matching documents found")

        # Step 3: Generate answer (from SQL or RAG)
        sql_results_formatted = None
        if sql_success and sql_results:
            # Format SQL results as table
            sql_results_formatted = sql_results
            table_text = _format_results_as_table(sql_results)
            context = f"Database Query Results:\n{table_text}"
            logger.info("using_sql_results", rows=len(sql_results))
        else:
            # Use RAG results
            context = "\n".join([f"- {doc.text}" for doc in retrieved[:3]])

        answer = await response_generator.generate_answer(
            query.query,
            context,
            routed.handler_type,
        )

        # Step 4: Verify grounding (skip for SQL results - already grounded)
        threshold = query.ab_threshold or 0.80
        if sql_success:
            # Database results are inherently grounded - 100% confidence
            verified = VerifiedResponse(
                answer=answer,
                is_grounded=True,
                confidence=1.0,
                sources=["database"],
                grounding_details=[],
                query=query.query,
            )
            is_grounded = True
        else:
            # RAG results need grounding verification
            verified = await grounding_verifier.verify_answer(
                answer,
                retrieved,
                query.query,
            )
            is_grounded = verified.confidence >= threshold

        response_time_ms = (time.time() - start_time) * 1000

        # Log metrics
        query_id = metrics.log_query(
            query=query.query,
            intent=routed.intent,
            truck_id=routed.truck_id,
            handler_type=routed.handler_type,
            confidence=verified.confidence,
            is_grounded=is_grounded,
            response_time_ms=response_time_ms,
            model="featherless",
        )

        # Log A/B test if threshold provided
        if query.ab_threshold:
            grounded_claims = sum(
                1 for c in verified.grounding_details if c["grounded"]
            )
            total_claims = len(verified.grounding_details)
            metrics.log_ab_test(
                query_id=query_id,
                threshold=query.ab_threshold,
                grounded_claims=grounded_claims,
                total_claims=total_claims,
            )

        logger.info(
            "query_completed",
            grounded=is_grounded,
            confidence=verified.confidence,
            response_time_ms=response_time_ms,
        )

        # Build sources from SQL results or RAG
        if sql_success and sql_results:
            sources_list = [
                {
                    "source": "database",
                    "type": "sql_result",
                    "rows": len(sql_results),
                    "grounded": True,
                }
            ]
        else:
            sources_list = verified.sources if query.include_sources else []

        # Format answer: ONLY show SQL results, hide web explanations
        if sql_success and sql_results_formatted:
            # SQL execution worked - show ONLY table, no theory
            answer_with_table = _format_results_as_table(sql_results_formatted)
        else:
            # No SQL results - show answer (will be from Tavily)
            answer_with_table = verified.answer

        response = QueryResponse(
            answer=answer_with_table,
            is_grounded=is_grounded,
            confidence=verified.confidence,
            sources=sources_list,
            query=query.query,
            response_time_ms=response_time_ms,
            sql_query=sql,  # Return generated SQL regardless of execution
            sql_results=sql_results_formatted if sql_success else None,  # Only return results if executed
        )

        # Cache successful responses
        try:
            query_cache.set(query.query, response.dict())
            logger.info("query_cached", query=query.query[:50])
        except Exception as cache_err:
            logger.warning("cache_store_failed", error=str(cache_err))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/upload")
@limiter.limit("10/minute")
async def batch_upload(request: Request, files: list[UploadFile] = File(...), token: str = Depends(verify_api_key)) -> BatchUploadResponse:
    """
    Batch upload fleet documents.

    Accepts: Multiple image files (PNG, JPG, PDF)
    Requires: Authorization: Bearer {api_key}
    """
    processed = 0
    failed = 0
    documents = []

    for file in files:
        try:
            # Save uploaded file temporarily
            temp_path = f"data/uploads/{file.filename}"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)

            # Process with ingestion pipeline
            result = await batch_processor.process_file(temp_path)

            # Log metrics
            metrics.log_document(
                doc_id=result.doc_id,
                doc_type=result.doc_type,
                truck_id=getattr(result, "truck_id", None),
                quality_score=getattr(result, "quality_score", 0),
                ocr_confidence=result.confidence,
                processing_time_ms=getattr(result, "processing_time_ms", 0),
            )

            documents.append({"filename": file.filename, "doc_id": result.doc_id})
            processed += 1

            # Cleanup
            os.remove(temp_path)

        except Exception as e:
            logger.error("batch_upload_file_failed", filename=file.filename, error=str(e))
            failed += 1

    return BatchUploadResponse(
        processed=processed,
        failed=failed,
        total=len(files),
        documents=documents,
    )


@app.get("/dashboard")
async def dashboard():
    """Serve dashboard HTML."""
    return FileResponse("frontend/dashboard.html")


@app.get("/database")
async def database_browser():
    """Serve database browser HTML."""
    return FileResponse("frontend/database.html")


@app.get("/architecture")
async def system_architecture():
    """Serve system architecture visualization."""
    return FileResponse("frontend/architecture.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0",
        "components": {
            "retrieval": "ready",
            "routing": "ready",
            "verification": "ready",
            "metrics": "ready",
        },
    }


@app.get("/tables")
async def list_tables():
    """List all tables in database."""
    try:
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        return {"tables": tables, "count": len(tables)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/table/{table_name}")
async def get_table_data(table_name: str, limit: int = 100):
    """Get all data from a specific table."""
    try:
        # Sanitize table name (prevent SQL injection)
        valid_tables = [
            "trucks",
            "drivers",
            "documents",
            "maintenance_records",
        ]
        if table_name not in valid_tables:
            return {"error": f"Table '{table_name}' not found"}

        cursor = db_conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {
            "table": table_name,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/schema")
async def get_schema():
    """Get database schema (all tables + columns)."""
    try:
        schema = {}
        tables = [
            "trucks",
            "drivers",
            "documents",
            "maintenance_records",
        ]

        for table_name in tables:
            cursor = db_conn.execute(f"PRAGMA table_info({table_name})")
            columns = [
                {"name": row[1], "type": row[2], "notnull": bool(row[3])}
                for row in cursor.fetchall()
            ]
            schema[table_name] = columns

        return schema
    except Exception as e:
        return {"error": str(e)}


@app.get("/stats")
async def stats():
    """Dashboard statistics."""
    return metrics.get_stats()


@app.get("/cache-stats")
async def cache_stats():
    """Query cache statistics."""
    return query_cache.stats()


@app.get("/ab-recommendations")
async def ab_recommendations():
    """Get A/B test recommendations for threshold tuning."""
    return metrics.get_ab_recommendations()


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions (auth, not found, etc)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail}
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    """Handle rate limit exceeded."""
    return JSONResponse(
        status_code=429,
        content={"status": "error", "detail": "Rate limit exceeded. Max 100 queries/minute, 10 uploads/minute."}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
