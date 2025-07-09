# api_server.py
"""
FastAPI server for the Advanced OCR Legal RAG System
Provides REST API, WebSocket support, and monitoring
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import uvicorn
from pathlib import Path
import aioredis
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import logging
from advanced_ocr_legal_rag import AdvancedOCRLegalRAG, Config

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced OCR Legal RAG API",
    description="Production-ready API for legal document search and analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
search_counter = Counter('rag_searches_total', 'Total number of searches')
search_histogram = Histogram('rag_search_duration_seconds', 'Search duration in seconds')
active_connections = Gauge('rag_websocket_connections', 'Active WebSocket connections')
build_counter = Counter('rag_builds_total', 'Total number of system builds')

# Global RAG instance
rag_system = None
config = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        active_connections.inc()
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        active_connections.dec()
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results to return")
    search_type: str = Field("hybrid", description="Search type: hybrid, semantic, keyword, citation")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")

class SearchResult(BaseModel):
    content: str
    page: int
    score: float
    citations: List[str]
    legal_terms: List[str]
    relevance_explanation: str
    highlighted_content: Optional[str]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    search_time: float
    search_type: str
    query_features: Dict[str, Any]

class BuildRequest(BaseModel):
    pdf_path: str
    output_dir: str = "rag_output"
    force_rebuild: bool = False
    config_overrides: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    status: str
    total_chunks: int
    total_pages: int
    unique_citations: int
    index_size_mb: float
    last_updated: str
    performance_metrics: Dict[str, Any]

class ExportRequest(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    format: str = Field("json", description="Export format: json, markdown, csv, pdf")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global rag_system, config
    
    # Load configuration
    config = Config("config.yaml")
    
    # Initialize RAG system
    rag_system = AdvancedOCRLegalRAG(config_path="config.yaml")
    
    # Try to load existing system
    try:
        rag_system.load_system(config.get('default_output_dir', 'rag_output'))
        logger.info("Loaded existing RAG system")
    except:
        logger.info("No existing system found, will build on first request")
    
    # Initialize Redis for pub/sub
    if config.get('cache.enabled'):
        app.state.redis = await aioredis.create_redis_pool(
            f"redis://{config.get('cache.redis_host')}:{config.get('cache.redis_port')}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(app.state, 'redis'):
        app.state.redis.close()
        await app.state.redis.wait_closed()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced OCR Legal RAG API",
        "version": "2.0.0",
        "status": "ready" if rag_system and rag_system.chunks else "not_initialized"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_loaded": bool(rag_system and rag_system.chunks)
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type="text/plain"
    )

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform search on the RAG system"""
    if not rag_system or not rag_system.chunks:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    search_counter.inc()
    
    with search_histogram.time():
        try:
            # Perform search
            results = rag_system.search(
                query=request.query,
                top_k=request.top_k,
                search_type=request.search_type
            )
            
            # Convert to response model
            search_results = []
            for r in results['results']:
                search_results.append(SearchResult(
                    content=r['content'],
                    page=r['page'],
                    score=r['score'],
                    citations=r['citations'],
                    legal_terms=r['legal_terms'],
                    relevance_explanation=r['relevance_explanation'],
                    highlighted_content=r.get('highlighted_content')
                ))
            
            response = SearchResponse(
                query=results['query'],
                results=search_results,
                total_found=results['total_found'],
                search_time=results['search_time'],
                search_type=results['search_type'],
                query_features=results['query_features']
            )
            
            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "search",
                "query": request.query,
                "results_count": len(search_results)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/build")
async def build_system(
    request: BuildRequest,
    background_tasks: BackgroundTasks
):
    """Build or rebuild the RAG system"""
    build_counter.inc()
    
    # Run build in background
    background_tasks.add_task(
        build_system_task,
        request.pdf_path,
        request.output_dir,
        request.force_rebuild,
        request.config_overrides
    )
    
    return {
        "message": "Build started",
        "status": "processing",
        "check_status_at": "/status"
    }

async def build_system_task(pdf_path: str, output_dir: str, 
                          force_rebuild: bool, config_overrides: Optional[Dict]):
    """Background task for building system"""
    global rag_system
    
    try:
        # Apply config overrides if provided
        if config_overrides:
            for key, value in config_overrides.items():
                rag_system.config.config[key] = value
        
        # Build system
        result = rag_system.build_system(
            pdf_path=pdf_path,
            output_dir=output_dir,
            force_rebuild=force_rebuild
        )
        
        # Broadcast completion
        await manager.broadcast({
            "type": "build_complete",
            "status": result['status'],
            "stats": result.get('stats', {})
        })
        
    except Exception as e:
        logger.error(f"Build error: {e}")
        await manager.broadcast({
            "type": "build_error",
            "error": str(e)
        })

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not rag_system.chunks:
        return SystemStatus(
            status="not_built",
            total_chunks=0,
            total_pages=0,
            unique_citations=0,
            index_size_mb=0,
            last_updated="",
            performance_metrics={}
        )
    
    # Calculate stats
    stats = rag_system._calculate_system_stats()
    metrics = rag_system.get_performance_metrics()
    
    return SystemStatus(
        status="ready",
        total_chunks=stats['total_chunks'],
        total_pages=stats['total_pages'],
        unique_citations=stats['unique_citations'],
        index_size_mb=stats['index_size_mb'],
        last_updated=datetime.now().isoformat(),
        performance_metrics=metrics
    )

@app.post("/export")
async def export_results(request: ExportRequest):
    """Export search results in various formats"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Prepare results dict
        results_dict = {
            'query': request.query,
            'results': request.results,
            'total_found': len(request.results),
            'export_time': datetime.now().isoformat()
        }
        
        # Export based on format
        if request.format == "pdf":
            # Generate PDF (requires additional library like reportlab)
            raise HTTPException(status_code=501, detail="PDF export not implemented")
        else:
            content = rag_system.export_results(
                results_dict,
                format=request.format
            )
            
            # Return appropriate response
            media_type = {
                "json": "application/json",
                "csv": "text/csv",
                "markdown": "text/markdown"
            }.get(request.format, "text/plain")
            
            return StreamingResponse(
                io.StringIO(content),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename=results.{request.format}"
                }
            )
    
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks
):
    """Upload a PDF document for processing"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    upload_path = Path("uploads")
    upload_path.mkdir(exist_ok=True)
    
    file_path = upload_path / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Trigger build in background
    background_tasks.add_task(
        build_system_task,
        str(file_path),
        "rag_output",
        True,
        None
    )
    
    return {
        "message": "File uploaded and processing started",
        "filename": file.filename,
        "status": "processing"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "search":
                # Perform search and send results
                if rag_system and rag_system.chunks:
                    results = rag_system.search(
                        query=data.get("query", ""),
                        top_k=data.get("top_k", 10)
                    )
                    await websocket.send_json({
                        "type": "search_results",
                        "results": results
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/suggest")
async def suggest_queries(
    partial: str = Query(..., description="Partial query for suggestions")
):
    """Get query suggestions based on partial input"""
    if not rag_system or not rag_system.chunks:
        return {"suggestions": []}
    
    # Simple suggestion based on legal terms and citations
    suggestions = set()
    
    # Search through chunks for matching terms
    partial_lower = partial.lower()
    for chunk in rag_system.chunks[:1000]:  # Limit for performance
        # Check legal terms
        for term in chunk.legal_terms:
            if partial_lower in term.lower():
                suggestions.add(term)
        
        # Check citations
        for citation in chunk.citations:
            if partial_lower in citation.lower():
                suggestions.add(citation)
    
    return {
        "suggestions": sorted(list(suggestions))[:10]
    }

@app.post("/feedback")
async def submit_feedback(
    query: str = Query(...),
    result_id: int = Query(...),
    relevant: bool = Query(...),
    comments: Optional[str] = Query(None)
):
    """Submit relevance feedback for search results"""
    # In production, store this in a database
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "result_id": result_id,
        "relevant": relevant,
        "comments": comments
    }
    
    # Log feedback (in production, save to database)
    logger.info(f"Feedback received: {feedback_data}")
    
    # Could trigger model retraining here
    
    return {"message": "Feedback recorded", "id": str(hash(str(feedback_data)))}

# CLI for running the server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR Legal RAG API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )