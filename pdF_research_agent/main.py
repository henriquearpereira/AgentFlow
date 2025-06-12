#!/usr/bin/env python3
"""
AI Research Agent - FastAPI Backend
Converted from CLI to web API with WebSocket support
"""

import os
import sys
import asyncio
import json
import time
import uuid
from pathlib import Path
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Set up HuggingFace cache
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = str(project_root / '.cache' / 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

from config.models import get_available_models, list_all_models
from models.local_models import LocalModelHandler
from models.api_models import APIModelHandler
from agents.research_agent import EnhancedResearchAgent

# Initialize FastAPI app
app = FastAPI(
    title="AI Research Agent API",
    description="Professional AI-powered research report generation with multiple model support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class QueryRefinementRequest(BaseModel):
    query: str
    scope: Optional[str] = "comprehensive"
    region: Optional[str] = "global"
    timeframe: Optional[str] = "current"
    audience: Optional[str] = "general"
    depth: Optional[str] = "detailed"
    focus_areas: Optional[str] = ""

class ResearchRequest(BaseModel):
    query: str
    provider: str
    model_key: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.2
    output_filename: Optional[str] = None
    refinements: Optional[Dict[str, Any]] = None

class ModelTestRequest(BaseModel):
    provider: str
    model_key: str

class ResearchProgress(BaseModel):
    session_id: str
    stage: str
    progress: int
    message: str
    elapsed_time: float
    estimated_remaining: Optional[float] = None

class ResearchResult(BaseModel):
    success: bool
    report_path: Optional[str] = None
    categories: List[str] = []
    report_structure: List[str] = []
    timing: Dict[str, float] = {}
    error: Optional[str] = None
    content_length: int = 0
    pdf_created: bool = False

# Global state management
active_sessions: Dict[str, Dict] = {}
connection_manager = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_progress(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except:
                # Connection might be closed
                self.disconnect(session_id)

manager = ConnectionManager()

def generate_output_filename(query: str) -> str:
    """Generate descriptive filename from query"""
    clean_query = re.sub(r'[^\w\s-]', '', query.lower())
    clean_query = re.sub(r'[-\s]+', '_', clean_query)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"reports/{clean_query}_{timestamp}.pdf"

def detect_query_type(query: str) -> Dict[str, Any]:
    """Detect query type and suggest refinements"""
    query_lower = query.lower()
    suggestions = {
        'type': 'general',
        'suggestions': []
    }
    
    if any(word in query_lower for word in ['salary', 'pay', 'wage', 'compensation']):
        suggestions['type'] = 'salary'
        suggestions['suggestions'] = [
            {'field': 'region', 'label': 'Geographic Focus', 'placeholder': 'Portugal, Europe, Global'},
            {'field': 'timeframe', 'label': 'Time Focus', 'placeholder': '2024, latest, trends'},
            {'field': 'experience', 'label': 'Experience Level', 'placeholder': 'junior, senior, all levels'}
        ]
    elif any(word in query_lower for word in ['vs', 'versus', 'compare', 'comparison']):
        suggestions['type'] = 'comparison'
        suggestions['suggestions'] = [
            {'field': 'criteria', 'label': 'Comparison Criteria', 'placeholder': 'features, pricing, performance'}
        ]
    elif any(word in query_lower for word in ['how to', 'guide', 'tutorial']):
        suggestions['type'] = 'tutorial'
        suggestions['suggestions'] = [
            {'field': 'level', 'label': 'Skill Level', 'placeholder': 'beginner, intermediate, advanced'}
        ]
    
    return suggestions

class ProgressTracker:
    """Progress tracker with WebSocket support"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
    
    async def update(self, message: str, percentage: int):
        """Send progress update via WebSocket"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Estimate remaining time
        remaining = None
        if percentage > 0:
            total_estimated = elapsed * 100 / percentage
            remaining = max(0, total_estimated - elapsed)
        
        progress_data = {
            'type': 'progress',
            'session_id': self.session_id,
            'stage': message,
            'progress': percentage,
            'elapsed_time': elapsed,
            'estimated_remaining': remaining
        }
        
        await manager.send_progress(self.session_id, progress_data)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Research Agent API",
        "version": "2.0.0",
        "endpoints": {
            "models": "/api/models",
            "research": "/api/research",
            "websocket": "/ws/{session_id}",
            "docs": "/docs"
        }
    }

@app.get("/api/models")
async def get_models():
    """Get all available models with their configurations"""
    try:
        models = get_available_models()
        
        # Add API key status for each provider
        for provider in models:
            if provider != "local":
                api_key_var = f"{provider.upper()}_API_KEY"
                has_key = bool(os.getenv(api_key_var))
                
                for model_key in models[provider]:
                    models[provider][model_key]["api_key_available"] = has_key
        
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/test")
async def test_model_connection(request: ModelTestRequest):
    """Test connection to a specific model"""
    try:
        provider = request.provider
        model_key = request.model_key
        
        if provider == "local":
            # For local models, just check if they can be initialized
            models = get_available_models()
            if model_key not in models["local"]:
                return {"success": False, "error": "Model not found"}
            
            return {"success": True, "message": "Local model available"}
        
        # Test API model
        api_key_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_var)
        
        if not api_key:
            return {"success": False, "error": f"API key {api_key_var} not found"}
        
        models = get_available_models()
        if provider not in models or model_key not in models[provider]:
            return {"success": False, "error": "Model not found"}
        
        model_name = models[provider][model_key]["name"]
        handler = APIModelHandler(provider, model_name, api_key)
        result = handler.test_connection()
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/query/analyze")
async def analyze_query(request: QueryRefinementRequest):
    """Analyze query and provide refinement suggestions"""
    try:
        suggestions = detect_query_type(request.query)
        
        return {
            "success": True,
            "query": request.query,
            "type": suggestions["type"],
            "suggestions": suggestions["suggestions"],
            "recommended_depth": "detailed",
            "estimated_time": "2-5 minutes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/research/start")
async def start_research(request: ResearchRequest):
    """Start a research session"""
    try:
        session_id = str(uuid.uuid4())
        
        # Store session data
        active_sessions[session_id] = {
            "request": request.dict(),
            "start_time": time.time(),
            "status": "initialized"
        }
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Research session created. Connect to WebSocket for progress updates."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket, session_id)
    
    try:
        if session_id not in active_sessions:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid session ID"
            })
            return
        
        # Get session data
        session_data = active_sessions[session_id]
        request_data = ResearchRequest(**session_data["request"])
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(session_id)
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "message": "Starting research process...",
            "session_id": session_id
        })
        
        # Execute research
        result = await execute_research(request_data, progress_tracker, session_id)
        
        # Send final result
        await websocket.send_json({
            "type": "complete",
            "result": result,
            "session_id": session_id
        })
        
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "disconnected"
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "session_id": session_id
        })
    finally:
        manager.disconnect(session_id)

async def execute_research(request: ResearchRequest, progress_tracker: ProgressTracker, session_id: str) -> Dict:
    """Execute the research process with progress tracking"""
    try:
        # Update session status
        active_sessions[session_id]["status"] = "running"
        
        await progress_tracker.update("Initializing model handler...", 10)
        
        # Create model handler
        if request.provider == "local":
            model_handler = LocalModelHandler(
                model_key=request.model_key,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                verbose=False
            )
        else:
            api_key_var = f"{request.provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_var)
            
            if not api_key:
                raise ValueError(f"API key {api_key_var} not found")
            
            model_handler = APIModelHandler(
                provider=request.provider,
                model_name=request.model_key,
                api_key=api_key,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                verbose=False
            )
        
        await progress_tracker.update("Setting up research agent...", 20)
        
        # Create research agent
        agent = EnhancedResearchAgent(model_handler)
        
        # Set up async progress callback
        async def progress_callback(message: str, percentage: int):
            await progress_tracker.update(message, percentage)
        
        agent.set_progress_callback(progress_callback)
        
        # Generate output filename
        output_filename = request.output_filename or generate_output_filename(request.query)
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        await progress_tracker.update("Starting research process...", 30)
        
        # Execute research (this will need to be made async in your research agent)
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            agent.conduct_research, 
            request.query, 
            str(output_path)
        )
        
        await progress_tracker.update("Research completed!", 100)
        
        # Update session
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["results"] = results
        
        return {
            "success": True,
            "report_path": str(output_path),
            "categories": results.get("categories", []),
            "report_structure": results.get("report_structure", []),
            "timing": results.get("timing", {}),
            "content_length": len(results.get("report_content", "")),
            "pdf_created": results.get("pdf_created", False)
        }
        
    except Exception as e:
        active_sessions[session_id]["status"] = "error"
        active_sessions[session_id]["error"] = str(e)
        
        await progress_tracker.update(f"Error: {str(e)}", 0)
        
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Cleanup
        if 'model_handler' in locals():
            model_handler.cleanup()

@app.get("/api/reports/{filename}")
async def download_report(filename: str):
    """Download a generated report"""
    try:
        file_path = Path("reports") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/pdf'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the status of a research session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "start_time": session["start_time"],
        "elapsed_time": time.time() - session["start_time"],
        "results": session.get("results"),
        "error": session.get("error")
    }

@app.delete("/api/sessions/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a research session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    manager.disconnect(session_id)
    
    return {"success": True, "message": "Session cleaned up"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "websocket_connections": len(manager.active_connections)
    }

# Serve static files (for potential frontend)
if Path("frontend/build").exists():
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")
    
    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Serve React frontend"""
        frontend_path = Path("frontend/build")
        file_path = frontend_path / path
        
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        else:
            # Return index.html for SPA routing
            return FileResponse(frontend_path / "index.html")

def create_app():
    """Factory function to create the app"""
    return app

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Research Agent FastAPI Server")
    print("=" * 50)
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    print("💻 Health Check: http://localhost:8000/api/health")
    print("🌐 WebSocket: ws://localhost:8000/ws/{session_id}")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)],
        log_level="info"
    )