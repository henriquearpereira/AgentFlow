# 🧠 AI Research Agent

A comprehensive AI-powered research automation platform that generates professional research reports with real-time WebSocket updates and multi-model support.

## 🌟 Features

- **Multi-Model Support**: OpenAI GPT, Anthropic Claude, and Local Models (Mistral, Llama)
- **Real-Time Updates**: WebSocket integration for live progress tracking
- **Professional Reports**: Automated PDF generation with structured content
- **Query Intelligence**: Smart query analysis and refinement suggestions
- **RESTful API**: Complete FastAPI backend with interactive documentation
- **Model Testing**: Built-in connection testing for all supported models
- **Session Management**: Robust session handling with cleanup capabilities

## 🏗️ Project Structure

```
AGENTFLOW/
├── frontend/                 # React frontend (optional)
│   ├── node_modules/
│   ├── public/
│   └── src/
├── agents/                   # AI agent implementations
│   ├── research_agent.py
│   └── pdf_research_agent.py
├── config/                   # Configuration files
│   ├── models.py            # Model configurations
│   └── settings.py          # Application settings
├── models/                   # Model handlers
│   ├── api_models.py        # API-based models (OpenAI, Anthropic)
│   └── local_models.py      # Local model implementations
├── utils/                    # Utility functions
│   ├── helpers.py
│   └── search.py
├── reports/                  # Generated PDF reports
├── main.py                   # FastAPI application entry point
├── requirements.txt          # Python dependencies
└── .env                     # Environment variables
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for frontend)
- API Keys for external models (OpenAI, Anthropic)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ai-research-agent
```

2. **Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables:**
Create a `.env` file in the root directory:
```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Cache
HF_HOME=./.cache/huggingface
```

4. **Start the server:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Alternative startup methods:

```bash
# Using uvicorn directly
uvicorn main:app --reload --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📚 API Documentation

### 🔗 Quick Links

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health
- **Available Reports**: http://localhost:8000/api/reports

### 🎯 Core Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-18T12:34:56.789Z",
  "active_sessions": 3,
  "websocket_connections": 2
}
```

#### List Available Models
```http
GET /api/models
```

**Response:**
```json
{
  "success": true,
  "models": {
    "openai": {
      "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "max_tokens": 4096,
        "api_key_available": true
      }
    },
    "anthropic": {
      "claude-3-sonnet": {
        "name": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "api_key_available": true
      }
    },
    "local": {
      "mistral-7b": {
        "path": "models/mistral-7b",
        "max_tokens": 2048
      }
    }
  }
}
```

#### Test Model Connection
```http
POST /api/models/test
```

**Request:**
```json
{
  "provider": "openai",
  "model_key": "gpt-4-turbo"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Connection successful"
}
```

### 🔍 Research Operations

#### Analyze Query
```http
POST /api/query/analyze
```

**Request:**
```json
{
  "query": "Software engineer salaries in Europe",
  "scope": "comprehensive",
  "region": "Europe",
  "timeframe": "2024",
  "audience": "professionals",
  "depth": "detailed",
  "focus_areas": "tech industry"
}
```

**Response:**
```json
{
  "success": true,
  "query": "Software engineer salaries in Europe",
  "type": "salary",
  "suggestions": [
    {
      "field": "region",
      "label": "Geographic Focus",
      "placeholder": "Portugal, Europe, Global"
    },
    {
      "field": "timeframe",
      "label": "Time Focus",
      "placeholder": "2024, latest, trends"
    }
  ],
  "recommended_depth": "detailed",
  "estimated_time": "2-5 minutes"
}
```

#### Start Research Session
```http
POST /api/research/start
```

**Request:**
```json
{
  "query": "Renewable energy trends in Portugal",
  "provider": "openai",
  "model_key": "gpt-4-turbo",
  "max_tokens": 1000,
  "temperature": 0.3,
  "output_filename": "renewable_energy_report.pdf"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "message": "Research session created. Connect to WebSocket for progress updates."
}
```

### 📊 Session Management

#### Get Session Status
```http
GET /api/sessions/{session_id}/status
```

**Response:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "status": "running",
  "start_time": 1687091200.123456,
  "elapsed_time": 42.7,
  "results": null,
  "error": null
}
```

#### Clean Up Session
```http
DELETE /api/sessions/{session_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Session cleaned up"
}
```

### 📄 Report Management

#### List Reports
```http
GET /api/reports
```

**Response:**
```json
{
  "reports": [
    {
      "filename": "renewable_energy_20240618_1234.pdf",
      "size": 24576,
      "created": "2025-06-18T12:34:56.789Z"
    }
  ]
}
```

#### Download Report
```http
GET /api/reports/{filename}
```

Returns the PDF file for download.

## 🔌 WebSocket Integration

### Connection
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${session_id}`);
```

### Message Types

#### Progress Update
```json
{
  "type": "progress",
  "session_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "stage": "Generating report content",
  "progress": 65,
  "elapsed_time": 42.7,
  "estimated_remaining": 22.3
}
```

#### Completion Message
```json
{
  "type": "complete",
  "result": {
    "success": true,
    "report_path": "renewable_energy_20240618_1234.pdf",
    "categories": ["Energy", "Technology", "Environment"],
    "report_structure": ["Introduction", "Current Trends", "Future Outlook"],
    "timing": {
      "research": 25.4,
      "analysis": 12.7,
      "writing": 30.2
    },
    "content_length": 12450,
    "pdf_created": true
  },
  "session_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8"
}
```

#### Error Message
```json
{
  "type": "error",
  "message": "API connection failed",
  "error_type": "ConnectionError",
  "session_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8"
}
```

## 🛠️ Frontend Integration

### React WebSocket Example
```javascript
const connectWebSocket = (sessionId) => {
  const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch (data.type) {
      case 'progress':
        updateProgress(data);
        break;
      case 'complete':
        handleCompletion(data.result);
        break;
      case 'error':
        handleError(data);
        break;
    }
  };
  
  return ws;
};
```

### Complete Research Flow
```javascript
// 1. Start research session
const startResearch = async (query, provider, modelKey) => {
  const response = await fetch('/api/research/start', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      provider,
      model_key: modelKey,
      max_tokens: 1000,
      temperature: 0.3
    })
  });
  
  const data = await response.json();
  return data.session_id;
};

// 2. Connect to WebSocket for progress
const sessionId = await startResearch(
  "AI trends in healthcare",
  "openai", 
  "gpt-4-turbo"
);

const ws = connectWebSocket(sessionId);
```

## 🧠 Supported Models

### OpenAI Models
- `gpt-4-turbo`: Latest GPT-4 Turbo model
- `gpt-4`: Standard GPT-4 model
- `gpt-3.5-turbo`: Fast and efficient model

### Anthropic Models
- `claude-3-sonnet`: Balanced performance model
- `claude-3-haiku`: Fast and efficient model
- `claude-3-opus`: Most capable model

### Local Models
- `mistral-7b`: Mistral 7B model
- `llama-2-7b`: Llama 2 7B model
- `codellama-7b`: Code-specialized Llama model

## 🐳 Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t ai-research-agent .
docker run -p 8000:8000 --env-file .env ai-research-agent
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: research-agent
  template:
    metadata:
      labels:
        app: research-agent
    spec:
      containers:
      - name: research-agent
        image: ai-research-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: anthropic-key
```

## 🔧 Configuration

### Model Configuration
Edit `config/models.py` to add or modify supported models:

```python
MODEL_CONFIG = {
    "openai": {
        "gpt-4-turbo": {
            "name": "gpt-4-turbo",
            "max_tokens": 4096,
            "requires_key": True
        }
    },
    "anthropic": {
        "claude-3-sonnet": {
            "name": "claude-3-sonnet-20240229",
            "max_tokens": 4096,
            "requires_key": True
        }
    },
    "local": {
        "mistral-7b": {
            "path": "models/mistral-7b",
            "max_tokens": 2048
        }
    }
}
```

### CORS Configuration
The API includes CORS middleware for frontend integration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📈 Performance & Monitoring

### Health Monitoring
The API includes comprehensive health checks:
- Active session tracking
- WebSocket connection monitoring
- Model availability status
- Report generation statistics

### Logging
Enable detailed logging for debugging:
```python
# Set environment variable
LOG_LEVEL=debug

# Or modify uvicorn startup
uvicorn main:app --log-level debug
```

## 🛡️ Error Handling

The API includes comprehensive error handling:

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (session, report, model)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "detail": "Model not found for provider: openai",
  "error_type": "ValueError"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Hugging Face for local model support
- FastAPI for the excellent web framework
- The open-source community for inspiration and tools

---

**Built with ❤️ by Henrique Pereira**