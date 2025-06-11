"""Configuration settings for the PDF Research Agent"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = Path(os.getenv('HF_HOME', r'D:\.cache\huggingface'))

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
MODEL_SETTINGS = {
    "local": {
        "default_device": "auto",  # auto, cpu, cuda
        "torch_dtype": "float32",  # float16, float32, bfloat16
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "max_new_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
    },
    "api": {
        "timeout": 30,  # seconds
        "max_retries": 3,
        "retry_delay": 1,  # seconds
        "max_tokens": 500,
        "temperature": 0.2,
    }
}

# Search settings
SEARCH_SETTINGS = {
    "max_results": 12,
    "timeout": 30,
    "min_content_length": 50,
    "relevant_keywords": [
        "python", "developer", "salary", "portugal", "lisbon", "porto",
        "programming", "software", "engineer", "wage", "compensation"
    ],
    "relevance_threshold": 2,
    "max_sources": 8,
}

# Report generation settings
REPORT_SETTINGS = {
    "sections": {
        "required": ["Key Statistics", "Trends", "Data Sources"],
        "optional": ["Methodology", "Recommendations", "Limitations"]
    },
    "formatting": {
        "max_salary_figures": 6,
        "max_sources_displayed": 5,
        "max_line_length": 80,
    },
    "validation": {
        "min_report_length": 200,
        "require_salary_data": False,
        "require_sources": True,
    }
}

# PDF settings
PDF_SETTINGS = {
    "page_size": "letter",  # letter, a4
    "margins": {
        "top": 72,
        "bottom": 72,
        "left": 72,
        "right": 72
    },
    "fonts": {
        "title": "Helvetica-Bold",
        "heading": "Helvetica-Bold", 
        "body": "Helvetica",
        "italic": "Helvetica-Oblique"
    },
    "styles": {
        "title_size": 18,
        "heading_size": 14,
        "body_size": 10,
        "line_spacing": 1.2
    }
}

# API Keys and endpoints
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "cohere": os.getenv("COHERE_API_KEY"),
}

API_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "huggingface": "https://api-inference.huggingface.co/models/",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "cohere": "https://api.cohere.ai/v1/generate",
}

# Performance settings
PERFORMANCE_SETTINGS = {
    "memory": {
        "max_ram_usage_gb": 16,
        "max_vram_usage_gb": 8,
        "gc_threshold": 0.8,  # Run garbage collection at 80% memory usage
    },
    "processing": {
        "max_concurrent_requests": 3,
        "request_timeout": 60,
        "batch_size": 1,
    },
    "caching": {
        "enable_model_cache": True,
        "cache_timeout": 3600,  # seconds
        "max_cache_size_gb": 5,
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "research_agent.log",
    "max_size_mb": 10,
    "backup_count": 5,
    "console_output": True,
}

# Default prompts
PROMPT_TEMPLATES = {
    "research_report": """Based on the research data below, create a professional salary report.

RESEARCH DATA:
{search_results}

TOPIC: {topic}

Create a report with these sections:

## Key Statistics
[List 4-5 specific salary figures and percentages from the data]

## Trends  
[Write 2-3 sentences about salary patterns and market trends]

## Data Sources
[List the main websites and sources]

Focus on actual numbers and be specific. Start your response with "## Key Statistics":""",

    "fallback_report": """Create a professional salary report for: {topic}

Use the following data points:
{data_points}

Structure the report with:
1. Key Statistics (salary ranges and figures)
2. Market Trends (analysis of patterns)
3. Data Sources (list of references)

Be specific and professional.""",

    "api_system": """You are a professional research analyst. Create detailed, accurate salary reports based on provided data. Focus on specific figures, market trends, and credible sources. Always structure your response with clear sections: Key Statistics, Trends, and Data Sources."""
}

# File extensions and formats
SUPPORTED_FORMATS = {
    "output": [".pdf", ".txt", ".md"],
    "input": [".txt", ".md", ".json"],
    "cache": [".pkl", ".json"]
}

# User interface settings
UI_SETTINGS = {
    "display": {
        "show_progress": True,
        "show_performance": True,
        "show_debug": False,
        "color_output": True,
    },
    "interaction": {
        "confirm_overwrite": True,
        "auto_open_pdf": False,
        "save_intermediate": False,
    },
    "model_selection": {
        "show_estimates": True,
        "show_requirements": True,
        "default_choice": "phi-2",
    }
}

# Error handling
ERROR_HANDLING = {
    "max_retries": 3,
    "retry_delay": 2,
    "fallback_enabled": True,
    "graceful_degradation": True,
    "save_error_reports": True,
}

# Validation rules
VALIDATION_RULES = {
    "query": {
        "min_length": 3,
        "max_length": 200,
        "required_keywords": [],
        "blocked_keywords": ["illegal", "hack", "exploit"]
    },
    "search_results": {
        "min_length": 100,
        "max_length": 50000,
        "required_elements": ["url", "title"],
    },
    "report": {
        "min_sections": 2,
        "required_sections": ["Key Statistics"],
        "min_length": 200,
        "max_length": 10000,
    }
}

# Default configurations for different use cases
USE_CASE_CONFIGS = {
    "quick": {
        "model": "phi-2",
        "search_results": 6,
        "max_tokens": 200,
        "sections": ["Key Statistics", "Data Sources"]
    },
    "standard": {
        "model": "phi-2",
        "search_results": 12,
        "max_tokens": 300,
        "sections": ["Key Statistics", "Trends", "Data Sources"]
    },
    "comprehensive": {
        "model": "zephyr",
        "search_results": 20,
        "max_tokens": 500,
        "sections": ["Key Statistics", "Trends", "Data Sources", "Methodology", "Recommendations"]
    }
}

# Environment-specific settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    LOGGING_CONFIG["level"] = "WARNING"
    UI_SETTINGS["display"]["show_debug"] = False
    ERROR_HANDLING["save_error_reports"] = False
elif ENVIRONMENT == "development":
    LOGGING_CONFIG["level"] = "DEBUG"
    UI_SETTINGS["display"]["show_debug"] = True
    ERROR_HANDLING["save_error_reports"] = True

# Version and metadata
VERSION = "1.0.0"
APP_NAME = "PDF Research Agent"
DESCRIPTION = "AI-powered research agent with PDF report generation"
AUTHOR = "AI Research Team"

# Export commonly used settings
__all__ = [
    "MODEL_SETTINGS",
    "SEARCH_SETTINGS", 
    "REPORT_SETTINGS",
    "PDF_SETTINGS",
    "API_KEYS",
    "API_ENDPOINTS",
    "PERFORMANCE_SETTINGS",
    "LOGGING_CONFIG",
    "PROMPT_TEMPLATES",
    "UI_SETTINGS",
    "ERROR_HANDLING",
    "VALIDATION_RULES",
    "USE_CASE_CONFIGS",
    "BASE_DIR",
    "REPORTS_DIR",
    "LOGS_DIR",
    "CACHE_DIR",
    "VERSION",
    "APP_NAME"
]

def get_setting(key_path: str, default: Any = None) -> Any:
    """Get a setting value using dot notation (e.g., 'model.local.temperature')"""
    keys = key_path.split('.')
    current = globals()
    
    try:
        for key in keys:
            if isinstance(current, dict):
                current = current[key]
            else:
                current = getattr(current, key)
        return current
    except (KeyError, AttributeError):
        return default

def update_setting(key_path: str, value: Any) -> bool:
    """Update a setting value using dot notation"""
    keys = key_path.split('.')
    current = globals()
    
    try:
        for key in keys[:-1]:
            if isinstance(current, dict):
                current = current[key]
            else:
                current = getattr(current, key)
        
        if isinstance(current, dict):
            current[keys[-1]] = value
        else:
            setattr(current, keys[-1], value)
        return True
    except (KeyError, AttributeError):
        return False

def validate_api_keys() -> Dict[str, bool]:
    """Validate which API keys are available"""
    return {
        provider: bool(key) 
        for provider, key in API_KEYS.items()
    }

def get_active_config(use_case: str = "standard") -> Dict[str, Any]:
    """Get configuration for a specific use case"""
    base_config = USE_CASE_CONFIGS.get(use_case, USE_CASE_CONFIGS["standard"])
    
    return {
        **base_config,
        "model_settings": MODEL_SETTINGS,
        "search_settings": SEARCH_SETTINGS,
        "report_settings": REPORT_SETTINGS,
        "pdf_settings": PDF_SETTINGS,
    }