"""Model configuration and definitions"""

MODEL_CONFIGS = {
    # Local Models
    "local": {
        "phi-2": {
            "name": "microsoft/phi-2",
            "description": "Phi-2 (Fast, lightweight, local)",
            "estimated_time": "15-30s",
            "memory_usage": "Low",
            "quality": "Good"
        },
        "zephyr": {
            "name": "HuggingFaceH4/zephyr-7b-beta", 
            "description": "Zephyr-7B (High quality, slower)",
            "estimated_time": "60-120s",
            "memory_usage": "High",
            "quality": "Excellent"
        },
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.1",
            "description": "Mistral-7B (Balanced performance)",
            "estimated_time": "45-90s", 
            "memory_usage": "Medium",
            "quality": "Very Good"
        }
    },
    
    # Enhanced Groq Models Configuration
    "groq": {
        "llama-3.1-70b-versatile": {
            "name": "llama-3.1-70b-versatile",
            "description": "Llama 3.1 70B Versatile - Most capable model for complex research and analysis",
            "estimated_time": "3-8s",
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Premium",
            "max_tokens": 8000,
            "recommended_temperature": 0.1,
            "best_for": ["research", "analysis", "detailed_reports", "technical_content"],
            "quality_tier": "premium"
        },
        "llama-3.1-8b-instant": {
            "name": "llama-3.1-8b-instant",
            "description": "Llama 3.1 8B Instant - Fast and efficient for quick research tasks",
            "estimated_time": "2-5s",
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Excellent",
            "max_tokens": 8000,
            "recommended_temperature": 0.2,
            "best_for": ["quick_summaries", "factual_queries", "structured_output"],
            "quality_tier": "standard"
        },
        "llama-3.2-90b-text-preview": {
            "name": "llama-3.2-90b-text-preview",
            "description": "Llama 3.2 90B Text Preview - Latest model with enhanced reasoning capabilities",
            "estimated_time": "4-10s",
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Premium",
            "max_tokens": 8000,
            "recommended_temperature": 0.1,
            "best_for": ["advanced_research", "complex_analysis", "academic_reports"],
            "quality_tier": "premium"
        },
        "mixtral-8x7b-32768": {
            "name": "mixtral-8x7b-32768",
            "description": "Mixtral 8x7B - Excellent for multilingual research and code analysis",
            "estimated_time": "3-8s", 
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Premium",
            "max_tokens": 32768,
            "recommended_temperature": 0.2,
            "best_for": ["code_analysis", "multilingual_content", "long_documents"],
            "quality_tier": "premium"
        },
        "gemma2-9b-it": {
            "name": "gemma2-9b-it",
            "description": "Gemma 2 9B IT - Specialized for technical and IT-related research",
            "estimated_time": "3-6s",
            "cost": "Free tier: 14,400 requests/day", 
            "quality": "Very Good",
            "max_tokens": 8192,
            "recommended_temperature": 0.15,
            "best_for": ["technical_research", "software_development", "IT_analysis"],
            "quality_tier": "standard"
        },
        "llama-3.2-3b-preview": {
            "name": "llama-3.2-3b-preview",
            "description": "Llama 3.2 3B Preview - Ultra-fast for basic research tasks",
            "estimated_time": "1-3s",
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Good",
            "max_tokens": 8192,
            "recommended_temperature": 0.3,
            "best_for": ["quick_facts", "basic_summaries", "simple_queries"],
            "quality_tier": "basic"
        },
        # Legacy models for backward compatibility
        "llama-3.1-8b": {
            "name": "llama-3.1-8b-instant",
            "description": "Llama 3.1 8B (Very fast API) - Legacy alias",
            "estimated_time": "2-5s",
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Excellent"
        },
        "mixtral-8x7b": {
            "name": "mixtral-8x7b-32768",
            "description": "Mixtral 8x7B (High quality) - Legacy alias",
            "estimated_time": "3-8s", 
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Excellent"
        },
        "gemma-7b": {
            "name": "gemma-7b-it",
            "description": "Gemma 7B (Google model) - Legacy alias",
            "estimated_time": "3-6s",
            "cost": "Free tier: 14,400 requests/day", 
            "quality": "Very Good"
        }
    },
    
    "together": {
        "llama-3.1-8b": {
            "name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "description": "Llama 3.1 8B Turbo",
            "estimated_time": "5-10s",
            "cost": "Free: $25 credits monthly",
            "quality": "Excellent"
        },
        "qwen-2.5-7b": {
            "name": "Qwen/Qwen2.5-7B-Instruct-Turbo", 
            "description": "Qwen 2.5 7B (Fast, multilingual)",
            "estimated_time": "4-8s",
            "cost": "Free: $25 credits monthly",
            "quality": "Very Good"
        }
    },
    
    "huggingface": {
        "zephyr-7b": {
            "name": "HuggingFaceH4/zephyr-7b-beta",
            "description": "Zephyr 7B (HF Inference API)",
            "estimated_time": "10-20s",
            "cost": "Free tier: Rate limited",
            "quality": "Excellent"
        },
        "mistral-7b": {
            "name": "mistralai/Mistral-7B-Instruct-v0.1", 
            "description": "Mistral 7B (HF Inference API)",
            "estimated_time": "8-15s",
            "cost": "Free tier: Rate limited", 
            "quality": "Very Good"
        }
    },
    
    "openrouter": {
        "llama-3.1-8b": {
            "name": "meta-llama/llama-3.1-8b-instruct:free",
            "description": "Llama 3.1 8B (Free tier)",
            "estimated_time": "5-15s",
            "cost": "Free tier: Limited requests",
            "quality": "Excellent"  
        },
        "mistral-7b": {
            "name": "mistralai/mistral-7b-instruct:free",
            "description": "Mistral 7B (Free tier)", 
            "estimated_time": "5-15s",
            "cost": "Free tier: Limited requests",
            "quality": "Very Good"
        }
    },
    
    "cohere": {
        "command": {
            "name": "command",
            "description": "Cohere Command (Good for research)",
            "estimated_time": "3-8s",
            "cost": "Free tier: 1000 requests/month",
            "quality": "Very Good"
        },
        "command-light": {
            "name": "command-light", 
            "description": "Cohere Command Light (Faster)",
            "estimated_time": "2-5s",
            "cost": "Free tier: 1000 requests/month",
            "quality": "Good"
        }
    }
}

# Model selection recommendations based on research type
RESEARCH_TYPE_RECOMMENDATIONS = {
    "academic": ["llama-3.2-90b-text-preview", "llama-3.1-70b-versatile"],
    "technical": ["mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.1-70b-versatile"],
    "business": ["llama-3.1-70b-versatile", "llama-3.2-90b-text-preview"],
    "quick_facts": ["llama-3.1-8b-instant", "llama-3.2-3b-preview"],
    "comprehensive": ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    "multilingual": ["mixtral-8x7b-32768", "llama-3.1-70b-versatile"]
}

def get_available_models():
    """Get all available models organized by provider"""
    return MODEL_CONFIGS

def get_model_info(provider, model_key):
    """Get specific model information"""
    return MODEL_CONFIGS.get(provider, {}).get(model_key, None)

def list_all_models():
    """List all models with their details"""
    all_models = []
    for provider, models in MODEL_CONFIGS.items():
        for model_key, model_info in models.items():
            all_models.append({
                'provider': provider,
                'key': model_key,
                'full_name': f"{provider}/{model_key}",
                **model_info
            })
    return all_models

def get_recommended_groq_model(query: str, research_type: str = None) -> str:
    """
    Recommend the best Groq model based on query complexity and type
    """
    query_lower = query.lower()
    
    # Detect research type from query if not provided
    if not research_type:
        if any(word in query_lower for word in ['code', 'programming', 'software', 'api', 'technical']):
            research_type = "technical"
        elif any(word in query_lower for word in ['academic', 'research', 'study', 'analysis', 'thesis']):
            research_type = "academic" 
        elif any(word in query_lower for word in ['business', 'market', 'strategy', 'company']):
            research_type = "business"
        elif len(query.split()) < 10:  # Short queries
            research_type = "quick_facts"
        else:
            research_type = "comprehensive"
    
    # Get recommendations for the research type
    recommended_models = RESEARCH_TYPE_RECOMMENDATIONS.get(research_type, ["llama-3.1-70b-versatile"])
    
    # Return the first available model
    return recommended_models[0]

def get_enhanced_groq_config(model_key: str) -> dict:
    """
    Get optimized configuration for specific Groq models
    """
    model_config = MODEL_CONFIGS.get("groq", {}).get(model_key, {})
    
    return {
        "max_tokens": model_config.get("max_tokens", 8000),
        "temperature": model_config.get("recommended_temperature", 0.2),
        "top_p": 0.9,
        "frequency_penalty": 0.1,  # Reduce repetition
        "presence_penalty": 0.1,   # Encourage diverse topics
        "model_name": model_key,
        "quality_tier": model_config.get("quality_tier", "standard")
    }

def create_enhanced_research_prompt(query: str, model_tier: str = "premium") -> str:
    """
    Create research prompts optimized for different model tiers
    """
    if model_tier == "premium":
        return f"""
        You are an expert research analyst. Conduct a comprehensive analysis of: "{query}"
        
        Please provide:
        1. Executive Summary (2-3 paragraphs)
        2. Detailed Analysis with multiple perspectives
        3. Key findings with supporting evidence
        4. Implications and recommendations
        5. Future outlook and trends
        6. Relevant statistics and data points
        7. Expert opinions and citations
        
        Use a professional, analytical tone with clear structure and logical flow.
        Ensure accuracy and provide actionable insights.
        """
    else:
        return f"""
        Research and analyze: "{query}"
        
        Please provide:
        1. Summary of key points
        2. Main findings
        3. Important statistics
        4. Conclusions
        
        Keep the response structured and informative.
        """

def get_groq_models_by_tier(tier: str = None) -> dict:
    """
    Get Groq models filtered by quality tier
    """
    groq_models = MODEL_CONFIGS.get("groq", {})
    
    if not tier:
        return groq_models
    
    filtered_models = {}
    for model_key, model_info in groq_models.items():
        if model_info.get("quality_tier") == tier:
            filtered_models[model_key] = model_info
    
    return filtered_models

def get_best_groq_model_for_task(task_type: str) -> str:
    """
    Get the best Groq model for a specific task type
    """
    task_mappings = {
        "research": "llama-3.1-70b-versatile",
        "analysis": "llama-3.2-90b-text-preview", 
        "code": "mixtral-8x7b-32768",
        "technical": "gemma2-9b-it",
        "quick": "llama-3.1-8b-instant",
        "basic": "llama-3.2-3b-preview"
    }
    
    return task_mappings.get(task_type.lower(), "llama-3.1-70b-versatile")