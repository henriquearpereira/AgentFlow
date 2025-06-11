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
    
    # API Models
    "groq": {
        "llama-3.1-8b": {
            "name": "llama-3.1-8b-instant",
            "description": "Llama 3.1 8B (Very fast API)",
            "estimated_time": "2-5s",
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Excellent"
        },
        "mixtral-8x7b": {
            "name": "mixtral-8x7b-32768",
            "description": "Mixtral 8x7B (High quality)",
            "estimated_time": "3-8s", 
            "cost": "Free tier: 14,400 requests/day",
            "quality": "Excellent"
        },
        "gemma-7b": {
            "name": "gemma-7b-it",
            "description": "Gemma 7B (Google model)",
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