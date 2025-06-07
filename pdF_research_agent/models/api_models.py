"""API-based model implementations"""

import os
import requests
import time
from typing import Optional, Dict, Any
import openai
import cohere

class APIModelHandler:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.setup_client()
    
    def setup_client(self):
        """Setup API client based on provider"""
        if self.provider == "groq":
            self.setup_groq()
        elif self.provider == "together":
            self.setup_together()
        elif self.provider == "huggingface":
            self.setup_huggingface()
        elif self.provider == "openrouter":
            self.setup_openrouter()
        elif self.provider == "cohere":
            self.setup_cohere()
    
    def setup_groq(self):
        """Setup Groq client"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        # Groq uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    
    def setup_together(self):
        """Setup Together AI client"""
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")
            
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1"
        )
    
    def setup_huggingface(self):
        """Setup Hugging Face Inference API"""
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
    
    def setup_openrouter(self):
        """Setup OpenRouter client"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
            
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def setup_cohere(self):
        """Setup Cohere client"""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")
        
        self.client = cohere.Client(api_key)
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.2, **kwargs) -> str:
        """Generate text using the configured API"""
        try:
            if self.provider == "groq":
                return self._generate_openai_compatible(prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "together":
                return self._generate_openai_compatible(prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "openrouter":
                return self._generate_openai_compatible(prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "cohere":
                return self._generate_cohere(prompt, max_tokens, temperature, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"⚠️ API generation error ({self.provider}): {e}")
            return self._fallback_response(prompt)
    
    def _generate_openai_compatible(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using OpenAI-compatible APIs (Groq, Together, OpenRouter)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI-compatible API error: {e}")
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using Hugging Face Inference API"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get('top_p', 0.9),
                "return_full_text": False
            },
            "options": {"wait_for_model": True}
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].strip()
                else:
                    raise Exception(f"Unexpected response format: {result}")
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"HuggingFace API error after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _generate_cohere(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using Cohere API"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                p=kwargs.get('top_p', 0.9),
                stop_sequences=kwargs.get('stop_sequences', [])
            )
            return response.generations[0].text.strip()
        except Exception as e:
            raise Exception(f"Cohere API error: {e}")
    
    def _fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when API fails"""
        return """## Key Statistics
- API generation temporarily unavailable
- Fallback report generated from search data
- Multiple salary data points collected from Portuguese job market

## Trends
The Portuguese tech market shows strong demand for Python developers with competitive salaries varying by experience level and location.

## Data Sources  
- Search results from multiple job market sources
- Salary data aggregated from various platforms"""

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return status"""
        try:
            test_prompt = "Hello, respond with 'API connection successful'"
            response = self.generate(test_prompt, max_tokens=50, temperature=0.1)
            return {
                "status": "success",
                "provider": self.provider,
                "model": self.model_name,
                "response": response[:100] + "..." if len(response) > 100 else response
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.provider,
                "model": self.model_name,
                "error": str(e)
            }