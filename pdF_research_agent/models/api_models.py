"""API-based model implementations - FIXED VERSION"""

import os
import requests
import time
from typing import Optional, Dict, Any

class APIModelHandler:
    def __init__(self, provider: str, model_name: str, api_key: str, max_tokens: int = 500, temperature: float = 0.2, verbose: bool = False):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.client = None
        
        if self.verbose:
            print(f"🔧 Initializing {provider} handler with model: {model_name}")
        
        self.setup_client()
    
    def setup_client(self):
        """Setup API client based on provider"""
        try:
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
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to setup {self.provider} client: {e}")
            raise
    
    def setup_groq(self):
        """Setup Groq client using requests (no openai dependency)"""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.verbose:
            print(f"✅ Groq client setup complete")
    
    def setup_together(self):
        """Setup Together AI client using requests"""
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is required")
            
        self.base_url = "https://api.together.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def setup_huggingface(self):
        """Setup Hugging Face Inference API"""
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def setup_openrouter(self):
        """Setup OpenRouter client using requests"""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
            
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def setup_cohere(self):
        """Setup Cohere client using requests"""
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is required")
        
        self.base_url = "https://api.cohere.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> str:
        """Generate text using the configured API"""
        # Use instance defaults if not provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        if self.verbose:
            print(f"🔄 Generating with {self.provider} ({self.model_name})")
            print(f"📝 Prompt length: {len(prompt)} chars")
        
        try:
            if self.provider in ["groq", "together", "openrouter"]:
                return self._generate_openai_compatible(prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "cohere":
                return self._generate_cohere(prompt, max_tokens, temperature, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            error_msg = f"API generation error ({self.provider}): {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            
            # Return a more informative fallback response
            return self._fallback_response(prompt, error_msg)
    
    def _generate_openai_compatible(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using OpenAI-compatible APIs (Groq, Together, OpenRouter)"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get('top_p', 0.9),
            "stream": False
        }
        
        if self.verbose:
            print(f"🌐 Making request to: {url}")
            print(f"📦 Payload model: {payload['model']}")
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload, 
                timeout=60
            )
            
            if self.verbose:
                print(f"📡 Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            if self.verbose:
                print(f"📄 Response keys: {list(result.keys())}")
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                if self.verbose:
                    print(f"✅ Generated {len(content)} characters")
                return content.strip()
            else:
                raise Exception(f"No choices in response: {result}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected response format, missing key: {e}")
        except Exception as e:
            raise Exception(f"OpenAI-compatible API error: {e}")
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using Hugging Face Inference API"""
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
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
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
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "p": kwargs.get('top_p', 0.9),
            "stop_sequences": kwargs.get('stop_sequences', [])
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if 'generations' in result and len(result['generations']) > 0:
                return result['generations'][0]['text'].strip()
            else:
                raise Exception(f"No generations in response: {result}")
                
        except Exception as e:
            raise Exception(f"Cohere API error: {e}")
    
    def _fallback_response(self, prompt: str, error_msg: str) -> str:
        """Generate a fallback response when API fails"""
        return f"""## API Error Response

**Error:** {error_msg}

**Provider:** {self.provider}
**Model:** {self.model_name}

## Fallback Information
Unable to generate AI response due to API connectivity issues. Please check:
1. API key configuration
2. Model availability
3. Network connectivity

## Next Steps
- Verify API key in environment variables
- Check model name spelling
- Try a different model or provider
- Review API quota and billing status

**Original prompt length:** {len(prompt)} characters
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return status"""
        test_prompt = "Hello! Please respond with 'API connection successful' to confirm the connection is working."
        
        try:
            if self.verbose:
                print(f"🧪 Testing {self.provider} connection...")
            
            response = self.generate(test_prompt, max_tokens=50, temperature=0.1)
            
            return {
                "success": True,
                "status": "connected",
                "provider": self.provider,
                "model": self.model_name,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "response_length": len(response)
            }
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "provider": self.provider,
                "model": self.model_name,
                "error": str(e)
            }
    
    def cleanup(self):
        """Cleanup resources (placeholder for future use)"""
        if self.verbose:
            print(f"🧹 Cleaning up {self.provider} handler")
        pass