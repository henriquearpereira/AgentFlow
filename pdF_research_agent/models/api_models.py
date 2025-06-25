"""API-based model implementations - UPDATED VERSION with better Groq models"""

import os
import requests
import time
import json
from typing import Optional, Dict, Any
import threading

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ huggingface_hub not installed. Run: pip install huggingface_hub")

# Global rate limiter for Together API
class TogetherRateLimiter:
    def __init__(self, max_calls=6, period=60):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = []  # list of timestamps

    def wait(self):
        with self.lock:
            now = time.time()
            # Remove calls older than period
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    print(f"⏳ Together API rate limit hit. Waiting {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                # After sleep, clean up again
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.period]
            self.calls.append(time.time())

together_rate_limiter = TogetherRateLimiter()

class APIModelHandler:
    def __init__(self, provider: str, model_name: str, api_key: str, max_tokens: int = 2000, temperature: float = 0.3, verbose: bool = False):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.client = None
        self.system_prompt = None  # Allow custom system prompts
        
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
            elif self.provider == "nebius":
                self.setup_nebius()
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
    
    def setup_nebius(self):
        """Setup Nebius client using Hugging Face SDK"""
        if not HF_AVAILABLE:
            raise ValueError("huggingface_hub package required for Nebius. Install with: pip install huggingface_hub")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required for Nebius provider")
        self.client = InferenceClient(provider="nebius", api_key=self.api_key)
        if self.verbose:
            print(f"✅ Nebius client setup complete")
    
    def set_parameters(self, temperature: float = None, max_tokens: int = None, system_prompt: str = None):
        """Set model parameters dynamically"""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if system_prompt is not None:
            self.system_prompt = system_prompt
    
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
            elif self.provider == "nebius":
                return self._generate_nebius(prompt, max_tokens, temperature, **kwargs)
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
        
        # Rate limit Together API calls
        if self.provider == "together":
            together_rate_limiter.wait()
        
        # Use custom system prompt if provided, otherwise use enhanced default
        if self.system_prompt:
            system_prompt = self.system_prompt
        else:
            system_prompt = """You are an expert research analyst and technical writer with deep domain knowledge across multiple fields. Your task is to create comprehensive, intelligent, and well-structured content that demonstrates deep understanding and provides valuable insights.

Key capabilities:
1. **Deep Reasoning**: Analyze complex topics with sophisticated understanding
2. **Original Insights**: Generate novel perspectives and forward-thinking analysis
3. **Comprehensive Coverage**: Address multiple dimensions (technical, business, social, ethical)
4. **Specific Examples**: Provide concrete cases, technologies, companies, and methodologies
5. **Actionable Recommendations**: Offer practical strategies and next steps
6. **Future-Focused**: Consider emerging trends and breakthrough opportunities

Quality standards:
- Write in professional, analytical tone with original insights
- Avoid generic placeholder language - be specific and detailed
- Include specific data points, technologies, and examples where relevant
- Provide balanced, objective analysis with multiple perspectives
- Structure content logically with clear sections and flow
- Demonstrate deep domain knowledge and understanding
- Consider global and local market dynamics
- Address both current state and future potential

Your responses should showcase intelligence, creativity, and practical value."""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
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
                timeout=120  # Increased timeout for larger models
            )
            if self.verbose:
                print(f"📡 Response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            if self.verbose:
                print(f"📄 Response keys: {list(result.keys())}")
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                raise Exception(f"No choices in response: {result}")
        except requests.exceptions.HTTPError as e:
            print(f"❌ API generation error ({self.provider}): {e}")
            print(f"🔎 Payload sent: {json.dumps(payload, indent=2)}")
            if e.response is not None:
                print(f"🔎 Error response body: {e.response.text}")
            raise
        except Exception as e:
            print(f"❌ API generation error ({self.provider}): {e}")
            print(f"🔎 Payload sent: {json.dumps(payload, indent=2)}")
            raise
    
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
    
    def _generate_nebius(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using Nebius provider through Hugging Face"""
        try:
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            if self.verbose:
                print(f"🌐 Making request to Nebius via HuggingFace")
                print(f"📦 Model: {self.model_name}")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            if self.verbose:
                print(f"📡 Response received successfully")
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Nebius API error: {e}")
            raise
    
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "handler_type": "APIModelHandler",
            "provider": self.provider,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt[:100] + "..." if self.system_prompt and len(self.system_prompt) > 100 else self.system_prompt,
            "verbose": self.verbose
        }

    async def generate_async(self, prompt: str) -> str:
        """Async version of generate"""
        return self.generate(prompt)

# Recommended model configurations for better performance
RECOMMENDED_GROQ_MODELS = {
    "best_quality": "llama-3.3-70b-versatile",  # Highest quality, slower
    "balanced": "llama-3.1-8b-instant",         # Good balance of speed and quality
    "experimental": "compound-beta",             # Latest model, high quality
    "specialized": "compound-beta",              # Good for complex reasoning and code
    "technical": "gemma2-9b-it",                # Specialized for technical content
    "fast": "llama-3.1-8b-instant"              # Ultra-fast for basic tasks
}

def get_recommended_model(use_case: str = "balanced") -> str:
    """Get recommended model based on use case"""
    return RECOMMENDED_GROQ_MODELS.get(use_case, "llama-3.1-8b-instant")

# Example usage with better model
# if __name__ == "__main__":
#     # Use the better 70B model for higher quality
#     model = APIModelHandler(
#         provider="groq",
#         model_name="llama-3.1-70b-versatile",
#         api_key=os.getenv("GROQ_API_KEY"),
#         max_tokens=2000,
#         temperature=0.3,
#         verbose=True
#     )
#     
#     # Test connection
#     result = model.test_connection()
#     print(f"Connection test: {result}")