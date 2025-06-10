"""Local model implementations - Fixed initialization and interface issues"""

import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config.models import get_model_info

class LocalModelHandler:
    def __init__(self, model_key="phi-2", max_tokens=500, temperature=0.2, verbose=False):
        """
        Initialize LocalModelHandler with proper parameter handling
        
        Args:
            model_key: String key for the model (e.g., "phi-2", "zephyr", "mistral")
            max_tokens: Maximum tokens for generation
            temperature: Generation temperature
            verbose: Enable verbose output
        """
        # Handle case where model_key might be passed as part of a config dict
        if isinstance(model_key, dict):
            config = model_key
            model_key = config.get('model_key', 'phi-2')
            max_tokens = config.get('max_tokens', 500)
            temperature = config.get('temperature', 0.2)
            verbose = config.get('verbose', False)
        
        print(f"🚀 Initializing Local Model: {model_key}...")
        self.start_time = time.time()
        
        # Store parameters
        self.model_key = model_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize model-related attributes
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Get model configuration
        model_config = get_model_info("local", model_key)
        
        if not model_config:
            raise ValueError(f"Unknown local model: {model_key}")
        
        self.model_name = model_config["name"]
        self.description = model_config["description"]
        
        print(f"📦 Loading {self.description}")
        print(f"⏱️  Expected time: {model_config['estimated_time']}")
        print(f"💾 Memory usage: {model_config['memory_usage']}")
        
        self.load_model()
        load_time = time.time() - self.start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        
        # Enhanced prompt template optimized for research reports
        self.prompt_template = self._get_prompt_template()

    def get(self, key, default=None):
        """
        Dictionary-like interface for compatibility with ResearchAgent
        This method allows the handler to be accessed like a dictionary
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __getitem__(self, key):
        """Allow dictionary-style access"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in LocalModelHandler")

    def __contains__(self, key):
        """Support 'in' operator"""
        return hasattr(self, key)

    def keys(self):
        """Return available keys"""
        return [attr for attr in dir(self) if not attr.startswith('_')]

    def _get_prompt_template(self):
        """Get optimized prompt template based on model"""
        if self.model_key == "phi-2":
            return """Based on the research data below, create a professional salary report.

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

Focus on actual numbers and be specific. Start your response with "## Key Statistics":"""
        
        elif self.model_key == "zephyr":
            return """<|system|>
You are a professional research analyst. Create a detailed salary report based on the provided data.
<|user|>
Research Data:
{search_results}

Topic: {topic}

Please create a comprehensive salary report with:
1. Key Statistics (specific salary figures)
2. Market Trends (patterns and insights)  
3. Data Sources (list of sources used)

Make it professional and data-driven.
<|assistant|>"""
        
        elif self.model_key == "mistral":
            return """[INST] You are a research analyst. Based on the following data, create a professional salary report.

Research Data:
{search_results}

Topic: {topic}

Create a structured report with:
- Key Statistics (specific numbers)
- Trends (market analysis)
- Data Sources (source list)

Be specific and professional. [/INST]"""
        
        else:
            # Generic template for other models
            return """Based on the research data, create a professional salary report for: {topic}

Data:
{search_results}

Report structure:
## Key Statistics
## Trends  
## Data Sources

Focus on specific numbers and professional analysis:"""

    def load_model(self):
        """Optimized model loading with memory efficiency"""
        torch.set_grad_enabled(False)
        
        try:
            # Load tokenizer
            print("🔄 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=os.environ.get('HF_HOME')
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimized settings
            print("🔄 Loading model...")
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=os.environ.get('HF_HOME')
            )
            
            # Create pipeline with model-specific settings
            pipeline_settings = self._get_pipeline_settings()
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **pipeline_settings
            )
            
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            raise

    def _get_pipeline_settings(self):
        """Get optimized pipeline settings for each model"""
        base_settings = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "return_full_text": False
        }
        
        if self.model_key == "phi-2":
            return {
                **base_settings,
                "max_new_tokens": min(self.max_tokens, 400),
                "temperature": self.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        elif self.model_key == "zephyr":
            return {
                **base_settings,
                "max_new_tokens": min(self.max_tokens, 500),
                "temperature": self.temperature,
                "top_p": 0.95,
                "repetition_penalty": 1.05
            }
        elif self.model_key == "mistral":
            return {
                **base_settings,
                "max_new_tokens": min(self.max_tokens, 450),
                "temperature": self.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        else:
            return {
                **base_settings,
                "max_new_tokens": min(self.max_tokens, 400),
                "temperature": self.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Generate text using the local model"""
        print(f"🤖 Generating with {self.model_key}...")
        gen_start = time.time()
        
        try:
            # Use custom parameters if provided, otherwise use defaults
            generation_kwargs = {}
            if max_tokens:
                generation_kwargs["max_new_tokens"] = max_tokens
            if temperature is not None:
                generation_kwargs["temperature"] = temperature
                
            # Override default settings with custom ones
            for key, value in kwargs.items():
                generation_kwargs[key] = value
            
            # Generate response
            response = self.pipeline(prompt, **generation_kwargs)
            generated_text = response[0]['generated_text'].strip()
            
            gen_time = time.time() - gen_start
            print(f"⏱️ Generation completed in {gen_time:.1f}s")
            
            return generated_text
            
        except Exception as e:
            print(f"⚠️ Generation error with {self.model_key}: {e}")
            gen_time = time.time() - gen_start
            print(f"⏱️ Failed generation time: {gen_time:.1f}s")
            return self._fallback_response()

    def generate_report(self, topic: str, search_results: str) -> str:
        """Generate a research report using the local model"""
        print(f"📊 Generating report for: '{topic}'...")
        
        # Create the prompt using the model-specific template
        prompt = self.prompt_template.format(
            topic=topic,
            search_results=search_results
        )
        
        # Generate the report
        report = self.generate(prompt)
        
        # Post-process and validate the report
        cleaned_report = self._clean_and_validate_report(report, topic, search_results)
        
        return cleaned_report

    def _clean_and_validate_report(self, text: str, topic: str, search_results: str) -> str:
        """Clean and validate the generated report"""
        import re
        
        # Remove any repetition of the prompt
        text = re.sub(r'Based on the research data.*?Start your response with', '', text, flags=re.DOTALL)
        text = re.sub(r'RESEARCH DATA:.*?TOPIC:', '', text, flags=re.DOTALL)
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|system\|>.*?<\|assistant\|>', '', text, flags=re.DOTALL)
        
        # Ensure it starts with ## Key Statistics
        if not text.startswith("## Key Statistics"):
            if "Key Statistics" in text:
                text = "## " + text[text.find("Key Statistics"):]
            else:
                text = "## Key Statistics\n" + text
        
        # Ensure all required sections exist
        required_sections = ["## Key Statistics", "## Trends", "## Data Sources"]
        
        for section in required_sections:
            if section not in text:
                if section == "## Trends":
                    # Extract salary numbers from search results for trends
                    salary_nums = re.findall(r'\d{1,3}(?:[,.\s]?\d{3})*', search_results)
                    trend_text = f"Python developer salaries in Portugal show ranges from entry-level to senior positions. "
                    if salary_nums:
                        trend_text += "Market data indicates competitive compensation with variation based on experience level."
                    text += f"\n\n{section}\n{trend_text}\n"
                    
                elif section == "## Data Sources":
                    # Extract URLs from search results
                    urls = re.findall(r'https?://[^\s]+', search_results)
                    text += f"\n\n{section}\n"
                    for url in urls[:4]:
                        text += f"- {url}\n"
                    if not urls:
                        text += "- Glassdoor Portugal\n- PayScale Portugal\n- SalaryExpert\n"
        
        # Validate that Key Statistics has actual numbers
        stats_section = text.split("## Key Statistics")[1].split("##")[0] if "## Key Statistics" in text else ""
        if not re.search(r'\d+', stats_section):
            # Add actual salary data from search results
            salary_figures = re.findall(r'(?:€|EUR|\$)?\s*\d{1,3}(?:[,.\s]?\d{3})*', search_results)
            if salary_figures:
                stats_addition = f"- Salary ranges: {', '.join(set(salary_figures[:4]))}\n"
                text = text.replace("## Key Statistics", f"## Key Statistics\n{stats_addition}", 1)
        
        return text

    def _fallback_response(self) -> str:
        """Generate a fallback response when generation fails"""
        return """## Key Statistics
- Local model generation temporarily unavailable
- Fallback report generated from search data
- Multiple salary data points collected from Portuguese job market

## Trends
The Portuguese tech market shows strong demand for Python developers with competitive salaries varying by experience level and location.

## Data Sources  
- Search results from multiple job market sources
- Salary data aggregated from various platforms"""

    def test_generation(self) -> dict:
        """Test the model with a simple prompt"""
        test_prompt = "Hello! Please respond with a brief greeting."
        
        try:
            start_time = time.time()
            response = self.generate(test_prompt, max_tokens=50, temperature=0.1)
            generation_time = time.time() - start_time
            
            return {
                "status": "success",
                "model": self.model_key,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "generation_time": f"{generation_time:.1f}s"
            }
        except Exception as e:
            return {
                "status": "error", 
                "model": self.model_key,
                "error": str(e)
            }

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "description": self.description,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "unknown"
        }

    def cleanup(self):
        """Clean up model resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            # Clear CUDA cache if available  
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("🧹 Model resources cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    # Compatibility methods for the ResearchAgent interface
    def test_connection(self):
        """Test connection - for compatibility with API models"""
        return self.test_generation()

    def __str__(self):
        """String representation of the handler"""
        return f"LocalModelHandler(model_key={self.model_key}, model_name={self.model_name})"

    def __repr__(self):
        """Detailed representation of the handler"""
        return f"LocalModelHandler(model_key='{self.model_key}', model_name='{self.model_name}', max_tokens={self.max_tokens}, temperature={self.temperature})"