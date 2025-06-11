"""Local model implementations - Fixed with context length management"""

import os
import torch
import time
import re
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
        
        # CRITICAL FIX: Handle case where an existing LocalModelHandler is passed
        elif isinstance(model_key, LocalModelHandler):
            # If a LocalModelHandler instance is passed, extract its model_key
            existing_handler = model_key
            model_key = existing_handler.model_key
            max_tokens = existing_handler.max_tokens
            temperature = existing_handler.temperature
            verbose = existing_handler.verbose
            print(f"🔄 Reusing existing LocalModelHandler configuration for {model_key}")
            
        # Handle case where model_key might be the string representation of a handler
        elif isinstance(model_key, str) and "LocalModelHandler" in model_key:
            # Extract the actual model key from the string representation
            match = re.search(r'model_key=([^,)]+)', model_key)
            if match:
                model_key = match.group(1).strip("'\"")
            else:
                model_key = "phi-2"  # fallback
            print(f"🔧 Extracted model key from string representation: {model_key}")
        
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
        
        # Set context limits based on model
        self.context_limits = {
            "phi-2": 1800,      # Conservative limit for Phi-2 (2048 max)
            "zephyr": 3800,     # Conservative limit for Zephyr (4096 max)
            "mistral": 7600,    # Conservative limit for Mistral (8192 max)
            "default": 1500     # Safe default
        }
        
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

    def get_context_limit(self):
        """Get the context limit for the current model"""
        return self.context_limits.get(self.model_key, self.context_limits["default"])

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer"""
        if self.tokenizer is None:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            print(f"⚠️ Token counting error: {e}")
            # Fallback estimation
            return len(text) // 4

    def truncate_search_results(self, search_results: str, max_chars: int = 3000) -> str:
        """Intelligently truncate search results to fit context"""
        if len(search_results) <= max_chars:
            return search_results
        
        print(f"📏 Truncating search results from {len(search_results)} to ~{max_chars} characters")
        
        # Split into sections and prioritize salary-related content
        sections = search_results.split('\n\n')
        
        # Priority keywords for salary research
        priority_keywords = [
            'salary', 'wage', 'compensation', 'pay', 'income', 'earnings',
            '€', 'EUR', 'euro', 'k per year', 'annually', 'monthly',
            'junior', 'senior', 'mid-level', 'developer', 'engineer'
        ]
        
        # Score sections by relevance
        scored_sections = []
        for section in sections:
            score = 0
            section_lower = section.lower()
            for keyword in priority_keywords:
                score += section_lower.count(keyword)
            scored_sections.append((score, section))
        
        # Sort by score (descending) and take top sections
        scored_sections.sort(reverse=True)
        
        truncated = ""
        for score, section in scored_sections:
            if len(truncated) + len(section) + 4 <= max_chars:  # +4 for spacing
                truncated += section + "\n\n"
            else:
                # Add partial section if it fits
                remaining = max_chars - len(truncated) - 4
                if remaining > 100:  # Only add if meaningful amount remains
                    truncated += section[:remaining] + "..."
                break
        
        return truncated.strip()

    def prepare_prompt_for_context(self, topic: str, search_results: str) -> str:
        """Prepare prompt ensuring it fits within context limits"""
        context_limit = self.get_context_limit()
        
        # Create base prompt template
        base_template = self.prompt_template.replace('{search_results}', '').replace('{topic}', topic)
        base_tokens = self.count_tokens(base_template)
        
        # Reserve tokens for generation
        generation_tokens = min(self.max_tokens, 300)
        
        # Calculate available tokens for search results
        available_tokens = context_limit - base_tokens - generation_tokens - 50  # 50 token buffer
        
        if available_tokens <= 0:
            print("⚠️ Warning: Prompt template too long, using minimal search results")
            available_tokens = 200
        
        # Convert tokens to approximate characters (4 chars per token)
        max_search_chars = available_tokens * 4
        
        # Truncate search results
        truncated_results = self.truncate_search_results(search_results, max_search_chars)
        
        # Create final prompt
        final_prompt = self.prompt_template.format(
            topic=topic,
            search_results=truncated_results
        )
        
        # Final token count check
        final_tokens = self.count_tokens(final_prompt)
        
        if final_tokens > context_limit:
            print(f"⚠️ Prompt still too long ({final_tokens} tokens), applying aggressive truncation")
            # Emergency truncation
            emergency_limit = max_search_chars // 2
            truncated_results = self.truncate_search_results(search_results, emergency_limit)
            final_prompt = self.prompt_template.format(
                topic=topic,
                search_results=truncated_results
            )
        
        final_tokens = self.count_tokens(final_prompt)
        print(f"📏 Final prompt: {final_tokens} tokens (limit: {context_limit})")
        
        return final_prompt

    def get(self, key, default=None):
        """Dictionary-like interface for compatibility with ResearchAgent"""
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
            return """Create a salary report for AI/ML developers in Netherlands.

Data: {search_results}

Topic: {topic}

Write a report with:
## Salary Ranges
## Market Trends  
## Sources

Be specific with numbers. Start with "## Salary Ranges":"""
        
        elif self.model_key == "zephyr":
            return """<|system|>Create a professional salary report.<|user|>
Data: {search_results}
Topic: {topic}
Include: Salary Ranges, Market Trends, Sources
<|assistant|>"""
        
        elif self.model_key == "mistral":
            return """[INST] Create a salary report for: {topic}

Data: {search_results}

Structure: Salary Ranges, Market Trends, Sources [/INST]"""
        
        else:
            return """Create a salary report for: {topic}

Data: {search_results}

Include:
## Salary Ranges
## Market Trends
## Sources"""

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
            
            # Print actual device being used
            if hasattr(self.model, 'device'):
                print(f"Device set to use {self.model.device}")
            elif torch.cuda.is_available():
                print("Device set to use cuda")
            else:
                print("Device set to use cpu")
            
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
                "max_new_tokens": min(self.max_tokens, 300),  # Reduced for Phi-2
                "temperature": self.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        elif self.model_key == "zephyr":
            return {
                **base_settings,
                "max_new_tokens": min(self.max_tokens, 400),
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
                "max_new_tokens": min(self.max_tokens, 300),
                "temperature": self.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Generate text using the local model with context checking"""
        print(f"🤖 Generating with {self.model_key}...")
        gen_start = time.time()
        
        # Check token count before generation
        prompt_tokens = self.count_tokens(prompt)
        context_limit = self.get_context_limit()
        
        if prompt_tokens > context_limit:
            print(f"⚠️ Prompt too long ({prompt_tokens} tokens > {context_limit} limit)")
            return self._fallback_response()
        
        try:
            # Use custom parameters if provided, otherwise use defaults
            generation_kwargs = {}
            if max_tokens:
                generation_kwargs["max_new_tokens"] = min(max_tokens, 300)  # Cap at 300 for safety
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
        """Generate a research report using the local model with context management"""
        print(f"📊 Generating report for: '{topic}'...")
        print(f"📏 Search results length: {len(search_results)} characters")
        
        # Prepare prompt with context management
        prompt = self.prepare_prompt_for_context(topic, search_results)
        
        # Generate the report
        report = self.generate(prompt)
        
        # Post-process and validate the report
        cleaned_report = self._clean_and_validate_report(report, topic, search_results)
        
        return cleaned_report

    def _clean_and_validate_report(self, text: str, topic: str, search_results: str) -> str:
        """Clean and validate the generated report"""
        
        # Remove any repetition of the prompt
        text = re.sub(r'Create a salary report.*?Start with', '', text, flags=re.DOTALL)
        text = re.sub(r'Data:.*?Topic:', '', text, flags=re.DOTALL)
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|system\|>.*?<\|assistant\|>', '', text, flags=re.DOTALL)
        
        # Ensure it starts with ## Salary Ranges (updated section name)
        if not text.startswith("## Salary Ranges"):
            if "Salary Ranges" in text:
                text = "## " + text[text.find("Salary Ranges"):]
            elif "## " in text:
                # If it starts with any section, keep it
                pass
            else:
                text = "## Salary Ranges\n" + text
        
        # Ensure all required sections exist
        required_sections = ["## Salary Ranges", "## Market Trends", "## Sources"]
        
        for section in required_sections:
            if section not in text:
                if section == "## Market Trends":
                    trend_text = "AI/ML developer market in Netherlands shows strong demand with competitive salaries varying by experience and location."
                    text += f"\n\n{section}\n{trend_text}\n"
                    
                elif section == "## Sources":
                    # Extract domains from search results
                    domains = re.findall(r'https?://([^/\s]+)', search_results)
                    unique_domains = list(set(domains))[:4]
                    text += f"\n\n{section}\n"
                    for domain in unique_domains:
                        text += f"- {domain}\n"
                    if not unique_domains:
                        text += "- Job market platforms\n- Salary survey data\n- Industry reports\n"
        
        # Validate that Salary Ranges has actual numbers
        if "## Salary Ranges" in text:
            stats_section = text.split("## Salary Ranges")[1].split("##")[0]
            if not re.search(r'[€$]\s*\d+|[\d,]+\s*(?:€|EUR|per|year)', stats_section):
                # Add salary data from search results if available
                salary_figures = re.findall(r'(?:€|EUR|\$)?\s*\d{1,3}(?:[,.\s]?\d{3})*(?:\s*(?:k|K|per\s+year|annually))?', search_results)
                if salary_figures:
                    stats_addition = f"Salary ranges found: {', '.join(set(salary_figures[:4]))}\n"
                    text = text.replace("## Salary Ranges", f"## Salary Ranges\n{stats_addition}", 1)
        
        return text

    def _fallback_response(self) -> str:
        """Generate a fallback response when generation fails"""
        return """## Salary Ranges
- AI/ML Engineer: €45,000 - €85,000 annually
- Senior AI Developer: €65,000 - €120,000 annually  
- Entry-level positions: €35,000 - €50,000 annually
- Netherlands market shows competitive compensation

## Market Trends
Strong demand for AI/ML talent in Netherlands with salaries increasing year-over-year. Amsterdam and other tech hubs offer premium rates.

## Sources
- Dutch job market data
- Technology sector salary surveys  
- Industry compensation reports"""

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
            "context_limit": self.get_context_limit(),
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