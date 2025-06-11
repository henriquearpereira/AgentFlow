"""Utility functions for the PDF Research Agent"""

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_huggingface_cache(cache_dir: str = None) -> str:
    """Setup HuggingFace cache directory"""
    if cache_dir is None:
        cache_dir = r'D:\.cache\huggingface'
    
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def extract_salary_data(text: str) -> List[str]:
    """Extract salary information from text using enhanced patterns"""
    salary_data = []
    text_lower = text.lower()
    
    # European salary patterns
    euro_patterns = [
        r'€\s*(\d{1,3}(?:[,.\s]?\d{3})*(?:[,.]?\d{2})?)',
        r'(\d{1,3}(?:[,.\s]?\d{3})*(?:[,.]?\d{2})?)\s*(?:eur|euro|euros)',
        r'(\d{1,3}(?:[,.\s]?\d{3})*)\s*(?:k|thousand)\s*(?:eur|euro|euros?)',
    ]
    
    # General salary patterns  
    salary_keywords = [
        r'salary.*?(\d{1,3}(?:[,.\s]?\d{3})*)',
        r'(?:earn|earning|makes?).*?(\d{1,3}(?:[,.\s]?\d{3})*)',
        r'(\d{1,3}(?:[,.\s]?\d{3})*)\s*(?:per year|annually|yearly)',
        r'average.*?(\d{1,3}(?:[,.\s]?\d{3})*)',
        r'range.*?(\d{1,3}(?:[,.\s]?\d{3})*)',
    ]
    
    # Extract salary information
    for pattern in euro_patterns + salary_keywords:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches[:3]:  # Limit matches
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            # Clean up the number
            clean_num = re.sub(r'[^\d,.]', '', str(match))
            if clean_num and len(clean_num) >= 3:  # Meaningful numbers
                salary_data.append(clean_num)
    
    return salary_data


def calculate_relevance_score(text: str, keywords: List[str]) -> int:
    """Calculate relevance score based on keyword presence"""
    text_lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in text_lower)


def clean_salary_figures(salary_figures: List[str]) -> List[str]:
    """Clean and deduplicate salary figures"""
    clean_salaries = []
    for salary in salary_figures:
        clean = re.sub(r'[^\d€$,.]', '', salary).strip()
        if clean and len(clean) >= 3:
            clean_salaries.append(salary.strip())
    
    return list(set(clean_salaries))


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)


def format_time_elapsed(start_time: float) -> str:
    """Format elapsed time in a readable format"""
    elapsed = time.time() - start_time
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_model_requirements(model_name: str) -> Tuple[bool, Optional[str]]:
    """Validate if model can be loaded with current system resources"""
    try:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        # Estimate memory requirements based on model name
        if "7b" in model_name.lower():
            min_ram_gb = 14  # Minimum RAM for 7B models
            min_vram_gb = 8   # Minimum VRAM if using GPU
        elif "phi-2" in model_name.lower():
            min_ram_gb = 4
            min_vram_gb = 3
        else:
            min_ram_gb = 8
            min_vram_gb = 6
        
        # Basic system check (simplified)
        import psutil
        available_ram = psutil.virtual_memory().available / (1024**3)  # GB
        
        if available_ram < min_ram_gb:
            return False, f"Insufficient RAM: {available_ram:.1f}GB available, {min_ram_gb}GB required"
        
        if cuda_available:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < min_vram_gb:
                    return False, f"Insufficient VRAM: {gpu_memory:.1f}GB available, {min_vram_gb}GB required"
            except:
                pass  # Continue with CPU
        
        return True, None
        
    except ImportError:
        # psutil not available, skip validation
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def setup_model_device_map(model_name: str) -> Dict[str, Any]:
    """Setup optimal device mapping for model loading"""
    device_map = {}
    
    if torch.cuda.is_available():
        # Check available GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 8:  # Sufficient VRAM
                device_map = "auto"
            else:
                device_map = {"": "cpu"}  # Fallback to CPU
        except:
            device_map = {"": "cpu"}
    else:
        device_map = {"": "cpu"}
    
    return device_map


def clean_generated_text(text: str, remove_prompt: bool = True) -> str:
    """Clean generated text by removing unwanted patterns"""
    if remove_prompt:
        # Remove any repetition of the prompt
        text = re.sub(r'Based on the research data.*?Start your response with', '', text, flags=re.DOTALL)
        text = re.sub(r'RESEARCH DATA:.*?TOPIC:', '', text, flags=re.DOTALL)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def ensure_directory_exists(file_path: str) -> None:
    """Ensure the directory for a file path exists"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return 0.0


def format_memory_usage() -> str:
    """Get current memory usage information"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return f"RAM: {memory_info.rss / 1024**2:.1f}MB"
    except ImportError:
        return "Memory info unavailable"


def print_performance_summary(timings: Dict[str, float], total_start_time: float) -> None:
    """Print a formatted performance summary"""
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("⏱️  PERFORMANCE SUMMARY")
    print("="*60)
    
    for stage, duration in timings.items():
        print(f"{stage:<18} {duration:.1f}s")
    
    print(f"{'Total runtime:':<18} {total_time:.1f}s")
    print(f"Memory usage:      {format_memory_usage()}")


def validate_search_results(search_results: str, min_length: int = 100) -> bool:
    """Validate if search results are sufficient for report generation"""
    if not search_results or len(search_results) < min_length:
        return False
    
    if "error" in search_results.lower():
        return False
    
    # Check for meaningful content
    meaningful_patterns = [
        r'\d+',  # Contains numbers
        r'salary|wage|pay|compensation',  # Contains salary-related terms
        r'https?://',  # Contains URLs
    ]
    
    for pattern in meaningful_patterns:
        if re.search(pattern, search_results, re.IGNORECASE):
            return True
    
    return False


def create_backup_filename(original_filename: str) -> str:
    """Create a backup filename with timestamp"""
    path = Path(original_filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
    return str(path.parent / backup_name)


def log_error(error: Exception, context: str = "") -> None:
    """Log error with context information"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    error_msg = f"[{timestamp}] ERROR in {context}: {type(error).__name__}: {str(error)}"
    
    # Print to console
    print(f"⚠️ {error_msg}")
    
    # Optionally log to file
    try:
        log_file = Path("logs/error.log")
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(error_msg + "\n")
    except:
        pass  # Don't fail if logging fails


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_model_identifier(model_id: str) -> Tuple[str, str]:
    """Parse provider/model identifier"""
    if "/" in model_id:
        provider, model = model_id.split("/", 1)
        return provider, model
    return "local", model_id


def get_default_output_filename(query: str) -> str:
    """Generate default output filename from query"""
    # Clean query for filename
    clean_query = re.sub(r'[^\w\s-]', '', query)
    clean_query = re.sub(r'\s+', '_', clean_query.strip())
    clean_query = clean_query[:50]  # Limit length
    
    timestamp = time.strftime("%Y%m%d_%H%M")
    return f"reports/{clean_query}_{timestamp}.pdf"


class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step_name: str = ""):
        """Update progress"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if self.current_step <= self.total_steps:
            percentage = (self.current_step / self.total_steps) * 100
            print(f"Progress: {percentage:.1f}% - {step_name} ({elapsed:.1f}s elapsed)")
    
    def complete(self):
        """Mark as complete"""
        total_time = time.time() - self.start_time
        print(f"✅ Completed in {total_time:.1f}s")