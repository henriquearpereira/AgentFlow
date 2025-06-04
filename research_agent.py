import os
os.environ['HF_HOME'] = r'D:\.cache\huggingface'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

import argparse
import time
import torch
import re
from pathlib import Path
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import ChatPromptTemplate
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

class ResearchAgent:
    def __init__(self, model_choice="phi-2"):
        print("🚀 Initializing Research Agent...")
        self.start_time = time.time()
        
        # Model selection
        self.model_choice = model_choice
        if model_choice == "zephyr":
            self.model_name = "HuggingFaceH4/zephyr-7b-beta"
            print("Using Zephyr-7B-Beta (Better quality, slower)")
        else:
            self.model_name = "microsoft/phi-2"
            print("Using Phi-2 (Faster, lighter)")
            
        self.load_model()
        print(f"✅ Model loaded in {time.time() - self.start_time:.1f}s")
        
        # Enhanced prompt template with strict formatting
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Create a data-focused executive briefing using ONLY these sections:\n"
             "1. Key Statistics: Bullet points with ONLY numbers, salaries, percentages, or years\n"
             "2. Trends: 1-2 sentences describing patterns\n"
             "3. Data Sources: List of URLs\n\n"
             "RULES:\n"
             "- Extract ONLY quantifiable facts\n"
             "- Ignore all opinions, marketing text, and non-numerical data\n"
             "- ABSOLUTELY NO introductions, conclusions, or explanations\n"
             "- MAX 100 words total\n"
             "- If insufficient data: 'Insufficient reliable data found'"),
            ("human", "TOPIC: {topic}\n\nSEARCH RESULTS:\n{search_results}")
        ])

    def load_model(self):
        """Optimized model loading with memory efficiency"""
        torch.set_grad_enabled(False)
        
        # Use float32 for CPU compatibility
        dtype = torch.float32
        
        try:
            # Load with optimized config
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline with optimized settings
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=100,          # Reduced tokens
                temperature=0.4,              # Less randomness
                top_k=30,                     # Narrower selection
                do_sample=True,
                num_return_sequences=1
            )
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            raise

    def run_search(self, query: str) -> str:
        """Focused search with salary-specific extraction"""
        print(f"🔍 Searching: '{query}'...")
        try:
            results = []
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=7)):
                    # Extract salary figures specifically
                    salary_pattern = r'(€?\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?\s?(?:per year|per month|/yr|/mo|annually|monthly))'
                    salaries = re.findall(salary_pattern, result['body'], re.IGNORECASE)
                    
                    # Extract other numerical data
                    other_nums = re.findall(r'\b\d{2,5}\b', result['body'])
                    
                    # Build result snippet
                    if salaries or other_nums:
                        snippet = f"Source {i+1}: "
                        if salaries:
                            # Convert set to list for subscripting
                            snippet += "Salaries: " + ", ".join(list(set(salaries))[:3]) + ". "
                        if other_nums:
                            snippet += "Numbers: " + ", ".join(list(set(other_nums))[:3])
                        snippet += f" | {result['href']}"
                        results.append(snippet)
                    
                    if len(results) >= 4:  # Use fewer but higher quality sources
                        break
            
            return "\n".join(results) if results else "No numerical data found"
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return "Search service unavailable"

    def generate_report(self, topic: str, search_results: str) -> str:
        """Faster report generation with strict formatting"""
        print(f"📊 Generating report for: '{topic}'...")
        gen_start = time.time()
        
        # Create prompt
        messages = self.prompt_template.format_messages(topic=topic, search_results=search_results)
        prompt_text = "\n".join([m.content for m in messages])
        
        try:
            # Generate with timeout
            response = self.pipeline(
                prompt_text,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # Disable beam search for speed
                truncation=True  # Add truncation to prevent long processing
            )
            
            # Extract generated text
            full_text = response[0]['generated_text']
            generated_text = full_text.split(prompt_text)[-1].strip()
            
            # Enforce section structure
            return self.enforce_structure(generated_text)
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return "Report generation failed"
        finally:
            print(f"⏱️  Generation time: {time.time() - gen_start:.1f}s")

    def enforce_structure(self, text: str) -> str:
        """Force the report into required sections"""
        # Initialize sections
        stats = []
        trends = []
        sources = []
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if re.match(r'(key stat|number|data point)', line, re.IGNORECASE):
                # Extract bullet points
                bullets = re.findall(r'[-•*]\s*(.*?)(?=\n|$)', line)
                if bullets:
                    stats.extend(bullets)
                elif any(char.isdigit() for char in line):
                    stats.append(line)
            elif re.match(r'(trend|pattern|change)', line, re.IGNORECASE):
                trends.append(line)
            elif 'http' in line or 'www.' in line:
                # Extract URLs
                urls = re.findall(r'https?://[^\s\)]+', line)
                if urls:
                    sources.extend(urls)
            elif any(char.isdigit() for char in line):
                stats.append(line)
            elif len(line.split()) < 8:  # Short lines likely metadata
                continue
            else:
                trends.append(line)
        
        # Build structured report
        report = ""
        if stats:
            report += "## Key Statistics\n"
            # Filter out non-numeric stats and limit to 5
            numeric_stats = [s for s in stats if any(char.isdigit() for char in s)][:5]
            report += "\n".join([f"- {s}" for s in numeric_stats]) + "\n\n"
        
        if trends:
            report += "## Trends\n"
            # Take first 2 unique trend statements
            unique_trends = list(set(trends))[:2]
            report += " ".join(unique_trends) + "\n\n"
        
        if sources:
            report += "## Data Sources\n"
            unique_sources = list(set(sources))[:3]
            report += "\n".join([f"- {s}" for s in unique_sources])
        
        # Fallback if structure missing
        if not report:
            return "⚠️ Insufficient reliable data found"
        
        return report.strip()

    def create_pdf(self, content: str, filename: str):
        """PDF creation with validation"""
        if "⚠️" in content or "insufficient" in content.lower():
            print("❌ Aborting PDF - no valid data")
            return False
            
        print(f"📄 Creating PDF: {filename}")
        try:
            # Create directory if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph("Research Report", styles['Title']))
            
            # Extract topic from filename (safe method)
            topic = Path(filename).stem.replace('_', ' ').title()
            story.append(Paragraph(f"Topic: {topic}", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Process content
            for section in content.split('\n\n'):
                if section.startswith('## '):
                    story.append(Paragraph(section[3:], styles['Heading2']))
                elif section.startswith('- ') and 'http' in section:
                    story.append(Paragraph(section[2:], styles['Italic']))
                elif section.startswith('- '):
                    story.append(Paragraph(section, styles['Bullet']))
                else:
                    story.append(Paragraph(section, styles['BodyText']))
                story.append(Spacer(1, 8))
                    
            doc.build(story)
            return True
        except Exception as e:
            print(f"⚠️ PDF creation failed: {e}")
            return False

def select_model():
    """Model selection with timing estimates"""
    print("\n" + "="*50)
    print("MODEL SELECTION")
    print("="*50)
    print("1. Zephyr-7B: Higher quality (~60s generation on CPU)")
    print("2. Phi-2: Faster (~15s generation on CPU)")
    print("3. Exit")
    
    while True:
        choice = input("\nChoose model (1-3): ").strip()
        if choice == "1":
            return "zephyr"
        elif choice == "2":
            return "phi-2"
        elif choice == "3":
            exit()
        else:
            print("Invalid choice. Please enter 1, 2, or 3")

if __name__ == "__main__":
    # User selects model
    model_choice = select_model()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='AI Research Agent')
    parser.add_argument('query', type=str, help='Research topic')
    parser.add_argument('-o', '--output', type=str, default='reports/report.pdf', help='Output PDF file')
    args = parser.parse_args()
    
    # Initialize agent
    start_time = time.time()
    agent = ResearchAgent(model_choice)
    model_load_time = time.time() - start_time
    
    # Execute workflow
    search_start = time.time()
    search_results = agent.run_search(args.query)
    search_time = time.time() - search_start
    
    report_start = time.time()
    report_content = agent.generate_report(args.query, search_results)
    report_time = time.time() - report_start
    
    pdf_start = time.time()
    pdf_success = agent.create_pdf(report_content, args.output)
    pdf_time = time.time() - pdf_start
    
    total_time = time.time() - start_time
    
    # Print diagnostics
    print("\n⏱️  PERFORMANCE DIAGNOSTICS:")
    print(f"  Model loading: {model_load_time:.1f}s")
    print(f"  Search: {search_time:.1f}s")
    print(f"  Report generation: {report_time:.1f}s")
    print(f"  PDF creation: {pdf_time:.1f}s")
    print(f"✅ Total time: {total_time:.1f}s")
    
    if pdf_success:
        print(f"📁 Output saved to: {os.path.abspath(args.output)}")
    else:
        print("❌ No PDF generated due to errors")
    
    # Print final report content
    print("\n📝 FINAL REPORT CONTENT:")
    print(report_content)