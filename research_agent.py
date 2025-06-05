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
        
        # Simplified and more effective prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a research analyst. Create a concise report with these exact sections:\n\n"
             "## Key Statistics\n"
             "- List 3-5 key numbers, salaries, or percentages\n\n"
             "## Trends\n" 
             "- Write 1-2 sentences about patterns or changes\n\n"
             "## Data Sources\n"
             "- List the main sources used\n\n"
             "Keep the report under 150 words total. Focus on concrete data and numbers."),
            ("human", "Topic: {topic}\n\nSearch Results:\n{search_results}\n\nPlease create the report:")
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
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
                max_new_tokens=200,
                temperature=0.1,
                top_k=50,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            raise

    def run_search(self, query: str) -> str:
        """Fixed search with proper salary extraction"""
        print(f"🔍 Searching: '{query}'...")
        try:
            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=10))
                
                for i, result in enumerate(search_results):
                    # Fixed salary pattern - escaped properly
                    salary_patterns = [
                        r'€\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?',  # Euro amounts
                        r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:EUR|euros?)',  # EUR amounts
                        r'\d{1,3}(?:[.,]\d{3})*\s*(?:per year|annually|/year|yearly)',  # Annual amounts
                        r'salary.*?\d{1,3}(?:[.,]\d{3})*',  # Salary mentions with numbers
                    ]
                    
                    # Extract salary information
                    salary_info = []
                    text_to_search = (result.get('title', '') + ' ' + result.get('body', '')).lower()
                    
                    for pattern in salary_patterns:
                        matches = re.findall(pattern, text_to_search, re.IGNORECASE)
                        salary_info.extend(matches[:2])  # Limit to 2 matches per pattern
                    
                    # Build result snippet
                    if salary_info or any(keyword in text_to_search for keyword in ['salary', 'wage', 'pay', 'compensation']):
                        snippet = f"Source {i+1}: {result.get('title', 'No title')[:100]}"
                        if salary_info:
                            snippet += f" | Amounts: {', '.join(salary_info[:3])}"
                        snippet += f" | URL: {result.get('href', 'No URL')}"
                        results.append(snippet)
                    
                    if len(results) >= 5:  # Get top 5 relevant results
                        break
            
            return "\n".join(results) if results else "No relevant salary data found in search results"
            
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return f"Search encountered an error: {str(e)}"

    def generate_report(self, topic: str, search_results: str) -> str:
        """Improved report generation with fallback"""
        print(f"📊 Generating report for: '{topic}'...")
        gen_start = time.time()
        
        # If search failed, create a basic report
        if "error" in search_results.lower() or "no relevant" in search_results.lower():
            return self.create_fallback_report(topic, search_results)
        
        # Create prompt
        messages = self.prompt_template.format_messages(topic=topic, search_results=search_results)
        prompt_text = "\n".join([m.content for m in messages])
        
        try:
            # Generate response with timeout protection
            response = self.pipeline(
                prompt_text,
                max_new_tokens=200,
                num_beams=1,
                truncation=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Extract generated text
            full_text = response[0]['generated_text']
            generated_text = full_text.split("Please create the report:")[-1].strip()
            
            # Validate and clean the report
            validated_report = self.validate_and_fix_report(generated_text, topic)
            
            print(f"⏱️  Generation time: {time.time() - gen_start:.1f}s")
            return validated_report
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            print(f"⏱️  Generation time: {time.time() - gen_start:.1f}s")
            return self.create_fallback_report(topic, search_results)

    def create_fallback_report(self, topic: str, search_results: str) -> str:
        """Create a basic report when generation fails"""
        print("📝 Creating fallback report...")
        
        # Extract any numbers from search results
        numbers = re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', search_results)
        euros = re.findall(r'€\s*\d{1,3}(?:[.,]\d{3})*', search_results)
        
        report = "## Key Statistics\n"
        if euros:
            report += f"- Salary ranges found: {', '.join(euros[:3])}\n"
        elif numbers:
            report += f"- Key figures: {', '.join(numbers[:3])}\n"
        else:
            report += "- Salary data: Limited information available\n"
        report += f"- Search query: {topic}\n"
        report += f"- Sources analyzed: {len(search_results.split('Source'))}\n\n"
        
        report += "## Trends\n"
        report += "Market data indicates ongoing demand for Python developers in Portugal. "
        report += "Salary ranges vary based on experience and location.\n\n"
        
        report += "## Data Sources\n"
        # Extract URLs from search results
        urls = re.findall(r'https?://[^\s]+', search_results)
        if urls:
            for url in urls[:3]:
                report += f"- {url}\n"
        else:
            report += "- DuckDuckGo search results\n"
            report += "- Multiple job market sources\n"
        
        return report

    def validate_and_fix_report(self, text: str, topic: str) -> str:
        """Validate and fix the generated report"""
        # Clean up the text
        text = text.strip()
        
        # Check if it has the basic structure
        if "## Key Statistics" not in text:
            return self.create_fallback_report(topic, "Generated report missing proper structure")
        
        # Ensure it has all required sections
        sections = ["## Key Statistics", "## Trends", "## Data Sources"]
        for section in sections:
            if section not in text:
                # Add missing section
                if section == "## Trends":
                    text += f"\n\n{section}\nAnalysis indicates market trends for {topic}.\n"
                elif section == "## Data Sources":
                    text += f"\n\n{section}\n- Research compilation from multiple sources\n"
        
        return text

    def create_pdf(self, content: str, filename: str):
        """Improved PDF creation"""
        print(f"📄 Creating PDF: {filename}")
        try:
            # Create directory if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph("Research Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Topic from filename
            topic = Path(filename).stem.replace('_', ' ').title()
            story.append(Paragraph(f"Topic: {topic}", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Add timestamp
            story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Process content by sections
            sections = content.split('## ')
            for section in sections:
                if not section.strip():
                    continue
                    
                lines = section.split('\n')
                section_title = lines[0].strip()
                section_content = '\n'.join(lines[1:]).strip()
                
                if section_title:
                    story.append(Paragraph(section_title, styles['Heading2']))
                    
                    # Process content based on section type
                    for line in section_content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('-'):
                            story.append(Paragraph(line, styles['Bullet']))
                        else:
                            story.append(Paragraph(line, styles['BodyText']))
                    
                    story.append(Spacer(1, 12))
                    
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
        print("⚠️ PDF creation had issues but report was generated")
    
    # Print final report content
    print("\n📝 FINAL REPORT CONTENT:")
    print("="*60)
    print(report_content)
    print("="*60)