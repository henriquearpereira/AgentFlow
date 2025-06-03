import os
# Set Hugging Face cache to D: drive with raw string
os.environ['HF_HOME'] = r'D:\.cache\huggingface'  # Fixed escape sequence
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

import argparse
import textwrap
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
        
        # Enhanced prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a data analyst creating executive briefings. Follow these rules:\n"
             "1. Extract ONLY numerical data, statistics, and quantifiable facts\n"
             "2. Ignore marketing fluff, opinions, and repeated information\n"
             "3. Structure in these sections:\n"
             "   - Key Statistics (bulleted numbers only)\n"
             "   - Trends (1-2 sentences)\n"
             "   - Data Sources (list URLs)\n"
             "4. ABSOLUTELY NO: introductions, conclusions, or filler text\n"
             "5. MAX 150 words total\n"
             "6. If data is insufficient, state: 'Insufficient reliable data found'"),
            ("human", "TOPIC: {topic}\n\nSEARCH RESULTS:\n{search_results}")
        ])

    def load_model(self):
        """Load model with memory optimization"""
        torch.set_grad_enabled(False)  # Reduce memory usage
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype="auto" if torch.cuda.is_available() else torch.float32
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=300,  # Shorter output
                temperature=0.3,     # Less random
                top_p=0.9,           # Focus on high probability words
                repetition_penalty=1.5  # Discourage repetition
            )
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            raise

    def run_search(self, query: str) -> str:
        """Get deduplicated and condensed search results"""
        print(f"🔍 Searching: '{query}'...")
        try:
            seen_domains = set()
            unique_results = []
            
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=10)):
                    # Skip duplicate domains
                    domain = result['href'].split('/')[2]
                    if domain in seen_domains:
                        continue
                    seen_domains.add(domain)
                    
                    # Extract key facts
                    body = result['body']
                    
                    # Find salary numbers, percentages, years
                    key_facts = []
                    if "salary" in query.lower():
                        salary_matches = re.findall(r'(\€?\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?\s?(?:per year|per month|/yr|/mo|€))', body, re.IGNORECASE)
                        if salary_matches:
                            key_facts.extend(salary_matches[:3])  # Max 3 salary matches
                    
                    # Find other numerical data
                    number_matches = re.findall(r'\b\d{2,4}\b', body)  # Find 2-4 digit numbers
                    if number_matches:
                        key_facts.extend(number_matches[:2])  # Max 2 other numbers
                    
                    # If no numbers found, use first meaningful sentence
                    if not key_facts:
                        sentences = [s.strip() for s in body.split('.') if s.strip() and len(s.split()) > 5]
                        if sentences:
                            key_facts.append(sentences[0][:120] + "...")
                    
                    # Build result snippet
                    snippet = f"[[Source {i+1}]] " + "; ".join(set(key_facts)) + f" ({result['href']})"
                    unique_results.append(snippet)
                    
                    # Stop when we have 5 high-quality sources
                    if len(unique_results) >= 5:
                        break
                        
            return "\n".join(unique_results)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return "No search results available."

    def generate_report(self, topic: str, search_results: str) -> str:
        """Generate concise research report"""
        attempts = 0
        best_report = ""
        
        while attempts < 2:  # Reduced to 2 attempts for speed
            messages = self.prompt_template.format_messages(topic=topic, search_results=search_results)
            prompt_text = "\n".join([m.content for m in messages])
            
            # Generate response
            response = self.pipeline(
                prompt_text,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=500  # Prevent excessively long outputs
            )
            
            # Extract new text
            full_text = response[0]['generated_text']
            generated_text = full_text.split(prompt_text)[-1].strip()
            
            # Post-process to enforce conciseness
            clean_report = self.clean_report(generated_text)
            
            # Quality check - keep best attempt
            if self.is_high_quality(clean_report) and (not best_report or len(clean_report) < len(best_report)):
                best_report = clean_report
                
            attempts += 1
        
        return best_report or "⚠️ Unable to generate quality report. Try refining your query."

    def clean_report(self, text: str) -> str:
        """Enforce strict formatting rules and remove repetition"""
        # Remove repetitive sentences using sentence hashing
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        unique_sentences = []
        seen = set()
        
        for s in sentences:
            # Simple hash: first 3 words
            words = s.split()
            if len(words) < 4:
                continue
            key = " ".join(words[:3]).lower()
            
            if key not in seen:
                seen.add(key)
                unique_sentences.append(s)
        
        # Rebuild with enforced structure
        report = ""
        stats_section = []
        trends_section = []
        sources_section = []
        
        for s in unique_sentences:
            if "key statistic" in s.lower():
                # Extract bullet points
                bullets = re.findall(r'[-•]\s*(.*?)(?=\n|$)', s)
                if bullets:
                    stats_section.extend(bullets)
            elif "trend" in s.lower():
                trends_section.append(s)
            elif "source" in s.lower():
                # Extract URLs
                urls = re.findall(r'https?://[^\s\)]+', s)
                if urls:
                    sources_section.extend(urls)
            elif any(char.isdigit() for char in s) and len(s) < 100:
                stats_section.append(s)
            elif "http" in s:
                sources_section.append(s)
        
        # Build structured report
        if stats_section:
            report += "## Key Statistics\n"
            report += "\n".join([f"- {stat}" for stat in set(stats_section)[:5]]) + "\n\n"
        
        if trends_section:
            report += "## Trends\n"
            report += " ".join(set(trends_section)[:2]) + "\n\n"
        
        if sources_section:
            report += "## Data Sources\n"
            report += "\n".join([f"- {src}" for src in set(sources_section)[:3]])
        
        # If no structure found, create minimal version
        if not report:
            stats = [s for s in unique_sentences if any(char.isdigit() for char in s)][:3]
            sources = [s for s in unique_sentences if "http" in s][:2]
            report = "## Key Statistics\n- " + "\n- ".join(stats) 
            if sources:
                report += "\n\n## Data Sources\n- " + "\n- ".join(sources)
        
        return report.strip()[:600]  # Hard character limit

    def is_high_quality(self, report: str) -> bool:
        """Check for essential quality markers"""
        if "insufficient" in report.lower() or "unable" in report.lower():
            return False
            
        has_numbers = any(char.isdigit() for char in report)
        has_sources = "http" in report or "www." in report
        not_too_short = len(report.split()) > 15
        not_repetitive = len(set(report.split())) > len(report.split()) * 0.5
        
        return has_numbers and has_sources and not_too_short and not_repetitive

    def create_pdf(self, content: str, filename: str):
        """Generate PDF report with optimized formatting"""
        print(f"📄 Creating PDF: {filename}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Research Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Special formatting for sections
        for line in content.split('\n'):
            if line.startswith('## '):
                story.append(Paragraph(line[3:], styles['Heading2']))
            elif line.startswith('- ') and ('http' in line or 'www.' in line):
                story.append(Paragraph(line[2:], styles['Italic']))
            elif line.startswith('- '):
                story.append(Paragraph(line, styles['Bullet']))
            else:
                story.append(Paragraph(line, styles['BodyText']))
            
            story.append(Spacer(1, 4))
                
        doc.build(story)

def select_model():
    """User-friendly model selection"""
    print("\n" + "="*50)
    print("MODEL SELECTION")
    print("="*50)
    print("1. Zephyr-7B-Beta: Better quality but slower (7GB)")
    print("2. Phi-2: Faster and lighter but simpler outputs (2.7GB)")
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
    agent = ResearchAgent(model_choice)
    
    # Execute workflow
    search_results = agent.run_search(args.query)
    report_content = agent.generate_report(args.query, search_results)
    agent.create_pdf(report_content, args.output)
    
    total_time = time.time() - agent.start_time
    print(f"✅ Research completed in {total_time:.1f}s")
    print(f"📁 Output saved to: {os.path.abspath(args.output)}")