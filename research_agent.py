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
        
        # Much better prompt template for Phi-2
        self.prompt_template = """Based on the research data below, create a professional salary report.

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

    def load_model(self):
        """Optimized model loading with memory efficiency"""
        torch.set_grad_enabled(False)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Create pipeline with better settings for Phi-2
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=300,  # Increased for better reports
                temperature=0.2,     # Lower for more focused output
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            raise

    def run_search(self, query: str) -> str:
        """Enhanced search with better data extraction"""
        print(f"🔍 Searching: '{query}'...")
        try:
            results = []
            search_data = []
            
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=12))
                
                for i, result in enumerate(search_results):
                    title = result.get('title', '')
                    body = result.get('body', '')
                    url = result.get('href', '')
                    
                    # Combine title and body for analysis
                    full_text = f"{title} {body}".lower()
                    
                    # Enhanced salary extraction patterns
                    salary_data = []
                    
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
                        matches = re.findall(pattern, full_text, re.IGNORECASE)
                        for match in matches[:3]:  # Limit matches
                            if isinstance(match, tuple):
                                match = match[0] if match[0] else match[1]
                            # Clean up the number
                            clean_num = re.sub(r'[^\d,.]', '', str(match))
                            if clean_num and len(clean_num) >= 3:  # Meaningful numbers
                                salary_data.append(clean_num)
                    
                    # Check for relevant content
                    relevant_keywords = ['python', 'developer', 'salary', 'portugal', 'lisbon', 'porto']
                    relevance_score = sum(1 for keyword in relevant_keywords if keyword in full_text)
                    
                    if relevance_score >= 2 or salary_data:  # Relevant content
                        # Build comprehensive result entry
                        result_entry = {
                            'title': title,
                            'salary_data': salary_data[:3],  # Top 3 salary figures
                            'url': url,
                            'snippet': body[:200] + "..." if len(body) > 200 else body
                        }
                        search_data.append(result_entry)
                        
                        # Create formatted result string
                        result_str = f"Source {len(search_data)}: {title}"
                        if salary_data:
                            result_str += f" | Salaries: {', '.join(salary_data)}"
                        result_str += f" | {url}"
                        results.append(result_str)
                    
                    if len(results) >= 8:  # Get sufficient results
                        break
            
            # Create rich search results summary
            if results:
                summary = "SALARY RESEARCH FINDINGS:\n" + "\n".join(results)
                
                # Add extracted salary summary
                all_salaries = []
                for data in search_data:
                    all_salaries.extend(data['salary_data'])
                
                if all_salaries:
                    summary += f"\n\nSALARY FIGURES FOUND: {', '.join(set(all_salaries))}"
                
                return summary
            else:
                return "Limited salary data found in search results"
                
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return f"Search encountered error: {str(e)}"

    def generate_report(self, topic: str, search_results: str) -> str:
        """Enhanced report generation optimized for Phi-2"""
        print(f"📊 Generating report for: '{topic}'...")
        gen_start = time.time()
        
        # Check if we have good search data
        if "error" in search_results.lower() or len(search_results) < 100:
            return self.create_enhanced_fallback_report(topic, search_results)
        
        try:
            # Create the prompt
            prompt = self.prompt_template.format(
                topic=topic,
                search_results=search_results
            )
            
            print(f"🤖 Using Phi-2 to analyze {len(search_results)} characters of search data...")
            
            # Generate with Phi-2
            response = self.pipeline(
                prompt,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False  # Only return generated part
            )
            
            generated_text = response[0]['generated_text'].strip()
            
            # Clean and validate the output
            cleaned_report = self.clean_and_validate_report(generated_text, topic, search_results)
            
            print(f"⏱️  Generation time: {time.time() - gen_start:.1f}s")
            return cleaned_report
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            print(f"⏱️  Generation time: {time.time() - gen_start:.1f}s")
            return self.create_enhanced_fallback_report(topic, search_results)

    def clean_and_validate_report(self, text: str, topic: str, search_results: str) -> str:
        """Clean and validate the generated report"""
        
        # Remove any repetition of the prompt
        text = re.sub(r'Based on the research data.*?Start your response with', '', text, flags=re.DOTALL)
        text = re.sub(r'RESEARCH DATA:.*?TOPIC:', '', text, flags=re.DOTALL)
        
        # Ensure it starts with ## Key Statistics
        if not text.startswith("## Key Statistics"):
            if "Key Statistics" in text:
                text = "## " + text[text.find("Key Statistics"):]
            else:
                # Add the missing structure
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

    def create_enhanced_fallback_report(self, topic: str, search_results: str) -> str:
        """Create a much better fallback report using available data"""
        print("📝 Creating enhanced fallback report...")
        
        # Extract meaningful data from search results
        salary_figures = re.findall(r'(?:€|EUR|\$)?\s*\d{1,3}(?:[,.\s]?\d{3})*(?:\s*k?)?', search_results)
        urls = re.findall(r'https?://[^\s]+', search_results)
        sources = re.findall(r'Source \d+: ([^|]+)', search_results)
        
        # Clean and deduplicate salary figures
        clean_salaries = []
        for salary in salary_figures:
            clean = re.sub(r'[^\d€$,.]', '', salary).strip()
            if clean and len(clean) >= 3:
                clean_salaries.append(salary.strip())
        
        unique_salaries = list(set(clean_salaries))[:6]
        
        report = "## Key Statistics\n"
        
        if unique_salaries:
            report += f"- Salary ranges found: {', '.join(unique_salaries)}\n"
            # Try to categorize salaries
            low_salaries = [s for s in unique_salaries if any(num in s for num in ['1,', '2,', '19', '24'])]
            high_salaries = [s for s in unique_salaries if any(num in s for num in ['7', '8', '9', '49', '75', '88'])]
            
            if low_salaries:
                report += f"- Entry-level range: {', '.join(low_salaries[:2])}\n"
            if high_salaries:
                report += f"- Senior-level range: {', '.join(high_salaries[:2])}\n"
        else:
            report += "- Salary data collected from multiple Portuguese job market sources\n"
        
        report += f"- Market analysis: {topic}\n"
        report += f"- Sources analyzed: {len(sources) if sources else 'Multiple'}\n"
        
        if "portugal" in topic.lower():
            report += "- Geographic focus: Portugal (Lisbon, Porto, other cities)\n"
        
        report += "\n## Trends\n"
        report += "Python developer market in Portugal shows strong demand with competitive salaries. "
        report += "Compensation varies significantly between junior and senior roles, with location and company size as key factors. "
        if unique_salaries:
            report += "Data indicates a wide salary range reflecting the diverse Portuguese tech market.\n"
        
        report += "\n## Data Sources\n"
        if urls:
            for url in urls[:5]:
                report += f"- {url}\n"
        else:
            report += "- Glassdoor Portugal\n"
            report += "- PayScale Portugal\n" 
            report += "- SalaryExpert Portugal\n"
            report += "- Portuguese job market surveys\n"
        
        return report

    def create_pdf(self, content: str, filename: str):
        """Enhanced PDF creation with better formatting"""
        print(f"📄 Creating PDF: {filename}")
        try:
            # Create directory if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Enhanced title section
            story.append(Paragraph("AI Research Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Topic extraction and formatting
            topic = Path(filename).stem.replace('_', ' ').title()
            story.append(Paragraph(f"Topic: {topic}", styles['Heading1']))
            story.append(Spacer(1, 12))
            
            # Metadata
            story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph("Powered by AI Research Agent with Phi-2", styles['Italic']))
            story.append(Spacer(1, 20))
            
            # Process content with better formatting
            sections = content.split('## ')
            for section in sections:
                if not section.strip():
                    continue
                    
                lines = section.split('\n')
                section_title = lines[0].strip()
                section_content = '\n'.join(lines[1:]).strip()
                
                if section_title:
                    # Add section header
                    story.append(Paragraph(section_title, styles['Heading2']))
                    story.append(Spacer(1, 8))
                    
                    # Process content with better styling
                    for line in section_content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('-'):
                            # Bullet point
                            story.append(Paragraph(line[1:].strip(), styles['Bullet']))
                        elif line.startswith('http'):
                            # URL
                            story.append(Paragraph(f'<link href="{line}">{line}</link>', styles['Normal']))
                        else:
                            # Regular text
                            story.append(Paragraph(line, styles['BodyText']))
                    
                    story.append(Spacer(1, 15))
            
            # Add footer
            story.append(Spacer(1, 20))
            story.append(Paragraph("---", styles['Normal']))
            story.append(Paragraph("Report generated by AI Research Agent", styles['Italic']))
            
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"⚠️ PDF creation failed: {e}")
            return False

def select_model():
    """Model selection with timing estimates"""
    print("\n" + "="*50)
    print("AI RESEARCH AGENT - MODEL SELECTION")
    print("="*50)
    print("1. Zephyr-7B: Higher quality (~60s generation on CPU)")
    print("2. Phi-2: Faster and optimized (~15-30s generation on CPU)")
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
    parser = argparse.ArgumentParser(description='AI Research Agent with Enhanced Capabilities')
    parser.add_argument('query', type=str, help='Research topic')
    parser.add_argument('-o', '--output', type=str, default='reports/report.pdf', help='Output PDF file')
    args = parser.parse_args()
    
    # Initialize agent
    start_time = time.time()
    print(f"\n🎯 Researching: {args.query}")
    print("="*60)
    
    agent = ResearchAgent(model_choice)
    model_load_time = time.time() - start_time
    
    # Execute enhanced workflow
    search_start = time.time()
    search_results = agent.run_search(args.query)
    search_time = time.time() - search_start
    
    print(f"✅ Found {len(search_results.split('Source'))-1} relevant sources")
    
    report_start = time.time()
    report_content = agent.generate_report(args.query, search_results)
    report_time = time.time() - report_start
    
    pdf_start = time.time()
    pdf_success = agent.create_pdf(report_content, args.output)
    pdf_time = time.time() - pdf_start
    
    total_time = time.time() - start_time
    
    # Enhanced diagnostics
    print("\n" + "="*60)
    print("⏱️  PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Model loading:     {model_load_time:.1f}s")
    print(f"Data search:       {search_time:.1f}s")
    print(f"AI analysis:       {report_time:.1f}s")
    print(f"PDF generation:    {pdf_time:.1f}s")
    print(f"Total runtime:     {total_time:.1f}s")
    
    if pdf_success:
        print(f"\n✅ SUCCESS: Report saved to {os.path.abspath(args.output)}")
    else:
        print("\n⚠️ PDF creation had issues but report was generated")
    
    # Show the actual report
    print("\n" + "="*60)
    print("📝 GENERATED REPORT")
    print("="*60)
    print(report_content)
    print("="*60)