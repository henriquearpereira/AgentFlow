# research_agent.py
import argparse
import os
import textwrap
import time
import torch
from pathlib import Path
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class ResearchAgent:
    def __init__(self):
        print("🚀 Initializing Research Agent...")
        self.start_time = time.time()
        
        # Model configuration
        self.model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
        self.load_model()
        print(f"✅ Model loaded in {time.time() - self.start_time:.1f}s")
        
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a senior research analyst. Create a professional report with:\n"
             "1. Executive Summary\n2. Key Findings\n3. Statistical Data\n4. Recommendations\n"
             "5. Sources\n\nFormat using markdown-style headers and bullet points."),
            ("human", "Research topic: {topic}\n\nSearch results:\n{search_results}")
        ])

    def load_model(self):
        """Load model with memory optimization for Windows"""
        try:
            # Try loading with GPU if available
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto"
            )
        except:
            # Fallback to CPU-only
            print("⚠️ GPU not available, using CPU")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.7
        )

    def run_search(self, query: str) -> str:
        """Get search results using DuckDuckGo"""
        print(f"🔍 Searching: '{query}'...")
        from duckduckgo_search import DDGS
        try:
            results = []
            ddgs = DDGS()
            for result in ddgs.text(query, max_results=5):
                results.append(result['body'])
            return "\n\n".join(results)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return "No search results available."

    def generate_report(self, topic: str, search_results: str) -> str:
        """Generate research report"""
        messages = self.prompt_template.format_messages(topic=topic, search_results=search_results)
        prompt_text = "\n".join([m.content for m in messages])
        
        print("🧠 Generating report...")
        response = self.pipeline(
            prompt_text,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return response[0]['generated_text'].split(prompt_text)[-1].strip()

    def create_pdf(self, content: str, filename: str):
        """Generate PDF report"""
        print(f"📄 Creating PDF: {filename}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Research Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Content
        for section in content.split('## '):
            if section.strip():
                header, *body = section.split('\n', 1)
                story.append(Paragraph(header, styles['Heading2']))
                if body:
                    wrapped = textwrap.fill(body[0], width=80)
                    story.append(Paragraph(wrapped, styles['BodyText']))
                story.append(Spacer(1, 12))
                
        doc.build(story)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Research Agent')
    parser.add_argument('query', type=str, help='Research topic')
    parser.add_argument('-o', '--output', type=str, default='reports/report.pdf', help='Output PDF file')
    args = parser.parse_args()
    
    agent = ResearchAgent()
    
    # Execute workflow
    search_results = agent.run_search(args.query)
    report_content = agent.generate_report(args.query, search_results)
    agent.create_pdf(report_content, args.output)
    
    total_time = time.time() - agent.start_time
    print(f"✅ Research completed in {total_time:.1f}s")
    print(f"📁 Output saved to: {os.path.abspath(args.output)}")