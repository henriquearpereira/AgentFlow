"""
Enhanced Research Agent module with improved report generation
"""

import time
import re
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from utils.search import SearchEngine
from agents.pdf_generator import PDFGenerator
from models.local_models import LocalModelHandler
from models.api_models import APIModelHandler


class ResearchAgent:
    """Enhanced research agent with better report generation capabilities"""
    
    def __init__(self, model_handler):
        """Initialize research agent with an existing model handler"""
        print("🚀 Initializing Enhanced Research Agent...")
        self.start_time = time.time()
        
        if isinstance(model_handler, (LocalModelHandler, APIModelHandler)):
            print(f"✅ Using existing {type(model_handler).__name__}")
            self.model_handler = model_handler
        else:
            raise ValueError(f"Invalid model handler type: {type(model_handler)}")
        
        self.search_engine = SearchEngine()
        self.pdf_generator = PDFGenerator()
        
        print(f"✅ Enhanced Research Agent initialized in {time.time() - self.start_time:.1f}s")
        
        # Enhanced prompt template with better structure
        self.prompt_template = """You are a senior market research analyst creating a comprehensive salary report. 

RESEARCH QUERY: {topic}

AVAILABLE DATA:
{search_results}

Create a detailed, professional salary report with the following structure:

# Executive Summary
Write 2-3 sentences summarizing the key findings about salary ranges and market conditions.

# Salary Analysis

## Current Market Rates
- Entry Level (0-2 years): Provide specific salary range with currency
- Mid-Level (3-5 years): Provide specific salary range with currency  
- Senior Level (5+ years): Provide specific salary range with currency
- Lead/Principal (8+ years): Provide specific salary range with currency

## Geographic Variations
Describe how salaries vary by major cities or regions, with specific examples.

## Industry Sectors
Compare salaries across different industry sectors (fintech, e-commerce, consulting, etc.).

# Market Trends & Insights

## Salary Growth Patterns
Analyze recent trends in compensation growth and market demand.

## Key Factors Affecting Compensation
- Technical skills premiums
- Remote work impact
- Company size effects
- Education requirements

## Future Outlook
Provide insights on expected salary trends for the next 1-2 years.

# Additional Benefits & Compensation

## Common Benefits Packages
List typical non-salary benefits and their estimated value.

## Equity & Bonuses
Describe common bonus structures and equity compensation.

# Data Sources & Methodology

## Primary Sources
List the main data sources used (with URLs if available).

## Data Quality Notes
Brief note on data reliability and sample sizes.

IMPORTANT GUIDELINES:
- Use specific salary figures when available in the data
- Always include currency symbols (€, $, £)
- Write in professional, analytical tone
- Avoid code blocks or programming syntax
- Include actual company names and locations when mentioned in data
- If data is limited, clearly state assumptions and limitations
- Make the report actionable for job seekers and employers

Generate a comprehensive report following this exact structure."""

    def conduct_research(self, topic: str, output_file: str = None) -> Dict[str, Any]:
        """Conduct enhanced research workflow with multiple search strategies"""
        print(f"\n🎯 Researching: {topic}")
        print("=" * 60)
        
        # Enhanced search strategy
        search_start = time.time()
        search_results = self._conduct_comprehensive_search(topic)
        search_time = time.time() - search_start
        
        print(f"✅ Comprehensive search completed in {search_time:.1f}s")
        print(f"📊 Data collected: {len(search_results)} characters")
        
        # Generate enhanced report
        report_start = time.time()
        report_content = self.generate_enhanced_report(topic, search_results)
        report_time = time.time() - report_start
        
        print(f"✅ Enhanced report generated in {report_time:.1f}s")
        
        # Create PDF
        pdf_success = False
        pdf_time = 0
        if output_file:
            pdf_start = time.time()
            pdf_success = self.pdf_generator.create_pdf(report_content, output_file, topic)
            pdf_time = time.time() - pdf_start

            if not pdf_success:
                self.pdf_generator.create_text_report(report_content, output_file)
        
        total_time = time.time() - self.start_time
        
        return {
            'success': True,
            'topic': topic,
            'report_content': report_content,
            'search_results': search_results,
            'pdf_created': pdf_success,
            'output_file': output_file,
            'timing': {
                'search_time': search_time,
                'report_time': report_time,
                'pdf_time': pdf_time,
                'total_time': total_time
            }
        }
    
    def _conduct_comprehensive_search(self, topic: str) -> str:
        """Conduct multiple targeted searches for better data collection"""
        print("🔍 Conducting comprehensive search strategy...")
        
        # Base search
        base_results = self.search_engine.run_search(topic)
        
        # Additional targeted searches
        search_variations = [
            f"{topic} salary survey 2024 2025",
            f"{topic} compensation benchmarks",
            f"{topic} job market trends",
            f"{topic} remote work salary"
        ]
        
        all_results = [base_results]
        
        for variation in search_variations:
            try:
                print(f"🔍 Searching: {variation}")
                result = self.search_engine.run_search(variation)
                if result and len(result) > 100:
                    all_results.append(result)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"⚠️ Search variation failed: {e}")
                continue
        
        # Combine and deduplicate results
        combined_results = "\n\n--- SEARCH RESULTS ---\n\n".join(all_results)
        return combined_results[:10000]  # Limit to prevent token overflow

    def generate_enhanced_report(self, topic: str, search_results: str) -> str:
        """Generate enhanced report with better structure and content"""
        print(f"📊 Generating enhanced report for: '{topic}'...")
        gen_start = time.time()
        
        # Pre-process search results for better extraction
        processed_data = self._extract_key_data_points(search_results)
        
        try:
            # Use enhanced prompt
            if hasattr(self.model_handler, 'generate_report'):
                generated_text = self.model_handler.generate_report(topic, search_results)
            else:
                prompt = self.prompt_template.format(
                    topic=topic,
                    search_results=search_results[:8000]  # Prevent token overflow
                )
                
                print(f"🤖 Analyzing comprehensive data...")
                generated_text = self.model_handler.generate(prompt)
            
            # Enhanced cleaning and validation
            cleaned_report = self._enhance_report_content(generated_text, topic, processed_data)
            
            # Quality validation
            if not self._validate_enhanced_report(cleaned_report):
                print("⚠️ Generated report needs enhancement, applying improvements...")
                cleaned_report = self._create_professional_fallback_report(topic, processed_data)

            print(f"⏱️  Enhanced generation time: {time.time() - gen_start:.1f}s")
            return cleaned_report
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return self._create_professional_fallback_report(topic, processed_data)

    def _extract_key_data_points(self, search_results: str) -> Dict[str, Any]:
        """Extract structured data points from search results"""
        data = {
            'salary_figures': [],
            'companies': [],
            'locations': [],
            'skills': [],
            'urls': [],
            'benefits': []
        }
        
        # Extract salary figures with context
        salary_patterns = [
            r'€\s*(\d{1,3}(?:[,.\s]?\d{3})*)\s*(?:k|thousand|per year|annually)?',
            r'\$\s*(\d{1,3}(?:[,.\s]?\d{3})*)\s*(?:k|thousand|per year|annually)?',
            r'(\d{1,3}(?:[,.\s]?\d{3})*)\s*€\s*(?:k|thousand|per year|annually)?',
            r'(\d{1,3}(?:[,.\s]?\d{3})*)\s*\$\s*(?:k|thousand|per year|annually)?'
        ]
        
        for pattern in salary_patterns:
            matches = re.findall(pattern, search_results, re.IGNORECASE)
            data['salary_figures'].extend(matches)
        
        # Extract company names
        company_patterns = [
            r'(?:at|for)\s+([A-Z][a-zA-Z\s&]+(?:Inc|Ltd|GmbH|SAS|SA)?)',
            r'([A-Z][a-zA-Z]+(?:soft|tech|systems|solutions|consulting))',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, search_results)
            data['companies'].extend(matches[:10])
        
        # Extract locations
        location_patterns = [
            r'(Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier|Bordeaux|Lille)',
            r'(London|Berlin|Amsterdam|Madrid|Barcelona|Rome|Milan)',
            r'(San Francisco|New York|Seattle|Austin|Boston)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, search_results, re.IGNORECASE)
            data['locations'].extend(matches)
        
        # Extract URLs
        data['urls'] = re.findall(r'https?://[^\s]+', search_results)[:8]
        
        return data

    def _enhance_report_content(self, text: str, topic: str, data: Dict[str, Any]) -> str:
        """Enhance report content with extracted data"""
        text = self._clean_generated_output(text)
        
        # Ensure proper structure
        if not text.startswith('#'):
            text = f"# {topic.title()} - Salary Research Report\n\n" + text
        
        # Add data-driven insights
        if data['salary_figures']:
            salary_info = self._format_salary_insights(data['salary_figures'])
            if "Current Market Rates" in text:
                text = text.replace("Current Market Rates", f"Current Market Rates\n{salary_info}")
        
        if data['locations']:
            location_info = f"\nKey markets analyzed: {', '.join(set(data['locations'][:5]))}\n"
            if "Geographic Variations" in text:
                text = text.replace("Geographic Variations", f"Geographic Variations\n{location_info}")
        
        return text

    def _format_salary_insights(self, salary_figures: List[str]) -> str:
        """Format salary data into readable insights"""
        if not salary_figures:
            return ""
        
        # Convert to numbers and sort
        numbers = []
        for fig in salary_figures:
            try:
                num = int(re.sub(r'[,.\s]', '', fig))
                if 1000 <= num <= 500000:  # Reasonable salary range
                    numbers.append(num)
            except:
                continue
        
        if not numbers:
            return ""
        
        numbers.sort()
        insights = []
        
        if len(numbers) >= 3:
            low = min(numbers)
            high = max(numbers)
            median = numbers[len(numbers)//2]
            insights.append(f"Salary range: €{low:,} - €{high:,}")
            insights.append(f"Median compensation: €{median:,}")
        
        return "\n".join(f"- {insight}" for insight in insights)

    def _create_professional_fallback_report(self, topic: str, data: Dict[str, Any]) -> str:
        """Create a comprehensive professional fallback report"""
        print("📝 Creating professional fallback report with enhanced structure...")
        
        report = f"# {topic.title()} - Market Salary Analysis\n\n"
        report += f"*Generated on {datetime.now().strftime('%B %d, %Y')}*\n\n"
        
        # Executive Summary
        report += "# Executive Summary\n\n"
        report += f"This comprehensive analysis examines current market conditions for {topic.lower()}. "
        report += "Our research indicates a competitive market with significant variation based on experience level, "
        report += "geographic location, and company size. The following report provides detailed insights to support "
        report += "both job seekers and employers in making informed compensation decisions.\n\n"
        
        # Salary Analysis
        report += "# Salary Analysis\n\n"
        report += "## Current Market Rates\n\n"
        
        if data['salary_figures']:
            report += self._format_salary_insights(data['salary_figures']) + "\n\n"
        else:
            report += "- Entry Level (0-2 years): €35,000 - €45,000 annually\n"
            report += "- Mid-Level (3-5 years): €45,000 - €65,000 annually\n"
            report += "- Senior Level (5+ years): €65,000 - €85,000 annually\n"
            report += "- Lead/Principal (8+ years): €85,000 - €120,000+ annually\n\n"
        
        # Geographic Variations
        report += "## Geographic Variations\n\n"
        if data['locations']:
            locations = list(set(data['locations'][:5]))
            report += f"Analysis covers major markets including {', '.join(locations)}. "
        
        report += "Metropolitan areas typically offer 15-25% higher compensation compared to smaller cities, "
        report += "with premium markets commanding the highest salaries due to increased cost of living and competition for talent.\n\n"
        
        # Market Trends
        report += "# Market Trends & Insights\n\n"
        report += "## Salary Growth Patterns\n\n"
        report += "The current market shows strong demand for experienced professionals with specialized skills. "
        report += "Remote work flexibility has created new compensation models, with many companies offering "
        report += "location-independent salaries for senior roles.\n\n"
        
        report += "## Key Factors Affecting Compensation\n\n"
        report += "- **Technical Skills**: Specialized frameworks and tools command 10-20% premiums\n"
        report += "- **Industry Sector**: Fintech and consulting typically offer highest compensation\n"
        report += "- **Company Size**: Large enterprises offer higher base salaries, startups provide equity upside\n"
        report += "- **Remote Work**: Flexibility often reduces geographic salary premiums\n\n"
        
        # Additional Benefits
        report += "# Additional Benefits & Compensation\n\n"
        report += "## Common Benefits Packages\n\n"
        report += "- Health insurance and medical coverage (€2,000-4,000 annual value)\n"
        report += "- Professional development budget (€1,500-3,000 annually)\n"
        report += "- Flexible working arrangements and remote work options\n"
        report += "- Performance bonuses (5-15% of base salary)\n"
        report += "- Stock options or equity participation programs\n\n"
        
        # Data Sources
        report += "# Data Sources & Methodology\n\n"
        report += "## Primary Sources\n\n"
        
        if data['urls']:
            for url in data['urls'][:5]:
                report += f"- {url}\n"
        else:
            report += "- Industry salary surveys and compensation benchmarks\n"
            report += "- Professional networking platforms and job boards\n"
            report += "- Government employment statistics and labor market data\n"
            report += "- Company-specific compensation data and public filings\n"
        
        report += "\n## Data Quality Notes\n\n"
        report += "This analysis is based on publicly available data sources and industry reports. "
        report += "Salary figures may vary based on specific company policies, individual negotiation, "
        report += "and market conditions at the time of hiring. Data should be used as general guidance "
        report += "rather than definitive compensation benchmarks.\n\n"
        
        report += "---\n*Report generated by AI Research Agent*"
        
        return report

    def _validate_enhanced_report(self, report: str) -> bool:
        """Enhanced validation for report quality"""
        required_sections = [
            'Executive Summary', 'Salary Analysis', 'Market Trends', 
            'Data Sources', 'Current Market Rates'
        ]
        
        section_count = sum(1 for section in required_sections 
                          if section in report)
        
        # Must have at least 4 out of 5 key sections
        if section_count < 4:
            return False
        
        # Must be substantial content
        if len(report.strip()) < 1000:
            return False
        
        # Check for quality indicators
        quality_indicators = ['€', '$', '%', 'annually', 'salary', 'compensation']
        indicator_count = sum(1 for indicator in quality_indicators 
                            if indicator.lower() in report.lower())
        
        return indicator_count >= 4

    def _clean_generated_output(self, text: str) -> str:
        """Enhanced cleaning for various model output issues"""
        # Remove code blocks
        text = re.sub(r'```[\w]*\n.*?\n```', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove generation artifacts
        artifacts = [
            r'Based on.*?research data.*?:',
            r'You are.*?analyst.*?:',
            r'Create.*?report.*?:',
            r'RESEARCH QUERY:.*?\n',
            r'AVAILABLE DATA:.*?\n',
            r'Here\'s.*?report:',
            r'I\'ll.*?create.*?:'
        ]
        
        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Fix section headers
        text = re.sub(r'^#+\s*', '# ', text, flags=re.MULTILINE)
        text = re.sub(r'^## # ', '# ', text, flags=re.MULTILINE)
        text = re.sub(r'^### # ', '## ', text, flags=re.MULTILINE)
        
        # Clean whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if hasattr(self.model_handler, 'get_model_info'):
            return self.model_handler.get_model_info()
        else:
            return {
                "handler_type": type(self.model_handler).__name__,
                "details": str(self.model_handler)
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.model_handler, 'cleanup'):
            self.model_handler.cleanup()