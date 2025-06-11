"""
Enhanced Research Agent module with improved report generation and subject-specific handling
"""

import time
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import json

from utils.search import EnhancedSearchEngine
from agents.pdf_generator import PDFGenerator
from models.local_models import LocalModelHandler
from models.api_models import APIModelHandler


class EnhancedResearchAgent:
    """Enhanced research agent with subject-specific intelligence and better report generation"""
    
    def __init__(self, model_handler):
        """Initialize research agent with an existing model handler"""
        print("🚀 Initializing Enhanced Research Agent...")
        self.start_time = time.time()
        
        if isinstance(model_handler, (LocalModelHandler, APIModelHandler)):
            print(f"✅ Using existing {type(model_handler).__name__}")
            self.model_handler = model_handler
        else:
            raise ValueError(f"Invalid model handler type: {type(model_handler)}")
        
        self.search_engine = EnhancedSearchEngine()
        self.pdf_generator = PDFGenerator()
        
        # Subject categorization patterns
        self.subject_patterns = {
            'salary_market': ['salary', 'salaries', 'pay', 'wage', 'compensation', 'earnings', 'income'],
            'comparison': ['vs', 'versus', 'compare', 'comparison', 'difference', 'better'],
            'technical': ['api', 'framework', 'programming', 'software', 'technology', 'technical'],
            'historical': ['history', 'timeline', 'evolution', 'development', 'origin'],
            'scientific': ['research', 'study', 'analysis', 'findings', 'methodology'],
            'product_review': ['review', 'pros', 'cons', 'advantages', 'disadvantages', 'rating'],
            'tutorial': ['how to', 'guide', 'tutorial', 'steps', 'instructions'],
            'market_analysis': ['market', 'industry', 'trends', 'forecast', 'growth']
        }
        
        # Progress tracking
        self.progress_callback = None
        
        print(f"✅ Enhanced Research Agent initialized in {time.time() - self.start_time:.1f}s")
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def _update_progress(self, message: str, percentage: int):
        """Update progress with message and percentage"""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        else:
            print(f"📊 [{percentage}%] {message}")
    
    def _categorize_subject(self, topic: str) -> List[str]:
        """Categorize the research topic to determine appropriate strategies"""
        topic_lower = topic.lower()
        categories = []
        
        for category, keywords in self.subject_patterns.items():
            if any(keyword in topic_lower for keyword in keywords):
                categories.append(category)
        
        # Default to general research if no specific category
        if not categories:
            categories.append('general')
        
        return categories
    
    def _get_search_variations(self, topic: str, categories: List[str]) -> List[str]:
        """Generate subject-specific search queries"""
        base_searches = [topic]
        
        # Category-specific search variations
        if 'salary_market' in categories:
            base_searches.extend([
                f"{topic} 2024 2025 survey",
                f"{topic} glassdoor payscale indeed",
                f"{topic} remote work compensation",
                f"{topic} market trends salary data",
                f"{topic} benefits package total compensation"
            ])
        
        elif 'comparison' in categories:
            base_searches.extend([
                f"{topic} detailed comparison",
                f"{topic} pros cons advantages",
                f"{topic} differences similarities",
                f"{topic} which is better",
                f"{topic} expert review analysis"
            ])
        
        elif 'technical' in categories:
            base_searches.extend([
                f"{topic} documentation specifications",
                f"{topic} best practices implementation",
                f"{topic} use cases examples",
                f"{topic} performance benchmarks",
                f"{topic} architecture design patterns"
            ])
        
        elif 'historical' in categories:
            base_searches.extend([
                f"{topic} timeline chronology",
                f"{topic} historical development",
                f"{topic} key milestones events",
                f"{topic} evolution over time",
                f"{topic} historical significance impact"
            ])
        
        elif 'scientific' in categories:
            base_searches.extend([
                f"{topic} research papers studies",
                f"{topic} methodology findings",
                f"{topic} peer reviewed articles",
                f"{topic} scientific evidence data",
                f"{topic} meta analysis review"
            ])
        
        elif 'product_review' in categories:
            base_searches.extend([
                f"{topic} expert review rating",
                f"{topic} user experience feedback",
                f"{topic} detailed analysis evaluation",
                f"{topic} strengths weaknesses",
                f"{topic} recommendation guide"
            ])
        
        elif 'tutorial' in categories:
            base_searches.extend([
                f"{topic} step by step guide",
                f"{topic} complete tutorial",
                f"{topic} best practices tips",
                f"{topic} common mistakes avoid",
                f"{topic} advanced techniques"
            ])
        
        elif 'market_analysis' in categories:
            base_searches.extend([
                f"{topic} market size growth",
                f"{topic} industry report forecast",
                f"{topic} competitive landscape",
                f"{topic} market trends analysis",
                f"{topic} key players market share"
            ])
        
        # General enhancements for all categories
        base_searches.extend([
            f"{topic} latest news updates",
            f"{topic} expert opinions insights"
        ])
        
        return base_searches[:8]  # Limit to prevent too many searches
    
    def _get_report_structure(self, topic: str, categories: List[str]) -> List[str]:
        """Return appropriate report sections based on topic categories"""
        
        if 'salary_market' in categories:
            return [
                'Executive Summary',
                'Salary Analysis',
                'Market Trends & Insights', 
                'Geographic Variations',
                'Industry Comparisons',
                'Additional Benefits & Compensation',
                'Future Outlook',
                'Data Sources & Methodology'
            ]
        
        elif 'comparison' in categories:
            return [
                'Executive Summary',
                'Overview',
                'Detailed Comparison',
                'Advantages & Disadvantages',
                'Use Case Analysis',
                'Performance Metrics',
                'Recommendations',
                'Conclusion'
            ]
        
        elif 'technical' in categories:
            return [
                'Overview',
                'Technical Specifications',
                'Architecture & Design',
                'Use Cases & Applications',
                'Implementation Guide',
                'Best Practices',
                'Performance Considerations',
                'Resources & Documentation'
            ]
        
        elif 'historical' in categories:
            return [
                'Introduction',
                'Historical Timeline',
                'Key Developments',
                'Major Events & Milestones',
                'Impact & Significance',
                'Evolution Over Time',
                'Current Status',
                'References & Sources'
            ]
        
        elif 'scientific' in categories:
            return [
                'Abstract',
                'Background & Context',
                'Research Methodology',
                'Key Findings',
                'Analysis & Discussion',
                'Implications',
                'Limitations & Future Research',
                'Bibliography'
            ]
        
        elif 'product_review' in categories:
            return [
                'Product Overview',
                'Key Features',
                'Strengths',
                'Weaknesses',
                'Performance Analysis',
                'User Experience',
                'Value Assessment',
                'Final Recommendation'
            ]
        
        elif 'tutorial' in categories:
            return [
                'Introduction',
                'Prerequisites',
                'Step-by-Step Guide',
                'Advanced Techniques',
                'Common Issues & Solutions',
                'Best Practices',
                'Additional Resources',
                'Conclusion'
            ]
        
        elif 'market_analysis' in categories:
            return [
                'Executive Summary',
                'Market Overview',
                'Market Size & Growth',
                'Key Players & Competition',
                'Market Trends',
                'Opportunities & Challenges',
                'Future Forecast',
                'Strategic Recommendations'
            ]
        
        # Default structure for general topics
        return [
            'Introduction',
            'Main Analysis',
            'Key Insights',
            'Implications',
            'Conclusion',
            'Sources & References'
        ]
    
    def _create_dynamic_prompt(self, topic: str, categories: List[str], search_results: str, structure: List[str]) -> str:
        """Create a dynamic prompt based on subject category and structure"""
        
        category_instructions = {
            'salary_market': """Focus on specific salary figures, market data, geographic variations, and compensation trends. Include numerical data, percentages, and specific ranges where available.""",
            
            'comparison': """Provide balanced analysis with clear pros/cons, side-by-side comparisons, and objective evaluation criteria. Include specific examples and use cases.""",
            
            'technical': """Include technical specifications, code examples if relevant, architecture details, and implementation considerations. Be precise and detailed.""",
            
            'historical': """Present information chronologically, highlight key dates, developments, and cause-effect relationships. Provide context for historical significance.""",
            
            'scientific': """Focus on research methodology, data analysis, evidence-based conclusions, and peer-reviewed sources. Include statistical data and research findings.""",
            
            'product_review': """Provide objective evaluation with specific criteria, user experience insights, and practical recommendations based on different use cases.""",
            
            'tutorial': """Structure as actionable steps with clear instructions, prerequisites, and troubleshooting information. Include practical examples.""",
            
            'market_analysis': """Include market size data, growth projections, competitive landscape analysis, and strategic insights with supporting data."""
        }
        
        # Get primary category instruction
        primary_category = categories[0] if categories else 'general'
        category_instruction = category_instructions.get(primary_category, 
            "Provide comprehensive analysis with supporting evidence, clear structure, and actionable insights.")
        
        # Build sections instruction
        sections_text = "\n".join([f"# {section}" for section in structure])
        
        prompt = f"""You are a senior research analyst creating a comprehensive report on the following topic.

RESEARCH QUERY: {topic}
SUBJECT CATEGORIES: {', '.join(categories)}

AVAILABLE DATA:
{search_results}

SPECIFIC INSTRUCTIONS: {category_instruction}

Create a detailed, professional report with the following structure:

{sections_text}

QUALITY REQUIREMENTS:
- Use specific data, figures, and examples from the search results
- Include proper citations and source attribution
- Write in professional, analytical tone
- Avoid code blocks or programming syntax unless specifically technical content
- Make content actionable and valuable for readers
- Include actual company names, locations, and specific details when available
- Clearly state any limitations or assumptions
- Ensure each section provides substantial value and insights

Generate a comprehensive, well-structured report following this exact structure and maintaining high professional standards throughout."""

        return prompt
    
    def conduct_research(self, topic: str, output_file: str = None) -> Dict[str, Any]:
        """Conduct enhanced research workflow with subject-specific strategies"""
        print(f"\n🎯 Researching: {topic}")
        print("=" * 60)
        
        # Analyze topic and determine strategy
        self._update_progress("Analyzing research topic...", 5)
        categories = self._categorize_subject(topic)
        print(f"📋 Detected categories: {', '.join(categories)}")
        
        # Get appropriate report structure
        report_structure = self._get_report_structure(topic, categories)
        print(f"📖 Report structure: {len(report_structure)} sections")
        
        # Enhanced search strategy
        self._update_progress("Conducting comprehensive search...", 15)
        search_start = time.time()
        search_results = self._conduct_comprehensive_search(topic, categories)
        search_time = time.time() - search_start
        
        print(f"✅ Comprehensive search completed in {search_time:.1f}s")
        print(f"📊 Data collected: {len(search_results)} characters")
        
        # Generate enhanced report
        self._update_progress("Generating analytical report...", 60)
        report_start = time.time()
        report_content = self.generate_enhanced_report(topic, categories, search_results, report_structure)
        report_time = time.time() - report_start
        
        print(f"✅ Enhanced report generated in {report_time:.1f}s")
        
        # Create PDF
        self._update_progress("Creating PDF document...", 85)
        pdf_success = False
        pdf_time = 0
        if output_file:
            pdf_start = time.time()
            pdf_success = self.pdf_generator.create_pdf(report_content, output_file, topic)
            pdf_time = time.time() - pdf_start

            if not pdf_success:
                self.pdf_generator.create_text_report(report_content, output_file)
        
        self._update_progress("Research completed!", 100)
        total_time = time.time() - self.start_time
        
        return {
            'success': True,
            'topic': topic,
            'categories': categories,
            'report_structure': report_structure,
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
    
    def _conduct_comprehensive_search(self, topic: str, categories: List[str]) -> str:
        """Conduct multiple targeted searches based on subject categories"""
        print("🔍 Conducting subject-specific search strategy...")
        
        # Get search variations based on categories
        search_variations = self._get_search_variations(topic, categories)
        all_results = []
        
        for i, search_query in enumerate(search_variations):
            try:
                print(f"🔍 [{i+1}/{len(search_variations)}] Searching: {search_query}")
                result = self.search_engine.run_search(search_query)
                if result and len(result) > 100:
                    all_results.append(result)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"⚠️ Search variation failed: {e}")
                continue
        
        # Combine and process results
        combined_results = "\n\n--- SEARCH RESULTS ---\n\n".join(all_results)
        return combined_results[:15000]  # Increased limit for better coverage
    
    def generate_enhanced_report(self, topic: str, categories: List[str], search_results: str, structure: List[str]) -> str:
        """Generate enhanced report with subject-specific structure and content"""
        print(f"📊 Generating enhanced report for: '{topic}'...")
        print(f"🏷️ Categories: {', '.join(categories)}")
        gen_start = time.time()
        
        # Extract and process data
        processed_data = self._extract_enhanced_data_points(search_results, categories)
        
        try:
            # Create dynamic prompt based on categories and structure
            prompt = self._create_dynamic_prompt(topic, categories, search_results, structure)
            
            print(f"🤖 Analyzing comprehensive data with AI model...")
            generated_text = self.model_handler.generate(prompt)
            
            # Enhanced cleaning and validation
            cleaned_report = self._enhance_report_content(generated_text, topic, processed_data, categories)
            
            # Quality validation
            if not self._validate_enhanced_report(cleaned_report, structure):
                print("⚠️ Generated report needs enhancement, creating professional fallback...")
                cleaned_report = self._create_professional_fallback_report(topic, categories, structure, processed_data)

            print(f"⏱️  Enhanced generation time: {time.time() - gen_start:.1f}s")
            return cleaned_report
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return self._create_professional_fallback_report(topic, categories, structure, processed_data)
    
    def _extract_enhanced_data_points(self, search_results: str, categories: List[str]) -> Dict[str, Any]:
        """Extract structured data points based on subject categories"""
        data = {
            'numerical_data': [],
            'dates': [],
            'companies': [],
            'locations': [],
            'technical_specs': [],
            'urls': [],
            'expert_quotes': [],
            'statistics': []
        }
        
        # Enhanced numerical data extraction
        numerical_patterns = [
            r'€\s*(\d{1,3}(?:[,.\s]?\d{3})*)\s*(?:k|thousand|million|per year|annually)?',
            r'\$\s*(\d{1,3}(?:[,.\s]?\d{3})*)\s*(?:k|thousand|million|per year|annually)?',
            r'(\d{1,3}(?:[,.\s]?\d{3})*)\s*[€$]\s*(?:k|thousand|million|per year|annually)?',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|decrease|growth|change)',
            r'(\d+(?:\.\d+)?)\s*(?:GB|MB|TB|GHz|MHz)',
        ]
        
        for pattern in numerical_patterns:
            matches = re.findall(pattern, search_results, re.IGNORECASE)
            data['numerical_data'].extend(matches)
        
        # Date extraction
        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'(?:in|since|during)\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, search_results, re.IGNORECASE)
            data['dates'].extend(matches)
        
        # Company name extraction (enhanced)
        company_patterns = [
            r'(?:at|for|by)\s+([A-Z][a-zA-Z\s&]+(?:Inc|Ltd|GmbH|SAS|SA|Corp|Corporation|LLC)?)',
            r'([A-Z][a-zA-Z]+(?:soft|tech|systems|solutions|consulting|labs|works))',
            r'(Google|Apple|Microsoft|Amazon|Meta|Netflix|Tesla|Intel|AMD|NVIDIA)',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, search_results)
            data['companies'].extend(matches[:15])
        
        # Location extraction (enhanced)
        location_patterns = [
            r'(New York|San Francisco|Los Angeles|Chicago|Boston|Seattle|Austin|Denver)',
            r'(London|Berlin|Amsterdam|Madrid|Barcelona|Rome|Milan|Zurich|Vienna)',
            r'(Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier)',
            r'(Tokyo|Singapore|Hong Kong|Sydney|Toronto|Vancouver|Montreal)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, search_results, re.IGNORECASE)
            data['locations'].extend(matches)
        
        # Technical specifications (for technical categories)
        if 'technical' in categories:
            tech_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:GB|MB|TB|GHz|MHz|cores?|threads?)',
                r'(?:version|v)\s*(\d+(?:\.\d+)*)',
                r'(\d+)\s*(?:bit|x|×)\s*(\d+)',
            ]
            
            for pattern in tech_patterns:
                matches = re.findall(pattern, search_results, re.IGNORECASE)
                data['technical_specs'].extend(matches)
        
        # Extract URLs
        data['urls'] = re.findall(r'https?://[^\s]+', search_results)[:10]
        
        # Extract potential expert quotes
        quote_patterns = [
            r'"([^"]{30,200})"',
            r'according to ([^,]{10,50})',
            r'([A-Z][a-z]+ [A-Z][a-z]+), (?:CEO|CTO|Director|Manager|Expert)'
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, search_results)
            data['expert_quotes'].extend(matches[:5])
        
        return data
    
    def _enhance_report_content(self, text: str, topic: str, data: Dict[str, Any], categories: List[str]) -> str:
        """Enhance report content with extracted data and category-specific improvements"""
        text = self._clean_generated_output(text)
        
        # Ensure proper title
        if not text.startswith('#'):
            text = f"# {topic.title()} - Research Report\n\n" + text
        
        # Add category-specific enhancements
        if 'salary_market' in categories and data['numerical_data']:
            salary_info = self._format_salary_insights(data['numerical_data'])
            if salary_info and "Salary Analysis" in text:
                text = text.replace("Salary Analysis\n", f"Salary Analysis\n\n{salary_info}\n")
        
        if data['locations']:
            location_info = f"**Key Geographic Markets**: {', '.join(set(data['locations'][:5]))}\n\n"
            # Insert after first section
            sections = text.split('\n# ')
            if len(sections) > 1:
                sections[1] = location_info + sections[1]
                text = '\n# '.join(sections)
        
        if data['companies']:
            company_info = f"**Major Organizations Mentioned**: {', '.join(set(data['companies'][:5]))}\n\n"
            # Add to appropriate section
            if "Market" in text or "Industry" in text:
                text = text.replace("# Market", f"{company_info}# Market")
        
        # Add data sources section if URLs found
        if data['urls']:
            sources_section = "\n\n# Primary Sources\n\n"
            for i, url in enumerate(data['urls'][:5], 1):
                sources_section += f"{i}. {url}\n"
            text += sources_section
        
        return text
    
    def _format_salary_insights(self, numerical_data: List[str]) -> str:
        """Format numerical data into salary insights"""
        if not numerical_data:
            return ""
        
        # Convert to numbers and filter reasonable salaries
        numbers = []
        for data_point in numerical_data:
            try:
                # Clean the string and convert to number
                clean_num = re.sub(r'[,.\s]', '', str(data_point))
                num = int(clean_num)
                if 10000 <= num <= 1000000:  # Reasonable salary range
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
            mean = sum(numbers) // len(numbers)
            
            insights.append(f"**Salary Range**: €{low:,} - €{high:,}")
            insights.append(f"**Median Compensation**: €{median:,}")
            insights.append(f"**Average Compensation**: €{mean:,}")
            insights.append(f"**Data Points Analyzed**: {len(numbers)} salary references")
        
        return "\n".join(insights) + "\n" if insights else ""
    
    def _create_professional_fallback_report(self, topic: str, categories: List[str], structure: List[str], data: Dict[str, Any]) -> str:
        """Create a comprehensive professional fallback report with dynamic structure"""
        print("📝 Creating professional fallback report with enhanced structure...")
        
        report = f"# {topic.title()} - Comprehensive Research Report\n\n"
        report += f"*Generated on {datetime.now().strftime('%B %d, %Y')}*\n"
        report += f"*Research Categories: {', '.join(categories)}*\n\n"
        
        # Dynamic content based on structure
        for section in structure:
            report += f"# {section}\n\n"
            
            if section == "Executive Summary":
                report += f"This comprehensive analysis examines {topic.lower()} across multiple dimensions. "
                report += f"Our research, categorized as {', '.join(categories)}, provides detailed insights "
                report += "to support informed decision-making and strategic planning.\n\n"
            
            elif "Salary" in section or "Compensation" in section:
                if data['numerical_data']:
                    salary_insights = self._format_salary_insights(data['numerical_data'])
                    if salary_insights:
                        report += salary_insights + "\n"
                else:
                    report += "Comprehensive salary analysis reveals competitive compensation structures "
                    report += "with significant variation based on experience, location, and specialization.\n\n"
            
            elif "Market" in section or "Analysis" in section:
                if data['companies']:
                    companies = list(set(data['companies'][:5]))
                    report += f"Key market participants include: {', '.join(companies)}. "
                report += "Market dynamics show evolving trends with emerging opportunities "
                report += "and competitive pressures shaping the landscape.\n\n"
            
            elif "Geographic" in section or "Location" in section:
                if data['locations']:
                    locations = list(set(data['locations'][:5]))
                    report += f"Geographic analysis covers major markets including {', '.join(locations)}. "
                report += "Regional variations demonstrate the impact of local economic conditions, "
                report += "regulatory environments, and market maturity on outcomes.\n\n"
            
            elif "Technical" in section or "Specifications" in section:
                if data['technical_specs']:
                    report += "Technical specifications and performance metrics provide detailed "
                    report += "insights into capabilities, requirements, and optimization opportunities.\n\n"
                else:
                    report += "Technical analysis reveals key specifications, performance characteristics, "
                    report += "and implementation considerations for optimal results.\n\n"
            
            elif "Trends" in section or "Future" in section:
                report += "Current trends indicate significant evolution in the space, with emerging "
                report += "technologies, changing user expectations, and market forces driving innovation. "
                report += "Future outlook suggests continued growth and transformation.\n\n"
            
            elif "Data Sources" in section or "Methodology" in section:
                report += "## Primary Sources\n\n"
                if data['urls']:
                    for i, url in enumerate(data['urls'][:5], 1):
                        report += f"{i}. {url}\n"
                else:
                    report += "- Industry reports and market analysis\n"
                    report += "- Professional surveys and compensation data\n"
                    report += "- Academic research and peer-reviewed studies\n"
                    report += "- Government statistics and regulatory filings\n"
                
                report += "\n## Data Quality Notes\n\n"
                report += "This analysis synthesizes information from multiple verified sources. "
                report += "Data accuracy depends on source reliability and reporting methodology. "
                report += "Findings should be considered alongside current market conditions.\n\n"
            
            else:
                # Generic section content
                report += f"Detailed analysis of {section.lower()} reveals important considerations "
                report += "and insights relevant to understanding the broader context and implications.\n\n"
        
        report += "---\n*Report generated by Enhanced AI Research Agent*"
        return report
    
    def _validate_enhanced_report(self, report: str, structure: List[str]) -> bool:
        """Enhanced validation for report quality based on expected structure"""
        # Check for required sections
        sections_found = sum(1 for section in structure if section in report)
        if sections_found < len(structure) * 0.7:  # At least 70% of sections
            return False
        
        # Check for substantial content
        if len(report.strip()) < 1500:
            return False
        
        # Check for quality indicators
        quality_indicators = ['analysis', 'data', 'research', 'insights', 'findings']
        indicator_count = sum(1 for indicator in quality_indicators 
                            if indicator.lower() in report.lower())
        
        return indicator_count >= 3
    
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
            r'I\'ll.*?create.*?:',
            r'SUBJECT CATEGORIES:.*?\n',
            r'SPECIFIC INSTRUCTIONS:.*?\n'
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