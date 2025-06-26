"""
Advanced Prompt Management module for the Enhanced Research Agent
Contains dynamic, context-aware prompting functions for comprehensive research reports
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class ReportType(Enum):
    """Enumeration of different report types"""
    SCIENTIFIC_TECHNICAL = "scientific_technical"
    BUSINESS_FINANCIAL = "business_financial"
    POLICY_REGULATORY = "policy_regulatory"
    CULTURAL_SOCIAL = "cultural_social"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TREND_ANALYSIS = "trend_analysis"
    CASE_STUDY = "case_study"
    HISTORICAL = "historical"
    GENERAL = "general"


class DomainType(Enum):
    """Enumeration of different domain types"""
    STEM = "stem"
    BUSINESS = "business"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    GOVERNMENT = "government"
    LEGAL = "legal"
    CULTURAL = "cultural"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    MILITARY = "military"
    RELIGIOUS = "religious"
    PHILOSOPHICAL = "philosophical"
    GENERAL = "general"


class AudienceType(Enum):
    """Enumeration of different audience types"""
    ACADEMIC = "academic"
    PROFESSIONAL = "professional"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    GENERAL_PUBLIC = "general_public"
    STUDENT = "student"
    EXPERT = "expert"


class AdvancedPromptManager:
    """Advanced prompt manager with dynamic, context-aware prompt generation"""
    
    def __init__(self):
        """Initialize the advanced prompt manager"""
        self.prompt_history = []
        self.quality_metrics = {}
        self.user_preferences = {}
        self.domain_keywords = self._initialize_domain_keywords()
        self.report_templates = self._initialize_report_templates()
        
    def _initialize_domain_keywords(self) -> Dict[DomainType, List[str]]:
        """Initialize domain-specific keywords for classification"""
        return {
            DomainType.STEM: [
                'research', 'study', 'analysis', 'experiment', 'methodology', 'data',
                'scientific', 'technical', 'engineering', 'mathematics', 'physics',
                'chemistry', 'biology', 'laboratory', 'hypothesis', 'theory', 'model',
                'algorithm', 'computation', 'simulation', 'statistics', 'peer-reviewed'
            ],
            DomainType.BUSINESS: [
                'market', 'industry', 'company', 'business', 'corporate', 'enterprise',
                'strategy', 'management', 'leadership', 'operations', 'sales', 'marketing',
                'customer', 'product', 'service', 'revenue', 'profit', 'growth', 'competition'
            ],
            DomainType.FINANCE: [
                'financial', 'investment', 'banking', 'economy', 'economic', 'market',
                'stock', 'bond', 'currency', 'trading', 'portfolio', 'risk', 'return',
                'valuation', 'analysis', 'forecast', 'budget', 'revenue', 'profit', 'loss'
            ],
            DomainType.HEALTHCARE: [
                'medical', 'health', 'healthcare', 'clinical', 'patient', 'treatment',
                'diagnosis', 'therapy', 'pharmaceutical', 'drug', 'medicine', 'hospital',
                'doctor', 'nurse', 'disease', 'symptom', 'prevention', 'wellness'
            ],
            DomainType.TECHNOLOGY: [
                'technology', 'software', 'hardware', 'digital', 'computer', 'programming',
                'algorithm', 'artificial intelligence', 'machine learning', 'data science',
                'cybersecurity', 'cloud', 'mobile', 'web', 'application', 'system'
            ],
            DomainType.GOVERNMENT: [
                'government', 'policy', 'political', 'legislation', 'regulation', 'law',
                'public', 'administration', 'bureaucracy', 'election', 'democracy',
                'constitution', 'parliament', 'congress', 'senate', 'executive', 'judicial'
            ],
            DomainType.LEGAL: [
                'legal', 'law', 'court', 'judge', 'attorney', 'lawyer', 'case', 'trial',
                'evidence', 'testimony', 'verdict', 'appeal', 'constitution', 'statute',
                'regulation', 'compliance', 'litigation', 'arbitration', 'mediation'
            ],
            DomainType.CULTURAL: [
                'culture', 'cultural', 'society', 'social', 'anthropology', 'sociology',
                'tradition', 'custom', 'belief', 'religion', 'philosophy', 'art', 'music',
                'literature', 'language', 'ethnicity', 'identity', 'heritage', 'values'
            ],
            DomainType.ENVIRONMENTAL: [
                'environment', 'environmental', 'climate', 'sustainability', 'ecology',
                'conservation', 'pollution', 'renewable', 'energy', 'carbon', 'emission',
                'biodiversity', 'ecosystem', 'natural', 'resource', 'green', 'clean'
            ],
            DomainType.MILITARY: [
                'military', 'army', 'navy', 'air force', 'defense', 'war', 'battle',
                'strategy', 'tactics', 'weapon', 'soldier', 'officer', 'command',
                'operation', 'mission', 'combat', 'security', 'intelligence', 'veteran'
            ],
            DomainType.RELIGIOUS: [
                'religion', 'religious', 'spiritual', 'faith', 'belief', 'worship',
                'church', 'temple', 'mosque', 'synagogue', 'prayer', 'ritual', 'ceremony',
                'sacred', 'divine', 'theology', 'philosophy', 'morality', 'ethics'
            ]
        }
    
    def _initialize_report_templates(self) -> Dict[ReportType, Dict[str, Any]]:
        """Initialize report templates for different types"""
        return {
            ReportType.SCIENTIFIC_TECHNICAL: {
                'structure': [
                    'Abstract',
                    'Introduction',
                    'Literature Review',
                    'Methodology',
                    'Results',
                    'Discussion',
                    'Conclusion',
                    'References'
                ],
                'focus': 'scientific rigor, methodology, data analysis, peer-reviewed sources',
                'tone': 'academic, precise, objective',
                'requirements': 'citations, methodology, statistical analysis, reproducibility'
            },
            ReportType.BUSINESS_FINANCIAL: {
                'structure': [
                    'Executive Summary',
                    'Market Analysis',
                    'Financial Performance',
                    'Competitive Landscape',
                    'Risk Assessment',
                    'Strategic Recommendations',
                    'Financial Projections',
                    'Appendices'
                ],
                'focus': 'market trends, financial metrics, competitive analysis, strategic insights',
                'tone': 'professional, analytical, actionable',
                'requirements': 'financial data, market research, competitive intelligence, actionable recommendations'
            },
            ReportType.POLICY_REGULATORY: {
                'structure': [
                    'Policy Overview',
                    'Current Regulatory Framework',
                    'Stakeholder Analysis',
                    'Impact Assessment',
                    'Compliance Requirements',
                    'Implementation Strategy',
                    'Risk Management',
                    'Monitoring and Evaluation'
                ],
                'focus': 'regulatory compliance, policy implications, stakeholder impact, implementation',
                'tone': 'formal, authoritative, comprehensive',
                'requirements': 'legal citations, regulatory references, stakeholder perspectives, compliance frameworks'
            },
            ReportType.CULTURAL_SOCIAL: {
                'structure': [
                    'Cultural Context',
                    'Social Dynamics',
                    'Historical Background',
                    'Contemporary Analysis',
                    'Stakeholder Perspectives',
                    'Cultural Impact',
                    'Social Implications',
                    'Future Outlook'
                ],
                'focus': 'cultural context, social dynamics, human behavior, societal impact',
                'tone': 'analytical, empathetic, culturally sensitive',
                'requirements': 'cultural context, social theory, ethnographic data, diverse perspectives'
            },
            ReportType.COMPARATIVE_ANALYSIS: {
                'structure': [
                    'Comparison Framework',
                    'Entity A Analysis',
                    'Entity B Analysis',
                    'Side-by-Side Comparison',
                    'Key Differences',
                    'Similarities and Patterns',
                    'Benchmarking Analysis',
                    'Recommendations'
                ],
                'focus': 'systematic comparison, benchmarking, pattern recognition, objective analysis',
                'tone': 'analytical, balanced, objective',
                'requirements': 'comparable metrics, standardized criteria, objective evaluation, balanced perspective'
            },
            ReportType.TREND_ANALYSIS: {
                'structure': [
                    'Current State Analysis',
                    'Historical Trends',
                    'Emerging Patterns',
                    'Driving Forces',
                    'Future Projections',
                    'Scenario Analysis',
                    'Opportunities and Threats',
                    'Strategic Implications'
                ],
                'focus': 'trend identification, pattern recognition, future forecasting, strategic implications',
                'tone': 'forward-looking, analytical, strategic',
                'requirements': 'trend data, forecasting models, scenario planning, strategic insights'
            },
            ReportType.CASE_STUDY: {
                'structure': [
                    'Case Background',
                    'Problem Statement',
                    'Context and Environment',
                    'Analysis and Investigation',
                    'Key Findings',
                    'Lessons Learned',
                    'Recommendations',
                    'Implications'
                ],
                'focus': 'detailed analysis, problem-solving, lessons learned, practical applications',
                'tone': 'detailed, analytical, practical',
                'requirements': 'detailed context, thorough analysis, practical insights, actionable lessons'
            },
            ReportType.HISTORICAL: {
                'structure': [
                    'Introduction',
                    'Historical Timeline',
                    'Key Developments',
                    'Major Events & Milestones',
                    'Impact & Significance',
                    'Evolution Over Time',
                    'Current Status',
                    'References & Sources'
                ],
                'focus': 'chronological accuracy, historical context, primary sources, scholarly analysis',
                'tone': 'academic, scholarly, chronological',
                'requirements': 'historical accuracy, primary sources, chronological structure, scholarly analysis'
            },
            ReportType.GENERAL: {
                'structure': [
                    'Introduction',
                    'Main Analysis',
                    'Key Insights',
                    'Implications',
                    'Conclusion',
                    'Sources & References'
                ],
                'focus': 'comprehensive overview, balanced analysis, clear presentation',
                'tone': 'professional, informative, accessible',
                'requirements': 'reliable sources, clear structure, balanced perspective, actionable insights'
            }
        }
    
    def create_context_aware_prompt(self, topic: str, report_type: Optional[ReportType] = None, 
                                   domain: Optional[DomainType] = None, 
                                   audience: AudienceType = AudienceType.PROFESSIONAL,
                                   user_intent: Optional[str] = None) -> str:
        """Create a dynamic, context-aware prompt based on multiple factors"""
        
        # Auto-detect domain and report type if not provided
        if domain is None:
            domain = self.detect_topic_domain(topic)
        
        if report_type is None:
            report_type = self.detect_report_type(topic, domain)
        
        # Get template for the report type
        template = self.report_templates.get(report_type, self.report_templates[ReportType.GENERAL])
        
        # Create base prompt
        base_prompt = self._create_base_prompt(topic, template, domain)
        
        # Customize for audience
        customized_prompt = self.customize_for_audience(base_prompt, audience)
        
        # Add user intent if provided
        if user_intent:
            customized_prompt = self._add_user_intent(customized_prompt, user_intent)
        
        # Add domain-specific enhancements
        enhanced_prompt = self._add_domain_specific_content(customized_prompt, domain, report_type)
        
        # Store prompt for learning
        self._store_prompt_for_learning(topic, report_type, domain, audience, enhanced_prompt)
        
        return enhanced_prompt
    
    def detect_topic_domain(self, topic: str) -> DomainType:
        """AI-powered domain classification based on topic keywords"""
        topic_lower = topic.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in topic_lower)
            domain_scores[domain] = score
        
        # Find domain with highest score
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        # Fallback to general domain
        return DomainType.GENERAL
    
    def detect_report_type(self, topic: str, domain: DomainType) -> ReportType:
        """Detect appropriate report type based on topic and domain"""
        topic_lower = topic.lower()
        
        # Check for specific report type indicators
        if any(word in topic_lower for word in ['compare', 'comparison', 'versus', 'vs', 'benchmark']):
            return ReportType.COMPARATIVE_ANALYSIS
        
        if any(word in topic_lower for word in ['trend', 'forecast', 'prediction', 'future', 'emerging']):
            return ReportType.TREND_ANALYSIS
        
        if any(word in topic_lower for word in ['case study', 'case', 'example', 'instance']):
            return ReportType.CASE_STUDY
        
        if any(word in topic_lower for word in ['policy', 'regulation', 'legal', 'compliance']):
            return ReportType.POLICY_REGULATORY
        
        if any(word in topic_lower for word in ['market', 'business', 'financial', 'economic']):
            return ReportType.BUSINESS_FINANCIAL
        
        if any(word in topic_lower for word in ['research', 'study', 'experiment', 'methodology']):
            return ReportType.SCIENTIFIC_TECHNICAL
        
        if any(word in topic_lower for word in ['culture', 'social', 'society', 'anthropology']):
            return ReportType.CULTURAL_SOCIAL
        
        if any(word in topic_lower for word in ['history', 'historical', 'timeline', 'century']):
            return ReportType.HISTORICAL
        
        # Default based on domain
        domain_report_mapping = {
            DomainType.STEM: ReportType.SCIENTIFIC_TECHNICAL,
            DomainType.BUSINESS: ReportType.BUSINESS_FINANCIAL,
            DomainType.FINANCE: ReportType.BUSINESS_FINANCIAL,
            DomainType.HEALTHCARE: ReportType.SCIENTIFIC_TECHNICAL,
            DomainType.TECHNOLOGY: ReportType.SCIENTIFIC_TECHNICAL,
            DomainType.GOVERNMENT: ReportType.POLICY_REGULATORY,
            DomainType.LEGAL: ReportType.POLICY_REGULATORY,
            DomainType.CULTURAL: ReportType.CULTURAL_SOCIAL,
            DomainType.SOCIAL: ReportType.CULTURAL_SOCIAL,
            DomainType.ENVIRONMENTAL: ReportType.SCIENTIFIC_TECHNICAL,
            DomainType.MILITARY: ReportType.HISTORICAL,
            DomainType.RELIGIOUS: ReportType.CULTURAL_SOCIAL,
            DomainType.PHILOSOPHICAL: ReportType.CULTURAL_SOCIAL
        }
        
        return domain_report_mapping.get(domain, ReportType.GENERAL)
    
    def customize_for_audience(self, base_prompt: str, audience: AudienceType) -> str:
        """Adjust complexity and focus based on target audience"""
        
        audience_customizations = {
            AudienceType.ACADEMIC: {
                'prefix': 'You are a senior academic researcher writing for a scholarly audience.',
                'style': 'Use formal academic language with extensive citations and theoretical frameworks.',
                'complexity': 'high',
                'focus': 'theoretical implications, methodological rigor, scholarly contribution'
            },
            AudienceType.PROFESSIONAL: {
                'prefix': 'You are a professional consultant creating a business report.',
                'style': 'Use professional, analytical language with practical insights and actionable recommendations.',
                'complexity': 'medium-high',
                'focus': 'practical applications, industry insights, strategic recommendations'
            },
            AudienceType.EXECUTIVE: {
                'prefix': 'You are a strategic advisor creating an executive summary report.',
                'style': 'Use clear, concise language with high-level insights and strategic implications.',
                'complexity': 'medium',
                'focus': 'strategic implications, key insights, executive summary, decision support'
            },
            AudienceType.TECHNICAL: {
                'prefix': 'You are a technical expert creating a detailed technical report.',
                'style': 'Use technical terminology and detailed analysis with specific technical recommendations.',
                'complexity': 'high',
                'focus': 'technical details, implementation specifics, technical best practices'
            },
            AudienceType.GENERAL_PUBLIC: {
                'prefix': 'You are a knowledgeable expert creating an accessible report for the general public.',
                'style': 'Use clear, engaging language that explains complex concepts in accessible terms.',
                'complexity': 'low-medium',
                'focus': 'clear explanations, practical relevance, public interest'
            },
            AudienceType.STUDENT: {
                'prefix': 'You are an educational expert creating a learning-focused report.',
                'style': 'Use educational language with clear explanations, examples, and learning objectives.',
                'complexity': 'low-medium',
                'focus': 'educational value, clear explanations, learning outcomes'
            },
            AudienceType.EXPERT: {
                'prefix': 'You are a domain expert creating a comprehensive expert analysis.',
                'style': 'Use advanced terminology and deep analysis with expert-level insights.',
                'complexity': 'very high',
                'focus': 'expert insights, advanced analysis, cutting-edge developments'
            }
        }
        
        customization = audience_customizations.get(audience, audience_customizations[AudienceType.PROFESSIONAL])
        
        customized_prompt = f"""{customization['prefix']}

{customization['style']}

AUDIENCE REQUIREMENTS:
- Complexity Level: {customization['complexity']}
- Primary Focus: {customization['focus']}
- Language Style: {customization['style']}

{base_prompt}

Ensure the content is appropriate for {audience.value} audience with {customization['complexity']} complexity level."""
        
        return customized_prompt
    
    def _create_base_prompt(self, topic: str, template: Dict[str, Any], domain: DomainType) -> str:
        """Create base prompt using template"""
        
        structure = template.get('structure', ['Introduction', 'Analysis', 'Conclusion'])
        focus = template.get('focus', 'comprehensive analysis')
        tone = template.get('tone', 'professional')
        requirements = template.get('requirements', 'detailed analysis')
        
        base_prompt = f"""Create a comprehensive {tone} report on: {topic}

DOMAIN: {domain.value.upper()}
FOCUS: {focus}
TONE: {tone}

REQUIRED STRUCTURE:
{chr(10).join([f'## {section}' for section in structure])}

QUALITY REQUIREMENTS:
- {requirements}
- Minimum 2000 words with substantial content in each section
- Include specific examples, data points, and evidence
- Provide actionable insights and recommendations
- Use appropriate terminology for the domain
- Ensure accuracy and reliability of information

WRITING GUIDELINES:
- Write in a {tone} style appropriate for {domain.value} domain
- Include relevant context and background information
- Provide clear analysis and interpretation
- Support claims with evidence and examples
- Consider multiple perspectives and implications
- Address potential challenges and limitations

Generate a comprehensive report that demonstrates deep understanding of {topic} within the {domain.value} domain."""
        
        return base_prompt
    
    def _add_user_intent(self, prompt: str, user_intent: str) -> str:
        """Add user intent to the prompt"""
        return f"""{prompt}

USER INTENT:
{user_intent}

Please ensure the report specifically addresses this user intent and provides relevant insights and recommendations."""
    
    def _add_domain_specific_content(self, prompt: str, domain: DomainType, report_type: ReportType) -> str:
        """Add domain-specific enhancements to the prompt"""
        
        domain_enhancements = {
            DomainType.STEM: """
STEM-SPECIFIC REQUIREMENTS:
- Include scientific methodology and research design
- Reference peer-reviewed sources and academic literature
- Provide statistical analysis where relevant
- Address reproducibility and validation
- Consider ethical implications of research
- Include technical specifications and parameters""",
            
            DomainType.BUSINESS: """
BUSINESS-SPECIFIC REQUIREMENTS:
- Include market analysis and competitive landscape
- Provide financial metrics and performance indicators
- Address strategic implications and business impact
- Consider stakeholder perspectives and interests
- Include risk assessment and mitigation strategies
- Provide actionable business recommendations""",
            
            DomainType.FINANCE: """
FINANCE-SPECIFIC REQUIREMENTS:
- Include financial analysis and metrics
- Provide market trends and economic indicators
- Address risk assessment and management
- Consider regulatory compliance and requirements
- Include investment implications and opportunities
- Provide financial projections and forecasts""",
            
            DomainType.HEALTHCARE: """
HEALTHCARE-SPECIFIC REQUIREMENTS:
- Include clinical evidence and medical research
- Address patient safety and outcomes
- Consider regulatory compliance (FDA, HIPAA, etc.)
- Include cost-effectiveness and value analysis
- Address ethical considerations and patient rights
- Provide clinical recommendations and guidelines""",
            
            DomainType.TECHNOLOGY: """
TECHNOLOGY-SPECIFIC REQUIREMENTS:
- Include technical specifications and architecture
- Address scalability and performance considerations
- Consider security and privacy implications
- Include implementation strategies and best practices
- Address emerging technologies and trends
- Provide technical recommendations and solutions""",
            
            DomainType.GOVERNMENT: """
GOVERNMENT-SPECIFIC REQUIREMENTS:
- Include policy analysis and implications
- Address stakeholder interests and public impact
- Consider regulatory compliance and legal requirements
- Include implementation strategies and timelines
- Address transparency and accountability
- Provide policy recommendations and next steps""",
            
            DomainType.LEGAL: """
LEGAL-SPECIFIC REQUIREMENTS:
- Include legal analysis and precedent
- Address regulatory compliance and requirements
- Consider risk assessment and liability
- Include case law and legal interpretations
- Address ethical considerations and professional standards
- Provide legal recommendations and compliance strategies""",
            
            DomainType.CULTURAL: """
CULTURAL-SPECIFIC REQUIREMENTS:
- Include cultural context and sensitivity
- Address social dynamics and human behavior
- Consider historical and contemporary perspectives
- Include diverse viewpoints and experiences
- Address cultural implications and impact
- Provide culturally appropriate recommendations""",
            
            DomainType.ENVIRONMENTAL: """
ENVIRONMENTAL-SPECIFIC REQUIREMENTS:
- Include environmental impact assessment
- Address sustainability and conservation
- Consider climate change and ecological factors
- Include regulatory compliance and standards
- Address long-term environmental implications
- Provide sustainable solutions and recommendations"""
        }
        
        enhancement = domain_enhancements.get(domain, "")
        if enhancement:
            return f"{prompt}\n{enhancement}"
        
        return prompt
    
    def _store_prompt_for_learning(self, topic: str, report_type: ReportType, domain: DomainType, 
                                  audience: AudienceType, prompt: str) -> None:
        """Store prompt for learning and optimization"""
        prompt_record = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'report_type': report_type.value,
            'domain': domain.value,
            'audience': audience.value,
            'prompt_length': len(prompt),
            'prompt': prompt
        }
        self.prompt_history.append(prompt_record)
        
        # Keep only last 100 prompts for memory management
        if len(self.prompt_history) > 100:
            self.prompt_history = self.prompt_history[-100:]
    
    def learn_from_output_quality(self, topic: str, prompt: str, output_quality_score: float, 
                                 feedback: Optional[str] = None) -> None:
        """Learn from output quality to improve future prompts"""
        
        # Store quality metrics
        quality_record = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'prompt_length': len(prompt),
            'quality_score': output_quality_score,
            'feedback': feedback
        }
        
        # Store in quality metrics
        topic_key = topic.lower().replace(' ', '_')
        if topic_key not in self.quality_metrics:
            self.quality_metrics[topic_key] = []
        self.quality_metrics[topic_key].append(quality_record)
        
        # Keep only last 50 quality records per topic
        if len(self.quality_metrics[topic_key]) > 50:
            self.quality_metrics[topic_key] = self.quality_metrics[topic_key][-50:]
    
    def optimize_prompt_based_on_history(self, topic: str, base_prompt: str) -> str:
        """Optimize prompt based on historical performance"""
        
        topic_key = topic.lower().replace(' ', '_')
        if topic_key not in self.quality_metrics:
            return base_prompt
        
        # Analyze historical performance
        quality_records = self.quality_metrics[topic_key]
        if not quality_records:
            return base_prompt
        
        # Find high-performing prompts
        high_quality_threshold = 0.8
        high_quality_prompts = [record for record in quality_records if record['quality_score'] >= high_quality_threshold]
        
        if high_quality_prompts:
            # Extract patterns from high-quality prompts
            avg_prompt_length = sum(record['prompt_length'] for record in high_quality_prompts) / len(high_quality_prompts)
            
            # Adjust current prompt based on successful patterns
            if len(base_prompt) < avg_prompt_length * 0.8:
                # Add more detail to match successful prompt length
                base_prompt += "\n\nADDITIONAL DETAIL REQUIREMENTS:\n- Provide more comprehensive analysis\n- Include additional examples and case studies\n- Expand on technical details and implications"
            elif len(base_prompt) > avg_prompt_length * 1.2:
                # Simplify prompt to match successful length
                base_prompt = self._simplify_prompt(base_prompt)
        
        return base_prompt
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify a prompt while maintaining essential elements"""
        # Remove redundant sections while keeping core requirements
        lines = prompt.split('\n')
        simplified_lines = []
        skip_section = False
        
        for line in lines:
            if 'ADDITIONAL' in line.upper() or 'ENHANCED' in line.upper():
                skip_section = True
            elif line.strip().startswith('##') or line.strip().startswith('#'):
                skip_section = False
            
            if not skip_section:
                simplified_lines.append(line)
        
        return '\n'.join(simplified_lines)
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompt usage and performance"""
        stats = {
            'total_prompts': len(self.prompt_history),
            'total_quality_records': sum(len(records) for records in self.quality_metrics.values()),
            'topics_analyzed': len(self.quality_metrics),
            'average_quality_score': 0.0,
            'most_common_domains': {},
            'most_common_report_types': {}
        }
        
        # Calculate average quality score
        all_scores = []
        for records in self.quality_metrics.values():
            all_scores.extend([record['quality_score'] for record in records])
        
        if all_scores:
            stats['average_quality_score'] = sum(all_scores) / len(all_scores)
        
        # Analyze domain and report type distribution
        domain_counts = {}
        report_type_counts = {}
        
        for record in self.prompt_history:
            domain = record['domain']
            report_type = record['report_type']
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            report_type_counts[report_type] = report_type_counts.get(report_type, 0) + 1
        
        stats['most_common_domains'] = dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        stats['most_common_report_types'] = dict(sorted(report_type_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return stats
    
    # Legacy methods for backward compatibility
    def create_intelligent_prompt(self, topic: str, categories: List[str], research_data: str, structure: List[str]) -> str:
        """Legacy method for backward compatibility"""
        domain = self.detect_topic_domain(topic)
        report_type = self.detect_report_type(topic, domain)
        return self.create_context_aware_prompt(topic, report_type, domain, AudienceType.PROFESSIONAL)
    
    def enhance_prompt_engineering(self, topic: str, categories: List[str], search_results: str, structure: List[str]) -> str:
        """Legacy method for backward compatibility"""
        domain = self.detect_topic_domain(topic)
        report_type = self.detect_report_type(topic, domain)
        base_prompt = self.create_context_aware_prompt(topic, report_type, domain, AudienceType.PROFESSIONAL)
        
        # Add search results context
        concrete_data = self._extract_concrete_data(search_results)
        data_context = self._create_data_context(concrete_data)
        
        enhanced_prompt = f"""{base_prompt}

AVAILABLE CONCRETE DATA:
{data_context}

SEARCH RESULTS WITH SPECIFIC INFORMATION:
{search_results[:2000]}

QUALITY STANDARDS:
- Every claim must be supported by specific data from search results
- Include actual company names, not "major companies" 
- Use specific numbers, not "significant growth"
- Cite URLs as sources: "According to [source URL]..."
- Write in active voice with concrete examples
- Each section minimum 200 words with unique, valuable insights"""
        
        return enhanced_prompt
    
    def _extract_concrete_data(self, text: str) -> Dict[str, List[str]]:
        """Extract concrete, specific data from search results"""
        # Implementation from original PromptManager
        data = {
            'salaries': [],
            'companies': [],
            'locations': [],
            'urls': [],
            'dates': [],
            'percentages': [],
            'specific_numbers': [],
            'numerical_data': []
        }
        
        # Enhanced salary extraction with context
        salary_patterns = [
            r'€\s*(\d{2,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|/year|k)',
            r'\$\s*(\d{2,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|/year|k)',
            r'£\s*(\d{2,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|/year|k)',
            r'salary.*?€\s*(\d{2,3}(?:,\d{3})*)',
            r'compensation.*?\$\s*(\d{2,3}(?:,\d{3})*)',
            r'earning.*?€\s*(\d{2,3}(?:,\d{3})*)',
            r'pay.*?\$\s*(\d{2,3}(?:,\d{3})*)',
            r'wage.*?€\s*(\d{2,3}(?:,\d{3})*)',
            r'(\d{2,3}(?:,\d{3})*)\s*(?:euro|eur|€)\s*(?:per year|annually|/year)',
            r'(\d{2,3}(?:,\d{3})*)\s*(?:dollar|usd|\$)\s*(?:per year|annually|/year)',
            r'(\d{2,3}(?:,\d{3})*)\s*(?:pound|gbp|£)\s*(?:per year|annually|/year)',
            r'(\d{2,3}(?:,\d{3})*)\s*k\s*(?:euro|eur|€)',
            r'(\d{2,3}(?:,\d{3})*)\s*k\s*(?:dollar|usd|\$)',
            r'(\d{2,3}(?:,\d{3})*)\s*k\s*(?:pound|gbp|£)',
        ]
        
        for pattern in salary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Determine currency based on pattern
                if '€' in pattern or 'euro' in pattern.lower() or 'eur' in pattern.lower():
                    formatted_salary = f"€{match}" if not match.startswith('€') else match
                elif '$' in pattern or 'dollar' in pattern.lower() or 'usd' in pattern.lower():
                    formatted_salary = f"${match}" if not match.startswith('$') else match
                elif '£' in pattern or 'pound' in pattern.lower() or 'gbp' in pattern.lower():
                    formatted_salary = f"£{match}" if not match.startswith('£') else match
                else:
                    # Default to Euro if no currency specified
                    formatted_salary = f"€{match}" if not match.startswith(('€', '$', '£')) else match
                
                data['salaries'].append(formatted_salary)
        
        # Company name extraction with context
        company_patterns = [
            r'(?:at|for|with|by)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:Inc|Ltd|GmbH|Corp|LLC))?)',
            r'(Google|Apple|Microsoft|Amazon|Meta|Netflix|Tesla|Intel|AMD|NVIDIA|IBM|Oracle|Salesforce)',
            r'(McKinsey|Deloitte|PwC|EY|KPMG|Accenture|BCG)',
            r'([A-Z][a-zA-Z]+(?:tech|soft|systems|solutions|consulting|labs|works|media))',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            data['companies'].extend([match for match in matches if len(match) > 2])
        
        # Geographic locations with salary context
        location_patterns = [
            r'(?:in|based|located)\s+(New York|San Francisco|Los Angeles|Chicago|Boston|Seattle|Austin|Denver)',
            r'(?:in|based|located)\s+(London|Berlin|Amsterdam|Madrid|Barcelona|Rome|Milan|Zurich|Vienna)',
            r'(?:in|based|located)\s+(Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg)',
            r'(?:in|based|located)\s+(Tokyo|Singapore|Hong Kong|Sydney|Toronto|Vancouver)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['locations'].extend(matches)
        
        # Extract all URLs
        data['urls'] = re.findall(r'https?://[^\s<>"]+', text)
        
        # Extract percentages and growth figures
        data['percentages'] = re.findall(r'(\d+(?:\.\d+)?%)', text)
        
        # Extract specific numbers with context
        number_patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s+(?:employees|workers|professionals|jobs|positions)',
            r'(\d{1,3}(?:,\d{3})*)\s*(?:companies|firms|organizations)',
            r'grew?\s+by\s+(\d+(?:\.\d+)?%)',
            r'increased?\s+by\s+(\d+(?:\.\d+)?%)',
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['specific_numbers'].extend(matches)
        
        # Combine percentages and specific numbers
        data['numerical_data'] = list(set(data['percentages'] + data['specific_numbers']))[:10]
        
        # Remove duplicates and limit results
        for key in data:
            data[key] = list(set(data[key]))[:10]
        
        return data
    
    def _create_data_context(self, concrete_data: Dict[str, List[str]]) -> str:
        """Create data context string from concrete data"""
        data_context = ""
        if concrete_data.get('salaries'):
            data_context += f"Available salary data: {', '.join(concrete_data['salaries'][:5])}\n"
        if concrete_data.get('companies'):
            data_context += f"Companies mentioned: {', '.join(concrete_data['companies'][:5])}\n"
        if concrete_data.get('locations'):
            data_context += f"Geographic locations: {', '.join(concrete_data['locations'][:5])}\n"
        if concrete_data.get('urls'):
            data_context += f"Source URLs available: {len(concrete_data['urls'])} sources\n"
        return data_context
    
    def improve_model_instructions(self) -> Dict[str, str]:
        """Enhanced model-specific instructions for better outputs"""
        return {
            'llama': {
                'temperature': 0.3,
                'system_prompt': """You are a professional research analyst creating detailed, specific reports. Your reports must contain:
                - Specific data points with numbers and percentages
                - Company names and locations with context
                - Source citations with URLs when available
                - No placeholder or template language
                - Minimum 1500 words with substantial content
                - Concrete examples and case studies, not generalizations
                - Technical details and methodologies where relevant
                - Actionable recommendations and insights""",
                'prefill': "Based on the comprehensive research data, here is a detailed analysis with specific figures, examples, and actionable insights:"
            },
            'claude': {
                'temperature': 0.4,
                'system_prompt': """Write a comprehensive research report with specific data points and concrete examples. 
                Avoid generic phrases and placeholder text. Include:
                - Specific company names and technologies
                - Numerical data and performance metrics
                - Real examples and case studies
                - Technical details and methodologies
                - Source citations with URLs
                - Actionable recommendations""",
                'prefill': "# Comprehensive Research Report\n\nBased on current market data and industry analysis, here is a detailed examination with specific examples and insights:"
            },
            'gpt': {
                'temperature': 0.3,
                'system_prompt': """You are an expert analyst creating data-driven reports with specific details. Requirements:
                - Use specific numbers and percentages from research data
                - Name actual companies, locations, and technologies
                - Cite sources with URLs when available
                - Write detailed sections (200+ words each) with substantial content
                - No generic or placeholder content
                - Include technical details and methodologies
                - Provide concrete examples and case studies
                - Offer actionable recommendations and insights""",
                'prefill': "Here is a detailed research report with specific data, examples, and analysis:"
            }
        }
    
    # Additional specialized prompt methods
    def create_historical_prompt(self, topic: str, structure: List[str]) -> str:
        """Create a specialized prompt for historical topics"""
        return self.create_context_aware_prompt(topic, ReportType.HISTORICAL, DomainType.GENERAL, AudienceType.ACADEMIC)
    
    def create_general_prompt(self, topic: str, structure: List[str]) -> str:
        """Create a general prompt for non-historical topics"""
        return self.create_context_aware_prompt(topic, ReportType.GENERAL, DomainType.GENERAL, AudienceType.PROFESSIONAL)
    
    def create_ai_insights_prompt(self, topic: str, categories: List[str]) -> str:
        """Create a prompt for generating AI insights"""
        domain = self.detect_topic_domain(topic)
        report_type = self.detect_report_type(topic, domain)
        return self.create_context_aware_prompt(topic, report_type, domain, AudienceType.EXPERT, "Generate comprehensive insights and analysis")
    
    def create_search_like_data_prompt(self, topic: str) -> str:
        """Create a prompt for generating search-like data"""
        return f"""Create a simulated search results page about: {topic}

Include 3-5 authoritative sources with:
- Fictional but plausible URLs (e.g., encyclopedia-portugal-history.edu, medieval-chronicles.org)
- Key facts with dates and figures
- Important events in chronological order
- Context and significance

Format as search results with:
- Title: [Descriptive title]
- URL: [plausible academic/historical URL]
- Snippet: [2-3 sentences with key information]

Focus on providing accurate information that would be found in reliable sources."""
    
    def create_section_content_prompt(self, section: str, topic: str) -> str:
        """Create a prompt for generating section content"""
        domain = self.detect_topic_domain(topic)
        report_type = self.detect_report_type(topic, domain)
        
        # Create a focused prompt for the specific section
        section_focus = f"Write a detailed section about {section.lower()} in relation to {topic}."
        
        return self.create_context_aware_prompt(
            topic, 
            report_type, 
            domain, 
            AudienceType.PROFESSIONAL, 
            section_focus
        )

    def detect_topic_domains(self, topic: str) -> List[Tuple[DomainType, float]]:
        """Enhanced multi-domain detection with confidence scores"""
        topic_lower = topic.lower()
        domain_scores = {}
        
        # Calculate scores for each domain
        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            keyword_matches = 0
            
            for keyword in keywords:
                if keyword.lower() in topic_lower:
                    keyword_matches += 1
                    # Weight by keyword length and specificity
                    score += len(keyword) * 0.1
            
            # Normalize score based on number of keywords in domain
            if len(keywords) > 0:
                score = (score / len(keywords)) * (keyword_matches / len(keywords))
                domain_scores[domain] = score
        
        # Sort domains by score and return with confidence
        ranked_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to confidence scores (0.0 to 1.0)
        if ranked_domains:
            max_score = ranked_domains[0][1]
            if max_score > 0:
                ranked_domains = [(domain, min(score / max_score, 1.0)) for domain, score in ranked_domains]
        
        # Filter domains with significant confidence (>0.1)
        significant_domains = [(domain, score) for domain, score in ranked_domains if score > 0.1]
        
        # Always include at least one domain
        if not significant_domains and ranked_domains:
            significant_domains = [(ranked_domains[0][0], 0.5)]  # Default confidence
        
        return significant_domains[:3]  # Return top 3 domains
    
    def generate_custom_template(self, topic: str, user_requirements: str, 
                                primary_domain: DomainType = None) -> Dict[str, Any]:
        """AI-powered dynamic template generation for unique topics"""
        
        # Determine primary domain if not provided
        if primary_domain is None:
            domains = self.detect_topic_domains(topic)
            primary_domain = domains[0][0] if domains else DomainType.GENERAL
        
        # Create custom template based on topic analysis
        custom_template = self._analyze_topic_for_template(topic, user_requirements, primary_domain)
        
        # Enhance with domain-specific elements
        custom_template = self._enhance_template_with_domain(custom_template, primary_domain)
        
        # Add user-specific requirements
        custom_template = self._incorporate_user_requirements(custom_template, user_requirements)
        
        return custom_template
    
    def _analyze_topic_for_template(self, topic: str, user_requirements: str, domain: DomainType) -> Dict[str, Any]:
        """Analyze topic to determine optimal template structure"""
        
        # Extract topic characteristics
        topic_words = topic.lower().split()
        topic_length = len(topic_words)
        
        # Determine complexity level
        complexity_indicators = ['advanced', 'complex', 'detailed', 'comprehensive', 'in-depth', 'thorough']
        is_complex = any(indicator in topic.lower() for indicator in complexity_indicators)
        
        # Determine if it's a comparison topic
        comparison_indicators = ['vs', 'versus', 'compare', 'comparison', 'difference', 'similarity']
        is_comparison = any(indicator in topic.lower() for indicator in comparison_indicators)
        
        # Determine if it's a trend analysis
        trend_indicators = ['trend', 'forecast', 'prediction', 'future', 'emerging', 'evolution']
        is_trend = any(indicator in topic.lower() for indicator in trend_indicators)
        
        # Generate appropriate structure
        if is_comparison:
            structure = [
                'Comparison Framework',
                'Entity A Analysis',
                'Entity B Analysis',
                'Side-by-Side Comparison',
                'Key Differences',
                'Similarities and Patterns',
                'Benchmarking Analysis',
                'Recommendations'
            ]
        elif is_trend:
            structure = [
                'Current State Analysis',
                'Historical Trends',
                'Emerging Patterns',
                'Driving Forces',
                'Future Projections',
                'Scenario Analysis',
                'Opportunities and Threats',
                'Strategic Implications'
            ]
        elif is_complex:
            structure = [
                'Executive Summary',
                'Background and Context',
                'Detailed Analysis',
                'Methodology and Approach',
                'Key Findings',
                'Critical Insights',
                'Implications and Impact',
                'Recommendations and Next Steps',
                'Technical Appendix'
            ]
        else:
            # Use domain-specific structure as base
            base_template = self.report_templates.get(ReportType.GENERAL, {})
            structure = base_template.get('structure', [
                'Introduction',
                'Main Analysis',
                'Key Insights',
                'Implications',
                'Conclusion',
                'Sources & References'
            ])
        
        # Determine focus and tone based on domain and complexity
        if domain in [DomainType.STEM, DomainType.TECHNOLOGY]:
            focus = 'technical analysis, methodology, data-driven insights'
            tone = 'precise, analytical, objective'
        elif domain in [DomainType.BUSINESS, DomainType.FINANCE]:
            focus = 'market analysis, strategic insights, actionable recommendations'
            tone = 'professional, analytical, actionable'
        elif domain in [DomainType.GOVERNMENT, DomainType.LEGAL]:
            focus = 'policy analysis, regulatory implications, compliance considerations'
            tone = 'formal, authoritative, comprehensive'
        else:
            focus = 'comprehensive analysis with practical insights'
            tone = 'professional, informative, balanced'
        
        return {
            'structure': structure,
            'focus': focus,
            'tone': tone,
            'complexity': 'high' if is_complex else 'medium',
            'domain': domain.value,
            'custom_generated': True
        }
    
    def _enhance_template_with_domain(self, template: Dict[str, Any], domain: DomainType) -> Dict[str, Any]:
        """Enhance template with domain-specific requirements"""
        
        domain_requirements = {
            DomainType.STEM: {
                'requirements': 'scientific methodology, peer-reviewed sources, statistical analysis, reproducibility',
                'additional_sections': ['Methodology', 'Data Analysis', 'Statistical Validation']
            },
            DomainType.BUSINESS: {
                'requirements': 'market research, competitive analysis, financial metrics, strategic recommendations',
                'additional_sections': ['Market Analysis', 'Competitive Landscape', 'Financial Impact']
            },
            DomainType.FINANCE: {
                'requirements': 'financial analysis, risk assessment, investment implications, regulatory compliance',
                'additional_sections': ['Financial Analysis', 'Risk Assessment', 'Investment Implications']
            },
            DomainType.HEALTHCARE: {
                'requirements': 'clinical evidence, patient outcomes, regulatory compliance, ethical considerations',
                'additional_sections': ['Clinical Evidence', 'Patient Impact', 'Regulatory Considerations']
            },
            DomainType.TECHNOLOGY: {
                'requirements': 'technical specifications, implementation details, performance analysis, security considerations',
                'additional_sections': ['Technical Architecture', 'Implementation Strategy', 'Performance Analysis']
            }
        }
        
        domain_config = domain_requirements.get(domain, {})
        
        # Add domain-specific requirements
        if 'requirements' in domain_config:
            template['requirements'] = domain_config['requirements']
        
        # Add domain-specific sections if not already present
        if 'additional_sections' in domain_config:
            existing_sections = set(section.lower() for section in template['structure'])
            for section in domain_config['additional_sections']:
                if section.lower() not in existing_sections:
                    template['structure'].append(section)
        
        return template
    
    def _incorporate_user_requirements(self, template: Dict[str, Any], user_requirements: str) -> Dict[str, Any]:
        """Incorporate user-specific requirements into template"""
        
        requirements_lower = user_requirements.lower()
        
        # Check for specific user requirements
        if 'executive' in requirements_lower or 'summary' in requirements_lower:
            template['structure'].insert(0, 'Executive Summary')
        
        if 'recommendations' in requirements_lower or 'action' in requirements_lower:
            if 'Recommendations' not in template['structure']:
                template['structure'].append('Recommendations')
        
        if 'technical' in requirements_lower or 'detailed' in requirements_lower:
            template['complexity'] = 'high'
            template['focus'] += ', technical specifications, detailed analysis'
        
        if 'simple' in requirements_lower or 'overview' in requirements_lower:
            template['complexity'] = 'low'
            template['focus'] = 'clear overview, essential information'
        
        if 'comparison' in requirements_lower:
            template['structure'] = [
                'Comparison Framework',
                'Entity A Analysis',
                'Entity B Analysis',
                'Side-by-Side Comparison',
                'Key Differences',
                'Recommendations'
            ]
        
        # Add custom user note
        template['user_requirements'] = user_requirements
        
        return template
    
    def predict_prompt_quality(self, prompt: str, topic: str) -> float:
        """Predict the likely quality/success of a prompt before generation"""
        
        quality_score = 0.0
        max_score = 100.0
        
        # Factor 1: Prompt length and completeness (20 points)
        prompt_length = len(prompt)
        if 500 <= prompt_length <= 2000:
            quality_score += 20
        elif 200 <= prompt_length < 500:
            quality_score += 15
        elif 2000 < prompt_length <= 3000:
            quality_score += 18
        else:
            quality_score += 10
        
        # Factor 2: Topic-prompt alignment (25 points)
        topic_words = set(topic.lower().split())
        prompt_words = set(prompt.lower().split())
        word_overlap = len(topic_words.intersection(prompt_words))
        alignment_score = min(word_overlap / max(len(topic_words), 1) * 25, 25)
        quality_score += alignment_score
        
        # Factor 3: Structure and organization (20 points)
        structure_indicators = ['structure', 'sections', 'requirements', 'guidelines', 'focus']
        structure_score = sum(1 for indicator in structure_indicators if indicator in prompt.lower()) * 4
        quality_score += min(structure_score, 20)
        
        # Factor 4: Specificity and detail (20 points)
        specificity_indicators = ['specific', 'detailed', 'concrete', 'examples', 'data', 'evidence']
        specificity_score = sum(1 for indicator in specificity_indicators if indicator in prompt.lower()) * 3.33
        quality_score += min(specificity_score, 20)
        
        # Factor 5: Historical performance (15 points)
        topic_key = topic.lower().replace(' ', '_')
        if topic_key in self.quality_metrics:
            recent_scores = [record['quality_score'] for record in self.quality_metrics[topic_key][-5:]]
            if recent_scores:
                avg_historical_score = sum(recent_scores) / len(recent_scores)
                quality_score += avg_historical_score * 0.15
        
        # Normalize to 0-1 scale
        normalized_score = min(quality_score / max_score, 1.0)
        
        return normalized_score
    
    def get_quality_prediction_breakdown(self, prompt: str, topic: str) -> Dict[str, float]:
        """Get detailed breakdown of quality prediction factors"""
        
        breakdown = {}
        
        # Prompt length analysis
        prompt_length = len(prompt)
        if 500 <= prompt_length <= 2000:
            breakdown['length_score'] = 1.0
        elif 200 <= prompt_length < 500:
            breakdown['length_score'] = 0.75
        elif 2000 < prompt_length <= 3000:
            breakdown['length_score'] = 0.9
        else:
            breakdown['length_score'] = 0.5
        
        # Topic alignment
        topic_words = set(topic.lower().split())
        prompt_words = set(prompt.lower().split())
        word_overlap = len(topic_words.intersection(prompt_words))
        breakdown['topic_alignment'] = word_overlap / max(len(topic_words), 1)
        
        # Structure quality
        structure_indicators = ['structure', 'sections', 'requirements', 'guidelines', 'focus']
        structure_count = sum(1 for indicator in structure_indicators if indicator in prompt.lower())
        breakdown['structure_quality'] = min(structure_count / 5, 1.0)
        
        # Specificity level
        specificity_indicators = ['specific', 'detailed', 'concrete', 'examples', 'data', 'evidence']
        specificity_count = sum(1 for indicator in specificity_indicators if indicator in prompt.lower())
        breakdown['specificity_level'] = min(specificity_count / 6, 1.0)
        
        # Historical performance
        topic_key = topic.lower().replace(' ', '_')
        if topic_key in self.quality_metrics:
            recent_scores = [record['quality_score'] for record in self.quality_metrics[topic_key][-5:]]
            if recent_scores:
                breakdown['historical_performance'] = sum(recent_scores) / len(recent_scores)
            else:
                breakdown['historical_performance'] = 0.5
        else:
            breakdown['historical_performance'] = 0.5
        
        return breakdown
    
    def optimize_prompt_for_quality(self, base_prompt: str, topic: str, target_quality: float = 0.8) -> str:
        """Optimize a prompt to achieve target quality score"""
        
        current_quality = self.predict_prompt_quality(base_prompt, topic)
        
        if current_quality >= target_quality:
            return base_prompt
        
        optimized_prompt = base_prompt
        
        # Get quality breakdown to identify weak areas
        breakdown = self.get_quality_prediction_breakdown(base_prompt, topic)
        
        # Improve structure quality if needed
        if breakdown.get('structure_quality', 0) < 0.7:
            optimized_prompt += "\n\nSTRUCTURE REQUIREMENTS:\n- Follow the specified report structure exactly\n- Ensure each section has clear objectives\n- Maintain logical flow between sections\n- Include appropriate transitions"
        
        # Improve specificity if needed
        if breakdown.get('specificity_level', 0) < 0.7:
            optimized_prompt += "\n\nSPECIFICITY REQUIREMENTS:\n- Include specific examples and case studies\n- Provide concrete data points and statistics\n- Name actual companies, technologies, or entities\n- Use specific dates, figures, and measurements\n- Avoid generic or placeholder language"
        
        # Improve topic alignment if needed
        if breakdown.get('topic_alignment', 0) < 0.6:
            optimized_prompt += f"\n\nTOPIC FOCUS:\n- Maintain consistent focus on: {topic}\n- Ensure all content directly relates to the topic\n- Avoid tangential or off-topic discussions\n- Keep analysis centered on the specified subject"
        
        # Add quality enforcement if overall score is still low
        new_quality = self.predict_prompt_quality(optimized_prompt, topic)
        if new_quality < target_quality:
            optimized_prompt += "\n\nQUALITY ENFORCEMENT:\n- Minimum 2000 words with substantial content\n- Each section must contain unique, valuable insights\n- Include source citations and references\n- Provide actionable recommendations\n- Ensure professional tone and formatting"
        
        return optimized_prompt
    
    def create_multi_domain_prompt(self, topic: str, domains: List[Tuple[DomainType, float]], 
                                  audience: AudienceType = AudienceType.PROFESSIONAL) -> str:
        """Create a prompt that addresses multiple domains for complex topics"""
        
        if not domains:
            return self.create_context_aware_prompt(topic)
        
        # Sort domains by confidence
        sorted_domains = sorted(domains, key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]
        secondary_domains = [domain for domain, score in sorted_domains[1:] if score > 0.3]
        
        # Create base prompt for primary domain
        base_prompt = self.create_context_aware_prompt(topic, domain=primary_domain, audience=audience)
        
        # Add multi-domain considerations
        multi_domain_section = "\n\nMULTI-DOMAIN CONSIDERATIONS:\n"
        multi_domain_section += f"Primary Domain: {primary_domain.value.upper()}\n"
        
        if secondary_domains:
            multi_domain_section += f"Secondary Domains: {', '.join([d.value.upper() for d in secondary_domains])}\n"
            multi_domain_section += "\nINTEGRATION REQUIREMENTS:\n"
            multi_domain_section += "- Address perspectives from all relevant domains\n"
            multi_domain_section += "- Show how different domains interact and influence each other\n"
            multi_domain_section += "- Provide insights that bridge multiple domains\n"
            multi_domain_section += "- Consider cross-domain implications and opportunities\n"
        
        # Add domain-specific requirements for each domain
        for domain, confidence in sorted_domains:
            if confidence > 0.2:  # Only include domains with significant relevance
                domain_requirements = self._get_domain_requirements(domain)
                if domain_requirements:
                    multi_domain_section += f"\n{domain.value.upper()} DOMAIN REQUIREMENTS:\n{domain_requirements}\n"
        
        return base_prompt + multi_domain_section
    
    def _get_domain_requirements(self, domain: DomainType) -> str:
        """Get specific requirements for a domain"""
        domain_requirements = {
            DomainType.STEM: "- Include scientific methodology and peer-reviewed sources\n- Provide statistical analysis and data validation\n- Address reproducibility and experimental design",
            DomainType.BUSINESS: "- Include market analysis and competitive landscape\n- Provide financial metrics and business impact\n- Address strategic implications and ROI",
            DomainType.FINANCE: "- Include financial analysis and risk assessment\n- Provide investment implications and market trends\n- Address regulatory compliance and financial projections",
            DomainType.HEALTHCARE: "- Include clinical evidence and patient outcomes\n- Address regulatory compliance (FDA, HIPAA)\n- Consider ethical implications and patient safety",
            DomainType.TECHNOLOGY: "- Include technical specifications and architecture\n- Address scalability and performance considerations\n- Consider security and implementation challenges",
            DomainType.GOVERNMENT: "- Include policy analysis and stakeholder impact\n- Address regulatory compliance and legal requirements\n- Consider public interest and transparency",
            DomainType.LEGAL: "- Include legal analysis and precedent\n- Address regulatory compliance and risk assessment\n- Consider liability and legal implications",
            DomainType.CULTURAL: "- Include cultural context and sensitivity\n- Address social dynamics and human behavior\n- Consider diverse perspectives and cultural implications"
        }
        
        return domain_requirements.get(domain, "")


# Backward compatibility - create an alias for the old class name
PromptManager = AdvancedPromptManager 