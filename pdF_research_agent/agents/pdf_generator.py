import datetime
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
from abc import ABC, abstractmethod

class DetailLevel(Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

class ReportType(Enum):
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    MARKET_STUDY = "market_study"
    TECHNICAL_REVIEW = "technical_review"
    FEASIBILITY = "feasibility"

@dataclass
class ReportConfig:
    detail_level: DetailLevel
    report_type: ReportType
    include_metrics: bool = True
    include_examples: bool = True
    target_audience: str = "business"
    custom_sections: List[str] = None

class ContentGenerator:
    """Dynamic content generation based on topic analysis"""
    
    def __init__(self, topic: str, config: ReportConfig):
        self.topic = topic
        self.config = config
        self.topic_analysis = self._analyze_topic()
        self.generated_at = datetime.datetime.now()
    
    def _analyze_topic(self) -> Dict[str, Any]:
        """Analyze the topic to determine content structure and focus areas"""
        topic_lower = self.topic.lower()
        
        # Detect comparison topics
        comparison_keywords = ['vs', 'versus', 'compare', 'comparison', 'against', 'between']
        is_comparison = any(keyword in topic_lower for keyword in comparison_keywords)
        
        # Extract main subjects for comparison
        subjects = []
        if is_comparison:
            # Split on common comparison separators
            for sep in [' vs ', ' versus ', ' against ', ' and ', ' or ']:
                if sep in topic_lower:
                    parts = topic_lower.split(sep)
                    subjects = [part.strip() for part in parts if part.strip()]
                    break
        
        # Detect technical vs business focus
        tech_keywords = ['api', 'database', 'software', 'programming', 'algorithm', 'framework', 'technology']
        business_keywords = ['market', 'strategy', 'business', 'revenue', 'cost', 'roi', 'investment']
        
        is_technical = any(keyword in topic_lower for keyword in tech_keywords)
        is_business = any(keyword in topic_lower for keyword in business_keywords)
        
        # Detect industry/domain
        industries = {
            'healthcare': ['health', 'medical', 'hospital', 'patient', 'clinical'],
            'finance': ['financial', 'banking', 'investment', 'trading', 'fintech'],
            'technology': ['tech', 'software', 'digital', 'ai', 'machine learning'],
            'retail': ['retail', 'ecommerce', 'shopping', 'consumer'],
            'manufacturing': ['manufacturing', 'industrial', 'production', 'supply chain']
        }
        
        detected_industry = 'general'
        for industry, keywords in industries.items():
            if any(keyword in topic_lower for keyword in keywords):
                detected_industry = industry
                break
        
        return {
            'is_comparison': is_comparison,
            'subjects': subjects,
            'is_technical': is_technical,
            'is_business': is_business,
            'industry': detected_industry,
            'main_focus': self._extract_main_focus(topic_lower),
            'suggested_sections': self._suggest_sections(is_comparison, is_technical, is_business)
        }
    
    def _extract_main_focus(self, topic_lower: str) -> str:
        """Extract the main focus/domain from the topic"""
        # Remove common words and extract meaningful terms
        stop_words = {'in', 'for', 'of', 'the', 'and', 'or', 'vs', 'versus', 'compare', 'comparison', 'analysis', 'report'}
        words = re.findall(r'\b\w+\b', topic_lower)
        important_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(important_words[:3])  # Take first 3 important words
    
    def _suggest_sections(self, is_comparison: bool, is_technical: bool, is_business: bool) -> List[str]:
        """Suggest appropriate sections based on topic analysis"""
        base_sections = ['overview', 'analysis']
        
        if is_comparison:
            base_sections.extend(['comparison_table', 'advantages_disadvantages', 'use_cases'])
        else:
            base_sections.extend(['key_findings', 'implications'])
        
        if is_technical:
            base_sections.extend(['technical_specifications', 'implementation_considerations'])
        
        if is_business:
            base_sections.extend(['market_analysis', 'cost_benefit_analysis', 'roi_projections'])
        
        base_sections.extend(['recommendations', 'conclusion'])
        return base_sections

class UniversalReportGenerator:
    """Universal report generator that adapts to any topic"""
    
    def __init__(self, topic: str, config: ReportConfig):
        self.topic = topic
        self.config = config
        self.content_gen = ContentGenerator(topic, config)
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate a complete report for any topic"""
        analysis = self.content_gen.topic_analysis
        
        # Determine sections to include
        sections = self.config.custom_sections or analysis['suggested_sections']
        
        report = {
            'title': self._generate_title(),
            'metadata': self._generate_metadata(),
            'executive_summary': self._generate_executive_summary(),
        }
        
        # Dynamically generate sections based on topic analysis
        for section in sections:
            if hasattr(self, f'_generate_{section}'):
                report[section] = getattr(self, f'_generate_{section}')()
            else:
                report[section] = self._generate_generic_section(section)
        
        return report
    
    def _generate_title(self) -> str:
        """Generate appropriate title based on topic and report type"""
        if self.content_gen.topic_analysis['is_comparison']:
            return f"Comparative Analysis: {self.topic.title()}"
        elif self.config.report_type == ReportType.MARKET_STUDY:
            return f"Market Study: {self.topic.title()}"
        elif self.config.report_type == ReportType.FEASIBILITY:
            return f"Feasibility Study: {self.topic.title()}"
        else:
            return f"Research Report: {self.topic.title()}"
    
    def _generate_metadata(self) -> Dict[str, str]:
        return {
            'generated_date': self.content_gen.generated_at.strftime("%B %d, %Y at %H:%M:%S"),
            'detail_level': self.config.detail_level.value.title(),
            'report_type': self.config.report_type.value.replace('_', ' ').title(),
            'target_audience': self.config.target_audience.title(),
            'topic_classification': self.content_gen.topic_analysis['industry'].title(),
            'focus_area': self.content_gen.topic_analysis['main_focus'].title()
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary adapted to any topic"""
        analysis = self.content_gen.topic_analysis
        
        # Base content templates by detail level
        if self.config.detail_level == DetailLevel.BRIEF:
            if analysis['is_comparison']:
                content = f"This analysis compares {' and '.join(analysis['subjects'])} across key dimensions. The comparison reveals distinct advantages and use cases for each option, with selection criteria depending on specific requirements and constraints."
            else:
                content = f"This report analyzes {self.topic} examining key factors, opportunities, and challenges. The analysis provides insights for decision-making and strategic planning in the {analysis['industry']} domain."
        
        elif self.config.detail_level == DetailLevel.STANDARD:
            if analysis['is_comparison']:
                content = f"This comprehensive analysis examines {self.topic} across multiple evaluation criteria. The research identifies key differentiators, performance characteristics, and optimal use cases for each option. Findings indicate that selection should be based on specific operational requirements, budget constraints, and strategic objectives within the {analysis['industry']} context."
            else:
                content = f"This detailed analysis of {self.topic} examines current state, trends, and future implications within the {analysis['industry']} sector. The research methodology combines quantitative analysis with qualitative insights to provide actionable recommendations for stakeholders and decision-makers."
        
        else:  # COMPREHENSIVE
            if analysis['is_comparison']:
                content = f"This comprehensive comparative analysis evaluates {self.topic} through systematic examination of technical capabilities, economic factors, and strategic implications. The research methodology incorporates industry best practices, expert insights, and empirical data to provide definitive guidance for {analysis['industry']} organizations. Key findings highlight the importance of alignment between solution characteristics and organizational requirements, with particular emphasis on long-term strategic value and operational sustainability."
            else:
                content = f"This extensive research report provides in-depth analysis of {self.topic} within the broader context of {analysis['industry']} industry dynamics. The comprehensive methodology combines primary research, secondary analysis, and expert consultation to deliver actionable insights for strategic decision-making. The analysis encompasses current market conditions, emerging trends, competitive landscape, and future growth projections to support informed planning and investment decisions."
        
        # Generate key points based on topic analysis
        key_points = self._generate_key_points(analysis)
        
        return {
            'content': content,
            'key_points': key_points
        }
    
    def _generate_key_points(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate relevant key points based on topic analysis"""
        points = []
        
        if analysis['is_comparison']:
            if len(analysis['subjects']) >= 2:
                points.append(f"{analysis['subjects'][0].title()} offers specific advantages in certain use cases")
                points.append(f"{analysis['subjects'][1].title()} provides alternative benefits for different scenarios")
            points.append("Selection criteria should align with specific organizational needs")
            points.append("Implementation considerations vary significantly between options")
        else:
            points.append(f"Current {analysis['main_focus']} landscape presents both opportunities and challenges")
            points.append(f"Key success factors identified for {analysis['industry']} organizations")
            points.append("Strategic recommendations provided for optimal outcomes")
        
        if analysis['is_technical']:
            points.append("Technical specifications and implementation requirements analyzed")
        
        if analysis['is_business']:
            points.append("Business impact and ROI considerations evaluated")
        
        if self.config.detail_level == DetailLevel.COMPREHENSIVE:
            points.extend([
                "Long-term strategic implications assessed",
                "Risk mitigation strategies identified",
                "Industry best practices incorporated"
            ])
        
        return points
    
    def _generate_comparison_table(self) -> Dict[str, Any]:
        """Generate comparison table for comparison topics"""
        analysis = self.content_gen.topic_analysis
        
        if not analysis['is_comparison'] or len(analysis['subjects']) < 2:
            return {"note": "Comparison table not applicable for this topic"}
        
        subjects = analysis['subjects'][:2]  # Take first two subjects
        
        # Generic comparison categories that work for most topics
        categories = {
            'primary_focus': f"Core strengths and intended use cases",
            'complexity': f"Implementation and operational complexity",
            'cost_structure': f"Associated costs and investment requirements",
            'scalability': f"Growth and expansion capabilities",
            'maintenance': f"Ongoing maintenance and support needs",
            'expertise_required': f"Skills and knowledge requirements"
        }
        
        comparison = {}
        for category, description in categories.items():
            comparison[category] = {
                subjects[0].title(): f"Optimized for specific {category.replace('_', ' ')} requirements",
                subjects[1].title(): f"Alternative approach to {category.replace('_', ' ')} considerations",
                'description': description
            }
        
        return comparison
    
    def _generate_advantages_disadvantages(self) -> Dict[str, Any]:
        """Generate advantages and disadvantages for comparison topics"""
        analysis = self.content_gen.topic_analysis
        
        if not analysis['is_comparison']:
            return self._generate_pros_cons()
        
        subjects = analysis['subjects'][:2] if len(analysis['subjects']) >= 2 else ['Option A', 'Option B']
        
        return {
            subjects[0].title(): {
                'advantages': [
                    f"Strong performance in specific {analysis['main_focus']} scenarios",
                    f"Well-established ecosystem and community support",
                    f"Proven track record in {analysis['industry']} implementations",
                    f"Comprehensive documentation and learning resources"
                ],
                'disadvantages': [
                    f"May have limitations in certain use cases",
                    f"Potential scalability constraints in specific scenarios",
                    f"Higher complexity for simple {analysis['main_focus']} needs",
                    f"Resource requirements may be significant"
                ]
            },
            subjects[1].title(): {
                'advantages': [
                    f"Flexibility and adaptability for diverse {analysis['main_focus']} needs",
                    f"Modern architecture and design principles",
                    f"Optimized for contemporary {analysis['industry']} requirements",
                    f"Streamlined approach to common challenges"
                ],
                'disadvantages': [
                    f"Newer technology with evolving best practices",
                    f"Smaller ecosystem compared to established alternatives",
                    f"May require specialized expertise",
                    f"Limited long-term track record in some scenarios"
                ]
            }
        }
    
    def _generate_pros_cons(self) -> Dict[str, Any]:
        """Generate pros and cons for non-comparison topics"""
        analysis = self.content_gen.topic_analysis
        
        return {
            'advantages': [
                f"Addresses key challenges in {analysis['industry']} sector",
                f"Provides measurable improvements in {analysis['main_focus']}",
                f"Aligns with current industry trends and best practices",
                f"Offers competitive advantages when properly implemented"
            ],
            'disadvantages': [
                f"Implementation complexity may require significant resources",
                f"Change management challenges in traditional {analysis['industry']} environments",
                f"Initial investment and learning curve considerations",
                f"Ongoing maintenance and optimization requirements"
            ],
            'considerations': [
                f"Organizational readiness for {analysis['main_focus']} initiatives",
                f"Integration with existing {analysis['industry']} systems",
                f"Long-term strategic alignment and sustainability",
                f"Risk mitigation and contingency planning"
            ]
        }
    
    def _generate_use_cases(self) -> Dict[str, Any]:
        """Generate use cases adapted to any topic"""
        analysis = self.content_gen.topic_analysis
        
        use_cases = {
            'primary_applications': [
                f"Organizations seeking to optimize {analysis['main_focus']} operations",
                f"{analysis['industry'].title()} companies with specific performance requirements",
                f"Enterprises planning digital transformation initiatives",
                f"Teams requiring enhanced {analysis['main_focus']} capabilities"
            ],
            'industry_specific': {
                analysis['industry']: [
                    f"Large-scale {analysis['industry']} operations with complex requirements",
                    f"Mid-market {analysis['industry']} companies seeking competitive advantages",
                    f"Startups in {analysis['industry']} sector requiring scalable solutions"
                ]
            }
        }
        
        if analysis['is_comparison']:
            subjects = analysis['subjects'][:2] if len(analysis['subjects']) >= 2 else ['Option A', 'Option B']
            use_cases.update({
                f'{subjects[0]}_optimal_for': [
                    f"Scenarios requiring proven {analysis['main_focus']} approaches",
                    f"Organizations with established {analysis['industry']} workflows",
                    f"Use cases prioritizing stability and reliability"
                ],
                f'{subjects[1]}_optimal_for': [
                    f"Dynamic environments requiring {analysis['main_focus']} flexibility",
                    f"Modern {analysis['industry']} applications with evolving needs",
                    f"Organizations prioritizing innovation and adaptability"
                ]
            })
        
        return use_cases
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations adapted to any topic"""
        analysis = self.content_gen.topic_analysis
        
        recommendations = {
            'key_recommendations': [
                f"Conduct thorough assessment of current {analysis['main_focus']} requirements",
                f"Develop clear success criteria and measurement frameworks",
                f"Ensure adequate {analysis['industry']} expertise and resources",
                f"Plan for gradual implementation with pilot programs"
            ],
            'implementation_strategy': [
                f"Phase 1: Assessment and planning for {analysis['main_focus']} initiative",
                f"Phase 2: Pilot implementation in controlled {analysis['industry']} environment",
                f"Phase 3: Full deployment with performance monitoring",
                f"Phase 4: Optimization and continuous improvement"
            ]
        }
        
        if analysis['is_comparison']:
            subjects = analysis['subjects'][:2] if len(analysis['subjects']) >= 2 else ['Option A', 'Option B']
            recommendations.update({
                'selection_criteria': [
                    f"Evaluate {subjects[0]} for established {analysis['main_focus']} requirements",
                    f"Consider {subjects[1]} for flexible and evolving needs",
                    f"Assess organizational readiness and expertise availability",
                    f"Analyze long-term strategic alignment with {analysis['industry']} goals"
                ]
            })
        
        if self.config.detail_level == DetailLevel.COMPREHENSIVE:
            recommendations.update({
                'strategic_considerations': [
                    f"Align {analysis['main_focus']} strategy with broader organizational objectives",
                    f"Develop {analysis['industry']}-specific success metrics and KPIs",
                    f"Establish governance framework for ongoing management",
                    f"Create change management plan for stakeholder adoption"
                ],
                'risk_mitigation': [
                    f"Identify potential risks in {analysis['main_focus']} implementation",
                    f"Develop contingency plans for {analysis['industry']} specific challenges",
                    f"Establish monitoring and early warning systems",
                    f"Create rollback procedures for critical scenarios"
                ]
            })
        
        return recommendations
    
    def _generate_conclusion(self) -> Dict[str, Any]:
        """Generate conclusion adapted to any topic"""
        analysis = self.content_gen.topic_analysis
        
        if self.config.detail_level == DetailLevel.BRIEF:
            summary = f"Analysis of {self.topic} reveals important considerations for {analysis['industry']} organizations. Success depends on careful evaluation of requirements and systematic implementation approach."
        elif self.config.detail_level == DetailLevel.STANDARD:
            summary = f"This analysis of {self.topic} provides comprehensive insights for {analysis['industry']} decision-makers. Key findings emphasize the importance of alignment between {analysis['main_focus']} requirements and organizational capabilities. Strategic implementation with proper planning and resources will drive optimal outcomes."
        else:  # COMPREHENSIVE
            summary = f"This comprehensive analysis of {self.topic} demonstrates the complexity and strategic importance of {analysis['main_focus']} decisions in the {analysis['industry']} sector. The research findings indicate that successful outcomes require careful consideration of organizational readiness, technical requirements, and long-term strategic alignment. Organizations that invest in proper planning, stakeholder engagement, and systematic implementation will realize significant competitive advantages."
        
        conclusion = {'summary': summary}
        
        if analysis['is_comparison']:
            conclusion['final_recommendation'] = f"The choice between {' and '.join(analysis['subjects'])} should be based on specific organizational needs, technical requirements, and strategic objectives rather than universal preferences."
        else:
            conclusion['final_recommendation'] = f"Organizations should approach {analysis['main_focus']} initiatives with careful planning, adequate resources, and clear success criteria to maximize value and minimize risks."
        
        if self.config.detail_level == DetailLevel.COMPREHENSIVE:
            conclusion['key_insights'] = [
                f"{analysis['main_focus'].title()} success requires strategic alignment and organizational commitment",
                f"{analysis['industry'].title()} specific factors significantly influence implementation approaches",
                f"Long-term value depends on continuous optimization and adaptation",
                f"Cross-functional collaboration essential for optimal outcomes"
            ]
        
        return conclusion
    
    def _generate_generic_section(self, section_name: str) -> Dict[str, Any]:
        """Generate content for any section name"""
        analysis = self.content_gen.topic_analysis
        formatted_name = section_name.replace('_', ' ').title()
        
        return {
            'content': f"Analysis of {formatted_name.lower()} reveals important implications for {self.topic}. Key factors include strategic alignment with {analysis['industry']} requirements, operational considerations for {analysis['main_focus']} implementation, and long-term sustainability within organizational context.",
            'key_points': [
                f"{formatted_name} directly impacts {analysis['main_focus']} outcomes",
                f"{analysis['industry'].title()} specific factors influence {formatted_name.lower()} considerations",
                f"Strategic planning essential for optimal {formatted_name.lower()} results"
            ]
        }

class PDFReportFormatter:
    """Format any report data into PDF-ready structure"""
    
    def __init__(self, report_data: Dict[str, Any], config: ReportConfig):
        self.data = report_data
        self.config = config
    
    def format_for_pdf(self) -> str:
        """Convert any report data to formatted text suitable for PDF generation"""
        sections = []
        
        # Title and metadata
        sections.append(f"# {self.data['title']}")
        sections.append(f"**Generated:** {self.data['metadata']['generated_date']}")
        sections.append(f"**Report Type:** {self.data['metadata']['report_type']}")
        sections.append(f"**Detail Level:** {self.data['metadata']['detail_level']}")
        sections.append(f"**Focus Area:** {self.data['metadata']['focus_area']}")
        sections.append("")
        
        # Executive Summary
        sections.append("## Executive Summary")
        sections.append(self.data['executive_summary']['content'])
        if 'key_points' in self.data['executive_summary']:
            sections.append("\n**Key Points:**")
            for point in self.data['executive_summary']['key_points']:
                sections.append(f"• {point}")
        sections.append("")
        
        # Dynamic sections
        section_order = ['overview', 'analysis', 'comparison_table', 'advantages_disadvantages', 
                        'use_cases', 'recommendations', 'conclusion']
        
        for section_key in section_order:
            if section_key in self.data:
                sections.append(self._format_section(section_key, self.data[section_key]))
        
        # Handle any additional sections not in standard order
        for key, value in self.data.items():
            if key not in ['title', 'metadata', 'executive_summary'] + section_order:
                sections.append(self._format_section(key, value))
        
        return "\n".join(sections)
    
    def _format_section(self, section_key: str, section_data: Any) -> str:
        """Format any section regardless of structure"""
        section_title = section_key.replace('_', ' ').title()
        lines = [f"## {section_title}"]
        
        if isinstance(section_data, str):
            lines.append(section_data)
        elif isinstance(section_data, dict):
            lines.extend(self._format_dict_section(section_data))
        elif isinstance(section_data, list):
            for item in section_data:
                lines.append(f"• {item}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_dict_section(self, data: Dict[str, Any]) -> List[str]:
        """Format dictionary data into readable text"""
        lines = []
        
        for key, value in data.items():
            if key in ['content', 'summary']:
                lines.append(value)
                lines.append("")
            elif isinstance(value, str):
                lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
            elif isinstance(value, list):
                lines.append(f"**{key.replace('_', ' ').title()}:**")
                for item in value:
                    lines.append(f"• {item}")
                lines.append("")
            elif isinstance(value, dict):
                lines.append(f"### {key.replace('_', ' ').title()}")
                lines.extend(self._format_dict_section(value))
        
        return lines

# Main API function for backend integration
def generate_universal_report(topic: str, 
                            detail_level: str = "standard",
                            report_type: str = "analysis",
                            include_metrics: bool = True,
                            target_audience: str = "business",
                            custom_sections: List[str] = None) -> str:
    """
    Generate a report for ANY topic with flexible configuration
    
    Args:
        topic: Any topic string (e.g., "Python vs JavaScript", "AI in Healthcare", "Market Analysis of EVs")
        detail_level: "brief", "standard", or "comprehensive"
        report_type: "comparison", "analysis", "research", "market_study", "technical_review", "feasibility"
        include_metrics: Whether to include performance/metrics sections
        target_audience: "business", "technical", or "executive"
        custom_sections: List of custom section names to include
    
    Returns:
        Formatted text ready for PDF generation
    """
    try:
        config = ReportConfig(
            detail_level=DetailLevel(detail_level),
            report_type=ReportType(report_type),
            include_metrics=include_metrics,
            target_audience=target_audience,
            custom_sections=custom_sections
        )
        
        generator = UniversalReportGenerator(topic, config)
        report_data = generator.generate_report()
        
        formatter = PDFReportFormatter(report_data, config)
        return formatter.format_for_pdf()
        
    except Exception as e:
        # Fallback for any errors
        return f"# Report Generation Error\n\nUnable to generate report for topic: {topic}\nError: {str(e)}\n\nPlease try with a different topic or configuration."

# Example usage and testing
if __name__ == "__main__":
    # Test with various topics
    test_topics = [
        "Python vs JavaScript for web development",
        "AI implementation in healthcare",
        "Electric vehicles market analysis",
        "Cloud computing migration strategies",
        "Sustainable energy solutions",
        "Remote work productivity tools"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*50}")
        print(f"TESTING: {topic}")
        print('='*50)
        
        report = generate_universal_report(
            topic=topic,
            detail_level="standard",
            report_type="analysis"
        )
        
        # Show first 500 characters
        print(report[:500] + "...")