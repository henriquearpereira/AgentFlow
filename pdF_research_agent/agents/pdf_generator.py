"""
Enhanced PDF Generation module for AI Research Agent - Dynamic Content Processing
"""

import time
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    HRFlowable, PageBreak, KeepTogether, Image
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Add matplotlib for charting
import matplotlib.pyplot as plt
import tempfile
import os


class EnhancedPDFGenerator:
    """Handles PDF report generation with dynamic content processing and user input adaptation"""
    
    def __init__(self, report_data=None, config=None):
        """
        Initialize Enhanced PDFGenerator
        
        Args:
            report_data: Optional report data
            config: Optional configuration
        """
        self.report_data = report_data
        self.config = config
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Dynamic section detection based on content
        self.content_type = None
        self.detected_sections = []
        
        # Quality detection patterns - now dynamic based on content type
        self.quality_patterns = {
            'historical': {
                'good_indicators': [
                    r'\b\d{3,4}\s*(CE|BCE|AD|BC)\b',
                    r'\b\d{1,2}\w{2}\s+century\b',
                    r'\b(battle|treaty|kingdom|empire|conquest|dynasty|reign)\s+of\s+\w+',
                    r'\b(founded|established|conquered|signed|declared)\s+in\s+\d+',
                    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?:\s+(?:I{1,3}|IV|V|VI{0,3}|IX|X))?'  # Names with roman numerals
                ],
                'bad_indicators': [
                    r'\b(detailed analysis|comprehensive examination).*?\b(reveals|uncovers|shows)\b',
                    r'\b(broader context|wider implications).*?\b(relevant to|impacting)\b'
                ],
                'required_elements': ['dates', 'historical_figures', 'events']
            },
            'technical': {
                'good_indicators': [
                    r'\b\d+%|\$\d+|€\d+|\d+\.\d+\b',
                    r'\b(API|SDK|framework|architecture|implementation)\b',
                    r'\bhttps?://[^\s]+\b',
                    r'\b(performance|scalability|security|efficiency)\b'
                ],
                'bad_indicators': [
                    r'\b(evolving trends|emerging opportunities|competitive pressures)\b',
                    r'\b(market dynamics|industry trends)\b'
                ],
                'required_elements': ['technical_details', 'specifications', 'data']
            },
            'business': {
                'good_indicators': [
                    r'\b\d+%|\$\d+|€\d+|\d+\.\d+[KMB]?\b',
                    r'\b(growth|revenue|profit|market share|ROI)\b',
                    r'\b(Q[1-4]|quarterly|annual|fiscal year)\b',
                    r'\b(strategy|analysis|forecast|projection)\b'
                ],
                'bad_indicators': [
                    r'\b(detailed analysis|important considerations)\b',
                    r'\b(broader context and implications)\b'
                ],
                'required_elements': ['metrics', 'data', 'analysis']
            }
        }
    
    def _setup_custom_styles(self):
        """Define custom styles with better hierarchy support"""
        # Enhanced title style
        self.styles.add(ParagraphStyle(
            name='TitleCustom',
            fontSize=28,
            spaceAfter=30,
            spaceBefore=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#2C3E50"),
            fontName='Helvetica-Bold'
        ))
        
        # Main section headers (H1)
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            fontSize=18,
            spaceAfter=15,
            spaceBefore=25,
            textColor=colors.HexColor("#34495E"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ))
        
        # Subsection headers (H2)
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            fontSize=16,
            spaceAfter=12,
            spaceBefore=18,
            textColor=colors.HexColor("#5D6D7E"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ))
        
        # Sub-subsection headers (H3)
        self.styles.add(ParagraphStyle(
            name='SubSubsectionHeader',
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor("#7F8C8D"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ))
        
        # Enhanced body text with better line spacing
        self.styles.add(ParagraphStyle(
            name='BodyTextCustom',
            fontSize=11,
            leading=17,
            spaceAfter=10,
            spaceBefore=3,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#2C3E50")
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletText',
            fontSize=11,
            leading=16,
            spaceAfter=6,
            spaceBefore=2,
            leftIndent=20,
            bulletIndent=10,
            textColor=colors.HexColor("#2C3E50")
        ))
        
        # Historical date/event highlight
        self.styles.add(ParagraphStyle(
            name='HistoricalHighlight',
            fontSize=12,
            leading=16,
            spaceAfter=10,
            spaceBefore=8,
            leftIndent=15,
            rightIndent=15,
            textColor=colors.HexColor("#8B4513"),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor("#D2B48C"),
            borderPadding=8,
            backColor=colors.HexColor("#FFF8DC")
        ))
        
        # Quality indicators
        self.styles.add(ParagraphStyle(
            name='QualityExcellent',
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor("#27AE60"),
            borderPadding=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='QualityGood',
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor("#3498DB"),
            borderPadding=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='QualityFair',
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor("#F39C12"),
            borderPadding=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='QualityPoor',
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor("#E74C3C"),
            borderPadding=6
        ))
        
        # Content warning styles
        self.styles.add(ParagraphStyle(
            name='ContentWarning',
            fontSize=10,
            leading=14,
            spaceAfter=8,
            spaceBefore=8,
            textColor=colors.HexColor("#D68910"),
            fontName='Helvetica-Oblique',
            borderWidth=1,
            borderColor=colors.HexColor("#F4D03F"),
            borderPadding=6,
            backColor=colors.HexColor("#FEF9E7")
        ))
        
        # Metadata style
        self.styles.add(ParagraphStyle(
            name='MetadataText',
            fontSize=10,
            spaceAfter=6,
            spaceBefore=3,
            textColor=colors.HexColor("#7F8C8D"),
            fontName='Helvetica'
        ))
    
    def _detect_content_type(self, content: str) -> str:
        """Dynamically detect content type based on content analysis"""
        content_lower = content.lower()
        
        # Historical content indicators
        historical_score = 0
        historical_indicators = [
            'history', 'historical', 'century', 'medieval', 'ancient', 'kingdom', 'empire',
            'battle', 'conquest', 'dynasty', 'reign', 'treaty', 'founded', 'established',
            'ce', 'bce', 'ad', 'bc', 'chronology', 'timeline', 'era', 'period'
        ]
        for indicator in historical_indicators:
            historical_score += content_lower.count(indicator)
        
        # Technical content indicators
        technical_score = 0
        technical_indicators = [
            'api', 'sdk', 'framework', 'architecture', 'development', 'software',
            'algorithm', 'database', 'programming', 'code', 'technical', 'system',
            'implementation', 'performance', 'scalability', 'security'
        ]
        for indicator in technical_indicators:
            technical_score += content_lower.count(indicator)
        
        # Business content indicators
        business_score = 0
        business_indicators = [
            'market', 'business', 'revenue', 'profit', 'growth', 'strategy',
            'analysis', 'forecast', 'industry', 'competition', 'customer',
            'sales', 'marketing', 'investment', 'financial'
        ]
        for indicator in business_indicators:
            business_score += content_lower.count(indicator)
        
        # Determine primary content type
        scores = {
            'historical': historical_score,
            'technical': technical_score,
            'business': business_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 5 else 'general'
    
    def _extract_section_structure(self, content: str) -> list:
        """Dynamically extract the actual section structure from content"""
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect headers (markdown style)
            header_match = re.match(r'^(#{1,6})\s*(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                sections.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content_start': i + 1
                })
            
            # Detect numbered sections (e.g., "1. Introduction", "2.1 Overview")
            elif re.match(r'^\d+\.(\d+\.)*\s+.+$', line):
                parts = line.split('.', 2)
                level = len([p for p in parts[:-1] if p.strip().isdigit()])
                title = parts[-1].strip()
                sections.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content_start': i + 1
                })
            
            # Detect bullet point style sections
            elif re.match(r'^•\s+.+\*$', line):  # Sections ending with *
                title = line[2:-1].strip()  # Remove bullet and asterisk
                sections.append({
                    'level': 2,
                    'title': title,
                    'line_number': i,
                    'content_start': i + 1
                })
        
        # Add content ranges to sections
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                section['content_end'] = sections[i + 1]['line_number']
            else:
                section['content_end'] = len(lines)
                
            # Extract section content
            content_lines = lines[section['content_start']:section['content_end']]
            section['content'] = '\n'.join(content_lines).strip()
        
        return sections
    
    def _assess_content_quality(self, content: str) -> dict:
        """Enhanced quality assessment based on detected content type"""
        content_type = self._detect_content_type(content)
        patterns = self.quality_patterns.get(content_type, self.quality_patterns['business'])
        
        issues = []
        score = 100
        
        # Check for good indicators
        good_count = 0
        for pattern in patterns['good_indicators']:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            good_count += matches
        
        if good_count < 3:
            issues.append(f"Limited specific {content_type} content indicators ({good_count} found)")
            score -= 20
        
        # Check for bad indicators (generic placeholder text)
        bad_count = 0
        for pattern in patterns['bad_indicators']:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            bad_count += matches
            
        if bad_count > 2:
            issues.append(f"Generic placeholder text detected ({bad_count} instances)")
            score -= bad_count * 10
        
        # Content-specific assessments
        if content_type == 'historical':
            # Check for historical specificity
            has_dates = bool(re.search(r'\b\d{3,4}\s*(CE|BCE|AD|BC)\b|\b\d{1,2}\w{2}\s+century\b', content, re.IGNORECASE))
            has_names = bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:I{1,3}|IV|V|VI{0,3}|IX|X))?\b', content))
            has_events = bool(re.search(r'\b(battle|treaty|conquest|founding|establishment)\s+of\s+\w+', content, re.IGNORECASE))
            
            if not has_dates:
                issues.append("No specific historical dates found")
                score -= 15
            if not has_names:
                issues.append("No specific historical figures mentioned")
                score -= 10
            if not has_events:
                issues.append("No specific historical events described")
                score -= 10
        
        # Section structure assessment
        sections = self._extract_section_structure(content)
        if len(sections) < 3:
            issues.append(f"Limited section structure ({len(sections)} sections)")
            score -= 15
        
        # Word count assessment
        word_count = len(content.split())
        if word_count < 500:
            issues.append(f"Content too brief ({word_count} words)")
            score -= 20
        elif word_count < 1000:
            issues.append(f"Content could be more comprehensive ({word_count} words)")
            score -= 10
        
        # Check for sources
        has_sources = bool(re.search(r'https?://', content))
        if not has_sources:
            issues.append("No source URLs provided")
            score -= 10
        
        score = max(0, min(100, score))
        quality_level = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"
        
        return {
            'score': score,
            'level': quality_level,
            'issues': issues,
            'content_type': content_type,
            'section_count': len(sections),
            'word_count': word_count,
            'has_sources': has_sources,
            'sections': sections
        }
    
    def _get_quality_style(self, score: int) -> str:
        """Return appropriate quality style based on score"""
        if score >= 85:
            return 'QualityExcellent'
        elif score >= 70:
            return 'QualityGood'
        elif score >= 50:
            return 'QualityFair'
        else:
            return 'QualityPoor'
    
    def _clean_title(self, title: str) -> str:
        """Enhanced title cleaning with better formatting"""
        if not title:
            return title
        
        # Remove markdown markers
        title = re.sub(r'^#{1,6}\s*', '', title)
        title = re.sub(r'\*+$', '', title)  # Remove trailing asterisks
        
        # Remove numbering if present
        title = re.sub(r'^\d+\.(\d+\.)*\s*', '', title)
        
        # Clean up whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Capitalize properly
        return title.title() if not title.isupper() else title
    
    def _should_page_break(self, section_title: str, level: int) -> bool:
        """Dynamic page break determination based on content and structure"""
        title_lower = section_title.lower().strip()
        
        # Always break for major sections (level 1)
        if level == 1:
            return True
            
        # Break for specific important sections regardless of level
        important_sections = [
            'introduction', 'conclusion', 'methodology', 'results', 'analysis',
            'historical timeline', 'major events', 'current status', 'references',
            'sources', 'bibliography', 'appendix', 'recommendations'
        ]
        
        return any(section in title_lower for section in important_sections)
    
    def create_pdf(self, content: str, filename: str, topic: str = None, user_input: str = None) -> bool:
        """
        Create PDF report with dynamic content processing based on user input and content type
        
        Args:
            content: Report content
            filename: Output PDF filename
            topic: Optional topic override
            user_input: Original user input for context
            
        Returns:
            bool: Success status
        """
        print(f"📄 Creating dynamic PDF report: {filename}")
        
        # Assess content quality and structure
        quality_assessment = self._assess_content_quality(content)
        content_type = quality_assessment['content_type']
        sections = quality_assessment['sections']
        
        print(f"📊 Content Type: {content_type.title()}")
        print(f"📊 Quality: {quality_assessment['level']} ({quality_assessment['score']}/100)")
        print(f"📊 Sections Found: {len(sections)}")
        
        if quality_assessment['issues']:
            print("⚠️ Quality Issues:")
            for issue in quality_assessment['issues'][:3]:  # Show top 3 issues
                print(f"  - {issue}")
        
        try:
            # Create directory if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            doc = SimpleDocTemplate(
                filename, 
                pagesize=letter,
                leftMargin=0.8*inch,
                rightMargin=0.8*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )
            story = []
            
            # Enhanced title section
            story.append(Paragraph("AI Research Report", self.styles['TitleCustom']))
            
            # Quality badge with appropriate styling
            quality_style = self._get_quality_style(quality_assessment['score'])
            story.append(Paragraph(
                f"Content Quality: {quality_assessment['level']} ({quality_assessment['score']}/100)", 
                self.styles[quality_style]
            ))
            story.append(Spacer(1, 20))
            
            # Dynamic topic extraction
            if not topic and user_input:
                # Extract topic from user input
                topic = self._extract_topic_from_input(user_input)
            elif not topic:
                topic = Path(filename).stem.replace('_', ' ')
            
            cleaned_topic = self._clean_title(topic)
            story.append(Paragraph(f"Research Topic: {cleaned_topic}", self.styles['SubsectionHeader']))
            story.append(Spacer(1, 15))
            
            # Content type indicator
            story.append(Paragraph(f"Content Type: {content_type.title()}", self.styles['MetadataText']))
            story.append(Spacer(1, 10))
            
            # Metadata section
            story.append(Paragraph(f"Generated: {time.strftime('%B %d, %Y at %H:%M:%S')}", self.styles['MetadataText']))
            story.append(Paragraph(f"Sections: {len(sections)}", self.styles['MetadataText']))
            story.append(Paragraph(f"Word Count: {quality_assessment['word_count']}", self.styles['MetadataText']))
            
            if user_input:
                # Truncate user input if too long
                display_input = user_input[:100] + "..." if len(user_input) > 100 else user_input
                story.append(Paragraph(f"Based on: {display_input}", self.styles['MetadataText']))
            
            story.append(Spacer(1, 20))
            
            # Add quality warning if needed
            if quality_assessment['score'] < 70:
                warning_text = f"⚠️ This {content_type} report shows quality issues. Consider providing more specific requirements or using a more capable AI model."
                story.append(Paragraph(warning_text, self.styles['ContentWarning']))
                story.append(Spacer(1, 15))
            
            # Divider
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BDC3C7")))
            story.append(Spacer(1, 20))
            
            # Process content dynamically based on detected structure
            self._process_dynamic_content(content, story, sections, content_type)
            
            # Quality assessment section
            if quality_assessment['issues']:
                story.append(PageBreak())
                story.append(Paragraph("Content Quality Assessment", self.styles['SectionHeader']))
                
                story.append(Paragraph("Issues Identified:", self.styles['SubsectionHeader']))
                for issue in quality_assessment['issues']:
                    story.append(Paragraph(f"• {issue}", self.styles['BulletText']))
                
                story.append(Spacer(1, 15))
                story.append(Paragraph("Recommendations:", self.styles['SubsectionHeader']))
                
                # Content-type specific recommendations
                if content_type == 'historical':
                    recs = [
                        "Include specific dates (CE/BCE format preferred)",
                        "Mention key historical figures with full names",
                        "Describe major events with context and significance",
                        "Add primary and secondary source references"
                    ]
                elif content_type == 'technical':
                    recs = [
                        "Include technical specifications and metrics",
                        "Add code examples or API documentation links", 
                        "Provide performance benchmarks and data",
                        "Include architectural diagrams or flowcharts"
                    ]
                else:
                    recs = [
                        "Add specific data points and metrics",
                        "Include credible source links and references",
                        "Provide more detailed analysis and insights",
                        "Structure content with clear sections and subsections"
                    ]
                
                for rec in recs:
                    story.append(Paragraph(f"• {rec}", self.styles['BulletText']))
            
            # Build PDF
            doc.build(story)
            print(f"✅ Dynamic PDF created successfully: {filename}")
            return True
            
        except Exception as e:
            print(f"⚠️ PDF creation failed: {e}")
            return False
    
    def _extract_topic_from_input(self, user_input: str) -> str:
        """Extract meaningful topic from user input"""
        # Remove common question words and phrases
        cleaned = re.sub(r'\b(tell me about|research|analyze|explain|describe|what is|how does|write about)\b', '', user_input, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(a|an|the|of|in|on|at|to|for|with|by)\b', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Take first few meaningful words
        words = cleaned.split()[:6]
        return ' '.join(words).title()
    
    def _process_dynamic_content(self, content: str, story: list, sections: list, content_type: str):
        """Process content dynamically based on detected structure and type"""
        if not sections:
            # Fallback: process as plain text with basic formatting
            self._process_plain_content(content, story)
            return
        
        # Process each section according to its structure
        for i, section in enumerate(sections):
            # Determine appropriate styling based on level
            if section['level'] == 1:
                header_style = self.styles['SectionHeader']
            elif section['level'] == 2:
                header_style = self.styles['SubsectionHeader']
            else:
                header_style = self.styles['SubSubsectionHeader']
            
            # Add page break for major sections
            if self._should_page_break(section['title'], section['level']) and i > 0:
                story.append(PageBreak())
            
            # Add section header
            story.append(Paragraph(section['title'], header_style))
            
            # Process section content
            if section['content'].strip():
                self._process_section_content(section['content'], story, content_type)
            
            # Add spacing between sections
            if i < len(sections) - 1:
                story.append(Spacer(1, 20))
    
    def _process_section_content(self, content: str, story: list, content_type: str):
        """Process individual section content with content-type awareness"""
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Check for special content patterns based on type
            if content_type == 'historical' and self._is_historical_highlight(paragraph):
                story.append(Paragraph(paragraph.strip(), self.styles['HistoricalHighlight']))
                story.append(Spacer(1, 8))
                continue
            
            # Process bullet points
            lines = paragraph.split('\n')
            if any(line.strip().startswith(('•', '-', '*')) for line in lines):
                for line in lines:
                    line = line.strip()
                    if line.startswith(('•', '-', '*')):
                        bullet_text = line[1:].strip()
                        story.append(Paragraph(f"• {bullet_text}", self.styles['BulletText']))
            else:
                # Regular paragraph
                combined_text = ' '.join(line.strip() for line in lines if line.strip())
                if combined_text:
                    story.append(Paragraph(combined_text, self.styles['BodyTextCustom']))
            
            story.append(Spacer(1, 8))
    
    def _is_historical_highlight(self, text: str) -> bool:
        """Check if text contains important historical information that should be highlighted"""
        highlight_patterns = [
            r'\b\d{3,4}\s*(CE|BCE|AD|BC)\b.*\b(battle|treaty|founded|established|conquered|signed)\b',
            r'\b(battle|treaty|conquest)\s+of\s+\w+.*\b\d{3,4}\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:I{1,3}|IV|V|VI{0,3}|IX|X))?\b.*\b(declared|proclaimed|crowned|defeated)\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in highlight_patterns)
    
    def _process_plain_content(self, content: str, story: list):
        """Fallback method for content without clear structure"""
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Simple formatting
            if paragraph.strip().isupper() or len(paragraph.strip()) < 50:
                # Treat as potential header
                story.append(Paragraph(paragraph.strip(), self.styles['SubsectionHeader']))
            else:
                # Regular paragraph
                story.append(Paragraph(paragraph.strip(), self.styles['BodyTextCustom']))
            
            story.append(Spacer(1, 10))

    def create_text_report(self, content: str, filename: str) -> bool:
        """Create plain text report as fallback with quality assessment"""
        try:
            txt_filename = Path(filename)
            if txt_filename.suffix.lower() != '.txt':
                txt_filename = txt_filename.with_suffix('.txt')
            txt_filename.parent.mkdir(parents=True, exist_ok=True)

            # Assess quality
            quality_assessment = self._assess_content_quality(content)

            topic = txt_filename.stem.replace('_', ' ')
            topic = re.sub(r'_\d{8}_\d{4}$', '', topic)
            cleaned_topic = self._clean_title(topic)

            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("AI RESEARCH REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Research Topic: {cleaned_topic}\n")
                f.write(f"Generated: {time.strftime('%B %d, %Y at %H:%M:%S')}\n")
                f.write("Powered by: AI Research Agent\n")
                f.write(f"Content Quality: {quality_assessment['level']} ({quality_assessment['score']}/100)\n")
                f.write("-" * 80 + "\n\n")

                if quality_assessment['issues']:
                    f.write("QUALITY ISSUES DETECTED:\n")
                    for issue in quality_assessment['issues']:
                        f.write(f"- {issue}\n")
                    f.write("\n" + "-" * 80 + "\n\n")

                f.write(content)
                f.write("\n\n" + "-" * 80 + "\n")
                f.write("This report was automatically generated by AI Research Agent\n")
                f.write("=" * 80 + "\n\n")

                # Add helpful guidance for PDF viewing
                f.write("\U0001F4CB PDF VIEWING INSTRUCTIONS:\n")
                f.write("-" * 40 + "\n")
                f.write("If you're seeing raw PDF data (gibberish text), you're viewing the PDF as text.\n")
                f.write("To properly view the formatted PDF report:\n")
                f.write("1. Right-click the .pdf file\n")
                f.write("2. Select 'Open with' 14 Choose a PDF viewer\n")
                f.write("3. Or double-click if associated with a PDF viewer\n")
                f.write("\nRecommended PDF viewers:\n")
                f.write("- Adobe Reader (free)\n")
                f.write("- Google Chrome (built-in PDF viewer)\n")
                f.write("- Mozilla Firefox (built-in PDF viewer)\n")
                f.write("- Microsoft Edge (built-in PDF viewer)\n")
                f.write("- Any PDF viewer application\n")
                f.write("\nThis text file (.txt) is provided for easier reading and searching.\n")

            print(f"\U0001F4D1 Text report with quality assessment saved: {txt_filename}")
            return True
        except Exception as e:
            print(f"⚠️ Text report creation failed: {e}")
            return False


# Backward compatibility
class PDFGenerator(EnhancedPDFGenerator):
    """Backward compatible wrapper"""
    pass