"""
PDF Generation module for AI Research Agent - Enhanced Version with Quality Detection
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


class PDFGenerator:
    """Handles PDF report generation with enhanced formatting and quality detection"""
    
    def __init__(self, report_data=None, config=None):
        """
        Initialize PDFGenerator with optional parameters to maintain compatibility
        
        Args:
            report_data: Optional report data (for compatibility)
            config: Optional configuration (for compatibility)
        """
        self.report_data = report_data
        self.config = config
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Define sections that should start on new pages
        self.page_break_sections = {
            'primary sources', 'data sources', 'methodology', 'references',
            'appendix', 'bibliography', 'conclusions', 'recommendations',
            'detailed analysis', 'technical specifications', 'future outlook'
        }
        
        # Quality detection patterns - improved for historical content
        self.low_quality_patterns = [
            r'\b(detailed analysis|comprehensive examination).*?\b(reveals|uncovers|shows)\b.*?\b(important considerations|key insights|significant factors)\b',
            r'\b(market dynamics|industry trends|current patterns).*?\b(show|demonstrate|indicate)\b.*?\b(evolving trends|changing patterns|developing characteristics)\b',
            r'\b(broader context|wider implications|overall perspective).*?\b(relevant to|impacting|affecting)\b.*?\b(understanding|comprehension|analysis)\b'
        ]
        
        self.placeholder_phrases = [
            'detailed analysis of',
            'important considerations and insights',
            'broader context and implications',
            'evolving trends with emerging opportunities',
            'competitive pressures shaping the landscape'
        ]
    
    def _setup_custom_styles(self):
        """Define custom styles for professional formatting with improved spacing"""
        # Custom title style with more space after
        self.styles.add(ParagraphStyle(
            name='TitleCustom',
            fontSize=28,
            spaceAfter=30,
            spaceBefore=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#2C3E50"),
            fontName='Helvetica-Bold'
        ))
        
        # Custom section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor("#34495E"),
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=0
        ))
        
        # Custom subsection header with better spacing
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor("#5D6D7E"),
            fontName='Helvetica-Bold'
        ))
        
        # Enhanced body text
        self.styles.add(ParagraphStyle(
            name='BodyTextCustom',
            fontSize=11,
            leading=16,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#2C3E50")
        ))
        
        # Highlighted text for key insights
        self.styles.add(ParagraphStyle(
            name='HighlightText',
            fontSize=12,
            leading=16,
            spaceAfter=10,
            spaceBefore=10,
            textColor=colors.HexColor("#E74C3C"),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor("#E74C3C"),
            borderPadding=8,
            backColor=colors.HexColor("#FCF3CF")
        ))
        
        # Warning style for low-quality content
        self.styles.add(ParagraphStyle(
            name='WarningText',
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
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='FooterText',
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.grey,
            fontName='Helvetica-Oblique'
        ))
        
        # Metadata style with better spacing
        self.styles.add(ParagraphStyle(
            name='MetadataText',
            fontSize=10,
            spaceAfter=6,
            spaceBefore=3,
            textColor=colors.HexColor("#7F8C8D"),
            fontName='Helvetica'
        ))
        
        # Topic title style (separate from section headers)
        self.styles.add(ParagraphStyle(
            name='TopicTitle',
            fontSize=18,
            spaceAfter=15,
            spaceBefore=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#34495E"),
            fontName='Helvetica-Bold'
        ))
        
        # Quality badge styles
        self.styles.add(ParagraphStyle(
            name='QualityGood',
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor("#27AE60"),
            borderPadding=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='QualityPoor',
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor("#E74C3C"),
            borderPadding=4
        ))
    
    def _assess_content_quality(self, content: str) -> dict:
        """
        Assess the quality of generated content
        
        Returns:
            dict: Quality assessment with score, issues, and recommendations
        """
        issues = []
        score = 100
        
        # Check for low-quality patterns
        for pattern in self.low_quality_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                issues.append(f"Generic placeholder text found ({matches} instances)")
                score -= matches * 15
        
        # Check for placeholder phrases
        placeholder_count = 0
        for phrase in self.placeholder_phrases:
            placeholder_count += len(re.findall(re.escape(phrase), content, re.IGNORECASE))
        
        if placeholder_count > 3:
            issues.append(f"High placeholder content ({placeholder_count} instances)")
            score -= placeholder_count * 5
        
        # Check for actual data presence - improved for historical content
        has_numbers = bool(re.search(r'\d+[.,]\d+|\$\d+|€\d+|%|\d+\s*(people|users|companies|CE|BCE|AD|years|century)', content))
        has_urls = bool(re.search(r'https?://', content))
        has_specific_names = bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', content))
        has_dates = bool(re.search(r'\b\d{3,4}\s*(CE|BCE|AD|BC)\b|\b\d{1,2}\w{2}\s+century\b', content, re.IGNORECASE))
        has_historical_terms = bool(re.search(r'\b(battle|treaty|kingdom|empire|dynasty|conquest|reconquest|monarchy|republic)\b', content, re.IGNORECASE))
        
        # For historical content, dates and historical terms are more important than URLs
        if not has_dates and not has_historical_terms:
            issues.append("No specific historical dates or terms found")
            score -= 15
        
        if not has_numbers and not has_dates:
            issues.append("No specific numerical data or dates found")
            score -= 10
        
        if not has_urls:
            issues.append("No source URLs provided")
            score -= 10  # Reduced penalty for historical content
        
        # Content length assessment
        word_count = len(content.split())
        if word_count < 200:
            issues.append(f"Content too brief ({word_count} words)")
            score -= 25
        elif word_count < 500:
            issues.append(f"Content could be more detailed ({word_count} words)")
            score -= 10
        
        # Repetitive content check
        sentences = content.split('.')
        unique_sentences = set(sentence.strip().lower() for sentence in sentences if len(sentence.strip()) > 10)
        if len(sentences) > 10 and len(unique_sentences) / len(sentences) < 0.7:
            issues.append("High content repetition detected")
            score -= 20
        
        # Check for section structure
        has_sections = bool(re.search(r'^#{1,3}\s+', content, re.MULTILINE))
        if not has_sections:
            issues.append("No clear section structure found")
            score -= 10
        
        score = max(0, min(100, score))
        
        quality_level = "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
        
        return {
            'score': score,
            'level': quality_level,
            'issues': issues,
            'has_data': has_numbers or has_dates,
            'has_sources': has_urls,
            'has_historical_content': has_dates or has_historical_terms,
            'word_count': word_count
        }
    
    def _clean_title(self, title: str) -> str:
        """Clean and improve title formatting with better hashtag removal"""
        if not title:
            return title
        
        # Remove ALL markdown hashtag markers (handles ###, ####, etc.)
        title = re.sub(r'^#{1,6}\s*', '', title)
        title = re.sub(r'#{1,6}\s*$', '', title)
        
        # Remove extra whitespace and normalize
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Handle special cases and improve readability
        title = re.sub(r'[_-]+', ' ', title)
        
        # Fix common patterns
        title_fixes = {
            r'\b(ai|ml|ai/ml)\b': 'AI/ML',
            r'\bapi\b': 'API',
            r'\bui\b': 'UI',
            r'\bux\b': 'UX',
            r'\bsdk\b': 'SDK',
            r'\baws\b': 'AWS',
            r'\bgcp\b': 'GCP',
            r'\bsql\b': 'SQL',
            r'\bnosql\b': 'NoSQL',
            r'\bmongodb\b': 'MongoDB',
            r'\bpostgresql\b': 'PostgreSQL',
            r'\bmysql\b': 'MySQL',
            r'\bredis\b': 'Redis',
            r'\breact\b': 'React',
            r'\bpython\b': 'Python',
            r'\bjava\b': 'Java',
            r'\bdevops\b': 'DevOps',
        }
        
        # Apply fixes (case insensitive)
        for pattern, replacement in title_fixes.items():
            title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)
        
        # Capitalize first letter of each word (title case) but preserve known acronyms
        words = title.split()
        title_cased_words = []
        
        for word in words:
            if word.isupper() and len(word) <= 4:
                title_cased_words.append(word)
            elif word in ['AI/ML', 'UI/UX', 'CI/CD']:
                title_cased_words.append(word)
            else:
                title_cased_words.append(word.capitalize())
        
        title = ' '.join(title_cased_words)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def _should_page_break(self, section_title: str) -> bool:
        """Determine if a section should start on a new page"""
        title_lower = section_title.lower().strip()
        return any(keyword in title_lower for keyword in self.page_break_sections)
    
    def _filter_low_quality_content(self, content: str) -> str:
        """
        Filter out or mark low-quality placeholder content
        
        Args:
            content: Original content
            
        Returns:
            str: Filtered content with quality warnings
        """
        filtered_lines = []
        
        for line in content.split('\n'):
            line_lower = line.lower().strip()
            
            # Check if line is generic placeholder
            is_placeholder = any(
                pattern in line_lower 
                for pattern in [
                    'detailed analysis of',
                    'reveals important considerations',
                    'broader context and implications',
                    'evolving trends with emerging',
                    'competitive pressures shaping'
                ]
            )
            
            if is_placeholder:
                # Replace with warning or skip entirely
                if len(line.strip()) > 20:  # Only warn for substantial placeholder text
                    filtered_lines.append(f"⚠️ [AI Generated Placeholder Content Detected]")
            else:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def generate_pdf(self, content: str, filename: str, topic: str = None) -> bool:
        """Alternative method name for compatibility - calls create_pdf"""
        return self.create_pdf(content, filename, topic)
    
    def create_pdf(self, content: str, filename: str, topic: str = None) -> bool:
        """
        Create PDF report with enhanced formatting and quality assessment
        
        Args:
            content: Report content in markdown format
            filename: Output PDF filename
            topic: Optional topic for title extraction
            
        Returns:
            bool: Success status
        """
        print(f"📄 Creating enhanced PDF with quality assessment: {filename}")
        
        # Assess content quality first
        quality_assessment = self._assess_content_quality(content)
        print(f"📊 Content Quality: {quality_assessment['level']} ({quality_assessment['score']}/100)")
        
        if quality_assessment['issues']:
            print("⚠️ Quality Issues Detected:")
            for issue in quality_assessment['issues']:
                print(f"  - {issue}")
        
        try:
            # Create directory if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            temp_images = []  # Track temp images for cleanup
            
            # Branding config
            logo_path = None
            org_name = None
            primary_color = None
            if self.config:
                logo_path = self.config.get('logo_path')
                org_name = self.config.get('org_name')
                primary_color = self.config.get('primary_color')
            
            doc = SimpleDocTemplate(
                filename, 
                pagesize=letter,
                leftMargin=0.8*inch,
                rightMargin=0.8*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )
            story = []
            
            # Table of Contents setup
            toc = TableOfContents()
            toc.levelStyles = [
                ParagraphStyle(fontSize=14, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=10, spaceAfter=8, leading=18),
                ParagraphStyle(fontSize=12, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=7, spaceAfter=6, leading=16),
            ]
            story.append(Paragraph("Table of Contents", self.styles['TitleCustom']))
            story.append(Spacer(1, 12))
            story.append(toc)
            story.append(PageBreak())
            
            # Enhanced title section with branding
            if logo_path and Path(logo_path).exists():
                story.append(Image(logo_path, width=2*inch, height=2*inch))
                story.append(Spacer(1, 10))
            story.append(Paragraph("AI Research Report", self.styles['TitleCustom']))
            if org_name:
                story.append(Paragraph(f"<b>{org_name}</b>", self.styles['MetadataText']))
                story.append(Spacer(1, 8))
            
            # Add quality badge
            quality_style = 'QualityGood' if quality_assessment['score'] >= 60 else 'QualityPoor'
            story.append(Paragraph(
                f"Content Quality: {quality_assessment['level']} ({quality_assessment['score']}/100)", 
                self.styles[quality_style]
            ))
            story.append(Spacer(1, 20))
            
            # Topic extraction and formatting
            if not topic:
                topic = Path(filename).stem.replace('_', ' ')
                topic = re.sub(r'_\d{8}_\d{4}$', '', topic)
            
            cleaned_topic = self._clean_title(topic)
            story.append(Paragraph(f"Research Topic: {cleaned_topic}", self.styles['TopicTitle']))
            story.append(Spacer(1, 15))
            
            # Add quality warning if needed
            if quality_assessment['score'] < 60:
                warning_text = "⚠️ This report contains AI-generated placeholder content. " \
                              "Consider regenerating with a more capable model for better results."
                story.append(Paragraph(warning_text, self.styles['WarningText']))
                story.append(Spacer(1, 10))
            elif quality_assessment['score'] < 80:
                improvement_text = "💡 This report could benefit from more specific historical details, dates, and source citations."
                story.append(Paragraph(improvement_text, self.styles['HighlightText']))
                story.append(Spacer(1, 10))
            
            # Add divider line
            story.append(HRFlowable(
                width="100%", 
                thickness=1, 
                color=colors.HexColor("#BDC3C7"),
                spaceBefore=5,
                spaceAfter=20
            ))
            
            # Metadata section
            story.append(Paragraph(f"<b>Generated:</b> {time.strftime('%B %d, %Y at %H:%M:%S')}", self.styles['MetadataText']))
            story.append(Paragraph("<b>Powered by:</b> AI Research Agent", self.styles['MetadataText']))
            
            # Add quality metrics
            story.append(Paragraph(f"<b>Word Count:</b> {quality_assessment['word_count']}", self.styles['MetadataText']))
            story.append(Paragraph(f"<b>Contains Data:</b> {'Yes' if quality_assessment['has_data'] else 'No'}", self.styles['MetadataText']))
            story.append(Paragraph(f"<b>Source URLs:</b> {'Yes' if quality_assessment['has_sources'] else 'No'}", self.styles['MetadataText']))
            story.append(Paragraph(f"<b>Historical Content:</b> {'Yes' if quality_assessment.get('has_historical_content', False) else 'No'}", self.styles['MetadataText']))
            story.append(Spacer(1, 30))
            
            # Filter and process content
            filtered_content = self._filter_low_quality_content(content)
            section_bookmarks = []
            self._process_content_with_toc(filtered_content, story, toc, temp_images, section_bookmarks)
            
            # Add quality issues section if any
            if quality_assessment['issues']:
                story.append(PageBreak())
                story.append(Paragraph("Quality Assessment Notes", self.styles['SectionHeader']))
                story.append(Paragraph(
                    "The following issues were detected in the generated content:",
                    self.styles['BodyTextCustom']
                ))
                for issue in quality_assessment['issues']:
                    story.append(Paragraph(f"• {issue}", self.styles['BodyTextCustom']))
                story.append(Spacer(1, 10))
                
                # Provide specific guidance based on content type
                if quality_assessment.get('has_historical_content', False):
                    story.append(Paragraph(
                        "For historical content, consider using a more advanced AI model (e.g., GPT-4, Claude-3, Llama-3) " \
                        "that can provide more specific dates, historical figures, and detailed events.",
                        self.styles['HighlightText']
                    ))
                else:
                    story.append(Paragraph(
                        "Consider using a more advanced AI model (e.g., GPT-4, Claude-3, Llama-3) for higher quality output.",
                        self.styles['HighlightText']
                    ))
            
            # Add footer section
            story.append(Spacer(1, 30))
            story.append(HRFlowable(
                width="100%", 
                thickness=0.5, 
                color=colors.HexColor("#BDC3C7")
            ))
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                "This report was automatically generated by AI Research Agent", 
                self.styles['FooterText']
            ))
            
            doc.build(story, onFirstPage=lambda c, d: self._add_toc_bookmarks(c, d, section_bookmarks, first_page=True), onLaterPages=lambda c, d: self._add_toc_bookmarks(c, d, section_bookmarks, first_page=False))
            # Clean up temp images
            for img_path in temp_images:
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            print(f"✅ Enhanced PDF created successfully: {filename}")
            return True
            
        except Exception as e:
            print(f"⚠️ PDF creation failed: {e}")
            return False
    
    def _process_content_with_toc(self, content: str, story: list, toc, temp_images=None, section_bookmarks=None):
        """Process markdown content, add section numbering, collect TOC entries, and record bookmarks for clickable TOC."""
        if temp_images is None:
            temp_images = []
        if section_bookmarks is None:
            section_bookmarks = []
        sections = re.split(r'\n(?=##)', content)
        section_num = 0
        subsection_num = 0
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            lines = section.split('\n')
            section_title = lines[0].strip()
            section_content = '\n'.join(lines[1:]).strip()
            if section_title:
                cleaned_title = self._clean_title(section_title)
                # Normalize header: remove stray '#' and extra whitespace
                cleaned_title = re.sub(r'^#+\s*', '', cleaned_title).strip()
                if not cleaned_title:
                    continue
                hashtag_match = re.match(r'^(#+)', section_title.strip())
                hashtag_count = len(hashtag_match.group()) if hashtag_match else 2
                if hashtag_count <= 2:
                    section_num += 1
                    subsection_num = 0
                    numbered_title = f"{section_num}. {cleaned_title}"
                    title_style = self.styles['SectionHeader']
                    toc.addEntry(0, numbered_title, len(story))
                    bookmark_name = f"section_{section_num}"
                    section_bookmarks.append((len(story), bookmark_name, numbered_title, 0))
                else:
                    subsection_num += 1
                    numbered_title = f"{section_num}.{subsection_num} {cleaned_title}"
                    title_style = self.styles['SubsectionHeader']
                    toc.addEntry(1, numbered_title, len(story))
                    bookmark_name = f"section_{section_num}_{subsection_num}"
                    section_bookmarks.append((len(story), bookmark_name, numbered_title, 1))
                if hashtag_count <= 2 and self._should_page_break(cleaned_title):
                    story.append(PageBreak())
                    story.append(Spacer(1, 12))
                # Add anchor for bookmark
                story.append(Paragraph(f'<a name="{bookmark_name}"/>{numbered_title}', title_style))
                # Check for salary data tables and chart
                if self._contains_salary_data(section_content):
                    salary_table, salary_values = self._extract_salary_table(section_content, return_values=True)
                    if salary_table:
                        story.append(Spacer(1, 10))
                        story.append(salary_table)
                        story.append(Spacer(1, 10))
                        # Generate and insert chart if values present
                        if salary_values:
                            chart_path = self._generate_salary_chart(salary_values)
                            if chart_path:
                                temp_images.append(chart_path)
                                story.append(Image(chart_path, width=4*inch, height=2.5*inch))
                                story.append(Spacer(1, 15))
                self._process_section_content(section_content, story)
                if i < len(sections) - 2:
                    story.append(Spacer(1, 25))
    
    def _process_section_content(self, content: str, story: list):
        """Process individual section content with improved historical content handling"""
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            lines = paragraph.split('\n')
            
            # Process bullet points
            if any(line.strip().startswith(('-', '*', '•')) for line in lines):
                for line in lines:
                    line = line.strip()
                    if line.startswith(('-', '*', '•')):
                        bullet_text = line[1:].strip()
                        bullet_text = self._clean_bullet_text(bullet_text)
                        
                        # Check if bullet point is placeholder - improved for historical content
                        is_placeholder = any(
                            phrase in bullet_text.lower() 
                            for phrase in self.placeholder_phrases
                        ) and not any(
                            historical_term in bullet_text.lower()
                            for historical_term in ['battle', 'treaty', 'kingdom', 'empire', 'dynasty', 'conquest', 'reconquest', 'monarchy', 'republic', 'CE', 'BCE', 'AD', 'BC']
                        )
                        
                        if is_placeholder:
                            story.append(Paragraph(f"• [Placeholder Content] {bullet_text}", self.styles['WarningText']))
                        elif self._is_key_insight(bullet_text):
                            story.append(Paragraph(f"• {bullet_text}", self.styles['HighlightText']))
                        else:
                            story.append(Paragraph(f"• {bullet_text}", self.styles['BodyTextCustom']))
                        story.append(Spacer(1, 4))
            
            # Process URLs
            elif any('http' in line for line in lines):
                for line in lines:
                    if 'http' in line:
                        url_pattern = r'(https?://[^\s]+)'
                        formatted_line = re.sub(url_pattern, r'<link href="\1">\1</link>', line)
                        story.append(Paragraph(formatted_line, self.styles['BodyTextCustom']))
                    else:
                        story.append(Paragraph(line, self.styles['BodyTextCustom']))
            
            # Regular paragraphs
            else:
                combined_text = ' '.join(line.strip() for line in lines if line.strip())
                if combined_text:
                    # Check for placeholder content in paragraphs - improved for historical content
                    is_placeholder = any(
                        re.search(pattern, combined_text, re.IGNORECASE) 
                        for pattern in self.low_quality_patterns
                    ) and not any(
                        historical_term in combined_text.lower()
                        for historical_term in ['battle', 'treaty', 'kingdom', 'empire', 'dynasty', 'conquest', 'reconquest', 'monarchy', 'republic', 'CE', 'BCE', 'AD', 'BC', 'century', 'medieval', 'ancient']
                    )
                    
                    if is_placeholder:
                        story.append(Paragraph(f"[Low Quality Content] {combined_text}", self.styles['WarningText']))
                    else:
                        story.append(Paragraph(combined_text, self.styles['BodyTextCustom']))
            
            story.append(Spacer(1, 10))
    
    def _clean_bullet_text(self, text: str) -> str:
        """Clean bullet point text, especially for bold/highlighted items"""
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        text = re.sub(r'\bsalary range\b', 'Salary Range', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmedian compensation\b', 'Median Compensation', text, flags=re.IGNORECASE)
        text = re.sub(r'\baverage compensation\b', 'Average Compensation', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdata points analyzed\b', 'Data Points Analyzed', text, flags=re.IGNORECASE)
        
        return text
    
    def _contains_salary_data(self, content: str) -> bool:
        """Check if content contains salary/compensation data"""
        salary_keywords = ['salary', 'compensation', '€', '$', 'wage', 'income', 'pay']
        return any(keyword.lower() in content.lower() for keyword in salary_keywords)
    
    def _extract_salary_table(self, content: str, return_values=False):
        """Extract and format salary data into a professional table. Optionally return values for charting."""
        salary_data = []
        values = {}
        range_match = re.search(r'€([\d,]+)\s*-\s*€([\d,]+)', content)
        if range_match:
            salary_data.append(['Salary Range', f"€{range_match.group(1)} - €{range_match.group(2)}"])
            try:
                values['Range Low'] = int(range_match.group(1).replace(',', ''))
                values['Range High'] = int(range_match.group(2).replace(',', ''))
            except Exception:
                pass
        median_match = re.search(r'[Mm]edian[:\s]*€([\d,]+)', content)
        if median_match:
            salary_data.append(['Median Compensation', f"€{median_match.group(1)}"])
            try:
                values['Median'] = int(median_match.group(1).replace(',', ''))
            except Exception:
                pass
        avg_match = re.search(r'[Aa]verage[:\s]*€([\d,]+)', content)
        if avg_match:
            salary_data.append(['Average Compensation', f"€{avg_match.group(1)}"])
            try:
                values['Average'] = int(avg_match.group(1).replace(',', ''))
            except Exception:
                pass
        if salary_data:
            table_data = [['Metric', 'Value']] + salary_data
            table = Table(table_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 11),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#BDC3C7")),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            if return_values:
                return table, values
            return table
        if return_values:
            return None, None
        return None

    def _generate_salary_chart(self, values: dict) -> str:
        """Generate a bar chart for salary data and return the image path."""
        if not values:
            return None
        labels = []
        data = []
        if 'Range Low' in values and 'Range High' in values:
            labels.extend(['Range Low', 'Range High'])
            data.extend([values['Range Low'], values['Range High']])
        if 'Median' in values:
            labels.append('Median')
            data.append(values['Median'])
        if 'Average' in values:
            labels.append('Average')
            data.append(values['Average'])
        if not labels or not data:
            return None
        fig, ax = plt.subplots(figsize=(4, 2.5))
        bars = ax.bar(labels, data, color=['#2980B9', '#27AE60', '#E67E22', '#8E44AD'][:len(labels)])
        ax.set_ylabel('€')
        ax.set_title('Salary/Compensation Overview')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'€{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(tmpfile.name, format='png')
        plt.close(fig)
        return tmpfile.name
    
    def _is_key_insight(self, text: str) -> bool:
        """Determine if text contains key insights that should be highlighted - improved for historical content"""
        key_indicators = [
            'salary', 'compensation', '€', '$', 'growth', 'increase', 'decrease',
            'trend', 'outlook', 'forecast', 'projected', 'expected',
            # Historical indicators
            'battle', 'treaty', 'kingdom', 'empire', 'dynasty', 'conquest', 'reconquest', 
            'monarchy', 'republic', 'CE', 'BCE', 'AD', 'BC', 'century', 'medieval', 'ancient',
            'founded', 'established', 'declared', 'conquered', 'defeated', 'signed',
            'reign', 'rule', 'dynasty', 'dynasty', 'coronation', 'abdication'
        ]
        return any(indicator.lower() in text.lower() for indicator in key_indicators)
    
    def _add_toc_bookmarks(self, canvas, doc, section_bookmarks, first_page=False):
        """Add bookmarks and outline entries for clickable TOC. Only add on first page to avoid duplication."""
        if not first_page:
            return
        for idx, bookmark_name, title, level in section_bookmarks:
            try:
                canvas.bookmarkPage(bookmark_name)
                canvas.addOutlineEntry(title, bookmark_name, level=level, closed=False)
            except Exception:
                pass
    
    def create_text_report(self, content: str, filename: str) -> bool:
        """Create plain text report as fallback with quality assessment"""
        try:
            txt_filename = Path(filename).with_suffix('.txt')
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
                f.write("📋 PDF VIEWING INSTRUCTIONS:\n")
                f.write("-" * 40 + "\n")
                f.write("If you're seeing raw PDF data (gibberish text), you're viewing the PDF as text.\n")
                f.write("To properly view the formatted PDF report:\n")
                f.write("1. Right-click the .pdf file\n")
                f.write("2. Select 'Open with' → Choose a PDF viewer\n")
                f.write("3. Or double-click if associated with a PDF viewer\n")
                f.write("\nRecommended PDF viewers:\n")
                f.write("- Adobe Reader (free)\n")
                f.write("- Google Chrome (built-in PDF viewer)\n")
                f.write("- Mozilla Firefox (built-in PDF viewer)\n")
                f.write("- Microsoft Edge (built-in PDF viewer)\n")
                f.write("- Any PDF viewer application\n")
                f.write("\nThis text file (.txt) is provided for easier reading and searching.\n")
            
            print(f"📄 Enhanced text report with quality assessment saved: {txt_filename}")
            return True
            
        except Exception as e:
            print(f"⚠️ Text report creation failed: {e}")
            return False