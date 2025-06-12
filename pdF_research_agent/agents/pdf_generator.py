"""
PDF Generation module for AI Research Agent - Enhanced Version with Fixed Formatting
"""

import time
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


class PDFGenerator:
    """Handles PDF report generation with enhanced formatting and fixed spacing"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Define sections that should start on new pages
        self.page_break_sections = {
            'primary sources', 'data sources', 'methodology', 'references',
            'appendix', 'bibliography', 'conclusions', 'recommendations',
            'detailed analysis', 'technical specifications', 'future outlook'
        }
    
    def _setup_custom_styles(self):
        """Define custom styles for professional formatting with improved spacing"""
        # Custom title style with more space after
        self.styles.add(ParagraphStyle(
            name='TitleCustom',
            fontSize=28,
            spaceAfter=30,  # Increased from 20 to 30
            spaceBefore=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#2C3E50"),
            fontName='Helvetica-Bold'
        ))
        
        # Custom section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            fontSize=16,
            spaceAfter=12,  # Increased from 10 to 12
            spaceBefore=20,  # Increased from 18 to 20
            textColor=colors.HexColor("#34495E"),
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=0
        ))
        
        # Custom subsection header with better spacing
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            fontSize=14,
            spaceAfter=10,  # Increased from 8 to 10
            spaceBefore=15,  # Increased from 12 to 15
            textColor=colors.HexColor("#5D6D7E"),
            fontName='Helvetica-Bold'
        ))
        
        # Enhanced body text
        self.styles.add(ParagraphStyle(
            name='BodyTextCustom',
            fontSize=11,
            leading=16,
            spaceAfter=8,  # Increased from 6 to 8
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#2C3E50")
        ))
        
        # Highlighted text for key insights
        self.styles.add(ParagraphStyle(
            name='HighlightText',
            fontSize=12,
            leading=16,
            spaceAfter=10,  # Increased from 8 to 10
            spaceBefore=10,  # Increased from 8 to 10
            textColor=colors.HexColor("#E74C3C"),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor("#E74C3C"),
            borderPadding=8,
            backColor=colors.HexColor("#FCF3CF")
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
            spaceAfter=6,  # Increased from 4 to 6
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
    
    def _clean_title(self, title: str) -> str:
        """Clean and improve title formatting with better hashtag removal"""
        if not title:
            return title
        
        # Remove ALL markdown hashtag markers (handles ###, ####, etc.)
        title = re.sub(r'^#{1,6}\s*', '', title)
        title = re.sub(r'#{1,6}\s*$', '', title)  # Also remove trailing hashtags
        
        # Remove extra whitespace and normalize
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Handle special cases and improve readability
        # Convert underscores and hyphens to spaces
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
            r'\bkubernetes\b': 'Kubernetes',
            r'\bdocker\b': 'Docker',
            r'\bjs\b': 'JavaScript',
            r'\bts\b': 'TypeScript',
            r'\bhtml\b': 'HTML',
            r'\bcss\b': 'CSS',
            r'\bsql\b': 'SQL',
            r'\bnosql\b': 'NoSQL',
            r'\bmongodb\b': 'MongoDB',
            r'\bpostgresql\b': 'PostgreSQL',
            r'\bmysql\b': 'MySQL',
            r'\bredis\b': 'Redis',
            r'\breact\b': 'React',
            r'\bvue\b': 'Vue.js',
            r'\bangular\b': 'Angular',
            r'\bnode\b': 'Node.js',
            r'\bpython\b': 'Python',
            r'\bjava\b': 'Java',
            r'\bc\+\+\b': 'C++',
            r'\bc#\b': 'C#',
            r'\bgo\b': 'Go',
            r'\brust\b': 'Rust',
            r'\bswift\b': 'Swift',
            r'\bkotlin\b': 'Kotlin',
            r'\bphp\b': 'PHP',
            r'\bruby\b': 'Ruby',
            r'\bscala\b': 'Scala',
            r'\bdevops\b': 'DevOps',
            r'\bcicd\b': 'CI/CD',
            r'\bgithub\b': 'GitHub',
            r'\bgitlab\b': 'GitLab',
            r'\bjenkins\b': 'Jenkins',
            r'\bansible\b': 'Ansible',
            r'\bterraform\b': 'Terraform',
        }
        
        # Apply fixes (case insensitive)
        for pattern, replacement in title_fixes.items():
            title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)
        
        # Proper case for countries and cities
        geographical_terms = {
            r'\bbelgium\b': 'Belgium',
            r'\bbrussels\b': 'Brussels',
            r'\bantwerp\b': 'Antwerp',
            r'\bghent\b': 'Ghent',
            r'\bliege\b': 'Liège',
            r'\bportugal\b': 'Portugal',
            r'\blisbon\b': 'Lisbon',
            r'\bporto\b': 'Porto',
            r'\bcoimbra\b': 'Coimbra',
            r'\bbraga\b': 'Braga',
            r'\bfrance\b': 'France',
            r'\bparis\b': 'Paris',
            r'\bgermany\b': 'Germany',
            r'\bberlin\b': 'Berlin',
            r'\bmunich\b': 'Munich',
            r'\bnetherlands\b': 'Netherlands',
            r'\bamsterdam\b': 'Amsterdam',
            r'\buk\b': 'UK',
            r'\bunited kingdom\b': 'United Kingdom',
            r'\blondon\b': 'London',
            r'\busa\b': 'USA',
            r'\bunited states\b': 'United States',
            r'\bcanada\b': 'Canada',
            r'\btoronto\b': 'Toronto',
            r'\bvancouver\b': 'Vancouver',
            r'\baustralia\b': 'Australia',
            r'\bsydney\b': 'Sydney',
            r'\bmelbourne\b': 'Melbourne',
            r'\bchina\b': 'China',
            r'\bbeijing\b': 'Beijing',
            r'\bshanghai\b': 'Shanghai',
            r'\bshenzhen\b': 'Shenzhen',
            r'\bsingapore\b': 'Singapore',
        }
        
        for pattern, replacement in geographical_terms.items():
            title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)
        
        # Capitalize first letter of each word (title case) but preserve known acronyms
        words = title.split()
        title_cased_words = []
        
        for word in words:
            # Don't change words that are already properly formatted (like API, AWS, etc.)
            if word.isupper() and len(word) <= 4:  # Likely acronym
                title_cased_words.append(word)
            elif word in ['AI/ML', 'UI/UX', 'CI/CD']:  # Special cases
                title_cased_words.append(word)
            else:
                title_cased_words.append(word.capitalize())
        
        title = ' '.join(title_cased_words)
        
        # Clean up extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def _should_page_break(self, section_title: str) -> bool:
        """Determine if a section should start on a new page"""
        title_lower = section_title.lower().strip()
        return any(keyword in title_lower for keyword in self.page_break_sections)
    
    def create_pdf(self, content: str, filename: str, topic: str = None) -> bool:
        """
        Create PDF report with enhanced formatting and fixed spacing
        
        Args:
            content: Report content in markdown format
            filename: Output PDF filename
            topic: Optional topic for title extraction
            
        Returns:
            bool: Success status
        """
        print(f"📄 Creating enhanced PDF: {filename}")
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
            
            # Enhanced title section with better spacing
            story.append(Paragraph("AI Research Report", self.styles['TitleCustom']))
            story.append(Spacer(1, 20))  # Extra space after main title
            
            # Topic extraction and formatting with cleanup
            if not topic:
                topic = Path(filename).stem.replace('_', ' ')
                # Remove timestamp from filename
                topic = re.sub(r'_\d{8}_\d{4}$', '', topic)
            
            # Clean the topic title
            cleaned_topic = self._clean_title(topic)
            story.append(Paragraph(f"Research Topic: {cleaned_topic}", self.styles['TopicTitle']))
            story.append(Spacer(1, 15))  # Space after topic title
            
            # Add subtle divider line
            story.append(HRFlowable(
                width="100%", 
                thickness=1, 
                color=colors.HexColor("#BDC3C7"),
                spaceBefore=5,
                spaceAfter=20  # Increased from 15 to 20
            ))
            
            # Metadata section with better spacing
            story.append(Paragraph(f"<b>Generated:</b> {time.strftime('%B %d, %Y at %H:%M:%S')}", self.styles['MetadataText']))
            story.append(Paragraph("<b>Powered by:</b> AI Research Agent", self.styles['MetadataText']))
            story.append(Spacer(1, 30))  # Increased from 25 to 30
            
            # Process content with enhanced formatting
            self._process_content(content, story)
            
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
            
            doc.build(story)
            print(f"✅ Enhanced PDF created successfully: {filename}")
            return True
            
        except Exception as e:
            print(f"⚠️ PDF creation failed: {e}")
            return False
    
    def _process_content(self, content: str, story: list):
        """Process markdown content with enhanced formatting and better title cleaning"""
        # Split by ## but also handle ### and other heading levels
        sections = re.split(r'\n(?=##)', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            lines = section.split('\n')
            section_title = lines[0].strip()
            section_content = '\n'.join(lines[1:]).strip()
            
            if section_title:
                # Clean the section title (removes ALL hashtag markers)
                cleaned_title = self._clean_title(section_title)
                
                # Skip empty titles after cleaning
                if not cleaned_title:
                    continue
                
                # Determine the heading level based on hashtags
                hashtag_count = len(re.match(r'^#+', section_title.strip()).group()) if re.match(r'^#+', section_title.strip()) else 2
                
                # Choose appropriate style based on heading level
                if hashtag_count <= 2:
                    title_style = self.styles['SectionHeader']
                else:
                    title_style = self.styles['SubsectionHeader']
                
                # Check if this section should start on a new page
                if hashtag_count <= 2 and self._should_page_break(cleaned_title):
                    story.append(PageBreak())
                    story.append(Spacer(1, 12))  # Add some space after page break
                
                # Add section header with enhanced styling
                story.append(Paragraph(cleaned_title, title_style))
                
                # Check if this section contains salary/compensation data
                if self._contains_salary_data(section_content):
                    salary_table = self._extract_salary_table(section_content)
                    if salary_table:
                        story.append(Spacer(1, 10))  # Space before table
                        story.append(salary_table)
                        story.append(Spacer(1, 15))  # Space after table
                
                # Process remaining content
                self._process_section_content(section_content, story)
                
                # Add spacing between sections (except for last section)
                if i < len(sections) - 2:
                    story.append(Spacer(1, 25))  # Increased from 20 to 25
    
    def _process_section_content(self, content: str, story: list):
        """Process individual section content with improved styling"""
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            lines = paragraph.split('\n')
            
            # Process bullet points
            if any(line.strip().startswith('-') or line.strip().startswith('*') or line.strip().startswith('•') for line in lines):
                for line in lines:
                    line = line.strip()
                    if line.startswith(('-', '*', '•')):
                        bullet_text = line[1:].strip()
                        # Clean bullet text titles
                        bullet_text = self._clean_bullet_text(bullet_text)
                        
                        # Highlight key financial data
                        if self._is_key_insight(bullet_text):
                            story.append(Paragraph(f"• {bullet_text}", self.styles['HighlightText']))
                        else:
                            story.append(Paragraph(f"• {bullet_text}", self.styles['BodyTextCustom']))
                        story.append(Spacer(1, 4))  # Increased from 3 to 4
            
            # Process URLs
            elif any('http' in line for line in lines):
                for line in lines:
                    if 'http' in line:
                        # Make URLs clickable
                        url_pattern = r'(https?://[^\s]+)'
                        formatted_line = re.sub(url_pattern, r'<link href="\1">\1</link>', line)
                        story.append(Paragraph(formatted_line, self.styles['BodyTextCustom']))
                    else:
                        story.append(Paragraph(line, self.styles['BodyTextCustom']))
            
            # Regular paragraphs
            else:
                combined_text = ' '.join(line.strip() for line in lines if line.strip())
                if combined_text:
                    story.append(Paragraph(combined_text, self.styles['BodyTextCustom']))
            
            story.append(Spacer(1, 10))  # Increased from 8 to 10
    
    def _clean_bullet_text(self, text: str) -> str:
        """Clean bullet point text, especially for bold/highlighted items"""
        # Remove extra asterisks used for markdown bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Clean up technical terms in bullet points
        text = re.sub(r'\bsalary range\b', 'Salary Range', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmedian compensation\b', 'Median Compensation', text, flags=re.IGNORECASE)
        text = re.sub(r'\baverage compensation\b', 'Average Compensation', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdata points analyzed\b', 'Data Points Analyzed', text, flags=re.IGNORECASE)
        
        return text
    
    def _contains_salary_data(self, content: str) -> bool:
        """Check if content contains salary/compensation data"""
        salary_keywords = ['salary', 'compensation', '€', '$', 'wage', 'income', 'pay']
        return any(keyword.lower() in content.lower() for keyword in salary_keywords)
    
    def _extract_salary_table(self, content: str) -> Table:
        """Extract and format salary data into a professional table"""
        # Look for salary patterns
        salary_patterns = [
            r'€([\d,]+)\s*-\s*€([\d,]+)',  # Range pattern
            r'€([\d,]+)',  # Single value pattern
            r'Median[:\s]+€([\d,]+)',
            r'Average[:\s]+€([\d,]+)'
        ]
        
        salary_data = []
        
        # Extract salary ranges
        range_match = re.search(r'€([\d,]+)\s*-\s*€([\d,]+)', content)
        if range_match:
            salary_data.append(['Salary Range', f"€{range_match.group(1)} - €{range_match.group(2)}"])
        
        # Extract median
        median_match = re.search(r'[Mm]edian[:\s]*€([\d,]+)', content)
        if median_match:
            salary_data.append(['Median Compensation', f"€{median_match.group(1)}"])
        
        # Extract average
        avg_match = re.search(r'[Aa]verage[:\s]*€([\d,]+)', content)
        if avg_match:
            salary_data.append(['Average Compensation', f"€{avg_match.group(1)}"])
        
        if salary_data:
            # Create table with headers
            table_data = [['Metric', 'Value']] + salary_data
            
            table = Table(table_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                
                # Data rows styling
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 11),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                
                # Borders and padding
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#BDC3C7")),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            return table
        
        return None
    
    def _is_key_insight(self, text: str) -> bool:
        """Determine if text contains key insights that should be highlighted"""
        key_indicators = [
            'salary', 'compensation', '€', '$', 'growth', 'increase', 'decrease',
            'trend', 'outlook', 'forecast', 'projected', 'expected'
        ]
        return any(indicator.lower() in text.lower() for indicator in key_indicators)
    
    def create_text_report(self, content: str, filename: str) -> bool:
        """
        Create plain text report as fallback
        
        Args:
            content: Report content
            filename: Output filename (will change extension to .txt)
            
        Returns:
            bool: Success status
        """
        try:
            txt_filename = Path(filename).with_suffix('.txt')
            txt_filename.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract and clean topic from filename
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
                f.write("-" * 80 + "\n\n")
                f.write(content)
                f.write("\n\n" + "-" * 80 + "\n")
                f.write("This report was automatically generated by AI Research Agent\n")
                f.write("=" * 80)
            
            print(f"📄 Enhanced text report saved: {txt_filename}")
            return True
            
        except Exception as e:
            print(f"⚠️ Text report creation failed: {e}")
            return False