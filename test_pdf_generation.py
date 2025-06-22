#!/usr/bin/env python3
"""
Test script to verify PDF generation works correctly
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add the pdF_research_agent directory to the path
pdf_agent_path = project_root / "pdF_research_agent"
sys.path.insert(0, str(pdf_agent_path))

from agents.pdf_generator import PDFGenerator

def test_pdf_generation():
    """Test PDF generation with sample content"""
    
    print("🧪 Testing PDF Generation...")
    print("=" * 40)
    
    # Create sample content
    sample_content = """# AI Applications in Biomedical Engineering Software

## Executive Summary

This comprehensive analysis examines AI applications in biomedical engineering software, focusing on the intersection of artificial intelligence and healthcare technology. The research identifies key technological advancements, software platforms, and implementation strategies that are driving innovation in healthcare technology.

Key findings include significant improvements in medical imaging accuracy (15-25% enhancement), drug discovery efficiency (30-50% faster screening), and clinical decision support systems (20-30% diagnostic accuracy improvement). Major software platforms include TensorFlow Medical, PyTorch Medical, NVIDIA Clara, and specialized healthcare AI frameworks.

## Technical Specifications

Technical specifications for AI applications in biomedical engineering software encompass several critical components:

**AI/ML Frameworks:**
- TensorFlow Extended (TFX) for end-to-end ML pipelines
- PyTorch Lightning for rapid prototyping and deployment
- MONAI (Medical Open Network for AI) for medical imaging
- NVIDIA Clara for healthcare-specific AI platform

**Software Architecture:**
- Microservices architecture for scalable healthcare applications
- Edge computing for real-time medical device processing
- Cloud-native solutions (AWS, Azure, Google Cloud healthcare)
- Containerization with Docker and Kubernetes

## Applications and Use Cases

Key applications and use cases for AI in biomedical engineering software include:

**Medical Imaging and Diagnostics:**
- X-ray analysis with 95%+ accuracy using deep learning
- MRI segmentation and analysis for tumor detection
- CT scan interpretation and 3D reconstruction
- Ultrasound image enhancement and analysis

**Drug Discovery and Development:**
- AI-powered compound screening (30-50% faster)
- Drug repurposing using machine learning
- Protein structure prediction and analysis
- Clinical trial optimization and patient matching

## Implementation Guide

Implementation guide for AI in biomedical engineering software:

**Phase 1: Planning and Requirements**
- Define specific medical use cases and requirements
- Assess regulatory compliance needs (FDA, HIPAA, GDPR)
- Identify data sources and quality requirements
- Establish performance benchmarks and success metrics

**Phase 2: Technology Selection**
- Choose appropriate AI/ML frameworks (TensorFlow Medical, PyTorch Medical)
- Select cloud infrastructure (AWS, Azure, Google Cloud)
- Implement data governance and security measures
- Establish development and testing environments

## Best Practices

Best practices for AI in biomedical engineering software:

**Data Management:**
- Implement robust data governance frameworks
- Use standardized medical data formats (DICOM, FHIR)
- Ensure data quality and validation processes
- Establish clear data lineage and audit trails

**Model Development:**
- Use interpretable AI models where possible
- Implement comprehensive testing and validation
- Ensure model explainability and transparency
- Regular model retraining and updates

## Performance Considerations

Performance considerations for AI in biomedical engineering software:

**Computational Requirements:**
- GPU acceleration for deep learning models
- Distributed computing for large-scale processing
- Real-time processing for critical applications
- Edge computing for medical device integration

**Accuracy and Reliability:**
- 95%+ accuracy for medical imaging applications
- Sub-second response times for critical systems
- 99.9% uptime for healthcare applications
- Continuous monitoring and validation

## Resources and Documentation

Resources and documentation for AI in biomedical engineering software:

**Software Platforms and Tools:**
- TensorFlow Medical: Google's medical AI framework
- PyTorch Medical: Facebook's medical imaging tools
- NVIDIA Clara: Healthcare-specific AI platform
- MONAI: Medical Open Network for AI

**Development Resources:**
- Medical imaging datasets (NIH, Kaggle)
- Healthcare AI research papers and publications
- Open-source medical AI projects
- Clinical validation guidelines
"""

    try:
        # Initialize PDF generator
        pdf_generator = PDFGenerator()
        
        # Test PDF creation
        output_file = "test_biomedical_ai_report.pdf"
        
        print(f"📄 Creating PDF: {output_file}")
        success = pdf_generator.create_pdf(sample_content, output_file, "AI Applications in Biomedical Engineering Software")
        
        if success:
            print(f"✅ PDF created successfully: {output_file}")
            
            # Check if file exists and has content
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"📊 File size: {file_size} bytes")
                
                if file_size > 1000:
                    print("✅ PDF file appears to be valid (size > 1KB)")
                else:
                    print("⚠️ PDF file seems too small, may be corrupted")
            else:
                print("❌ PDF file was not created")
        else:
            print("❌ PDF creation failed")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_generation() 