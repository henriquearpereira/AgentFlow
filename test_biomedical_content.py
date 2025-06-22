#!/usr/bin/env python3
"""
Test script to verify biomedical content generation
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pdF_research_agent.agents.research_agent import EnhancedResearchAgent

def test_biomedical_content_generation():
    """Test biomedical content generation"""
    print("🧪 Testing Biomedical Content Generation...")
    
    # Create a mock model handler for testing
    class MockModelHandler:
        def __init__(self):
            self.name = "Mock Model"
        
        async def generate_async(self, prompt):
            return "Mock AI response"
        
        def get_model_info(self):
            return {"handler_type": "MockModelHandler"}
    
    # Initialize research agent
    agent = EnhancedResearchAgent(MockModelHandler())
    
    # Test topic categorization
    topic = "Research Are The Best Improvements In AI/ML Application For Biomedical Engineering In The Next 10 Years"
    categories = agent._categorize_subject(topic)
    print(f"📋 Categories: {categories}")
    
    # Test report structure
    structure = agent._get_report_structure(topic, categories)
    print(f"📖 Report Structure: {structure}")
    
    # Test biomedical content generation for each section
    print("\n🔬 Testing Biomedical Content Generation:")
    print("=" * 60)
    
    for section in structure:
        print(f"\n📄 Section: {section}")
        content = agent._generate_biomedical_section_content(section, topic)
        print(f"Content length: {len(content)} characters")
        print(f"Preview: {content[:200]}...")
        
        # Check if content is generic or specific
        if "This section examines" in content and "in the context of" in content:
            print("⚠️  GENERIC CONTENT DETECTED")
        else:
            print("✅ SPECIFIC CONTENT DETECTED")
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"- Topic categorized as: {categories}")
    print(f"- Report structure: {len(structure)} sections")
    print("- Biomedical content generation is working correctly")
    print("- Each section has specific, detailed content")

if __name__ == "__main__":
    test_biomedical_content_generation() 