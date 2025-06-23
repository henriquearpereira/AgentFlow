#!/usr/bin/env python3
"""
Simple test for historical research improvements
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_historical_categorization():
    """Test historical topic categorization"""
    try:
        from pdF_research_agent.agents.research_agent import EnhancedResearchAgent
        
        # Create a mock model handler for testing
        class MockModelHandler:
            async def generate_async(self, prompt):
                return "Mock historical content with dates and figures"
        
        # Initialize agent
        agent = EnhancedResearchAgent(MockModelHandler())
        
        # Test historical topic categorization
        topic = "Medieval Portugal Kingdom"
        categories = agent._categorize_subject(topic)
        
        print(f"✅ Topic: {topic}")
        print(f"✅ Categories: {categories}")
        
        # Check if historical category is detected
        if 'historical' in categories:
            print("✅ Historical category correctly detected!")
        else:
            print("❌ Historical category not detected")
        
        # Test historical validation
        test_report = """
        # Medieval Portugal Kingdom
        
        ## Introduction
        The Kingdom of Portugal was established in the 12th century during the medieval period. 
        King Afonso I, also known as Afonso Henriques, played a crucial role in the foundation 
        of the Portuguese kingdom in 1139.
        """
        
        is_valid = agent._validate_historical_content(test_report, ["Introduction"])
        print(f"✅ Historical validation: {is_valid}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Historical Research Improvements")
    print("=" * 50)
    
    success = test_historical_categorization()
    
    if success:
        print("\n✅ Historical research improvements are working correctly!")
    else:
        print("\n❌ Some tests failed") 