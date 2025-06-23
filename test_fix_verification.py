#!/usr/bin/env python3
"""
Test to verify the fix for the undefined variable issue in _structure_existing_insights
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_structure_existing_insights():
    """Test the _structure_existing_insights method with proper parameters"""
    try:
        from pdF_research_agent.agents.research_agent import EnhancedResearchAgent
        
        # Create a mock model handler for testing
        class MockModelHandler:
            async def generate_async(self, prompt):
                return "Mock content"
        
        # Initialize agent
        agent = EnhancedResearchAgent(MockModelHandler())
        
        # Test data
        research_data = """# HYBRID RESEARCH DATA

## AI-GENERATED INSIGHTS AND ANALYSIS
This is some AI-generated content about Portugal.

## SEARCH VALIDATION AND ADDITIONAL DATA
Some search results here.
"""
        structure = ["Introduction", "Background"]
        topic = "Portugal History"
        categories = ["historical"]
        data = {}
        
        # Test the method
        result = agent._structure_existing_insights(research_data, structure, topic, categories, data)
        
        print("✅ _structure_existing_insights method executed successfully!")
        print(f"✅ Result length: {len(result)} characters")
        print(f"✅ Contains topic: {'Portugal History' in result}")
        print(f"✅ Contains sections: {'Introduction' in result and 'Background' in result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Fix for _structure_existing_insights")
    print("=" * 50)
    
    success = test_structure_existing_insights()
    
    if success:
        print("\n✅ The fix is working correctly!")
        print("✅ The undefined variable issue has been resolved.")
    else:
        print("\n❌ The fix needs further attention.") 