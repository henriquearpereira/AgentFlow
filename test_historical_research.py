#!/usr/bin/env python3
"""
Test script for enhanced historical research capabilities
Demonstrates the improved research agent with historical knowledge prioritization
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdF_research_agent.agents.research_agent import EnhancedResearchAgent
from pdF_research_agent.models.local_models import LocalModelHandler
from pdF_research_agent.models.api_models import APIModelHandler

async def test_historical_research():
    """Test the enhanced historical research capabilities"""
    
    print("🧪 Testing Enhanced Historical Research Agent")
    print("=" * 60)
    
    # Initialize model handler (you can use either local or API)
    try:
        # Try local model first
        model_handler = LocalModelHandler()
        print("✅ Using local model handler")
    except Exception as e:
        print(f"⚠️ Local model failed: {e}")
        try:
            # Fallback to API model
            model_handler = APIModelHandler()
            print("✅ Using API model handler")
        except Exception as e:
            print(f"❌ Both model handlers failed: {e}")
            return
    
    # Initialize enhanced research agent
    agent = EnhancedResearchAgent(model_handler)
    
    # Test historical topics
    historical_topics = [
        "Medieval Portugal Kingdom",
        "Ancient Roman Military Tactics", 
        "Renaissance Art in Florence",
        "Medieval Castles and Fortifications"
    ]
    
    for topic in historical_topics:
        print(f"\n🎯 Testing: {topic}")
        print("-" * 40)
        
        try:
            # Conduct research
            result = await agent.conduct_research(
                topic=topic,
                output_file=f"reports/{topic.replace(' ', '_').lower()}.pdf"
            )
            
            if result['success']:
                print(f"✅ Research completed successfully!")
                print(f"📊 Categories: {', '.join(result['categories'])}")
                print(f"📄 Report length: {len(result['report_content'].split())} words")
                print(f"⏱️ Total time: {result['timing']['total_time']:.1f}s")
                
                # Check if historical validation was used
                if 'historical' in result['categories']:
                    print("📜 Historical validation applied")
                
                # Show a snippet of the report
                content = result['report_content']
                if len(content) > 200:
                    print(f"📝 Report preview: {content[:200]}...")
                
            else:
                print("❌ Research failed")
                
        except Exception as e:
            print(f"❌ Error during research: {e}")
        
        print("\n" + "="*60)
    
    # Cleanup
    agent.cleanup()
    print("🧹 Cleanup completed")

async def test_search_strategy():
    """Test the enhanced search strategy for historical topics"""
    
    print("\n🔍 Testing Enhanced Search Strategy")
    print("=" * 60)
    
    try:
        model_handler = LocalModelHandler()
        agent = EnhancedResearchAgent(model_handler)
        
        # Test historical topic
        topic = "Medieval Portugal Kingdom"
        categories = agent._categorize_subject(topic)
        
        print(f"Topic: {topic}")
        print(f"Categories: {categories}")
        
        # Test enhanced search
        search_results = await agent._conduct_enhanced_search(topic, categories)
        
        print(f"Search results length: {len(search_results)} characters")
        if "MODEL-GENERATED SEARCH RESULTS" in search_results:
            print("✅ Model-generated search data used")
        else:
            print("✅ Web search results used")
        
        # Test AI insights generation
        ai_insights = await agent._generate_ai_insights(topic, categories)
        print(f"AI insights length: {len(ai_insights)} characters")
        
        # Test historical validation
        test_report = f"""
        # {topic}
        
        ## Introduction
        
        The Kingdom of Portugal was established in the 12th century during the medieval period. 
        King Afonso I, also known as Afonso Henriques, played a crucial role in the foundation 
        of the Portuguese kingdom in 1139. The medieval period saw significant developments 
        in Portuguese history, including the establishment of feudal systems and the 
        construction of numerous castles and fortifications.
        
        ## Historical Background
        
        During the medieval era, Portugal emerged as an independent kingdom from the 
        County of Portugal, which was part of the Kingdom of León. The Treaty of Zamora 
        in 1143 recognized Portugal's independence, marking a significant milestone in 
        medieval European history.
        """
        
        is_valid = agent._validate_historical_content(test_report, ["Introduction", "Historical Background"])
        print(f"Historical validation result: {is_valid}")
        
        agent.cleanup()
        
    except Exception as e:
        print(f"❌ Error during search strategy test: {e}")

async def main():
    """Main test function"""
    print("🚀 Starting Enhanced Historical Research Tests")
    print("=" * 60)
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Run tests
    await test_historical_research()
    await test_search_strategy()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 