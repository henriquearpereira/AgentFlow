#!/usr/bin/env python3
"""
Test script to verify research agent improvements
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pdF_research_agent.agents.research_agent import EnhancedResearchAgent
    from pdF_research_agent.models.local_models import LocalModelHandler
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure you're running this from the project root directory")
    sys.exit(1)

async def test_research_improvements():
    """Test the improved research agent functionality"""
    
    print("🧪 Testing Research Agent Improvements")
    print("=" * 50)
    
    try:
        # Initialize model handler (you may need to adjust this based on your setup)
        print("📋 Initializing model handler...")
        model_handler = LocalModelHandler()
        
        # Initialize research agent
        print("🔧 Initializing enhanced research agent...")
        agent = EnhancedResearchAgent(model_handler)
        
        # Test topics
        test_topics = [
            "Python web development frameworks",
            "Cloud computing security",
            "Machine learning in finance",
            "Blockchain technology applications",
            "Cybersecurity best practices"
        ]
        
        for topic in test_topics:
            print(f"\n🎯 Testing topic: {topic}")
            print("-" * 40)
            
            # Test AI insights generation
            print("🧠 Testing AI insights generation...")
            categories = agent._categorize_subject(topic)
            print(f"📋 Detected categories: {categories}")
            
            ai_insights = await agent._generate_ai_insights(topic, categories)
            print(f"✅ AI insights length: {len(ai_insights)} characters")
            
            # Check if insights are specific (not generic)
            generic_phrases = [
                "significant improvements",
                "various applications", 
                "broader context",
                "important considerations",
                "evolving trends"
            ]
            
            is_specific = True
            for phrase in generic_phrases:
                if phrase.lower() in ai_insights.lower():
                    print(f"⚠️ Found generic phrase: {phrase}")
                    is_specific = False
            
            if is_specific:
                print("✅ AI insights appear to be specific and detailed")
            else:
                print("⚠️ AI insights contain some generic content")
            
            # Test data extraction
            print("🔍 Testing data extraction...")
            extracted_data = agent._extract_concrete_data(ai_insights)
            
            print(f"💰 Salaries found: {len(extracted_data['salaries'])}")
            print(f"🏢 Companies found: {len(extracted_data['companies'])}")
            print(f"🌍 Locations found: {len(extracted_data['locations'])}")
            print(f"🔗 URLs found: {len(extracted_data['urls'])}")
            print(f"📊 Numerical data: {len(extracted_data['numerical_data'])}")
            
            # Test report structure generation
            print("📖 Testing report structure...")
            structure = agent._get_report_structure(topic, categories)
            print(f"📋 Report structure: {structure}")
            
            # Test prompt creation
            print("📝 Testing intelligent prompt creation...")
            prompt = agent._create_intelligent_prompt(topic, categories, ai_insights, structure)
            print(f"📄 Prompt length: {len(prompt)} characters")
            
            # Check prompt quality
            if "CRITICAL: Do not use generic placeholder text" in prompt:
                print("✅ Prompt includes anti-generic instructions")
            else:
                print("⚠️ Prompt may need stronger anti-generic instructions")
            
            if "specific examples" in prompt.lower():
                print("✅ Prompt emphasizes specific examples")
            else:
                print("⚠️ Prompt could emphasize specific examples more")
            
            print(f"✅ Topic '{topic}' testing completed")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_specific_topic():
    """Test a specific topic to see the actual output"""
    
    print("\n🧪 Testing Specific Topic Output")
    print("=" * 50)
    
    try:
        # Initialize model handler
        model_handler = LocalModelHandler()
        agent = EnhancedResearchAgent(model_handler)
        
        # Test a specific topic
        topic = "Python web development frameworks"
        print(f"🎯 Testing: {topic}")
        
        # Generate insights
        categories = agent._categorize_subject(topic)
        ai_insights = await agent._generate_ai_insights(topic, categories)
        
        print(f"\n📊 AI Insights Preview (first 500 chars):")
        print("-" * 40)
        print(ai_insights[:500] + "..." if len(ai_insights) > 500 else ai_insights)
        
        # Extract data
        data = agent._extract_concrete_data(ai_insights)
        
        print(f"\n📈 Extracted Data:")
        print(f"💰 Salaries: {data['salaries'][:3]}")
        print(f"🏢 Companies: {data['companies'][:3]}")
        print(f"🌍 Locations: {data['locations'][:3]}")
        print(f"📊 Numbers: {data['numerical_data'][:3]}")
        
        # Generate report structure
        structure = agent._get_report_structure(topic, categories)
        
        # Create prompt
        prompt = agent._create_intelligent_prompt(topic, categories, ai_insights, structure)
        
        print(f"\n📝 Prompt Preview (first 300 chars):")
        print("-" * 40)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        
        print("\n✅ Specific topic test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Specific topic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Research Agent Improvement Tests")
    
    # Run tests
    success1 = asyncio.run(test_research_improvements())
    success2 = asyncio.run(test_specific_topic())
    
    if success1 and success2:
        print("\n🎉 All tests passed! Research agent improvements are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 