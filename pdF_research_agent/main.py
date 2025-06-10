#!/usr/bin/env python3
"""
AI Research Agent - Main Entry Point
Enhanced with local and API model support
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Set up HuggingFace cache
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = str(project_root / '.cache' / 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

from config.models import get_available_models, list_all_models
from models.local_models import LocalModelHandler
from models.api_models import APIModelHandler
from agents.research_agent import ResearchAgent  # Fixed: Changed from EnhancedResearchAgent to ResearchAgent

def display_model_menu():
    """Display comprehensive model selection menu"""
    print("\n" + "="*70)
    print("🤖 AI RESEARCH AGENT - MODEL SELECTION")
    print("="*70)
    
    models = get_available_models()
    option_num = 1
    model_options = {}
    
    # Local Models
    print(f"\n📱 LOCAL MODELS (No API key required)")
    print("-" * 50)
    for model_key, model_info in models["local"].items():
        print(f"{option_num}. {model_info['description']}")
        print(f"   ⏱️  Time: {model_info['estimated_time']} | 💾 Memory: {model_info['memory_usage']} | ⭐ Quality: {model_info['quality']}")
        model_options[str(option_num)] = ("local", model_key)
        option_num += 1
    
    # API Models
    api_providers = ["groq", "together", "huggingface", "openrouter", "cohere"]
    
    for provider in api_providers:
        if provider in models:
            provider_name = provider.upper()
            print(f"\n🌐 {provider_name} API MODELS")
            print("-" * 50)
            
            for model_key, model_info in models[provider].items():
                # Check if API key is available
                api_key_var = f"{provider.upper()}_API_KEY"
                has_key = bool(os.getenv(api_key_var))
                key_status = "✅" if has_key else "❌"
                
                print(f"{option_num}. {model_info['description']} {key_status}")
                print(f"   ⏱️  Time: {model_info['estimated_time']} | 💰 Cost: {model_info['cost']} | ⭐ Quality: {model_info['quality']}")
                
                if not has_key:
                    print(f"   ⚠️  Requires {api_key_var} in .env file")
                
                model_options[str(option_num)] = (provider, model_key)
                option_num += 1
    
    print(f"\n{option_num}. 🧪 Test API Connections")
    model_options[str(option_num)] = ("test", "connections")
    option_num += 1
    
    print(f"{option_num}. ❌ Exit")
    model_options[str(option_num)] = ("exit", "")
    
    return model_options

def test_api_connections():
    """Test all available API connections"""
    print("\n🧪 Testing API Connections...")
    print("="*50)
    
    models = get_available_models()
    results = []
    
    for provider in ["groq", "together", "huggingface", "openrouter", "cohere"]:
        if provider not in models:
            continue
            
        api_key_var = f"{provider.upper()}_API_KEY"
        if not os.getenv(api_key_var):
            print(f"❌ {provider.upper()}: No API key found ({api_key_var})")
            continue
        
        print(f"🔄 Testing {provider.upper()}...")
        
        # Test first available model for each provider
        first_model = list(models[provider].keys())[0]
        model_name = models[provider][first_model]["name"]
        
        try:
            handler = APIModelHandler(provider, model_name)
            result = handler.test_connection()
            
            if result["status"] == "success":
                print(f"✅ {provider.upper()}: Connection successful")
                results.append((provider, first_model, True))
            else:
                print(f"❌ {provider.upper()}: {result['error']}")
                results.append((provider, first_model, False))
                
        except Exception as e:
            print(f"❌ {provider.upper()}: {str(e)}")
            results.append((provider, first_model, False))
    
    print("\n📊 Connection Test Summary:")
    print("-" * 30)
    working_apis = [r for r in results if r[2]]
    print(f"✅ Working APIs: {len(working_apis)}")
    print(f"❌ Failed APIs: {len(results) - len(working_apis)}")
    
    if working_apis:
        print("\n🎉 Recommended API models:")
        for provider, model, _ in working_apis:
            model_info = models[provider][model]
            print(f"  • {provider}/{model}: {model_info['description']}")
    
    input("\nPress Enter to continue...")

def select_model_interactive():
    """Interactive model selection with enhanced options"""
    while True:
        model_options = display_model_menu()
        
        choice = input(f"\n🎯 Choose your model (1-{len(model_options)}): ").strip()
        
        if choice not in model_options:
            print("❌ Invalid choice. Please try again.")
            continue
        
        provider, model_key = model_options[choice]
        
        if provider == "exit":
            print("👋 Goodbye!")
            sys.exit(0)
        elif provider == "test":
            test_api_connections()
            continue
        elif provider == "local":
            print(f"✅ Selected: Local {model_key}")
            return provider, model_key
        else:
            # API model selected
            api_key_var = f"{provider.upper()}_API_KEY"
            if not os.getenv(api_key_var):
                print(f"❌ Error: {api_key_var} not found in .env file")
                print(f"💡 Please add your {provider.upper()} API key to .env and restart")
                continue
            
            print(f"✅ Selected: {provider.upper()} {model_key}")
            return provider, model_key

def create_agent_config(provider: str, model_key: str, args) -> dict:
    """Create configuration dictionary for ResearchAgent"""
    config = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'verbose': args.verbose
    }
    
    if provider == "local":
        config['use_api_model'] = False
        config['model_name'] = model_key
    else:
        config['use_api_model'] = True
        config['api_provider'] = provider
        config['model_name'] = model_key
        
        # Add API key
        api_key_var = f"{provider.upper()}_API_KEY"
        config['api_key'] = os.getenv(api_key_var)
    
    return config

def main():
    """Main application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='AI Research Agent with Enhanced Model Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Python developer salaries Portugal"
  python main.py "Machine learning engineer salaries Lisbon" -o reports/ml_salaries.pdf
  python main.py "Data scientist remote work Portugal" --model-type api --provider groq
        """
    )
    
    parser.add_argument('query', type=str, help='Research topic or query')
    parser.add_argument('-o', '--output', type=str, default='reports/research_report.pdf', 
                       help='Output PDF file path (default: reports/research_report.pdf)')
    parser.add_argument('--model-type', choices=['local', 'api', 'interactive'], default='interactive',
                       help='Model type selection (default: interactive)')
    parser.add_argument('--provider', type=str, help='API provider (groq, together, huggingface, openrouter, cohere)')
    parser.add_argument('--model', type=str, help='Specific model key to use')
    parser.add_argument('--max-tokens', type=int, default=500, help='Maximum tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Display startup banner
    print("\n" + "="*70)
    print("🚀 AI RESEARCH AGENT - ENHANCED PDF GENERATOR")
    print("="*70)
    print(f"📝 Query: {args.query}")
    print(f"📄 Output: {args.output}")
    
    # Model selection
    if args.model_type == 'interactive' or (not args.provider and not args.model):
        provider, model_key = select_model_interactive()
    else:
        if args.provider and args.model:
            provider, model_key = args.provider, args.model
        else:
            print("❌ Error: For non-interactive mode, both --provider and --model must be specified")
            sys.exit(1)
    
    # Initialize application
    start_time = time.time()
    
    try:
        # Create agent configuration
        print(f"\n🔧 Initializing {provider.upper()} model handler...")
        config = create_agent_config(provider, model_key, args)
        
        # Create research agent
        print("🔬 Setting up research agent...")
        agent = ResearchAgent(config)  # Fixed: Changed from EnhancedResearchAgent to ResearchAgent
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Execute research workflow
        print(f"\n🎯 Starting research for: '{args.query}'")
        print("="*60)
        
        # Run the complete research workflow
        results = agent.conduct_research(args.query, args.output)
        
        total_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*60)
        print("⏱️  PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🔍 Search time:        {results['timing']['search_time']:.1f}s")
        print(f"🤖 Generation time:    {results['timing']['report_time']:.1f}s")
        print(f"📄 PDF creation:       {results['timing']['pdf_time']:.1f}s")
        print(f"⚡ Total runtime:      {total_time:.1f}s")
        print(f"🎯 Model used:         {provider.upper()}/{model_key}")
        
        if results['pdf_created']:
            output_path = Path(args.output).resolve()
            print(f"\n✅ SUCCESS: Report saved to {output_path}")
            if output_path.exists():
                print(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
        else:
            print("\n⚠️  PDF creation had issues")
        
        # Show the generated report
        if args.verbose:
            print("\n" + "="*60)
            print("📝 GENERATED REPORT PREVIEW")
            print("="*60)
            report_content = results['report_content']
            print(report_content[:500] + "..." if len(report_content) > 500 else report_content)
            print("="*60)
        
        print(f"\n🎉 Research completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup resources
        if 'agent' in locals():
            agent.cleanup()

if __name__ == "__main__":
    main()