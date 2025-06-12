#!/usr/bin/env python3
"""
AI Research Agent - Enhanced Main Entry Point
With interactive query refinement and improved user experience
"""

import os
import sys
import argparse
import time
from pathlib import Path
import re 
from datetime import datetime
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
from agents.research_agent import EnhancedResearchAgent

def generate_output_filename(query: str) -> str:
    """Generate descriptive filename from query"""
    clean_query = re.sub(r'[^\w\s-]', '', query.lower())
    clean_query = re.sub(r'[-\s]+', '_', clean_query)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"reports/{clean_query}_{timestamp}.pdf"

def refine_query_interactive(initial_query: str) -> dict:
    """Interactive query refinement to improve research quality"""
    print("\n" + "="*70)
    print("🎯 QUERY REFINEMENT - Let's optimize your research!")
    print("="*70)
    
    print(f"📝 Your query: {initial_query}")
    
    # Detect query type for targeted questions
    query_lower = initial_query.lower()
    refinements = {
        'query': initial_query,
        'scope': 'comprehensive',
        'region': 'global',
        'timeframe': 'current',
        'audience': 'general',
        'depth': 'detailed'
    }
    
    # Ask clarifying questions based on query type
    if any(word in query_lower for word in ['salary', 'pay', 'wage', 'compensation']):
        print("\n💰 Detected: Salary Research")
        refinements['scope'] = input("🌍 Geographic focus (e.g., 'Portugal', 'Europe', 'Global'): ").strip() or 'Global'
        refinements['timeframe'] = input("📅 Time focus (e.g., '2024', 'latest', 'trends'): ").strip() or 'latest'
        
        experience = input("👨‍💻 Experience level focus (e.g., 'junior', 'senior', 'all levels'): ").strip()
        if experience:
            refinements['query'] += f" {experience} level"
            
    elif any(word in query_lower for word in ['vs', 'versus', 'compare', 'comparison']):
        print("\n⚖️ Detected: Comparison Research")
        criteria = input("📊 Key comparison criteria (e.g., 'features', 'pricing', 'performance'): ").strip()
        if criteria:
            refinements['query'] += f" {criteria} comparison"
            
    elif any(word in query_lower for word in ['how to', 'guide', 'tutorial']):
        print("\n📚 Detected: Tutorial/Guide Research")
        level = input("🎓 Target skill level (e.g., 'beginner', 'intermediate', 'advanced'): ").strip()
        if level:
            refinements['query'] += f" {level} guide"
            
    else:
        print("\n🔍 General Research Query")
        
    # Common refinement options
    print(f"\n📋 Current refined query: {refinements['query']}")
    
    depth_choice = input("📖 Report depth (1=Summary, 2=Detailed, 3=Comprehensive): ").strip()
    depth_map = {'1': 'summary', '2': 'detailed', '3': 'comprehensive'}
    refinements['depth'] = depth_map.get(depth_choice, 'detailed')
    
    # Ask about specific focus areas
    print("\n🎯 Any specific focus areas? (comma-separated, or press Enter to skip)")
    focus_areas = input("   Examples: remote work, startups, trends, benefits: ").strip()
    if focus_areas:
        refinements['query'] += f" {focus_areas}"
    
    # Confirm final query
    print(f"\n✅ Final refined query: {refinements['query']}")
    confirm = input("👍 Proceed with this query? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes']:
        manual_query = input("✏️  Enter your preferred query: ").strip()
        if manual_query:
            refinements['query'] = manual_query
    
    return refinements

class ProgressTracker:
    """Enhanced progress tracking with visual feedback"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, message: str, percentage: int):
        """Update progress with enhanced visual feedback"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Estimate remaining time
        if percentage > 0:
            total_estimated = elapsed * 100 / percentage
            remaining = max(0, total_estimated - elapsed)
            time_info = f"⏱️  {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining"
        else:
            time_info = f"⏱️  {elapsed:.1f}s elapsed"
        
        print(f"\r🔄 [{bar}] {percentage:3d}% | {message} | {time_info}", end='', flush=True)
        
        if percentage >= 100:
            print()  # New line when complete
        
        self.last_update = current_time

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

def main():
    """Enhanced main application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced AI Research Agent with Interactive Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Python developer salaries Portugal"
  python main.py "Machine learning engineer salaries Lisbon" -o reports/ml_salaries.pdf
  python main.py "Data scientist remote work Portugal" --model-type api --provider groq
  python main.py "Docker vs Kubernetes comparison" --interactive
        """
    )
    
    parser.add_argument('query', type=str, help='Research topic or query')
    parser.add_argument('-o', '--output', type=str,
                        help='Output PDF file path (auto-generated if not specified)')
    parser.add_argument('--model-type', choices=['local', 'api', 'interactive'], default='interactive',
                       help='Model type selection (default: interactive)')
    parser.add_argument('--provider', type=str, help='API provider (groq, together, huggingface, openrouter, cohere)')
    parser.add_argument('--model', type=str, help='Specific model key to use')
    parser.add_argument('--max-tokens', type=int, default=500, help='Maximum tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive query refinement')
    parser.add_argument('--no-refinement', action='store_true', help='Skip query refinement')
    
    args = parser.parse_args()
    
    # Display startup banner
    print("\n" + "="*70)
    print("🚀 ENHANCED AI RESEARCH AGENT - PROFESSIONAL REPORT GENERATOR")
    print("="*70)
    
    # Interactive query refinement (unless disabled)
    final_query = args.query
    if not args.no_refinement and (args.interactive or not any([args.provider, args.model])):
        refinement_data = refine_query_interactive(args.query)
        final_query = refinement_data['query']
    
    print(f"📝 Final Query: {final_query}")
    
    # Generate filename if not provided
    if not args.output:
        args.output = generate_output_filename(final_query)
        print(f"📄 Auto-generated output: {args.output}")
    else:
        print(f"📄 Output file: {args.output}")
    
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
    progress_tracker = ProgressTracker()
    
    try:
        # Create model handler
        print(f"\n🔧 Initializing {provider.upper()} model handler...")
        
        if provider == "local":
            model_handler = LocalModelHandler(
                model_key=model_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )
        else:
            # API model handler
            api_key_var = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_var)
            
            if not api_key:
                raise ValueError(f"API key {api_key_var} not found in environment")
            
            model_handler = APIModelHandler(
                provider=provider,
                model_name=model_key,
                api_key=api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose
            )
        
        if args.verbose:
            print(f"🔍 Debug - Model handler created: {type(model_handler)}")
        
        # Create enhanced research agent with progress tracking
        print("🔬 Setting up enhanced research agent...")
        agent = EnhancedResearchAgent(model_handler)
        agent.set_progress_callback(progress_tracker.update)
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Execute enhanced research workflow
        print(f"\n🎯 Starting enhanced research for: '{final_query}'")
        print("="*70)
        
        # Run the complete enhanced research workflow
        results = agent.conduct_research(final_query, args.output)
        
        total_time = time.time() - start_time
        
        # Display enhanced results
        print("\n" + "="*70)
        print("📊 RESEARCH ANALYSIS SUMMARY")
        print("="*70)
        print(f"🏷️  Topic Categories:    {', '.join(results['categories'])}")
        print(f"📖 Report Structure:    {len(results['report_structure'])} sections")
        print(f"🔍 Search Results:      {len(results['search_results'])} characters")
        print(f"📄 Content Generated:   {len(results['report_content'])} characters")
        
        print("\n" + "="*70)
        print("⏱️  PERFORMANCE METRICS")
        print("="*70)
        print(f"🔍 Search Phase:        {results['timing']['search_time']:.1f}s")
        print(f"🤖 AI Generation:       {results['timing']['report_time']:.1f}s")
        print(f"📄 PDF Creation:        {results['timing']['pdf_time']:.1f}s")
        print(f"⚡ Total Runtime:       {total_time:.1f}s")
        print(f"🎯 Model Used:          {provider.upper()}/{model_key}")
        
        if results['pdf_created']:
            output_path = Path(args.output).resolve()
            print(f"\n✅ SUCCESS: Professional report saved to {output_path}")
            if output_path.exists():
                print(f"📊 File Size: {output_path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"\n⚠️  PDF creation had issues, text report available")
        
        # Show report preview
        if args.verbose:
            print("\n" + "="*70)
            print("📝 REPORT STRUCTURE PREVIEW")
            print("="*70)
            for i, section in enumerate(results['report_structure'], 1):
                print(f"{i}. {section}")
            
            print("\n" + "="*70)
            print("📄 CONTENT PREVIEW (First 800 characters)")
            print("="*70)
            content_preview = results['report_content'][:800]
            print(content_preview + "..." if len(results['report_content']) > 800 else content_preview)
            print("="*70)
        
        print(f"\n🎉 Enhanced research completed successfully!")
        print(f"🏆 Quality Score: {len(results['categories'])} categories analyzed")
        
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
        if 'model_handler' in locals():
            model_handler.cleanup()
        if 'agent' in locals() and hasattr(agent, 'cleanup'):
            agent.cleanup()

if __name__ == "__main__":
    main()