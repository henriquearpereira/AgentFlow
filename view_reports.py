#!/usr/bin/env python3
"""
Script to help view research reports and provide PDF viewing guidance
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def list_recent_reports():
    """List recent research reports"""
    reports_dir = Path("pdF_research_agent/reports")
    
    if not reports_dir.exists():
        print("❌ Reports directory not found")
        return
    
    # Get all PDF and TXT files
    pdf_files = list(reports_dir.glob("*.pdf"))
    txt_files = list(reports_dir.glob("*.txt"))
    
    # Sort by modification time (newest first)
    all_files = []
    for pdf_file in pdf_files:
        all_files.append((pdf_file, 'pdf', pdf_file.stat().st_mtime))
    for txt_file in txt_files:
        all_files.append((txt_file, 'txt', txt_file.stat().st_mtime))
    
    all_files.sort(key=lambda x: x[2], reverse=True)
    
    print("📋 Recent Research Reports:")
    print("=" * 60)
    
    for i, (file_path, file_type, mtime) in enumerate(all_files[:10], 1):
        size = file_path.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
        
        print(f"{i:2d}. {file_path.name}")
        print(f"    Type: {file_type.upper()}, Size: {size_str}")
        print()
    
    return all_files

def open_file_with_default_viewer(file_path):
    """Open file with default system viewer"""
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        else:  # Linux
            subprocess.run(["xdg-open", file_path])
        return True
    except Exception as e:
        print(f"❌ Failed to open file: {e}")
        return False

def show_pdf_guidance():
    """Show guidance for viewing PDF files"""
    print("\n📖 PDF VIEWING GUIDANCE:")
    print("=" * 50)
    print("If you're seeing raw PDF data (gibberish text), follow these steps:")
    print()
    print("1. RIGHT-CLICK the PDF file")
    print("2. Select 'Open with'")
    print("3. Choose a PDF viewer:")
    print("   - Adobe Reader (recommended)")
    print("   - Google Chrome")
    print("   - Mozilla Firefox")
    print("   - Microsoft Edge")
    print("   - Any PDF viewer application")
    print()
    print("💡 TIP: Text files (.txt) are provided for easier reading")
    print("   and searching. They contain the same content as PDFs.")

def main():
    """Main function"""
    print("🔍 Research Report Viewer")
    print("=" * 50)
    
    # List recent reports
    files = list_recent_reports()
    
    if not files:
        print("No reports found.")
        return
    
    # Show guidance
    show_pdf_guidance()
    
    # Interactive file opening
    print("\n🎯 Quick Actions:")
    print("1. View most recent PDF")
    print("2. View most recent text file")
    print("3. List all files")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                # Find most recent PDF
                pdf_files = [(f, t, m) for f, t, m in files if t == 'pdf']
                if pdf_files:
                    most_recent_pdf = pdf_files[0][0]
                    print(f"📄 Opening: {most_recent_pdf.name}")
                    if open_file_with_default_viewer(most_recent_pdf):
                        print("✅ File opened successfully")
                    else:
                        print("❌ Failed to open file")
                else:
                    print("❌ No PDF files found")
            
            elif choice == "2":
                # Find most recent text file
                txt_files = [(f, t, m) for f, t, m in files if t == 'txt']
                if txt_files:
                    most_recent_txt = txt_files[0][0]
                    print(f"📄 Opening: {most_recent_txt.name}")
                    if open_file_with_default_viewer(most_recent_txt):
                        print("✅ File opened successfully")
                    else:
                        print("❌ Failed to open file")
                else:
                    print("❌ No text files found")
            
            elif choice == "3":
                print("\n📋 All Available Files:")
                print("=" * 40)
                for i, (file_path, file_type, mtime) in enumerate(files, 1):
                    size = file_path.stat().st_size
                    size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                    print(f"{i:2d}. {file_path.name} ({file_type.upper()}, {size_str})")
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 