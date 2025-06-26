#!/usr/bin/env python3
"""
Test script for Cold War research to see if the fix works
"""

import asyncio
import json
import requests
import time

async def test_cold_war_research():
    """Test the Cold War research functionality"""
    
    # Test the research API
    url = "http://localhost:8000/api/research/start"
    
    payload = {
        "query": "report about the cold war",
        "provider": "local",
        "model_key": "phi-2",
        "search_engine": "free_web",
        "max_tokens": 1000,
        "temperature": 0.2
    }
    
    print("🚀 Starting Cold War research test...")
    print(f"📋 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Research started: {json.dumps(data, indent=2)}")
            
            session_id = data.get('session_id')
            if session_id:
                print(f"🆔 Session ID: {session_id}")
                
                # Wait a bit and check status
                await asyncio.sleep(5)
                
                status_url = f"http://localhost:8000/api/sessions/{session_id}/status"
                status_response = requests.get(status_url)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📊 Status: {json.dumps(status_data, indent=2)}")
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
        else:
            print(f"❌ Research start failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_cold_war_research()) 