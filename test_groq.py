import requests
import json
import os

try:
    # Try to load from secrets.toml
    from streamlit import secrets
    api_key = secrets.GROQ_API_KEY
    print("✅ Loaded key from Streamlit secrets")
except:
    # Try to load from environment or hardcode for testing
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Manually enter your key for testing
        api_key = input("Enter your Groq API key: ").strip().strip('"').strip("'")
    print("✅ Loaded key from environment/input")

# Test the API key
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test 1: List available models
print("\n📡 Testing API connection...")
try:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers=headers,
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        models = response.json()
        print("✅ API Key is VALID!")
        print(f"Available models: {[m['id'] for m in models.get('data', [])[:3]]}")
    elif response.status_code == 401:
        print("❌ INVALID API KEY - Unauthorized")
        print("Get a new key at: https://console.groq.com")
    elif response.status_code == 429:
        print("⚠️ Rate limit exceeded")
    else:
        print(f"❌ Error: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Try a simple chat completion
print("\n🧪 Testing chat completion...")
try:
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello' in one word."}
        ],
        "temperature": 0.7,
        "max_tokens": 10
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        reply = result['choices'][0]['message']['content']
        print(f"✅ Chat test successful: {reply}")
    else:
        print(f"❌ Chat test failed: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🔑 Your API key (first 10 chars):", api_key[:10] + "..." if len(api_key) > 10 else api_key)