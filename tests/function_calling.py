import requests
import json

# vLLM server endpoint (adjust port if needed)
BASE_URL = "https://llama3-llama.apps.gpu.osdu.opdev.io"

# Define test functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculate",
            "description": "Perform basic math operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    }
]

# Test function calling
def test_function_calling():
    # Request with function definitions
    payload = {
        "model": "llama3",  # Adjust model name if different
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": tools,
        "tool_choice": "auto"  # Let model decide when to use functions
    }
    
    try:
        # Make request to vLLM
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Response:", json.dumps(result, indent=2))
            
            # Check if model made a function call
            if result["choices"][0]["message"].get("tool_calls"):
                print("\n✓ Function calling works!")
                print("Tool calls:", result["choices"][0]["message"]["tool_calls"])
            else:
                print("\n✗ No function call made")
                print("Response:", result["choices"][0]["message"]["content"])
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

# Quick connectivity test
def test_models_endpoint():
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        print("Available models:", response.json())
    except Exception as e:
        print(f"Cannot connect to vLLM server: {e}")

if __name__ == "__main__":
    # Test server connection first
    test_models_endpoint()
    print("\n" + "="*50 + "\n")
    
    # Test function calling
    test_function_calling()
