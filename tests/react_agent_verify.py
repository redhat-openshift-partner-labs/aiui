import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ReAct agent implementation for llama-stack with vLLM backend
class ActionType(Enum):
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    FINISH = "finish"

@dataclass
class ReActStep:
    """Single step in ReAct chain"""
    type: ActionType
    content: str
    result: Optional[str] = None

class ReActAgent:
    def __init__(self, vllm_base_url: str = "https://llama3-llama.apps.gpu.osdu.opdev.io", model_name: str = "llama3"):
        """Initialize ReAct agent with vLLM endpoint"""
        self.base_url = vllm_base_url
        self.model_name = model_name
        self.max_steps = 10
        
        # Define available tools/functions
        self.tools = {
            "calculator": self._calculator,
            "get_weather": self._mock_weather,
            "search": self._mock_search
        }
        
    def _calculator(self, expression: str) -> str:
        """Simple calculator tool"""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error: Invalid expression"
    
    def _mock_weather(self, location: str) -> str:
        """Mock weather API"""
        return f"Weather in {location}: Sunny, 72°F"
    
    def _mock_search(self, query: str) -> str:
        """Mock search API"""
        return f"Search results for '{query}': [Sample result 1, Sample result 2]"
    
    def _create_prompt(self, query: str, history: List[ReActStep]) -> str:
        """Create ReAct prompt with function calling format"""
        tools_desc = "\n".join([f"- {name}: {func.__doc__}" for name, func in self.tools.items()])
        
        prompt = f"""You are a helpful AI assistant that uses the ReAct framework to solve problems.
You have access to the following tools:
{tools_desc}

Follow this format for each step:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"parameter": "value"}}

When you have the final answer, use:
Thought: [Final reasoning]
Answer: [Your final answer]

Question: {query}

"""
        # Add conversation history
        for step in history:
            if step.type == ActionType.THINK:
                prompt += f"Thought: {step.content}\n"
            elif step.type == ActionType.ACT:
                prompt += f"Action: {step.content}\n"
                if step.result:
                    prompt += f"Observation: {step.result}\n"
        
        return prompt
    
    def _call_vllm(self, prompt: str) -> str:
        """Call vLLM server with prompt"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "stop": ["Observation:", "\nQuestion:"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error calling vLLM: {str(e)}"
    
    def _parse_action(self, response: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse action and parameters from LLM response"""
        lines = response.strip().split('\n')
        action = None
        params = {}
        
        for i, line in enumerate(lines):
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                # Extract JSON parameters
                try:
                    json_str = line.replace("Action Input:", "").strip()
                    params = json.loads(json_str)
                except:
                    # Try to extract from next lines if multiline JSON
                    if i + 1 < len(lines):
                        try:
                            json_str = '\n'.join(lines[i:]).replace("Action Input:", "").strip()
                            params = json.loads(json_str)
                        except:
                            params = {"input": json_str}
        
        return action, params
    
    def validate_function_calling(self) -> bool:
        """Validate that the model can handle function calling format"""
        test_prompt = """You are testing function calling capabilities.
Available tools:
- test_tool: A test tool

Respond with:
Thought: Testing function calling
Action: test_tool
Action Input: {"param": "test"}"""
        
        try:
            response = self._call_vllm(test_prompt)
            action, params = self._parse_action(response)
            
            # Check if model can generate proper format
            has_action = action is not None
            has_params = bool(params)
            
            print(f"Function Calling Validation:")
            print(f"- Can generate actions: {has_action}")
            print(f"- Can generate parameters: {has_params}")
            print(f"- Response: {response}")
            
            return has_action
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return False
    
    def run(self, query: str) -> str:
        """Execute ReAct loop for given query"""
        history = []
        
        for step in range(self.max_steps):
            # Generate next step
            prompt = self._create_prompt(query, history)
            response = self._call_vllm(prompt)
            
            # Check if final answer
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
                return answer
            
            # Parse thought and action
            thought = None
            if "Thought:" in response:
                thought = response.split("Thought:")[1].split("\n")[0].strip()
                history.append(ReActStep(ActionType.THINK, thought))
            
            # Parse and execute action
            action, params = self._parse_action(response)
            if action and action in self.tools:
                history.append(ReActStep(ActionType.ACT, f"{action}\nAction Input: {json.dumps(params)}"))
                
                # Execute tool
                tool_func = self.tools[action]
                try:
                    # Extract first parameter value
                    param_value = list(params.values())[0] if params else ""
                    result = tool_func(param_value)
                    history[-1].result = result
                except Exception as e:
                    history[-1].result = f"Error: {str(e)}"
        
        return "Maximum steps reached without final answer"

# Example usage
if __name__ == "__main__":
    # Initialize agent with your vLLM server
    agent = ReActAgent(vllm_base_url="https://llama3-llama.apps.gpu.osdu.opdev.io")
    
    # Validate function calling capability
    print("=== Validating Function Calling ===")
    is_capable = agent.validate_function_calling()
    print(f"Function calling capable: {is_capable}\n")
    
    # Test queries
    test_queries = [
        "What is 25 * 4 + 10?",
        "What's the weather in San Francisco?",
        "Search for information about Python programming"
    ]
    
    print("=== Running Test Queries ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer = agent.run(query)
        print(f"Answer: {answer}")
        print("-" * 50)
