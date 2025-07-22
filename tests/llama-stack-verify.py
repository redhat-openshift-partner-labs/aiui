import asyncio
import json
from typing import List, Dict, Any, Optional
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import (
    CompletionRequest,
    ToolDefinition,
    ToolCallDelta,
    FunctionCallArguments,
    UserMessage,
    SystemMessage
)

# ReAct agent using llama-stack with vLLM distribution
class ReActAgent:
    def __init__(self, host: str = "localhost", port: int = 5000):
        """Initialize ReAct agent with llama-stack client"""
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.model_id = "llama3"  # Model served by vLLM
        
        # Define tools for function calling
        self.tool_definitions = [
            ToolDefinition(
                tool_name="calculator",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            ToolDefinition(
                tool_name="weather",
                description="Get weather information for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or location name"
                        }
                    },
                    "required": ["location"]
                }
            )
        ]
        
        # Tool implementations
        self.tool_implementations = {
            "calculator": self._calculator,
            "weather": self._mock_weather
        }
    
    def _calculator(self, expression: str) -> str:
        """Calculator tool implementation"""
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _mock_weather(self, location: str) -> str:
        """Mock weather API"""
        return f"Weather in {location}: Sunny, 72°F"
    
    async def validate_function_calling(self) -> bool:
        """Validate that llama-stack + vLLM supports function calling"""
        try:
            # Create test messages
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that uses tools to answer questions."
                ),
                UserMessage(
                    content="What is 25 + 17?"
                )
            ]
            
            # Test completion with tools
            request = CompletionRequest(
                model=self.model_id,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=256
            )
            
            # Stream response to check for tool calls
            has_tool_calls = False
            async for chunk in self.client.inference.completion(request):
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_calls'):
                    if chunk.delta.tool_calls:
                        has_tool_calls = True
                        print(f"Tool call detected: {chunk.delta.tool_calls}")
            
            print(f"\nFunction calling validation: {'✓ Supported' if has_tool_calls else '✗ Not supported'}")
            return has_tool_calls
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool with given arguments"""
        if tool_name in self.tool_implementations:
            func = self.tool_implementations[tool_name]
            # Extract first argument value for simplicity
            arg_value = list(arguments.values())[0] if arguments else ""
            return func(arg_value)
        return f"Error: Unknown tool {tool_name}"
    
    async def run_react(self, query: str) -> str:
        """Execute ReAct loop using llama-stack"""
        # System prompt with ReAct format
        system_prompt = """You are a helpful AI assistant that uses the ReAct framework.
For each step, think about what to do, then act using available tools.

Format your responses as:
Thought: [Your reasoning]
Action: [Use a tool if needed]
Observation: [I will provide this]
... (repeat until you have the answer)
Answer: [Final answer]
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=query)
        ]
        
        max_iterations = 5
        for i in range(max_iterations):
            # Create completion request
            request = CompletionRequest(
                model=self.model_id,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=512
            )
            
            # Collect response
            full_response = ""
            tool_calls = []
            
            async for chunk in self.client.inference.completion(request):
                if hasattr(chunk, 'delta'):
                    # Collect text
                    if hasattr(chunk.delta, 'content') and chunk.delta.content:
                        full_response += chunk.delta.content
                    
                    # Collect tool calls
                    if hasattr(chunk.delta, 'tool_calls') and chunk.delta.tool_calls:
                        tool_calls.extend(chunk.delta.tool_calls)
            
            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": full_response
            })
            
            # Check for final answer
            if "Answer:" in full_response:
                return full_response.split("Answer:")[-1].strip()
            
            # Execute tool calls if any
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.tool_name
                    # Parse arguments - handle different formats
                    if isinstance(tool_call.arguments, str):
                        try:
                            args = json.loads(tool_call.arguments)
                        except:
                            args = {"input": tool_call.arguments}
                    else:
                        args = tool_call.arguments
                    
                    # Execute tool
                    result = await self._execute_tool(tool_name, args)
                    
                    # Add observation to messages
                    observation = f"Observation: {result}"
                    messages.append({
                        "role": "user",
                        "content": observation
                    })
        
        return "Maximum iterations reached without final answer"
    
    async def run_simple(self, query: str) -> str:
        """Simple completion without ReAct loop"""
        messages = [
            UserMessage(content=query)
        ]
        
        request = CompletionRequest(
            model=self.model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=256
        )
        
        response = ""
        async for chunk in self.client.inference.completion(request):
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                if chunk.delta.content:
                    response += chunk.delta.content
        
        return response

# Example usage
async def main():
    # Initialize agent - adjust host/port for your llama-stack server
    agent = ReActAgent(host="localhost", port=5000)
    
    # Validate function calling
    print("=== Validating Function Calling ===")
    is_capable = await agent.validate_function_calling()
    
    if is_capable:
        # Test ReAct agent
        print("\n=== Testing ReAct Agent ===")
        queries = [
            "What is 42 * 3 + 15?",
            "What's the weather in Paris?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            answer = await agent.run_react(query)
            print(f"Answer: {answer}")
    else:
        # Fallback to simple completion
        print("\n=== Function calling not supported, using simple completion ===")
        response = await agent.run_simple("What is 42 * 3?")
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
