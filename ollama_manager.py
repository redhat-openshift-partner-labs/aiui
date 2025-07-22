import json
import yaml

from ollama import Client, ChatResponse
from typing import Dict


# Load configuration
def load_config() -> Dict:
    """
    Load configuration from the config.yaml file.

    Returns:
        Dict: Configuration dictionary
    """
    with open("config.yaml", "r") as file:
        config_file = yaml.safe_load(file)

    return config_file

config = load_config()


class OllamaManager:
    """Manages the persistent Ollama client with tool/function support"""

    def __init__(self, host, model, options, tools=None):
        self.client = Client(host=host)
        self.model = model
        self.options = options

        with open("tools.json", "r") as f:
            self.tools = json.load(f)  # Load tools from the JSON file

    def chat(self, messages) -> ChatResponse:
        """Chat with optional tool support"""
        # Build request parameters
        params = {
            'model': self.model,
            'messages': messages,
            #'tools': self.tools,
            'options': self.options
        }

        response = self.client.chat(**params)
        print(response)
        return response

    def chat_stream(self, messages, tools=None):
        """Stream chat responses with tool support"""
        params = {
            'model': self.model,
            'messages': messages,
            'options': self.options,
            'stream': True
        }

        if tools or self.tools:
            params['tools'] = tools or self.tools

        return self.client.chat(**params)

    def set_tools(self, tools):
        """Update default tools"""
        self.tools = tools