#!/usr/bin/env python3
import asyncio
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
from toolbox_core import ToolboxSyncClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolboxServer:
    def __init__(self, toolbox_url: str = "http://127.0.0.1:5000", tools_config_path: str = "tools.yaml"):
        self.server = Server("toolbox-mcp-server")
        self.toolbox_url = toolbox_url
        self.tools_config_path = tools_config_path
        self.toolbox_client = None
        self.tools_config = None
        
    async def load_tools_config(self):
        """Load tools configuration from YAML file"""
        try:
            config_path = Path(self.tools_config_path)
            if not config_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path("mcpsrv/tools.yaml"),
                    Path("../tools.yaml"),
                    Path("tools.json")
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        config_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Tools config file not found: {self.tools_config_path}")
            
            with open(config_path, 'r') as f:
                if config_path.suffix == '.json':
                    # Handle JSON format
                    json_data = json.load(f)
                    # Convert JSON tool format to our YAML-like structure
                    self.tools_config = {
                        'tools': {},
                        'toolsets': {'default': []}
                    }
                    for tool_def in json_data:
                        if tool_def.get('type') == 'function':
                            func = tool_def['function']
                            tool_name = func['name']
                            self.tools_config['tools'][tool_name] = {
                                'kind': 'function',
                                'description': func['description'],
                                'parameters': func.get('parameters', {})
                            }
                            self.tools_config['toolsets']['default'].append(tool_name)
                else:
                    # Handle YAML format
                    self.tools_config = yaml.safe_load(f)
                    
            logger.info(f"Loaded tools config from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load tools config: {e}")
            return False
    
    def connect_to_toolbox(self):
        """Connect to the toolbox service"""
        try:
            self.toolbox_client = ToolboxSyncClient(self.toolbox_url)
            logger.info(f"Connected to toolbox at {self.toolbox_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to toolbox: {e}")
            return False
    
    def register_tools(self):
        """Register tools with the MCP server"""
        if not self.tools_config or 'tools' not in self.tools_config:
            logger.warning("No tools configuration found")
            return
            
        for tool_name, tool_config in self.tools_config['tools'].items():
            # Convert tool config to MCP Tool format
            mcp_tool = Tool(
                name=tool_name,
                description=tool_config.get('description', f'Tool: {tool_name}'),
                inputSchema=self._convert_parameters_to_schema(tool_config.get('parameters', {}))
            )
            
            # Register the tool with MCP server
            @self.server.call_tool()
            async def handle_tool_call(name: str, arguments: dict) -> list[TextContent]:
                if name == tool_name:
                    return await self._execute_tool(name, arguments)
                raise ValueError(f"Unknown tool: {name}")
                
        logger.info(f"Registered {len(self.tools_config['tools'])} tools")
    
    def _convert_parameters_to_schema(self, parameters: Dict) -> Dict:
        """Convert tool parameters to JSON schema format"""
        if isinstance(parameters, dict) and 'type' in parameters:
            # Already in JSON schema format
            return parameters
            
        # Convert from our YAML format to JSON schema
        if isinstance(parameters, list):
            # Handle list of parameter definitions
            schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param in parameters:
                if isinstance(param, dict) and 'name' in param:
                    param_name = param['name']
                    param_type = param.get('type', 'string')
                    param_desc = param.get('description', '')
                    
                    schema['properties'][param_name] = {
                        "type": param_type,
                        "description": param_desc
                    }
                    
                    if param.get('required', True):
                        schema['required'].append(param_name)
            
            return schema
        
        # Default empty schema
        return {"type": "object", "properties": {}}
    
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> list[TextContent]:
        """Execute a tool using the toolbox client"""
        try:
            if not self.toolbox_client:
                raise Exception("Toolbox client not connected")
                
            # Load and execute the tool
            tool = self.toolbox_client.load_tool(tool_name)
            result = tool(**arguments)
            
            # Convert result to TextContent
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, indent=2)
                
            return [TextContent(type="text", text=content)]
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            return [TextContent(type="text", text=error_msg)]
    
    async def list_tools(self) -> list[Tool]:
        """List available tools"""
        tools = []
        
        if not self.tools_config or 'tools' not in self.tools_config:
            return tools
            
        for tool_name, tool_config in self.tools_config['tools'].items():
            tool = Tool(
                name=tool_name,
                description=tool_config.get('description', f'Tool: {tool_name}'),
                inputSchema=self._convert_parameters_to_schema(tool_config.get('parameters', {}))
            )
            tools.append(tool)
            
        return tools
    
    async def run(self):
        """Run the MCP server"""
        # Load configuration and connect to toolbox
        if not await self.load_tools_config():
            logger.error("Failed to load tools config, exiting")
            return
            
        if not self.connect_to_toolbox():
            logger.error("Failed to connect to toolbox, exiting")
            return
        
        # Register MCP handlers
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return await self.list_tools()
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            return await self._execute_tool(name, arguments)
        
        logger.info("Starting MCP server...")
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    """Main entry point"""
    server = MCPToolboxServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())