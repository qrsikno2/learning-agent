from typing import Callable, Dict, Any, Optional

from abc import ABC, abstractmethod

class Tool(ABC):
    name: str
    description: str
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, params: dict) -> str:
        pass

    def __str__(self):
        return f"Tool(name={self.name}, description={self.description})"

    def _generate_info(self) -> Dict[str, Any]:
        return f"{self.name}: {self.description}"

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, tool: Tool) -> bool:
        if tool.name in self.tools:
            return False
        self.tools[tool.name] = tool
        return True

    def unregister(self, name: str) -> bool:
        if name not in self.tools:
            return False
        del self.tools[name]
        return True
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def get_tools_description(self) -> str:
        return "\n".join(
            [tool._generate_info() for tool in self.tools.values()]
        )
        
    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def execute(self, name: str, params) -> str:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        return tool.run(params)
