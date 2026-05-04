from typing import Optional
from .tool import Tool, ToolRegistry


class ToolCallMixin:
    tool_registry: ToolRegistry
    enable_tool_calling: bool

    def add_tools(self, tool: Tool) -> None:
        if not hasattr(self, 'tool_registry') or self.tool_registry is None:
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True
        self.tool_registry.register(tool)

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None and bool(self.tool_registry.tools)

    def remove_tool(self, tool_name: str) -> bool:
        if not self.tool_registry:
            return False
        return self.tool_registry.unregister(tool_name)

    def list_tools(self) -> list:
        if not self.tool_registry:
            return []
        return self.tool_registry.list_tools()
