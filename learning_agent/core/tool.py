from typing import Callable, Dict, Any, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

class ToolParamter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    
    def get_description(self) -> str:
        return f"name={self.name} (type:{self.type}), description: {self.description}, {'required' if self.required else 'optional'}, default={self.default}"

class Tool(ABC):
    name: str
    description: str
    
    def __init__(self, name: str, description: str, parameters: Optional[list[ToolParamter]] = None):
        self.name = name
        self.description = description
        self._parameters = parameters or []

    @abstractmethod
    def run(self, params: dict) -> str:
        pass
    
    def get_parameters(self) -> list[ToolParamter]:
        return self._parameters

    def __str__(self):
        return f"Tool(name={self.name}, description={self.description})"

    def _generate_info(self) -> str:
        return f"Tool(name={self.name}): description={self.description}, Parameters: [{', '.join([param.get_description() for param in self.get_parameters()])}]"

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema 格式

        用于 FunctionCallAgent,使工具能够被 OpenAI 原生 function calling 使用

        Returns:
            符合 OpenAI function calling 标准的 schema
        """
        parameters = self.get_parameters()

        # 构建 properties
        properties = {}
        required = []

        for param in parameters:
            # 基础属性定义
            prop = {
                "type": param.type,
                "description": param.description
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"

            # 如果是数组类型，添加 items 定义
            if param.type == "array":
                prop["items"] = {"type": "string"}  # 默认字符串数组

            properties[param.name] = prop

            # 收集必需参数
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        } 

class SimpleFunctionTool(Tool):
    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        super().__init__(name, description)
        self.func = func

    def run(self, param: str) -> str:
        return self.func(param)

    def get_parameters(self) -> list[ToolParamter]:
        return [ToolParamter(name="input", type="string", description="输入参数")]

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool) -> bool:
        if tool.name in self._tools:
            return False
        self._tools[tool.name] = tool
        return True

    def unregister(self, name: str) -> bool:
        if name not in self._tools:
            return False
        del self._tools[name]
        return True
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_tools_description(self) -> str:
        return None if self._tools is None else "\n".join(
            [tool._generate_info() for tool in self._tools.values()]
        )
        
    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, params) -> str:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        return tool.run(params)

