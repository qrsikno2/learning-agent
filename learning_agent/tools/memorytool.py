from learning_agent.core import Tool, ToolParameter

class MemoryTool(Tool):
    def __init__(self):
        super().__init__(
            name="MemoryTool", 
            description="用于管理和操作记忆的工具. 你可以总是使用此工具来管理自己的记忆.",
            parameters=[
                ToolParameter(name="action", type="string", description="操作类型，支持 'add' 和 'query'"),
                ToolParameter(name="content", type="string", description="记忆内容或查询内容"),
            ]
        )
        self.memory_store: list[str] = []

    def run(self, param: str) -> str:
        """
        执行记忆操作。

        支持:
        1. add 添加记忆: {"action": "add", "content": "记忆内容"}
        2. query 查询记忆: {"action": "query", "content": "查询内容"}
        """
        action = param.get("action")

        if action == "add":
            return self._handle_add(param)
        elif action == "query":
            return self._handle_query(param)
        else:
            return f"unsupported action '{action}'. Supported actions are 'add' and 'query'."
    
    def _handle_add(self, param: dict) -> str:
        content = param.get("content")
        
        if content:
            self.memory_store.append(content)
            return "success"
        else:
            return "'content' is required for add action"
    
    def _handle_query(self, param: dict) -> str:
        content = param.get("content")
        
        if content:
            results = [mem for mem in self.memory_store if content in mem]
            return "[" + ','.join(results) + "]" if results else "no matching memory found"
        else:
            return "'content' is required for query action"
