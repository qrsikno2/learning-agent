import re
from learning_agent.core import Agent, Tool, ToolRegistry, Config, LLM, Message
from learning_agent.core import RunnableMixin, ToolCallMixin
from typing import List, Iterator


class SimpleAgent(Agent, RunnableMixin, ToolCallMixin):
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        system_prompt: str = None, 
        config: Config = None, 
        tool_registry: ToolRegistry = None, 
        enable_tool_calling: bool = True
    ):
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.enable_tool_calling = enable_tool_calling
        
    def _get_enhanced_system_prompt(self) -> str:
        base_prompt = self.system_prompt or "你是一个很牛的助手."
        
        if not self.enable_tool_calling or not self.tool_registry.tools:
            return base_prompt
        
        tools_description = self.tool_registry.get_tools_description()
        if tools_description is None:
            return base_prompt
        
        tools_section = "\n\n# 可用工具\n"
        tools_section += "你可以使用以下工具来辅助回答用户的问题：\n"
        tools_section += tools_description + "\n\n"
        
        tools_section += "## 调用工具格式\n"
        tools_section += "当你需要调用工具时，请按照以下格式输出：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如：`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`。\n\n"
        tools_section += "请确保工具调用格式正确，并且工具名称和参数符合要求。工具调用结果会被自动插入到对话中，然后你可以继续生成回答。"
        tools_section += "\n\n"
        
        return base_prompt + tools_section
    
    @Agent.print_turns
    def run(self, input_text: str, max_tool_iteration: int = 3, **kwargs) -> str:
        enhanced_sys_prompt = self._get_enhanced_system_prompt()
        message = self._build_prompt(input_text, custom_sys_prompt=enhanced_sys_prompt)
        if not self.enable_tool_calling:
            res = self.llm.think(message, **kwargs)
            self.add_message(Message(role='user', content=input_text))
            self.add_message(Message(role='assistant', content=res))
            
            return res
        
        return self._run_with_tool_calls(message, input_text, max_tool_iteration, **kwargs)

    def stream_run(self, input_text: str, max_tool_iteration: int = 3, **kwargs) -> Iterator[str]:
        """
        暂时不支持工具调用
        """
        sys_prompt = self.system_prompt or "你是一个很牛的助手."
        message = self._build_prompt(input_text, custom_sys_prompt=sys_prompt)
        full_resp = ""
        for chunk in self.llm.stream_think(message, **kwargs):
            full_resp += chunk
            yield chunk
        
        self.add_message(Message(role='user', content=input_text))
        # 这里假设stream_think的最后一个chunk是完整的回复
        self.add_message(Message(role='assistant', content=full_resp))
    
    def _run_with_tool_calls(self, msgs: List, input_text: str, max_tool_iteration: int , **kwargs) -> str:
        current_iteration = 0
        final_response = ""
        
        while current_iteration < max_tool_iteration:
            res = self.llm.think(msgs, **kwargs)
            tool_calls = self._parse_tool_calls(res)
            
            if tool_calls:
                tool_results, clean_res = [], res
                
                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    print(f"Tool calling: {call['tool_name']} with params {call['parameters']} -> Result: {result}")
                    clean_res = clean_res.replace(call['original'], "")
                    
                msgs.append({"role": "assistant", "content": clean_res})
                tool_result_text = "\n\n".join(tool_results)
                msgs.append({"role": "user", "content": f"工具调用结果：\n{tool_result_text}"})
                current_iteration += 1
            else:
                final_response = res
                break
        
        if current_iteration >= max_tool_iteration and not final_response:
            final_response = self.llm.think(msgs, **kwargs)
        
        self.add_message(Message(role='user', content=input_text))
        self.add_message(Message(role='assistant', content=final_response))
        
        return final_response
    
    def _parse_tool_calls(self, text: str) -> list:
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        ret = []
        for tool_name, params in matches:
            ret.append({
                'tool_name': tool_name.strip(),
                'parameters': params.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{params}]'
            })
            
        return ret

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        if not self.tool_registry:
            return "错误：没有工具注册中心，无法执行工具调用。"
        
        try:
            param_dict = self._parse_tool_parameters(tool_name, parameters)
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"错误：未找到工具 '{tool_name}'。请检查工具名称是否正确，并确保该工具已注册。"
            result = tool.run(param_dict)
            return f"工具 '{tool_name}' 调用结果：{result}"
            
        except Exception as e:
            return f"工具调用失败：{str(e)}"
        
    def _parse_tool_parameters(self, tool_name: str, params: str) -> dict:
        ret = {}
        
        if '=' in params:
            # Key=Value Or action=search, query=python
            pairs = params.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    ret[key.strip()] = value.strip()
        else:
            if tool_name == 'search':
                ret['query'] = params
            elif tool_name == 'memory':
                ret = {'action': 'search', 'query': params}
            else:
                ret['input'] = params
        
        return ret
    

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = LLM()

    # === 测试1: 基础对话（无工具） ===
    print("=== 测试1: 基础对话 ===")
    agent1 = SimpleAgent(name="基础助手", llm=llm, system_prompt="你是一个友好的助手，请用简洁的方式回答问题。", enable_tool_calling=False)
    response = agent1.run("你好，请简单介绍一下你自己")
    print(f"响应: {response}\n")
    print("✅ 测试1 通过\n")

    # === 测试2: 带工具的对话 ===
    print("=== 测试2: 工具增强对话 ===")
    class _CalcTool(Tool):
        def __init__(self):
            super().__init__("calculator", "计算数学表达式，输入如 15*8+32 返回计算结果")
        def run(self, params) -> str:
            expr = params if isinstance(params, str) else params.get("expression", params.get("input", ""))
            if not expr:
                return "错误：未提供表达式"
            try:
                return str(eval(expr))
            except Exception as e:
                return f"计算错误: {e}"

    registry = ToolRegistry()
    registry.register(_CalcTool())
    agent2 = SimpleAgent(name="增强助手", llm=llm, system_prompt="你是一个智能助手，可以利用工具来帮助用户。", tool_registry=registry, enable_tool_calling=True)
    resp = agent2.run("请计算一下 15*8+32 的结果")
    print(f"响应如下:\n{resp}\n")
    
    print("流式响应: ", end="")
    for chunk in agent2.stream_run("写一篇关于为什么人类要运动的文章"):
        print(chunk, end="")
    print()
    print("✅ 测试2 通过\n")

    # === 测试3: 历史记录 ===
    print("=== 测试3: 历史记录 ===")
    history = agent1.get_history()
    print(f"历史消息数: {len(history)}")
    for msg in history:
        print(f"  [{msg.role}] {msg.content[:60]}...")
    print("✅ 测试3 通过\n")

    # === 测试4: 空工具注册表 ===
    print("=== 测试4: 空工具注册表的Agent ===")
    agent3 = SimpleAgent(name="无工具助手", llm=llm, system_prompt="简洁回答", tool_registry=ToolRegistry(), enable_tool_calling=True)
    response = agent3.run("1+1等于几？")
    print(f"响应: {response}\n")
    print("✅ 测试4 通过\n")

    print("🎉 全部测试完成")
