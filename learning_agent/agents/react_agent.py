import re
from typing import Dict, List, Optional

from learning_agent.core import Agent, RunnableMixin, ToolRegistry, Config, LLM, Message
from learning_agent.tools import DateTimeTool, TavilySearchTool

REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循: 工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""


class ReActAgent(Agent, RunnableMixin):
    def __init__(
        self,
        name: str,
        llm: LLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 10,
        custom_prompt: Optional[str] = None,
    ):
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else REACT_PROMPT

    def _parse_response(self, response: str) -> Dict[str, str]:
        thought_match = re.match(
            r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL
        )
        action_match = re.search(r"Action:\s*(.*)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action_text = action_match.group(1).strip() if action_match else ""
        action = self._parse_action(action_text) if action_text else {}
        return {"thought": thought, "action": action}

    def _parse_action(self, action_str: str) -> Dict[str, str]:
        match = re.match(r"(\w+)\[(.*)\]", action_str, re.DOTALL)
        if match:
            return {"name": match.group(1), "input": match.group(2)}
        return {}

    def _parse_action_input(self, action_str: str) -> str:
        match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    @Agent.print_turns
    def run(self, input_text: str, **kwargs) -> str:
        self.current_history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1

            tools_desc = self.tool_registry.get_tools_description() or "无可用工具"
            history_str = "\n".join(self.current_history) or "（尚未执行任何操作）"
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str,
            )

            messages = [{"role": "user", "content": prompt}]
            response = self.llm.think(messages, **kwargs)

            if not response:
                self.current_history.append("（LLM未返回有效响应）")
                continue

            parsed = self._parse_response(response)
            thought = parsed.get("thought", "")
            action = parsed.get("action") or {}

            if not action:
                self.current_history.append(f"Invalid Response: {response}")
                continue

            if thought:
                self.current_history.append(f"Thought: {thought}")
            if self.config.debug:
                print(f"LLM Thought: {thought}")
            action_name = action.get("name", "")
            action_input = action.get("input", "")

            if action_name == "Finish":
                self.add_message(Message(content=input_text, role="user"))
                self.add_message(Message(content=action_input, role="assistant"))
                return action_input

            if not action_name:
                self.current_history.append(f"Invalid Action: {action}")
                continue

            self.current_history.append(f"Action: {action_name}[{action_input}]")
            if self.config.debug:
                print(f"LLM Action: {action_name}[{action_input}]")

            try:
                result = self.tool_registry.execute(action_name, action_input)
                if self.config.debug:
                    print(f"工具执行结果: {result}")
            except Exception as e:
                result = f"工具执行失败: {e}"

            self.current_history.append(f"Observation: {result}")

        fallback_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=fallback_answer, role="assistant"))
        return fallback_answer

    def add_tools(self, tool) -> None:
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True
        
        self.tool_registry.register(tool)
    
    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None
    
    def remove_tool(self, tool_name: str) -> bool:
        return False if not self.tool_registry else self.tool_registry.unregister(tool_name)
    
    def list_tools(self) -> list:
        return [] if not self.tool_registry else self.tool_registry.list_tools()
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = LLM()
    registry = ToolRegistry()
    registry.register(DateTimeTool())
    registry.register(TavilySearchTool())
    # config = Config(debug=True, log_level="DEBUG")
    config = Config()
    
    agent = ReActAgent(name="ReActAgent", llm=llm, tool_registry=registry, config=config)
    question = "帮我制定一个明天北京的旅游计划？"
    answer = agent.run(question)
    print(f"最终回答: {answer}")
    