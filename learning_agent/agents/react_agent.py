import json
import logging
import re
from typing import Dict, List, Optional, Iterator

from learning_agent.core import Agent, RunnableMixin, StreamableMixin, ToolRegistry, Config, LLM, Message, ToolCallMixin
from learning_agent.core.base import LLMResponse
from learning_agent.core.console import print_tool_call, print_tool_result, print_final_answer

logger = logging.getLogger(__name__)

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

FC_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。
请分析问题，调用合适的工具获取信息，最终给出准确的答案。
每一步请先说明你的推理过程，然后决定是否需要调用工具。
当你有足够信息时，直接给出最终答案。

## 当前任务
**Question:** {question}

## 执行历史
{history}
"""


class ReActAgent(Agent, RunnableMixin, StreamableMixin, ToolCallMixin):
    def __init__(
        self,
        name: str,
        llm: LLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 10,
        custom_prompt: Optional[str] = None,
        enable_function_calling: bool = False,
    ):
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = True
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else REACT_PROMPT
        self.enable_function_calling = enable_function_calling

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
        if self.enable_function_calling:
            return self._run_with_function_calling(input_text, **kwargs)
        return self._run_with_regex(input_text, **kwargs)

    def _run_with_regex(self, input_text: str, **kwargs) -> str:
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
                print_tool_call("思考", thought)
                logger.debug("LLM Thought: %s", thought)
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
            print_tool_call(action_name, action_input)
            logger.debug("LLM Action: %s[%s]", action_name, action_input)

            try:
                result = self.tool_registry.execute(action_name, action_input)
                print_tool_result(action_name, result)
                logger.debug("工具执行结果: %s", result)
            except Exception as e:
                result = f"工具执行失败: {e}"

            self.current_history.append(f"Observation: {result}")

        fallback_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=fallback_answer, role="assistant"))
        return fallback_answer

    def _run_with_function_calling(self, input_text: str, **kwargs) -> str:
        self.current_history = []
        tools_schema = self.tool_registry.get_openai_tools()
        messages: List[dict] = []

        for step in range(self.max_steps):
            history_str = "\n".join(self.current_history) 
            prompt = FC_REACT_PROMPT.format(question=input_text, history=history_str)
            messages = [{"role": "user", "content": prompt}]

            response = self.llm.think_with_tools(messages, tools=tools_schema, **kwargs)

            if response is None:
                fallback = self._fallback_one_step(input_text, **kwargs)
                if fallback is not None:
                    return fallback
                continue

            if not response.has_tool_calls:
                answer = response.get_text()
                if not answer:
                    continue
                self.add_message(Message(content=input_text, role="user"))
                self.add_message(Message(content=answer, role="assistant"))
                return answer

            tool_calls = response.tool_calls
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args_raw = tc.function.arguments

                try:
                    tool_args = json.loads(tool_args_raw)
                except (json.JSONDecodeError, TypeError):
                    tool_args = tool_args_raw

                print_tool_call(tool_name, tool_args)
                logger.debug("Function Call: %s(%s)", tool_name, tool_args)

                self.current_history.append(f"Action: {tool_name}({tool_args})")

                try:
                    result = self.tool_registry.execute(tool_name, tool_args)
                except Exception as e:
                    result = f"工具执行失败: {e}"

                print_tool_result(tool_name, result)
                logger.debug("工具执行结果: %s", result)

                self.current_history.append(f"Observation: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        fallback_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=fallback_answer, role="assistant"))
        return fallback_answer

    def _fallback_one_step(self, input_text: str, **kwargs) -> Optional[str]:
        logger.debug("Function calling 失败，降级回正则模式重试当前步骤")

        tools_desc = self.tool_registry.get_tools_description() or "无可用工具"
        history_str = "\n".join(self.current_history) or "（尚未执行任何操作）"
        prompt = REACT_PROMPT.format(
            tools=tools_desc,
            question=input_text,
            history=history_str,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.think(messages, **kwargs)

        if not response:
            return None

        parsed = self._parse_response(response)
        thought = parsed.get("thought", "")
        action = parsed.get("action") or {}

        if not action:
            return None

        if thought:
            self.current_history.append(f"Thought: {thought}")

        action_name = action.get("name", "")
        action_input = action.get("input", "")

        if action_name == "Finish":
            self.add_message(Message(content=input_text, role="user"))
            self.add_message(Message(content=action_input, role="assistant"))
            return action_input

        if not action_name:
            return None

        self.current_history.append(f"Action: {action_name}[{action_input}]")

        try:
            result = self.tool_registry.execute(action_name, action_input)
        except Exception as e:
            result = f"工具执行失败: {e}"

        self.current_history.append(f"Observation: {result}")
        return None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
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
            
            full_response = ""
            yield "[思考]: "
            for chunk in self.llm.stream_think(messages, **kwargs):
                if chunk:
                    full_response += chunk
                    yield chunk

            if not full_response:
                self.current_history.append("（LLM未返回有效响应）")
                continue

            parsed = self._parse_response(full_response)
            thought = parsed.get("thought", "")
            action = parsed.get("action") or {}

            if not action:
                self.current_history.append(f"Invalid Response: {full_response}")
                continue

            if thought:
                self.current_history.append(f"Thought: {thought}")
            action_name = action.get("name", "")
            action_input = action.get("input", "")

            if action_name == "Finish":
                self.add_message(Message(content=input_text, role="user"))
                self.add_message(Message(content=action_input, role="assistant"))
                yield f"[最终回答]: {action_input}\n"
                return 
                
            if not action_name:
                self.current_history.append(f"Invalid Action: {action}")
                continue

            self.current_history.append(f"Action: {action_name}[{action_input}]")
            yield f"[调用工具: {action_name}]\n"

            try:
                result = self.tool_registry.execute(action_name, action_input)
                yield f"[工具结果: name={action_name}, result={result}]\n"
            except Exception as e:
                result = f"工具执行失败: {e}"
            self.current_history.append(f"Observation: {result}")

        fallback_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=fallback_answer, role="assistant"))
        yield fallback_answer

if __name__ == "__main__":
    from dotenv import load_dotenv
    from learning_agent.tools import DateTimeTool, TavilySearchTool
    load_dotenv()

    logger = logging.getLogger("learning_agent.agents.react_agent")

    llm = LLM()
    registry = ToolRegistry()
    registry.register(DateTimeTool())
    registry.register(TavilySearchTool())
    config = Config().from_env()
    config.setup_logging()
    
    agent = ReActAgent(name="ReActAgent", llm=llm, tool_registry=registry, config=config, enable_function_calling=True)
    question = "帮我制定一个明天北京的旅游计划？"
    answer = agent.run(question)
    print_final_answer(answer)
    
    # agent2 = ReActAgent(name="ReActAgentStream", llm=llm, tool_registry=registry, config=config)
    # question = "帮我制定一个明天南京的旅游计划？"
    # print("Streaming回答:")
    # for chunk in agent2.stream_run(question):
    #     print(chunk, end="")
