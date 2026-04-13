import logging
import re
from typing import Any, Dict, List, cast

from openai.types.chat import ChatCompletionMessageParam

from core import LLM, Toolset


SYSTEM_PROMPT_TEMPLATE = """
你是一个由emofer研发的AI助手, 你需要仔细的逐步的来回答用户的问题。
你可以使用以下工具来获取你不知道或不确定的信息：
{tools_list}

请严格按照以下格式来进行对话:
Thought: 你的思考过程
Action: 你这一阶段决定执行的工具，必须满足以下格式:
- `{{tool_name}}[{{tool_input}}]`，其中{{tool_name}}必须是上面工具列表中的一个工具名称，{{tool_input}}是你要传递给工具的输入。
- `Finish[{{final_answer}}]`，当你认为已经有足够的信息来回答用户的问题时，使用这个格式来结束对话，其中{{final_answer}}是你要给用户的最终答案。
- 当你已经完成了一个工具的调用后，你需要继续进行思考并决定下一步的行动，直到你认为可以给出最终答案为止。
"""

USER_PROMPT_TEMPLATE = """
请根据上面的系统提示来回答用户的问题。你需要在每一步都清晰地展示你的思考过程，并且合理地使用工具来获取信息，直到你认为可以给出最终答案为止。
历史对话: 
{history}

用户的问题是: {question}
请根据上述进展，给出你下一步的Thought和Action.
"""


class ReActAgent:
    def __init__(
        self,
        model: LLM,
        toolset: Toolset,
        max_iterations=1000,
        max_history_items: int = 10,
        max_history_entry_chars: int = 200,
        max_tool_result_chars: int = 400,
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.toolset = toolset
        self.max_iterations = max_iterations
        self.max_history_items = max_history_items
        self.max_history_entry_chars = max_history_entry_chars
        self.max_tool_result_chars = max_tool_result_chars
        self.logger = logger or logging.getLogger(__name__)
        self.history: List[str] = []

    def _append_history(self, entry: str) -> None:
        self.history.append(entry)
        if len(self.history) > self.max_history_items:
            self.history = self.history[-self.max_history_items :]

    def _truncate_history_entry(self, text: Any) -> str:
        return str(text)[: self.max_history_entry_chars]

    def _summarize_tool_result(self, tool_name: str, tool_result: Any) -> str:
        text = str(tool_result).strip()
        compact_text = " ".join(text.split())

        if not compact_text:
            return f"{tool_name}: empty result"

        if tool_name == "search":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            summary_parts: List[str] = []
            if lines:
                summary_parts.append(lines[0])

            numbered_lines = [line for line in lines if re.match(r"^\d+\.\s+", line)]
            summary_parts.extend(numbered_lines[:2])
            summary = " | ".join(summary_parts)
            if summary:
                return summary[: self.max_tool_result_chars]

        return compact_text[: self.max_tool_result_chars]

    def _parse_response(self, response: str) -> Dict[str, Any]:
        thought_match = re.match(
            r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL
        )
        action_match = re.search(r"Action:\s*(.*)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action_text = action_match.group(1).strip() if action_match else ""
        action = self._parse_action(action_text) if action_text else {}
        return {"thought": thought, "action": action}

    def _parse_action(self, action_str: str) -> Dict[str, str]:
        mymatch = re.match(r"(\w+)\[(.*)\]", action_str)
        if mymatch:
            return {"name": mymatch.group(1), "input": mymatch.group(2)}

        return {}

    def run(self, question: str):
        self.history = []
        for _ in range(self.max_iterations):
            history_string = "\n".join(self.history)
            msg: List[ChatCompletionMessageParam] = cast(
                List[ChatCompletionMessageParam],
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_TEMPLATE.format(
                            tools_list=self.toolset.get_available_tools()
                        ),
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            question=question, history=history_string
                        ),
                    },
                ],
            )

            response = self.model.think(msg)
            self.logger.debug(f"LLM Response: {response}")

            if not response:
                self.logger.error("LLM did not return a response.")
                break

            parsed_response = self._parse_response(response)
            if not parsed_response:
                self.logger.error("Failed to parse LLM response. Retrying...")
                continue

            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action") or {}

            if not action:
                self.logger.error(
                    "Failed to parse Action from LLM response. Retrying..."
                )
                if thought:
                    self.logger.info(f"Thought: {thought}")
                    self._append_history(
                        f"Thought: {self._truncate_history_entry(thought)}"
                    )
                self._append_history(
                    f"Invalid Response: {self._truncate_history_entry(response)}"
                )
                continue

            if action.get("name") == "Finish":
                final_answer = action.get("input", "")
                self.logger.info(f"Final Answer: {final_answer}")
                return final_answer

            if thought:
                self.logger.info(f"Thought: {thought}")
                self._append_history(
                    f"Thought: {self._truncate_history_entry(thought)}"
                )

            tool_name = action.get("name")
            tool_input = action.get("input", "")

            if not tool_name:
                self.logger.error("Parsed action is missing tool name. Retrying...")
                self._append_history(
                    f"Invalid Action: {self._truncate_history_entry(action)}"
                )
                continue

            self.logger.info(f"Action: {tool_name}[{tool_input}]")
            self._append_history(
                f"Action: {tool_name}[{self._truncate_history_entry(tool_input)}]"
            )

            tool_func = self.toolset.get(tool_name)
            if tool_func:
                tool_result = tool_func(tool_input)
                if not tool_result:
                    tool_result = "工具执行失败或没有返回结果。"
                self.logger.debug(f"Tool Result: {tool_result}")
                summarized_result = self._summarize_tool_result(tool_name, tool_result)
                self._append_history(f"Tool Result Summary: {summarized_result}")
            else:
                self._append_history(f"Tool {tool_name} not found in toolset.")

        self.logger.warning("Reached maximum iterations without finishing.")
        return None
