import logging
import os, re
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
from typing import Any, Callable, Dict, List, Optional, cast
from serpapi import SerpApiClient
from datetime import datetime
import requests
from bs4 import BeautifulSoup


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",  # 蓝色
        logging.INFO: "\033[92m",  # 绿色
        logging.WARNING: "\033[93m",  # 黄色
        logging.ERROR: "\033[91m",  # 红色
        logging.CRITICAL: "\033[1m\033[91m",  # 粗体红色
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{str(record.msg)}{self.RESET}"
        return super().format(record)


logger = logging.getLogger("LearningAgent")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = ColorFormatter(
    "%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class LLM:
    def __init__(self, base_url, apikey, model_name):
        self.base_url = base_url
        self.apikey = apikey
        self.model_name = model_name

        self.client = OpenAI(base_url=self.base_url, api_key=self.apikey)

    def think(self, prompt: List[ChatCompletionMessageParam]) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                # stream=True,
                stream=False,
            )

            result = response.choices[0].message.content
            if not result:
                finish_reason = response.choices[0].finish_reason
                logger.warning(
                    f"LLM response has no content. Finish reason: {finish_reason}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in LLM think: {e}")
            return None


class Toolset:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, func):
        self.tools[name] = {"description": description, "func": func}

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        return self.tools.get(name, {}).get("func", None)

    def get_available_tools(self) -> str:
        return "\n".join(
            [f"{name}: {info['description']}" for name, info in self.tools.items()]
        )


def format_serpapi_results_for_llm(json_data):
    context_parts = []

    # 1. 优先提取直接答案 (Answer Box)
    if "answer_box" in json_data:
        answer_text = json_data["answer_box"].get("snippet") or json_data[
            "answer_box"
        ].get("answer")
        if answer_text:
            context_parts.append(f"[直接答案]: {answer_text}")

    if "knowledge_graph" in json_data:
        kg = json_data["knowledge_graph"]
        desc = kg.get("description", "")
        if desc:
            context_parts.append(f"[知识图谱介绍]: {desc}")
        facts = [
            f"{k}: {v}"
            for k, v in kg.items()
            if isinstance(v, str)
            and not k.endswith("link")
            and k not in ["title", "type", "description"]
        ]
        if facts:
            context_parts.append(f"[实体属性]: {', '.join(facts)}")

    if "organic_results" in json_data:
        context_parts.append("[网络搜索结果]:")
        for idx, result in enumerate(json_data["organic_results"][:3]):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            context_parts.append(
                f"{idx + 1}. {title}\n   摘要: {snippet}\n   来源: {link}"
            )

    return "\n\n".join(context_parts) if context_parts else "没有找到相关信息。"


def search(query: str):
    try:
        logger.debug(f"正在执行搜索 {query}")
        api_key = os.environ.get("SERPAPI_API_KEY", None)
        if not api_key:
            logger.warning("API KEY not configured")
            return None

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "jp",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params_dict=params)
        result = client.get_dict()
        logger.debug("搜索完毕, 正在格式化结果...")
        result = format_serpapi_results_for_llm(result)
        logger.debug("结果格式化完毕.")
        return result

    except Exception as e:
        logger.error("Error in searching %s: %s" % (query, str(e)))
        return None


def timenow(dummyinput) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def curl(url: str) -> str:
    try:
        logger.debug(f"正在浏览 {url}")
        response = requests.get(url, timeout=10)
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text[:65535] if text else "无法获取页面内容"

    except Exception as e:
        logger.error(f"Error in browsing {url}: {e}")
        return "浏览页面时发生错误： " + str(e)


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
    def __init__(self, model: LLM, toolset: Toolset, max_iterations=1000):
        self.model = model
        self.toolset = toolset
        self.max_iterations = max_iterations
        self.history = []

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
        """
        运行ReActAgent来回答用户的问题。这个方法会根据系统提示模板构建对话历史，并不断调用LLM来获取思考和行动，直到得到最终答案或达到最大迭代次数。
        """

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
            logger.debug(f"LLM Response: {response}")

            if not response:
                logger.error("LLM did not return a response.")
                break

            parsed_response = self._parse_response(response)
            if not parsed_response:
                logger.error("Failed to parse LLM response. Retrying...")
                continue

            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action") or {}

            if not action:
                logger.error("Failed to parse Action from LLM response. Retrying...")
                if thought:
                    logger.info(f"Thought: {thought}")
                    self.history.append(f"Thought: {thought}")
                self.history.append(f"Invalid Response: {response}")
                continue

            if action.get("name") == "Finish":
                final_answer = action.get("input", "")
                logger.info(f"Final Answer: {final_answer}")
                return final_answer

            if thought:
                logger.info(f"Thought: {thought}")
                self.history.append(f"Thought: {thought}")

            if action:
                tool_name = action.get("name")
                tool_input = action.get("input", "")

                if not tool_name:
                    logger.error("Parsed action is missing tool name. Retrying...")
                    self.history.append(f"Invalid Action: {action}")
                    continue

                logger.info(f"Action: {tool_name}[{tool_input}]")
                self.history.append(f"Action: {tool_name}[{tool_input}]")

                tool_func = self.toolset.get(tool_name)
                if tool_func:
                    tool_result = tool_func(tool_input)
                    if not tool_result:
                        tool_result = "工具执行失败或没有返回结果。"
                    logger.debug(f"Tool Result: {tool_result}")
                    self.history.append(f"Tool Result: {tool_result}")
                else:
                    self.history.append(f"Tool {tool_name} not found in toolset.")

        logger.warning("Reached maximum iterations without finishing.")
        return None


if __name__ == "__main__":
    load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL")
    API_key = os.environ.get("LLM_API_KEY")
    model_name = os.environ.get("LLM_MODEL_ID")
    model = LLM(base_url=base_url, apikey=API_key, model_name=model_name)
    toolset = Toolset()

    toolset.register(
        "search",
        "Use this tool to search the web for information THAT you don't know or NOT SURE.",
        search,
    )
    toolset.register("timenow", "Use this tool to get the current time.", timenow)
    toolset.register(
        "curl", "Use this tool to curl the content of a webpage given its URL.", curl
    )

    agent = ReActAgent(model=model, toolset=toolset, max_iterations=10000)

    while True:
        question = input("请输入您的问题: ")
        result = agent.run(question)
        logger.info(f"最终结果: {result}")
