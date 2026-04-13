import logging
import os
from dotenv import load_dotenv
from typing import Any
from serpapi import SerpApiClient
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from core import LLM, Toolset
from react_agent import ReActAgent


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


def timenow(_) -> str:
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


PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个可以由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的，可执行的子任务，并且按照逻辑顺序排列.
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

请严格按照以下格式来输出你的计划，用```python和```作为前后缀来包裹你的输出是必要的:
```python
["子任务1的描述","子任务2的描述",...]
```

以下是你需要解决的用户问题: 
{question}
请开始你的规划，并输出你的行动计划。
"""


class PlannerAgent:
    def __init__(self, llm_client: LLM):
        self.llm_client = llm_client


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

    agent = ReActAgent(
        model=model, toolset=toolset, max_iterations=10000, logger=logger
    )

    while True:
        question = input("请输入您的问题: ")
        result = agent.run(question)
        logger.info(f"最终结果: {result}")
