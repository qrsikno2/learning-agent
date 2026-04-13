import logging
import os
from serpapi import SerpApiClient
from datetime import datetime
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

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
