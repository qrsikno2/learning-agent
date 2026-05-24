import os
from learning_agent.core import Tool, ToolParameter


class TavilySearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="search",
            description="搜索互联网获取最新信息，输入搜索关键词，返回搜索结果摘要",
            parameters=[
                ToolParameter(name="query", type="string", description="搜索关键词", required=True)
            ]
        )

    def run(self, params) -> str:
        try:
            from tavily import TavilyClient
        except ImportError:
            return "Tavily 客户端未安装。请运行: pip install tavily-python"

        if isinstance(params, dict):
            query = params.get("query", "") or params.get("q", "") or params.get("input", "")
        else:
            query = str(params)

        if not query:
            return "错误：未提供搜索关键词"

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "错误：TAVILY_API_KEY 环境变量未设置"

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                num_results=5,
                include_answer=True,
            )
            answer = response.get("answer", "")
            results = response.get("results", [])
            if results:
                snippets = []
                for i, r in enumerate(results[:3], 1):
                    title = r.get("title", "")
                    content = r.get("content", "")[:120].replace("\n", " ").strip()
                    url = r.get("url", "")
                    snippets.append(f"[{i}] {title}\n    {content}\n    {url}")
                refs = "\n".join(snippets)
                if answer:
                    return f"{answer}\n\nReferences:\n{refs}"
                return refs
            return answer or "没有找到相关信息"
        except Exception as e:
            return f"搜索执行失败: {e}"
