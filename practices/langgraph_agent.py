from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END

import readline
import os, numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: int 
    
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL_ID"),
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
    temperature=0.7,
)

tavily_client = TavilyClient(
    api_key="tvly-dev-" + os.environ.get("TAVILY_API_KEY")
)

def understand_query_node(state: SearchState) -> dict:
    user_message = state["messages"][-1].content
    
    understand_prompt = f"""
    分析用户的查询: '{user_message}'
    
    请完成两个任务：
    1. 简洁总结用户想要什么
    2. 生成最适合搜索引擎的关键词，中英文均可，要精准

    格式：
    理解: [用户需求总结]
    搜索词: [最佳的搜索关键词]
    """

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content.strip()
    
    search_query = user_message
    if "搜索词:" in response_text:
        search_query = response_text.split("搜索词:")[1].strip()
    
    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索: {search_query}")]
    }
    

def tavily_search_node(state: SearchState) -> dict:
    search_query = state["search_query"]
    
    try:
        print(f"正在使用 Tavily 搜索: {search_query}")
        response = tavily_client.search(query=search_query, search_depth="basic", num_results=5, include_answer=True)
        
        # 现在只先处理其中的answer部分，后续可以把这些材料拼起来.
        search_results = response.get("answer", "没有找到相关信息")

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content=f"搜索完成，正在整理答案, 相关信息:{search_results[:100]}... ")]
        }
    except Exception as e:
        print(f"Tavily 搜索出错: {e}")
        search_results = f"搜索失败，请稍后再试: {e}"
        return {
            "search_results": search_results,
            "step": "search_failed",
            "messages": [AIMessage(content=f"搜索遇到问题... {e}")]
        } 

def generate_answer_node(state: SearchState) -> dict:
    if state["step"] == "search_failed":
        prompt = f"""
        搜索结果暂时不可用，请给予您的个人知识回答用户的问题:
        用户问题: {state['user_query']}
        """
    else:
        prompt = f"""
        给予以下搜索结果，为用户的问题提供完整准确的答案:
        用户问题: {state['user_query']}
        搜索结果: {state['search_results']}
        请综合以上信息，生成一个清晰、准确且有用的回答给用户。
        """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "final_answer": response.content.strip(),
        "step": "completed",
        "messages": [AIMessage(content=f"回答已生成，准备发送给用户: {response.content.strip()}")]
    }


def create_assistant() -> StateGraph:
    workflow = StateGraph(SearchState)
    
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)   
    
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    app = workflow.compile()
    return app


def agent_main():
    question = input("请输入您的问题：")
        
    assistant = create_assistant()
    input_msg = SearchState(
        messages=[HumanMessage(content=question)],
        user_query="",
        search_query="",
        search_results="",
        final_answer="",
        step=0
    )

    for event in assistant.stream(input_msg):
        for node, state_update in event.items():
            print(f"\n[运行节点:{node}]")
            
            if "messages" in state_update and len(state_update["messages"]) > 0:
                print(f"-> {state_update['messages'][-1].content}")

if __name__ == "__main__":
    agent_main()
