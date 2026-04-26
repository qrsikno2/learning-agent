import os, asyncio, json
from dotenv import load_dotenv
from agentscope.agent import ReActAgent, AgentBase
from agentscope.model import OpenAIChatModel, OllamaChatModel
from agentscope.formatter import OpenAIChatFormatter, OllamaMultiAgentFormatter
from agentscope.message import Msg
from pydantic import BaseModel, Field

load_dotenv()  

llm = OpenAIChatModel(
    model_name=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"), 
    client_kwargs={"base_url": os.getenv("LLM_BASE_URL")},
    stream=True,
)

# 创建一个 ReAct 智能体
agent = ReActAgent(
    name="Jarvis",
    sys_prompt="你是一个名为 Jarvis 的有用助手。",
    model=llm,
    formatter=OpenAIChatFormatter(),
)

# 结构化模型
class Model(BaseModel):
    name: str = Field(description="人物的姓名")
    description: str = Field(description="人物的一句话描述")
    age: int = Field(description="年龄")
    honor: list[str] = Field(description="人物荣誉列表")

async def example_structured_output() -> None:
    """结构化输出示例"""
    res = await agent(
        Msg(
            "user",
            "介绍爱因斯坦",
            "user",
        ),
        structured_model=Model,
    )
    print("\n结构化输出：")
    print(json.dumps(res.metadata, indent=4, ensure_ascii=False))


asyncio.run(example_structured_output())
