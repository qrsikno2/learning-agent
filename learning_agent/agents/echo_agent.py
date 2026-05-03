from typing import Iterator

from dotenv import load_dotenv

from learning_agent.core import Agent, RunnableMixin, SteamableMixin
from learning_agent.core import LLM

class EchoAgent(Agent, RunnableMixin):
    def run(self, input_text: str, **kwargs) -> Iterator[str]:
        message = self._build_prompt(input_text)
        return self.llm.stream_think(message, **kwargs)

class StreamEchoAgent(Agent, SteamableMixin):
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        message = self._build_prompt(input_text)
        return self.llm.stream_think(message, **kwargs)
    
if __name__ == "__main__":
    load_dotenv()
    llm = LLM()
    
    agent = StreamEchoAgent(name="EchoAgent", llm=llm, system_prompt="你是一个回声代理，你会调戏用户")
    
    for response in agent.stream_run("你好，我是🐖猪猪, 请写一篇10000字的文章歌颂我"):
        print(response, end="")
