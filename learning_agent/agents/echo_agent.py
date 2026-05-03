from typing import Iterator

from core.agent import Agent
from core.base import LLM

class EchoAgent(Agent):
    def run(self, input_text: str, **kwargs) -> Iterator[str]:
        message = self._build_prompt(input_text)
        return self.llm.stream_think(message, **kwargs)
    
if __name__ == "__main__":
    llm = LLM()
    
    agent = EchoAgent(name="EchoAgent", llm=llm, system_prompt="你是一个回声代理，你会调戏用户")
    
    for response in agent.run("你好，我是🐖猪猪"):
        print(response, end="")
