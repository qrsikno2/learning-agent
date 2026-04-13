from core import LLM

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
