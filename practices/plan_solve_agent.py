from practices.core import LLM, Toolset
import logging
import ast

logger = logging.getLogger("LearningAgent.PlanSolveAgent")

PLANNER_PROMPT_TEMPLATE_SYSTEM = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个可以由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的，可执行的子任务，并且按照逻辑顺序排列.
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

请严格按照以下格式来输出你的计划，用```python和```作为前后缀来包裹你的输出是必要的:
```python
["子任务1的描述","子任务2的描述",...]
```
"""

PLANNER_PROMPT_TEMPLATE_USER = """
以下是你需要解决的用户问题: 
{question}
请开始你的规划，并输出你的行动计划。
"""

EXECUTOR_PROMPT_TEMPLATE_SYSTEM = """
你是一个顶级的AI执行专家。你的任务是根据指定的行动计划，一步步的解决用户的问题.
你将受到原始问题，完整的计划，以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅仅输出该步骤的最终答案，不要输出任何额外的解释或者对话。
# 原始问题:
{question}

# 完整的计划:
{plan}
"""

EXECUTOR_PROMPT_TEMPLATE_USER = """
以下是这一步的信息
# 已经完成的步骤和结果:
{history}

# 当前步骤:
{current_step}

接下来请你执行当前步骤，并输出该步骤的最终答案。
"""

class PlannerAgent:
    def __init__(self, model: LLM, tools: Toolset):
        self.model = model
        self.tools = tools
    
    def plan(self, question: str) -> list[str]:
        """
        根据用户的问题生成一个行动计划，返回一个字符串列表，每个字符串描述一个子任务。
        """
        prompt = PLANNER_PROMPT_TEMPLATE_USER.format(question=question)

        messages = [
            {"role": "system", "content": PLANNER_PROMPT_TEMPLATE_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        logger.debug(f"将信息发送到LLM: {prompt}")
        response = self.model.think(messages) or ""
        try:
            logger.debug(f"接收到plan的结果: {response}")
            plans = response.strip().split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plans)
            
            return plan if isinstance(plan, list) else []
        except Exception as e:
            logger.error(f"解析计划失败: {e}")
            return []

class ExecutorAgent:
    def __init__(self, model: LLM, tools: Toolset):
        self.model = model
        self.tools = tools

    def execute(self, question:str, plan: list[str]) -> str:
        """
        根据原始问题，完整的计划，以及到目前为止已经完成的步骤和结果，执行当前步骤，并输出该步骤的最终答案。
        """
        history = ""

        logger.debug("正在执行计划...")
        for idx, step in enumerate(plan):
            logger.debug(f"正在执行步骤 {idx + 1}/{len(plan)}: {step}")
            
            prompt_sys = EXECUTOR_PROMPT_TEMPLATE_SYSTEM.format(question=question, plan=plan)
            prompt_user = EXECUTOR_PROMPT_TEMPLATE_USER.format(history=history, current_step=step)
            messages = [
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user}
            ]
            
            response = self.model.think(messages) or ""
            
            history += f"步骤 {idx + 1}: {step}\n结果: {response}\n\n"
            
            logger.debug(f"步骤 {idx + 1} 的结果: {response}")
        
        final_answer = response.strip()
        return final_answer

class PlanSolveAgent:
    def __init__(self, model: LLM, tools: Toolset):
        self.planner = PlannerAgent(model, tools)
        self.executor = ExecutorAgent(model, tools)
        self.model = model
        self.tools = tools

    def run(self, question: str) -> str:
        plan = self.planner.plan(question)
        if not plan:
            logger.error("未能生成有效的计划.")
            return "抱歉，我无法为这个问题生成一个计划。"

        final_answer = self.executor.execute(question, plan)
        return final_answer
