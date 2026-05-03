from typing import List, Dict, Any, Optional, final
import logging
from weakref import ref
from core import LLM

INITIAL_PROMPT = """
你是一位资深的中文程序员与架构师。请根据以下要求，快速给出编写一份简单的代码来完成任务。
你的代码必须包含完整的函数前面，文档字符串和注释，以便于理解和维护，并遵循编码规范。

要求: {task}

请直接输出代码，不要包含任何解释或额外的文本。
"""

REFLECT_PROMPT = """
你是一位资深的极其严谨和严格的代码评审专家和资深算法工程师(中文)，对代码质量有极高的要求。
你的任务是审查以下代码，评审上一轮的代码执行结果，并给出详细的反思和改进建议，特别是在</strong>代码结构、性能和可维护性</strong>方面。

# 原始任务:
{task}

# 需要被审查的代码:
```python
{code}
```

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的中文程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

logger = logging.getLogger("LearningAgent.reflect_agent")

class Memory:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
    
    def add_record(self, record_type: str, content: str):
        record = {
            'type': record_type,
            'content': content
        }
        self.records.append(record)
    
    def get_trajectory(self) -> str:
        temp_list = []
        for record in self.records:
            if record['type'] == 'execution':
                temp_list.append(f"上一轮尝试: {record['content']}")
            if record['type'] == 'reflection':
                temp_list.append(f"上一轮评审员反思: {record['content']}")
        
        return '\n\n'.join(temp_list)

    def get_last_execution(self) -> Optional[str]:
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None
    
class ReflectAgent:
    def __init__(self, model: LLM, max_iterations: int = 5):
        self.model = model
        self.memory = Memory()
        self.max_iterations = max_iterations
    
    def _get_model_response(self, prompt: str) -> str:
        msg = [
            {"role": "user", "content": prompt}
        ]
        return self.model.think(msg) or ""
        
    def run(self, task: str) -> str:
        logger.info(f"开始处理任务: {task}")
        initial_prompt = INITIAL_PROMPT.format(task=task)
        initial_code = self._get_model_response(initial_prompt)
        self.memory.add_record('execution', initial_code)
        logger.info(f"初始迭代代码:\n{initial_code}")
        
        for i in range(self.max_iterations):
            logger.info(f"第 {i+1} 轮评审和优化")

            last_code = self.memory.get_last_execution() or ""
            reflect_prompt = REFLECT_PROMPT.format(task=task, code=last_code)
            feedback = self._get_model_response(reflect_prompt)
            self.memory.add_record('reflection', feedback)
            
            if "改进" not in feedback and "优化" not in feedback:
                logger.info("评审员没有提出改进建议，停止迭代")
                break
            
            print(f"评审员反馈:\n{feedback}, 正在生成优化代码...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_model_response(refine_prompt)
            self.memory.add_record('execution', refined_code)
            logger.info(f"优化后的代码:\n{refined_code}")

        final_code = self.memory.get_last_execution() or ""
        return final_code
