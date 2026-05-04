import datetime
from learning_agent.core import Tool

class DateTimeTool(Tool):
    def __init__(self):
        super().__init__(name="datetime", description="获取当前日期和时间，参数格式：datetime[]")

    def run(self, params: dict) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")