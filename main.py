import logging
import os
from dotenv import load_dotenv
from core import LLM, Toolset
from react_agent import ReActAgent
from plan_solve_agent import PlanSolveAgent
from tools import timenow, curl, search

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",  # 蓝色
        logging.INFO: "\033[92m",  # 绿色
        logging.WARNING: "\033[93m",  # 黄色
        logging.ERROR: "\033[91m",  # 红色
        logging.CRITICAL: "\033[1m\033[91m",  # 粗体红色
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{str(record.msg)}{self.RESET}"
        return super().format(record)


logger = logging.getLogger("LearningAgent")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = ColorFormatter(
    "%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == "__main__":
    load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL")
    API_key = os.environ.get("LLM_API_KEY")
    model_name = os.environ.get("LLM_MODEL_ID")
    model = LLM(base_url=base_url, apikey=API_key, model_name=model_name)
    toolset = Toolset()

    toolset.register(
        "search",
        "Use this tool to search the web for information THAT you don't know or NOT SURE.",
        search,
    )
    toolset.register("timenow", "Use this tool to get the current time.", timenow)
    toolset.register(
        "curl", "Use this tool to curl the content of a webpage given its URL.", curl
    )

    # agent = ReActAgent(
    #     model=model, toolset=toolset, max_iterations=10000, logger=logger
    # )
    agent = PlanSolveAgent(model=model, toolset=toolset)

    while True:
        question = input("请输入您的问题: ")
        result = agent.run(question)
        logger.info(f"最终结果: {result}")
