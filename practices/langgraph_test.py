import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from practices.core import LLM

load_dotenv()

class AgentState(TypedDict):
    messages: List[str] # 对话历史
    current_task: str # 当前任务
    final_answer: str # 最终答案

llm = LLM(
    base_url=os.environ.get("LLM_BASE_URL"),
    apikey=os.environ.get("LLM_API_KEY"),
    model_name=os.environ.get("LLM_MODEL_ID")
)
    
def planner_node(state: AgentState) -> AgentState:
    current_task = state['current_task']
    
    print("Planner Node: 生成计划中...")
    plan = f"你是一个规划专家，请为任务 '{current_task}' 生成制定的分布计划..."
    
    response = llm.think([{"role": "system", "content": plan}])
    
    real_plan = response if response else "未能生成计划"
    
    state['messages'].append(f"Planner Node: {real_plan}")
    
    return state

def executor_node(state: AgentState) -> AgentState:
    latest_plan = state['messages'][-1] if state['messages'] else "没有计划"
    
    print("Executor Node: 执行任务中...")
    execution = f"你是一个执行专家，请根据以下计划执行任务：{latest_plan}"
    
    result = llm.think([{"role": "system", "content": execution}])

    state['messages'].append(f"Executor Node: {result}")

    return state

def should_continue(state: AgentState) -> str:
    if len(state["messages"]) < 3:
        return "continue_to_planner"
    else:
        state['final_answer'] = state['messages'][-1]
        return "end_workflow"

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")

workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue_to_planner": "planner",
        "end_workflow": END
    }
)

app = workflow.compile()

inputs = """
阅读下面的材料，根据要求写作。（60分）

德国哲学家康德曾提出过一条著名的道德律令：“你的行动，要把你自己人身中的人性，和其他人身中的人性，在任何时候都同样看作是目的，永远不能只看作是手段。”

然而，在现代社会的高速运转中，我们常常看到不同的图景：
在效率至上的系统里，外卖骑手或网约车司机有时被异化为算法指令下的“履约工具”；在激烈的升学或职场竞争中，有人将同伴视为攀登的“垫脚石”；甚至很多时候，我们自己也会把自己当成获取分数、积攒履历或兑换财富的“手段”，在疲于奔命间，模糊了生活的本来面目。

“人究竟是目的，还是手段？”两百多年前的哲学叩问，在今天依然振聋发聩。

以上材料引发了你怎样的联想和思考？请结合时代背景与自身体验，写一篇文章。

要求： 1. 选准角度，确定立意，明确文体，自拟标题；
2. 不要套作，不得抄袭；
3. 不得泄露个人真实信息；
4. 字数不少于800字。
"""

input_msg = AgentState(
    messages=[],
    current_task=inputs,
    final_answer=""
)

for event in app.stream(input_msg):
    for node, state_update in event.items():
        print(f"\n[运行节点:{node}]")
        
        if "messages" in state_update and len(state_update["messages"]) > 0:
            print(f"-> {state_update['messages'][-1]}")

if "final_answer" in state_update:
    print(f"\n最终答案: {state_update['final_answer']}")
