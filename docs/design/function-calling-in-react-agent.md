# ReActAgent 支持 Function Calling 设计文档

## 背景

### 正则解析模式（原有实现）

ReActAgent 的原始实现要求 LLM 按固定文本格式输出工具调用：

```
Thought: 我需要搜索天气信息
Action: search[北京明天天气]
```

然后通过正则表达式从 LLM 返回的自由文本中提取工具调用信息：

```python
# 从响应中提取 Thought 和 Action
re.match(r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL)
re.search(r"Action:\s*(.*)", response, re.DOTALL)

# 从 Action 中提取工具名和参数
re.match(r"(\w+)\[(.*)\]", action_str, re.DOTALL)
```

**问题**：
- 依赖 LLM 严格遵守格式，但 LLM 经常输出格式偏差
- 正则表达式脆弱，边界情况多（缺括号、多换行、多余引号等）
- 解析失败时浪费迭代步数重试
- 所有参数都是字符串类型，不支持嵌套对象、数组等复杂类型
- 每次只能解析一个工具调用，不支持并行调用

### Function Calling 模式（新增实现）

现代 LLM（OpenAI、DeepSeek 等）支持原生 function calling。通过 API 的 `tools` 参数传递工具定义，LLM 返回结构化的 `tool_calls` 响应，无需正则解析。

**优势**：
- 格式可靠性：LLM 生成时使用受限解码（constrained decoding），从解码层面保证输出一定是合法 JSON
- 参数类型完整：支持 string、number、boolean、array、object 等 JSON 类型
- 并行调用：LLM 可在一次响应中返回多个 tool_calls
- Prompt 简洁：工具定义通过 `tools` 参数独立传递，prompt 只需描述任务

## 受限解码（Constrained Decoding）

LLM 生成文本是逐 token 预测的过程，每一步输出词表中所有 token 的概率分布。

**普通生成**：所有 token 都有可能被选中。

**受限解码**：在每一步，先用 mask 把不符合约束的 token 概率设为 0，只从合法 token 中采样。

例如生成 `name` 字段时，已注册工具只有 `search` 和 `calculate`：

```
词表: [", {, search, calculate, hello, weather, ...]
普通生成概率:  [0.05, 0.02, 0.15, 0.1, 0.08, 0.12, ...]
受限解码 mask: [  0,    0,     1,    1,    0,    0,   ...]
受限后概率:    [  0,    0,   0.6,  0.4,   0,    0,   ...]
```

LLM 只能输出 `search` 或 `calculate`，不可能输出 `hello`。

这不是 LLM 本身的功能，而是推理框架在解码时加的限制（LogitsProcessor）。每一步都有对应的 JSON schema 约束，确保最终输出一定是合法格式。

**与正则解析的本质区别**：
- 正则解析：生成**之后**从任意文本中提取（事后检查）
- 受限解码：生成**过程中**强制只输出合法格式（事前约束）

## 架构设计

### 两条路径并存

```
ReActAgent
├── enable_function_calling=False（默认）
│   └── _run_with_regex() → REACT_PROMPT + 正则解析
└── enable_function_calling=True
    └── _run_with_function_calling() → FC_REACT_PROMPT + think_with_tools()
        └── Fallback: 失败时降级回 _fallback_one_step()（REACT_PROMPT + 正则）
```

### 核心组件

| 组件 | 文件 | 职责 |
|---|---|---|
| `LLMResponse` | `core/base.py` | 封装 LLM 响应，包含 content 和 tool_calls |
| `think_with_tools()` | `core/base.py` | 调用 LLM API 时传入 tools 参数，返回 LLMResponse |
| `ToolRegistry.get_openai_tools()` | `core/tool.py` | 将注册工具转为 OpenAI tools schema 列表 |
| `Message` | `core/message.py` | 支持 tool_calls 和 tool_call_id 字段 |
| `ReActAgent` | `agents/react_agent.py` | 两条路径的调度和执行逻辑 |

### Function Calling 执行流程

```
while step < max_steps:
    1. 构建 messages（FC_REACT_PROMPT，简化版，无格式说明）
    2. 获取 tools schema: registry.get_openai_tools()
    3. 调用 llm.think_with_tools(messages, tools=tools_schema)
    4. 解析 LLMResponse:
       - has_tool_calls() == True:
           for each tool_call:
             解析 tool_name 和 arguments
             执行工具 → 得到 result
             追加 assistant message（带 tool_calls）
             追加 tool message（role="tool", content=result, tool_call_id=...）
       - has_tool_calls() == False:
           返回 content 作为最终答案
    5. Fallback: think_with_tools() 返回 None 时
       → 切换到 REACT_PROMPT + 正则解析重试当前步骤
       → 后续步骤继续尝试 function calling（不全局降级）
```

## Prompt 设计

### 正则模式 Prompt（REACT_PROMPT）

包含详细的格式说明，要求 LLM 按 `Thought: ... Action: name[input]` 格式输出。

### Function Calling 模式 Prompt（FC_REACT_PROMPT）

简化版，只描述任务和要求推理过程，不包含格式说明：

```
你是一个具备推理和行动能力的AI助手。
请分析问题，调用合适的工具获取信息，最终给出准确的答案。
每一步请先说明你的推理过程，然后决定是否需要调用工具。
当你有足够信息时，直接给出最终答案。
```

ReAct 行为（思考→行动→观察→循环）靠循环代码保证，不靠 prompt 格式。

## Fallback 降级策略

当 `enable_function_calling=True` 时，如果 function calling 失败（模型不支持 tools 参数、API 报错、tool_calls 解析失败）：

1. 只重试当前这一步，不全局降级
2. 切换到 REACT_PROMPT（带格式说明）+ 普通 `llm.think()` + 正则解析
3. 下一步继续尝试 function calling

这样既保证了 function calling 的优先使用，又兼容不支持的模型。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `learning_agent/core/base.py` | 新增 `LLMResponse` 数据类；新增 `think_with_tools()` 方法 |
| `learning_agent/core/message.py` | 新增 `tool_calls`、`tool_call_id` 字段；`content` 改为 Optional |
| `learning_agent/core/tool.py` | 新增 `ToolRegistry.get_openai_tools()` |
| `learning_agent/core/__init__.py` | 导出 `LLMResponse` |
| `learning_agent/agents/react_agent.py` | 新增 `enable_function_calling` 参数；新增 `_run_with_function_calling()` 和 `_fallback_one_step()` |
| `learning_agent/tools/tavily_search.py` | 补充 `ToolParameter` 定义（query 参数） |

## 使用方式

```python
from learning_agent.core import LLM, ToolRegistry, Config
from learning_agent.agents import ReActAgent
from learning_agent.tools import DateTimeTool, TavilySearchTool

llm = LLM()
registry = ToolRegistry()
registry.register(DateTimeTool())
registry.register(TavilySearchTool())
config = Config()

# 正则模式（默认，向后兼容）
agent = ReActAgent(name="react", llm=llm, tool_registry=registry, config=config)
answer = agent.run("北京明天天气怎么样？")

# Function calling 模式
agent_fc = ReActAgent(name="react", llm=llm, tool_registry=registry, config=config, enable_function_calling=True)
answer = agent_fc.run("北京明天天气怎么样？")

# 调试模式
config_debug = Config(debug=True)
agent_debug = ReActAgent(name="react", llm=llm, tool_registry=registry, config=config_debug, enable_function_calling=True)
answer = agent_debug.run("北京明天天气怎么样？")
```

## 后续计划

- `stream_run()` 支持 function calling（需要处理 tool_calls 的 chunk 拼接）
- `SimpleAgent` 支持 function calling
