# 3. Agents（智能体）

Agents 是 LangChain 最强大的组件，它让 LLM 从被动回答升级为主动的决策和执行系统。

## 3.1 Agent 核心概念

### ReAct 模式（Reasoning + Acting）

Agent 基于 ReAct 模式工作：

```
Thought（思考） -> Action（行动） -> Observation（观察） -> Thought（再思考）
```

```python
# Agent 执行循环示例
"""
Human: 今天北京的天气怎么样？

AI Thought: 用户询问天气，我需要使用天气工具获取信息
AI Action: 调用 weather_tool(city="北京")
AI Observation: {"temperature": 25, "condition": "晴朗"}

AI Thought: 我已获得天气信息，可以回答用户
AI Final Answer: 今天北京天气晴朗，温度25°C
"""
```

## 3.2 Agent 类型

### 3.2.1 主要 Agent 类型对比

| Agent 类型 | 适用模型 | 特点 | 使用场景 |
|------------|----------|------|----------|
| **ReAct** | 通用 LLM | 推理+行动，最通用 | 通用场景 |
| **OpenAI Tools** | OpenAI 模型 | 原生工具调用 | OpenAI 模型 |
| **Tool Calling** | 支持工具调用的模型 | 结构化工具调用 | 新模型 |
| **Self-Ask** | 通用 LLM | 自问自答分解问题 | 复杂推理 |
| **Plan-and-Execute** | 通用 LLM | 先计划后执行 | 复杂任务 |

### 3.2.2 创建基础 Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 定义工具
tools = [search_tool, calculator_tool, weather_tool]

# 2. 创建 Agent
llm = ChatOpenAI(model="gpt-4o-mini")

# ReAct Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示执行过程
    handle_parsing_errors=True  # 处理解析错误
)

# 执行
result = agent_executor.invoke({"input": "今天北京天气怎么样？"})
```

### 3.2.3 Tool Calling Agent（推荐）

```python
from langchain.agents import create_tool_calling_agent

# 创建 Tool Calling Agent（适用于支持工具调用的模型）
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "计算 123 * 456"})
```

## 3.3 Agent 提示词模板

### 3.3.1 标准 ReAct 模板

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ReAct Agent 提示词模板
react_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，可以使用以下工具：

{tools}

使用以下格式：

Question: 需要回答的问题
Thought: 思考如何解决问题
Action: 要采取的行动（必须是以下之一: [{tool_names}]）
Action Input: 工具的输入
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道答案了
Final Answer: 问题的最终答案

开始！"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

### 3.3.2 结构化 Agent 提示词

```python
structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的数据分析助手。

你的职责：
1. 分析用户的数据相关问题
2. 使用适当的工具获取或处理数据
3. 提供清晰、准确的回答

可用工具：
{tools}

注意：
- 对于简单问题，直接回答
- 对于复杂问题，先思考再行动
- 如果不确定，使用搜索工具获取信息
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

## 3.4 Agent 执行器配置

### 3.4.1 AgentExecutor 参数

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    
    # 执行控制
    max_iterations=10,           # 最大迭代次数
    max_execution_time=60,       # 最大执行时间（秒）
    early_stopping_method="force",  # 提前停止方法
    
    # 错误处理
    handle_parsing_errors=True,  # 处理解析错误
    
    # 日志
    verbose=True,                # 显示详细日志
    
    # 返回格式
    return_intermediate_steps=True,  # 返回中间步骤
)
```

### 3.4.2 带记忆的 Agent

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 带记忆的 Agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 多轮对话
agent_executor.invoke({"input": "你好，我叫小明"})
agent_executor.invoke({"input": "我叫什么名字？"})  # 能记住上下文
```

## 3.5 高级 Agent 模式

### 3.5.1 Plan-and-Execute Agent

```python
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

# 规划器
planner = load_chat_planner(llm)

# 执行器
executor = load_agent_executor(llm, tools, verbose=True)

# Plan-and-Execute Agent
agent = PlanAndExecute(planner=planner, executor=executor)

# 执行复杂任务
result = agent.invoke("帮我制定一个北京3日游计划，包括景点和美食")
```

### 3.5.2 Self-Ask with Search

```python
from langchain.agents import initialize_agent, AgentType

# Self-Ask Agent - 通过自问自答分解复杂问题
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True
)

result = agent.invoke("谁是美国第一任总统的妻子？")
```

### 3.5.3 结构化输出 Agent

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class AgentResponse(BaseModel):
    answer: str = Field(description="问题的答案")
    confidence: float = Field(description="置信度 0-1")
    sources: list = Field(description="信息来源")

# 使用结构化输出
structured_agent = agent_executor.with_types(output_type=AgentResponse)
```

## 3.6 LangGraph Agent（推荐）

LangGraph 是 LangChain 的扩展，用于构建复杂的多步骤 Agent 工作流。

### 3.6.1 基础 LangGraph Agent

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 定义节点
def agent_node(state):
    """Agent 决策节点"""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 定义条件边
def should_continue(state):
    """判断是否应该继续执行"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果有工具调用，继续
    if last_message.tool_calls:
        return "continue"
    return "end"

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# 添加边
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# 编译
app = workflow.compile()

# 执行
result = app.invoke({
    "messages": [("human", "今天北京天气怎么样？")]
})
```

### 3.6.2 带状态的 LangGraph

```python
from langgraph.checkpoint.memory import MemorySaver

# 添加记忆
checkpointer = MemorySaver()

# 编译时添加检查点
app = workflow.compile(checkpointer=checkpointer)

# 执行（带线程ID以维持状态）
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(
    {"messages": [("human", "你好")]},
    config=config
)

# 继续对话（状态会自动恢复）
result = app.invoke(
    {"messages": [("human", "刚才我说了什么？")]},
    config=config
)
```

---

## 最佳实践

### ✅ Do

- 为工具编写清晰、具体的描述
- 限制工具数量（通常 3-5 个最佳）
- 设置合理的 max_iterations 防止无限循环
- 使用 verbose=True 调试 Agent 行为
- 对复杂任务使用 Plan-and-Execute

### ❌ Don't

- 工具描述不要过于笼统
- 不要给 Agent 太多相似功能的工具
- 不要忽略错误处理
- 不要在生产环境使用过高的 temperature

---

## 参考资源

- [LangChain Agent 文档](https://python.langchain.com/docs/modules/agents/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Plan-and-Execute Paper](https://arxiv.org/abs/2305.04091)
