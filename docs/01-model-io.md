# 1. Model I/O（模型输入输出）

Model I/O 是 LangChain 中与 LLM 交互的基础组件，它提供了统一的接口来调用不同的语言模型。

## 1.1 核心概念

Model I/O 包含三个核心模块：

| 模块 | 作用 | 关键类 |
|------|------|--------|
| **Models** | 语言模型接口 | `ChatOpenAI`, `ChatAnthropic`, `ChatOllama` |
| **Prompts** | 提示词模板管理 | `PromptTemplate`, `ChatPromptTemplate` |
| **Output Parsers** | 输出结构化处理 | `StrOutputParser`, `JsonOutputParser` |

## 1.2 模型（Models）

### 1.2.1 聊天模型 vs LLM

LangChain 区分两种模型类型：

- **Chat Models**（推荐）：接收消息列表，返回消息，适合对话场景
- **LLMs**：接收字符串，返回字符串，适合简单文本生成

```python
from langchain_openai import ChatOpenAI, OpenAI

# 聊天模型（推荐）
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# LLM（传统）
llm = OpenAI(model="gpt-3.5-turbo-instruct")
```

### 1.2.2 常用模型提供商

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# 本地模型 (Ollama)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")

# 智谱 AI
from langchain_zhipuai import ChatZhipuAI
llm = ChatZhipuAI(model="glm-4")

# 通义千问
from langchain_community.chat_models import ChatTongyi
llm = ChatTongyi(model="qwen-turbo")
```

### 1.2.3 模型参数详解

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",        # 模型名称
    temperature=0.7,            # 温度：控制随机性 (0-2)
    max_tokens=2000,            # 最大输出 token 数
    top_p=1.0,                  # 核采样概率
    frequency_penalty=0.0,      # 频率惩罚 (-2 到 2)
    presence_penalty=0.0,       # 存在惩罚 (-2 到 2)
    timeout=None,               # 请求超时时间
    max_retries=2,              # 最大重试次数
    api_key="...",              # API Key（也可通过环境变量）
    base_url="..."              # 自定义 API 地址
)
```

**参数说明**：
- **temperature**：值越低越确定，越高越有创造性
- **max_tokens**：控制生成长度和成本
- **frequency_penalty**：降低重复词的概率
- **presence_penalty**：鼓励讨论新话题

## 1.3 提示词模板（Prompt Templates）

### 1.3.1 基础模板

```python
from langchain_core.prompts import PromptTemplate

# 简单模板
template = "告诉我一个关于 {topic} 的 {style} 笑话"
prompt = PromptTemplate.from_template(template)

# 格式化
formatted = prompt.format(topic="程序员", style="冷")
print(formatted)  # 告诉我一个关于 程序员 的 冷 笑话
```

### 1.3.2 聊天模板

```python
from langchain_core.prompts import ChatPromptTemplate

# 多消息模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role}，用 {tone} 的语气回答问题"),
    ("human", "{question}"),
    ("ai", "我理解你的问题是关于 {context}"),
    ("human", "{follow_up}")
])

# 使用
messages = prompt.format_messages(
    role="技术专家",
    tone="友好",
    question="什么是 LangChain？",
    context="LLM 框架",
    follow_up="能给我举个例子吗？"
)
```

### 1.3.3 少样本提示（Few-shot）

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# 定义示例
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "5*3", "output": "15"},
    {"input": "10/2", "output": "5"}
]

# 创建少样本模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 组合到主提示词
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个计算器助手"),
    few_shot_prompt,
    ("human", "{input}")
])
```

### 1.3.4 消息占位符

```python
from langchain_core.prompts import MessagesPlaceholder

# 动态消息列表
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 helpful 助手"),
    MessagesPlaceholder(variable_name="history"),  # 动态插入历史消息
    ("human", "{input}")
])

from langchain_core.messages import HumanMessage, AIMessage

messages = prompt.format_messages(
    history=[
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮你的？")
    ],
    input="今天天气怎么样？"
)
```

## 1.4 输出解析器（Output Parsers）

### 1.4.1 字符串解析器

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
# 将模型输出转换为纯字符串
```

### 1.4.2 JSON 解析器

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 定义输出结构
class Joke(BaseModel):
    setup: str = Field(description="笑话铺垫")
    punchline: str = Field(description="笑点")
    rating: int = Field(description="好笑程度 1-10")

parser = JsonOutputParser(pydantic_object=Joke)

# 在提示词中包含格式说明
prompt = PromptTemplate.from_template(
    "讲一个关于 {topic} 的笑话。\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### 1.4.3 结构化输出解析器

```python
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

# 定义响应结构
response_schemas = [
    ResponseSchema(name="answer", description="问题的答案"),
    ResponseSchema(name="source", description="信息来源"),
    ResponseSchema(name="confidence", description="置信度 (0-1)")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate.from_template(
    "回答问题并提供相关信息。\n{format_instructions}\n问题: {question}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### 1.4.4 列表解析器

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate.from_template(
    "列出5个关于 {topic} 的关键词。\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

## 1.5 完整工作流示例

### 1.5.1 基础链式调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术作家"),
    ("human", "请用 {style} 的风格解释什么是 {topic}")
])
parser = StrOutputParser()

# 2. 构建链（使用 LCEL）
chain = prompt | llm | parser

# 3. 执行
result = chain.invoke({
    "topic": "LangChain",
    "style": "通俗易懂"
})
print(result)
```

### 1.5.2 流式输出

```python
# 流式输出
for chunk in chain.stream({"topic": "Python", "style": "幽默"}):
    print(chunk, end="", flush=True)
```

### 1.5.3 批量处理

```python
# 批量处理
inputs = [
    {"topic": "Python", "style": "简洁"},
    {"topic": "JavaScript", "style": "详细"},
    {"topic": "Rust", "style": "技术"}
]
results = chain.batch(inputs)
for result in results:
    print(result)
```

## 1.6 最佳实践

### ✅ Do

- 使用 `ChatPromptTemplate` 而不是简单的字符串拼接
- 利用 `partial_variables` 预设固定的提示词内容
- 使用 Pydantic 模型定义结构化输出
- 设置合理的 `temperature` 和 `max_tokens`

### ❌ Don't

- 不要直接在代码中硬编码 API Keys
- 避免过长的单轮提示词，考虑使用 Few-shot 或 Chain
- 不要在生产环境使用过高的 temperature

---

## 参考资源

- [LangChain Model I/O 文档](https://python.langchain.com/docs/concepts/#model-i-o)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI API 文档](https://platform.openai.com/docs/introduction)
