# 6. Chains（链）

Chains 是将多个组件组合成可复用工作流的核心机制。LangChain Expression Language (LCEL) 提供了声明式的方式来构建链。

## 6.1 LCEL 基础

### 6.1.1 管道操作符

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 组件
prompt = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 使用 | 操作符构建链
chain = prompt | llm | parser

# 执行
result = chain.invoke({"topic": "程序员"})
```

### 6.1.2 链的执行方法

```python
# 同步执行
result = chain.invoke({"topic": "程序员"})

# 批量执行
results = chain.batch([
    {"topic": "程序员"},
    {"topic": "产品经理"},
    {"topic": "设计师"}
])

# 流式执行
for chunk in chain.stream({"topic": "程序员"}):
    print(chunk, end="", flush=True)

# 异步执行
result = await chain.ainvoke({"topic": "程序员"})
```

## 6.2 Runnable 组件

### 6.2.1 核心 Runnable 类型

| Runnable | 作用 | 示例 |
|----------|------|------|
| `RunnableSequence` | 顺序执行 | `a | b | c` |
| `RunnableParallel` | 并行执行 | `{"x": a, "y": b}` |
| `RunnablePassthrough` | 透传/赋值 | `RunnablePassthrough.assign(...)` |
| `RunnableLambda` | 包装函数 | `RunnableLambda(func)` |
| `RunnableBranch` | 条件分支 | `RunnableBranch(...)` |

### 6.2.2 RunnableSequence

```python
from langchain_core.runnables import RunnableSequence

# 显式创建
sequence = RunnableSequence([prompt, llm, parser])

# 或使用管道符（隐式创建）
sequence = prompt | llm | parser

# 执行
result = sequence.invoke({"topic": "程序员"})
```

### 6.2.3 RunnableParallel（并行执行）

```python
from langchain_core.runnables import RunnableParallel

# 并行生成笑话和诗歌
joke_chain = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话") | llm | parser
poem_chain = ChatPromptTemplate.from_template("写一首关于 {topic} 的诗") | llm | parser

# 方式1: 显式创建
parallel = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain
)

# 方式2: 使用字典（推荐）
parallel = {
    "joke": joke_chain,
    "poem": poem_chain
}

# 执行
result = parallel.invoke({"topic": "春天"})
print(result["joke"])   # 笑话
print(result["poem"])   # 诗歌
```

### 6.2.4 RunnablePassthrough

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# 透传原始输入
chain = RunnablePassthrough() | prompt | llm

# 使用 assign 添加新字段
chain = (
    RunnablePassthrough.assign(
        topic_length=lambda x: len(x["topic"])
    )
    | prompt
    | llm
)

# 使用 itemgetter
chain = (
    RunnablePassthrough.assign(
        topic=itemgetter("topic"),
        upper_topic=lambda x: x["topic"].upper()
    )
    | prompt
    | llm
)
```

### 6.2.5 RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

# 包装普通函数
def double(x: int) -> int:
    return x * 2

def add_one(x: int) -> int:
    return x + 1

double_runnable = RunnableLambda(double)
add_one_runnable = RunnableLambda(add_one)

# 构建链
chain = double_runnable | add_one_runnable
result = chain.invoke(5)  # (5 * 2) + 1 = 11

# 直接使用 lambda
chain = RunnableLambda(lambda x: x * 2) | RunnableLambda(lambda x: x + 1)
```

### 6.2.6 RunnableBranch（条件分支）

```python
from langchain_core.runnables import RunnableBranch

# 定义分支
branch = RunnableBranch(
    # (条件, 分支链)
    (lambda x: "python" in x["topic"].lower(), python_chain),
    (lambda x: "javascript" in x["topic"].lower(), js_chain),
    # 默认分支
    general_chain
)

# 使用
result = branch.invoke({"topic": "python 编程"})  # 走 python_chain
result = branch.invoke({"topic": "golang"})       # 走 general_chain
```

## 6.3 复杂链构建

### 6.3.1 RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# RAG Pipeline
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 执行
result = rag_chain.invoke("什么是 LangChain？")
```

### 6.3.2 多步骤处理链

```python
# 步骤1: 生成大纲
outline_prompt = ChatPromptTemplate.from_template(
    "为关于 {topic} 的文章生成大纲"
)
outline_chain = outline_prompt | llm | parser

# 步骤2: 根据大纲写文章
write_prompt = ChatPromptTemplate.from_template(
    """根据以下大纲写文章：
大纲: {outline}

要求: 内容详实，语言流畅"""
)
write_chain = write_prompt | llm | parser

# 步骤3: 润色文章
polish_prompt = ChatPromptTemplate.from_template(
    """润色以下文章，使其更加专业：
{article}"""
)
polish_chain = polish_prompt | llm | parser

# 组合链
article_chain = (
    outline_chain
    | (lambda x: {"outline": x})
    | write_chain
    | (lambda x: {"article": x})
    | polish_chain
)

# 执行
article = article_chain.invoke({"topic": "人工智能"})
```

### 6.3.3 Map-Reduce 链

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Map 阶段：对每个文档进行处理
map_prompt = ChatPromptTemplate.from_template(
    "总结以下文档的主要内容:\n{doc}"
)
map_chain = map_prompt | llm | parser

# Reduce 阶段：合并所有摘要
reduce_prompt = ChatPromptTemplate.from_template(
    """基于以下摘要，生成最终总结：
{summaries}"""
)
reduce_chain = reduce_prompt | llm | parser

# 完整的 Map-Reduce
map_reduce_chain = (
    # Map: 并行处理所有文档
    (lambda docs: [{"doc": d.page_content} for d in docs])
    | RunnableParallel({f"doc_{i}": map_chain for i in range(len(docs))})
    # Reduce: 合并结果
    | (lambda x: {"summaries": "\n".join(x.values())})
    | reduce_chain
)
```

## 6.4 链的组合与复用

### 6.4.1 子链组合

```python
# 基础组件链
translate_chain = (
    ChatPromptTemplate.from_template("将以下文本翻译成英文: {text}")
    | llm
    | parser
)

summarize_chain = (
    ChatPromptTemplate.from_template("总结以下文本: {text}")
    | llm
    | parser
)

# 组合成新链
translate_then_summarize = (
    translate_chain
    | (lambda x: {"text": x})
    | summarize_chain
)

# 并行执行两个任务
translate_and_summarize = RunnableParallel(
    translation=translate_chain,
    summary=(RunnablePassthrough() | (lambda x: {"text": x["text"]}) | summarize_chain)
)
```

### 6.4.2 链的调试

```python
# 打印中间结果
def debug_print(x):
    print(f"[DEBUG] {x}")
    return x

chain = (
    prompt
    | debug_print  # 查看格式化后的提示词
    | llm
    | debug_print  # 查看模型输出
    | parser
)

# 或使用回调
from langchain_core.callbacks import StdOutCallbackHandler

result = chain.invoke(
    {"topic": "程序员"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

## 6.5 可视化链

```python
# 打印链的结构
print(chain.get_graph().draw_ascii())

# 导出为图片
chain.get_graph().draw_png("chain.png")

# 获取 JSON 表示
json_repr = chain.get_graph().to_json()
```

---

## 最佳实践

### ✅ Do

- 使用 LCEL 替代旧的 Chain 类
- 利用 RunnableParallel 并行化独立任务
- 使用 RunnablePassthrough.assign 添加中间状态
- 为复杂的链添加类型注解
- 使用回调函数进行调试和监控

### ❌ Don't

- 不要嵌套过深的链（超过 5 层考虑重构）
- 不要在链中处理过多的数据转换（使用专用函数）
- 不要忽视错误处理（特别是外部调用）
- 不要在生产环境保留调试输出

---

## 参考资源

- [LCEL 文档](https://python.langchain.com/docs/expression_language/)
- [LCEL 速查表](https://python.langchain.com/docs/expression_language/cheatsheet/)
- [Interface 文档](https://python.langchain.com/docs/expression_language/interface/)
- [How-to Guides](https://python.langchain.com/docs/expression_language/how_to/)
