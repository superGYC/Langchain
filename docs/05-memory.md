# 5. Memory（记忆）

Memory 组件让 LLM 应用具备"记忆能力"，能够在多轮对话中保持上下文连贯性。

## 5.1 记忆类型

### 5.1.1 记忆类型对比

| 记忆类型 | 特点 | 适用场景 | 存储内容 |
|----------|------|----------|----------|
| **Buffer** | 保存完整对话 | 短对话 | 所有消息 |
| **Buffer Window** | 保存最近 N 轮 | 中等长度对话 | 最近 K 轮 |
| **Summary** | 摘要历史 | 长对话 | 摘要文本 |
| **Entity** | 提取实体 | 需要记忆关键信息 | 实体信息 |
| **Vector Store** | 语义检索 | 超长相册 | 向量化历史 |

## 5.2 基础记忆类

### 5.2.1 ConversationBufferMemory

保存完整对话历史。

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 在提示词中的变量名
    return_messages=True         # 返回消息对象列表而非字符串
)

# 保存上下文
memory.save_context(
    {"input": "你好，我叫小明"},
    {"output": "你好小明！很高兴认识你"}
)

# 加载记忆
memory_variables = memory.load_memory_variables({})
print(memory_variables)
# {'chat_history': [HumanMessage(...), AIMessage(...)]}
```

### 5.2.2 ConversationBufferWindowMemory

只保存最近 N 轮对话，控制 token 使用。

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 3 轮对话
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,  # 保留轮数
    return_messages=True
)

# 模拟多轮对话
for i in range(5):
    memory.save_context(
        {"input": f"问题 {i}"},
        {"output": f"回答 {i}"}
    )

# 只保留最近 3 轮
print(memory.load_memory_variables({}))
```

### 5.2.3 ConversationSummaryMemory

对历史对话进行摘要，适合长对话。

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# 需要 LLM 来生成摘要
llm = ChatOpenAI(model="gpt-4o-mini")

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# 保存长对话
memory.save_context(
    {"input": "我想了解 Python 编程"},
    {"output": "Python 是一种高级编程语言..."}
)
memory.save_context(
    {"input": "它有什么特点？"},
    {"output": "Python 的特点包括简洁的语法..."}
)

# 查看摘要
print(memory.buffer)
```

### 5.2.4 ConversationEntityMemory

提取和记忆对话中的实体信息。

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(
    llm=llm,
    memory_key="chat_history"
)

# 保存包含实体的对话
memory.save_context(
    {"input": "我的名字叫张三，我在百度工作"},
    {"output": "你好张三，很高兴认识你"}
)

# 会提取实体: 张三(人名), 百度(公司)
print(memory.load_memory_variables({"input": "我的公司怎么样？"}))
```

## 5.3 在 Chain 中使用记忆

### 5.3.1 基础用法

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# 创建记忆和模型
memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-4o-mini")

# 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 多轮对话
conversation.predict(input="你好，我叫小明")
conversation.predict(input="我叫什么名字？")  # 能记住上下文
conversation.predict(input="1+1等于几？")
conversation.predict(input="再加上10呢？")  # 能记住上一轮计算
```

### 5.3.2 在 LCEL 中使用记忆

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from operator import itemgetter

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

# 创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 helpful 助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 构建链
chain = (
    RunnablePassthrough.assign(
        history=lambda x: memory.load_memory_variables(x)["history"]
    )
    | prompt
    | llm
)

# 使用
inputs = {"input": "你好，我叫小明"}
response = chain.invoke(inputs)
memory.save_context(inputs, {"output": response.content})

# 第二轮
inputs = {"input": "我叫什么名字？"}
response = chain.invoke(inputs)  # 能记住名字
```

## 5.4 高级记忆模式

### 5.4.1 向量存储记忆

基于语义的长期记忆，可以检索相关的历史对话。

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))

# 创建向量记忆
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="history"
)

# 保存记忆（自动向量化）
memory.save_context(
    {"input": "我喜欢 Python 编程"},
    {"output": "Python 是非常流行的编程语言"}
)

# 检索相关记忆
relevant_history = memory.load_memory_variables({"input": "推荐一门编程语言"})
```

### 5.4.2 组合记忆

结合多种记忆类型。

```python
from langchain.memory import CombinedMemory

# 创建多种记忆
buffer_memory = ConversationBufferWindowMemory(
    memory_key="recent_history",
    k=3,
    return_messages=True
)

summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="summary",
    input_key="input"
)

# 组合记忆
combined_memory = CombinedMemory(
    memories=[buffer_memory, summary_memory]
)

# 在提示词中同时使用
prompt = ChatPromptTemplate.from_messages([
    ("system", "对话摘要: {summary}"),
    MessagesPlaceholder(variable_name="recent_history"),
    ("human", "{input}")
])
```

### 5.4.3 自定义记忆类

```python
from langchain.memory import BaseMemory
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class CustomMemory(BaseMemory, BaseModel):
    """自定义记忆类示例"""
    
    # 存储
    conversations: List[Dict] = Field(default_factory=list)
    max_items: int = 10
    
    # 必需的类属性
    memory_key: str = "custom_history"
    
    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量"""
        return {self.memory_key: self.conversations}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存上下文"""
        self.conversations.append({
            "input": inputs.get("input"),
            "output": outputs.get("output")
        })
        # 保持最大数量
        if len(self.conversations) > self.max_items:
            self.conversations.pop(0)
    
    def clear(self) -> None:
        """清空记忆"""
        self.conversations = []
```

## 5.5 记忆持久化

### 5.5.1 保存到文件

```python
import json

class PersistentMemory(ConversationBufferMemory):
    """持久化记忆"""
    
    def __init__(self, file_path: str = "memory.json", **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
        self._load()
    
    def _load(self):
        """从文件加载"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chat_memory.messages = [
                    HumanMessage(content=m["content"]) if m["type"] == "human"
                    else AIMessage(content=m["content"])
                    for m in data.get("messages", [])
                ]
        except FileNotFoundError:
            pass
    
    def _save(self):
        """保存到文件"""
        data = {
            "messages": [
                {"type": "human" if isinstance(m, HumanMessage) else "ai", 
                 "content": m.content}
                for m in self.chat_memory.messages
            ]
        }
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        self._save()
```

### 5.5.2 Redis 记忆

```python
from langchain.memory import RedisChatMessageHistory

# 使用 Redis 存储对话历史
message_history = RedisChatMessageHistory(
    session_id="user_123",
    url="redis://localhost:6379/0"
)

memory = ConversationBufferMemory(
    chat_memory=message_history,
    memory_key="chat_history",
    return_messages=True
)
```

---

## 最佳实践

### ✅ Do

- 根据对话长度选择合适的记忆类型
- 对于长对话使用 SummaryMemory
- 控制记忆窗口大小以节省 token
- 对敏感对话考虑持久化加密
- 定期清理或归档旧的历史记录

### ❌ Don't

- 不要在记忆窗口中保留过多轮数
- 不要存储敏感信息（如密码）在记忆中
- 不要让单轮对话内容过长
- 不要忽视记忆带来的 token 消耗

---

## 参考资源

- [LangChain Memory 文档](https://python.langchain.com/docs/modules/memory/)
- [Memory Types](https://python.langchain.com/docs/modules/memory/types/)
- [Chat Message History](https://python.langchain.com/docs/modules/memory/chat_messages/)
