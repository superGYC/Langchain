# 2. Retrieval（检索）

Retrieval 组件让 LLM 能够访问和利用外部知识体系，是构建 RAG（Retrieval-Augmented Generation，检索增强生成）系统的核心。

## 2.1 RAG 核心流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  文档加载   │ -> │  文本分割   │ -> │  向量化     │ -> │  向量存储   │
│  Loaders    │    │  Splitters  │    │  Embeddings │    │  VectorStore│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  生成回答   │ <- │  LLM        │ <- │  上下文组装 │ <- │  相似度检索 │
│  Generate   │    │             │    │  Context    │    │  Retrieval  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 2.2 文档加载器（Document Loaders）

### 2.2.1 常用加载器

```python
from langchain_community.document_loaders import (
    TextLoader,           # 文本文件
    PyPDFLoader,          # PDF
    UnstructuredWordDocumentLoader,  # Word
    CSVLoader,            # CSV
    WebBaseLoader,        # 网页
    DirectoryLoader       # 整个目录
)

# 加载文本文件
text_loader = TextLoader("data/article.txt", encoding="utf-8")
docs = text_loader.load()

# 加载 PDF
pdf_loader = PyPDFLoader("data/document.pdf")
pages = pdf_loader.load_and_split()  # 自动按页分割

# 加载网页
web_loader = WebBaseLoader([
    "https://python.langchain.com/docs/get_started/introduction"
])
web_docs = web_loader.load()

# 加载整个目录
dir_loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",  # 匹配模式
    loader_cls=TextLoader
)
dir_docs = dir_loader.load()
```

### 2.2.2 自定义加载器

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator

class CustomLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            yield Document(
                page_content=content,
                metadata={"source": self.file_path, "type": "custom"}
            )
```

## 2.3 文本分割器（Text Splitters）

### 2.3.1 常用分割器

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # 递归字符分割（推荐）
    CharacterTextSplitter,            # 字符分割
    TokenTextSplitter,                # Token 分割
    MarkdownHeaderTextSplitter,       # Markdown 标题分割
    HTMLHeaderTextSplitter,           # HTML 标题分割
    SemanticChunker                   # 语义分割
)

# 递归字符分割器（最常用）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 每个块的最大字符数
    chunk_overlap=200,      # 块之间的重叠字符数
    length_function=len,    # 长度计算函数
    separators=["\n\n", "\n", "。", " ", ""]  # 分割符优先级
)
chunks = text_splitter.split_documents(docs)

# 按 Token 分割
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    encoding_name="cl100k_base"  # OpenAI 的 token 编码
)
```

### 2.3.2 分割策略选择

| 场景 | 推荐分割器 | 说明 |
|------|-----------|------|
| 通用文本 | RecursiveCharacterTextSplitter | 保持语义完整 |
| 代码 | RecursiveCharacterTextSplitter | 配合代码语言参数 |
| Markdown | MarkdownHeaderTextSplitter | 按标题层级分割 |
| HTML | HTMLHeaderTextSplitter | 按标签层级分割 |
| 需要语义 | SemanticChunker | 基于语义相似度 |

### 2.3.3 代码分割示例

```python
from langchain_text_splitters import Language

# Python 代码分割
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

# 支持的语言：PYTHON, JS, TS, JAVA, CPP, GO, RUBY 等
```

## 2.4 嵌入模型（Embeddings）

### 2.4.1 常用嵌入模型

```python
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    DashScopeEmbeddings
)

# OpenAI
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 或 text-embedding-3-large
    dimensions=1536  # 可选：降维
)

# Ollama 本地模型
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# HuggingFace
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"  # 中文推荐
)

# 通义千问
dashscope_embeddings = DashScopeEmbeddings(
    model="text-embedding-v1"
)
```

### 2.4.2 嵌入模型对比

| 模型 | 维度 | 语言 | 特点 |
|------|------|------|------|
| text-embedding-3-small | 1536 | 多语言 | 速度快，成本低 |
| text-embedding-3-large | 3072 | 多语言 | 精度高 |
| bge-large-zh | 1024 | 中文 | 中文表现优秀 |
| nomic-embed-text | 768 | 多语言 | 开源，可本地部署 |
| m3e-base | 768 | 中文 | 中文开源推荐 |

## 2.5 向量存储（Vector Stores）

### 2.5.1 常用向量数据库

```python
from langchain_community.vectorstores import (
    Chroma,        # 本地，轻量
    FAISS,         # Facebook，高效
    Pinecone,      # 云端，企业级
    Qdrant,        # 开源，功能丰富
    Milvus,        # 企业级，分布式
    Weaviate       # 云原生
)

# Chroma（本地开发推荐）
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化路径
)

# FAISS（内存型，速度快）
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")  # 保存
vectorstore = FAISS.load_local("faiss_index", embeddings)  # 加载
```

### 2.5.2 向量数据库对比

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| Chroma | 本地/嵌入式 | 易用，无需部署 | 开发、小型项目 |
| FAISS | 内存 | 极快，纯检索 | 内存充足，纯搜索 |
| Qdrant | 本地/云端 | 功能全，支持过滤 | 中小型生产 |
| Pinecone | 云服务 | 托管，无需运维 | 生产环境 |
| Milvus | 分布式 | 企业级，高并发 | 大规模应用 |

## 2.6 检索器（Retrievers）

### 2.6.1 基础检索

```python
# 从向量存储创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",      # 相似度检索
    search_kwargs={"k": 5}         # 返回前5个结果
)

# 执行检索
docs = retriever.invoke("什么是 LangChain？")
for doc in docs:
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content[:200]}...")
    print()
```

### 2.6.2 检索类型

```python
# 1. 相似度检索（默认）
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 2. MMR (最大边际相关性) - 平衡相关性和多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "lambda_mult": 0.5}
)

# 3. 相似度阈值 - 过滤低相似度结果
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
```

### 2.6.3 高级检索器

```python
from langchain.retrievers import (
    ContextualCompressionRetriever,  # 上下文压缩
    MultiQueryRetriever,            # 多查询检索
    BM25Retriever,                  # 关键词检索
    EnsembleRetriever               # 混合检索
)

# 多查询检索 - 生成多个查询变体
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# 上下文压缩 - 只保留相关部分
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 混合检索 - 结合向量检索和关键词检索
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

## 2.7 完整 RAG 实现

### 2.7.1 基础 RAG Chain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 组件初始化
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG Prompt
template = """基于以下上下文回答问题。如果上下文没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

# 格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建 RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("什么是 RAG？")
print(answer)
```

### 2.7.2 带源引用的 RAG

```python
from langchain_core.runnables import RunnableParallel

# 返回检索到的文档和答案
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(
    answer=(
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
)

result = rag_chain_with_source.invoke("什么是 RAG？")
print(f"答案: {result['answer']}")
print(f"来源: {[doc.metadata['source'] for doc in result['context']]}")
```

## 2.8 高级 RAG 技术

### 2.8.1 查询重写

```python
# 使用 LLM 优化用户查询
rewrite_template = """将以下问题改写为更适合向量检索的形式。
保持原意，但使其更具体、包含更多关键词。

原问题: {question}
改写后的问题:"""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# 在 RAG 中使用
rag_chain = (
    {
        "context": (lambda x: rewrite_chain.invoke({"question": x})) | retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

### 2.8.2 重排序（Re-ranking）

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 加载重排序模型
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

reranker = CrossEncoderReranker(model=model, top_n=3)

# 创建带重排序的检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)
```

---

## 最佳实践

### ✅ Do

- 根据文档类型选择合适的分割器
- 保持适当的 chunk_overlap（通常 10-20%）
- 使用 MMR 增加检索结果的多样性
- 添加元数据以便追踪来源
- 对检索结果进行重排序提升质量

### ❌ Don't

- chunk_size 不要过小（会丢失上下文）
- chunk_size 不要过大（会降低检索精度）
- 不要忽略元数据的存储
- 不要在没有评估的情况下直接上线

---

## 参考资源

- [LangChain RAG 文档](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- [Embedding Models Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
