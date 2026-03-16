"""
RAG 基础示例
演示基本的检索增强生成流程
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()


def create_sample_documents():
    """创建示例文档"""
    texts = [
        """LangChain 是一个用于开发大语言模型应用程序的框架。
        它提供了模块化组件，帮助开发者轻松构建复杂的 LLM 应用。
        LangChain 的核心组件包括 Model I/O、Retrieval、Agents、Chains、Memory 和 Tools。""",
        
        """RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
        它首先从知识库中检索相关信息，然后让 LLM 基于这些信息生成回答。
        RAG 可以有效解决 LLM 的知识局限和幻觉问题。""",
        
        """向量数据库是 RAG 系统的核心组件之一。
        它将文本转换为向量（嵌入）并存储，支持基于语义的相似度检索。
        常用的向量数据库包括 Chroma、FAISS、Pinecone 等。"""
    ]
    
    return [Document(page_content=text, metadata={"source": f"doc_{i}"}) 
            for i, text in enumerate(texts)]


def setup_vectorstore(documents):
    """设置向量存储"""
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"📄 文档分割为 {len(chunks)} 个块")
    
    # 创建嵌入和向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore


def create_rag_chain(vectorstore):
    """创建 RAG 链"""
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    # 创建提示词模板
    template = """基于以下上下文回答问题。如果上下文没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # 格式化文档函数
    def format_docs(docs):
        return "\n\n".join(f"[来源: {doc.metadata['source']}]\n{doc.page_content}" 
                         for doc in docs)
    
    # 构建 RAG 链
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def main():
    """主函数"""
    print("=" * 60)
    print("RAG 基础示例")
    print("=" * 60)
    
    # 1. 准备文档
    print("\n📚 准备文档...")
    documents = create_sample_documents()
    print(f"   创建了 {len(documents)} 个文档")
    
    # 2. 设置向量存储
    print("\n🔧 设置向量存储...")
    vectorstore = setup_vectorstore(documents)
    print("   ✓ 向量存储创建完成")
    
    # 3. 创建 RAG 链
    print("\n🔗 创建 RAG 链...")
    rag_chain = create_rag_chain(vectorstore)
    print("   ✓ RAG 链创建完成")
    
    # 4. 测试问答
    print("\n" + "=" * 60)
    print("开始问答")
    print("=" * 60)
    
    questions = [
        "什么是 LangChain？",
        "RAG 是什么技术？",
        "向量数据库有什么作用？",
        "谁发明了电灯泡？"  # 知识库外的问题
    ]
    
    for question in questions:
        print(f"\n❓ 问题: {question}")
        answer = rag_chain.invoke(question)
        print(f"💡 回答: {answer}")
    
    print("\n" + "=" * 60)
    print("✅ RAG 示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
