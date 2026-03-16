"""
RAG 知识库问答系统

一个完整的 RAG 应用，支持文档上传、向量化存储和智能问答。
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader
)


@dataclass
class RAGConfig:
    """RAG 配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_k: int = 4
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    persist_directory: str = "./chroma_db"


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", " ", ""]
        )
    
    def load_documents(self, path: str) -> List[Document]:
        """加载文档"""
        if os.path.isdir(path):
            loader = DirectoryLoader(
                path,
                glob="**/*.{txt,pdf,md}",
                loader_cls=TextLoader
            )
        elif path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding='utf-8')
        
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self.text_splitter.split_documents(documents)


class RAGSystem:
    """RAG 系统"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.processor = DocumentProcessor(self.config)
        
        # 初始化组件
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model
        )
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=0.7
        )
        
        # 向量存储
        self.vectorstore: Optional[Chroma] = None
        
        # 提示词模板
        self.prompt = ChatPromptTemplate.from_template("""基于以下上下文回答问题。
如果上下文没有相关信息，请说"根据现有资料无法回答"。

上下文：
{context}

问题：{question}

请提供详细的回答，并注明信息来源。""")
    
    def ingest_documents(self, path: str) -> None:
        """摄入文档"""
        print(f"📚 加载文档: {path}")
        documents = self.processor.load_documents(path)
        print(f"   加载了 {len(documents)} 个文档")
        
        print("🔪 分割文档...")
        chunks = self.processor.split_documents(documents)
        print(f"   分割为 {len(chunks)} 个块")
        
        print("💾 存储到向量数据库...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config.persist_directory
        )
        print("   ✓ 完成")
    
    def load_existing(self) -> bool:
        """加载已有的向量存储"""
        if os.path.exists(self.config.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.config.persist_directory,
                embedding_function=self.embeddings
            )
            return True
        return False
    
    def query(self, question: str) -> dict:
        """查询"""
        if not self.vectorstore:
            raise ValueError("请先摄入文档或加载已有的向量存储")
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.search_k}
        )
        
        # 格式化文档
        def format_docs(docs):
            return "\n\n".join(
                f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
                for doc in docs
            )
        
        # 构建 RAG 链
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 获取检索到的文档
        retrieved_docs = retriever.invoke(question)
        
        # 执行链
        answer = rag_chain.invoke(question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata.get('source', '未知') for doc in retrieved_docs]
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 知识库问答系统")
    parser.add_argument("--ingest", help="摄入文档路径")
    parser.add_argument("--query", help="查询问题")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    # 初始化 RAG 系统
    config = RAGConfig()
    rag = RAGSystem(config)
    
    if args.ingest:
        rag.ingest_documents(args.ingest)
    
    elif args.query:
        if rag.load_existing():
            result = rag.query(args.query)
            print(f"\n❓ 问题: {result['question']}")
            print(f"\n💡 回答:\n{result['answer']}")
            print(f"\n📚 来源: {', '.join(set(result['sources']))}")
        else:
            print("❌ 没有找到向量存储，请先使用 --ingest 摄入文档")
    
    elif args.interactive:
        if rag.load_existing():
            print("=" * 60)
            print("RAG 问答系统 - 交互模式")
            print("输入 'quit' 或 'exit' 退出")
            print("=" * 60)
            
            while True:
                question = input("\n❓ 问题: ").strip()
                if question.lower() in ['quit', 'exit', '退出']:
                    break
                if not question:
                    continue
                
                try:
                    result = rag.query(question)
                    print(f"\n💡 回答:\n{result['answer']}")
                except Exception as e:
                    print(f"❌ 错误: {e}")
        else:
            print("❌ 没有找到向量存储，请先使用 --ingest 摄入文档")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
