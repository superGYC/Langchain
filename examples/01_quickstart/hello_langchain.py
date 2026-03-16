"""
LangChain 快速入门示例
第一个 LangChain 程序
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()


def main():
    """主函数"""
    print("=" * 50)
    print("LangChain 快速入门")
    print("=" * 50)
    
    # 初始化模型
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 方式1: 使用消息列表
    messages = [
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="你好！请介绍一下自己")
    ]
    
    print("\n📝 发送消息...")
    response = llm.invoke(messages)
    print(f"\n🤖 AI回复: {response.content}")
    
    # 方式2: 使用字符串
    print("\n" + "=" * 50)
    print("使用字符串调用")
    print("=" * 50)
    
    response = llm.invoke("用一句话解释什么是 LangChain")
    print(f"\n🤖 AI回复: {response.content}")
    
    # 流式输出
    print("\n" + "=" * 50)
    print("流式输出演示")
    print("=" * 50)
    
    print("\n🤖 AI回复: ", end="", flush=True)
    for chunk in llm.stream("讲一个简短的笑话"):
        print(chunk.content, end="", flush=True)
    print()
    
    print("\n✅ 示例完成！")


if __name__ == "__main__":
    main()
