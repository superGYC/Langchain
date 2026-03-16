"""
Agent 基础示例
演示如何创建和使用智能体
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


# 定义工具
@tool
def calculator(expression: str) -> str:
    """执行数学计算。
    
    Args:
        expression: 数学表达式，如 "123 * 456" 或 "2 ** 10"
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


def create_agent():
    """创建 Agent"""
    # 初始化工具
    search = DuckDuckGoSearchRun()
    tools = [search, calculator, get_current_time]
    
    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个 helpful 助手，可以使用以下工具帮助用户：

- search: 搜索最新信息
- calculator: 进行数学计算  
- get_current_time: 获取当前时间

请遵循以下原则：
1. 对于简单问题直接回答
2. 需要最新信息时使用搜索
3. 数学计算使用计算器
4. 回答要简洁明了"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


def main():
    """主函数"""
    print("=" * 60)
    print("Agent 基础示例")
    print("=" * 60)
    
    # 创建 Agent
    print("\n🔧 初始化 Agent...")
    agent = create_agent()
    print("✅ Agent 创建完成\n")
    
    # 测试用例
    test_cases = [
        "你好，请介绍一下你自己",
        "1234 乘以 5678 等于多少？",
        "现在几点了？",
        "2024年AI领域有什么重大进展？",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {query}")
        print('='*60)
        
        try:
            result = agent.invoke({"input": query})
            print(f"\n📝 最终回答: {result['output']}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Agent 示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
