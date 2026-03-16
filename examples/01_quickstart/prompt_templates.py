"""
提示词模板示例
演示各种 PromptTemplate 的用法
"""

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage


def basic_template_demo():
    """基础模板示例"""
    print("\n" + "=" * 50)
    print("基础 PromptTemplate")
    print("=" * 50)
    
    # 创建模板
    template = "将以下文本翻译成{language}: {text}"
    prompt = PromptTemplate.from_template(template)
    
    # 格式化
    formatted = prompt.format(language="英文", text="你好，世界")
    print(f"\n格式化后的提示词:\n{formatted}")


def chat_template_demo():
    """聊天模板示例"""
    print("\n" + "=" * 50)
    print("ChatPromptTemplate")
    print("=" * 50)
    
    # 创建多消息模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{skill}"),
        ("human", "{question}")
    ])
    
    # 格式化
    messages = prompt.format_messages(
        role="Python 专家",
        skill="数据分析",
        question="如何用 pandas 读取 CSV？"
    )
    
    print("\n格式化后的消息:")
    for msg in messages:
        print(f"  [{msg.type}]: {msg.content}")


def few_shot_demo():
    """少样本提示示例"""
    print("\n" + "=" * 50)
    print("Few-Shot Prompting")
    print("=" * 50)
    
    # 定义示例
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "5*6", "output": "30"},
        {"input": "10-3", "output": "7"}
    ]
    
    # 示例模板
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    
    # 少样本模板
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    # 完整模板
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个计算器"),
        few_shot_prompt,
        ("human", "{input}")
    ])
    
    messages = final_prompt.format_messages(input="15/3")
    print("\n少样本提示结果:")
    for msg in messages:
        print(f"  [{msg.type}]: {msg.content}")


def placeholder_demo():
    """消息占位符示例"""
    print("\n" + "=" * 50)
    print("MessagesPlaceholder")
    print("=" * 50)
    
    # 创建带占位符的模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 helpful 助手"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 历史消息
    history = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮你的？"),
        HumanMessage(content="我想学 Python")
    ]
    
    messages = prompt.format_messages(history=history, input="从哪里开始？")
    print("\n带历史消息的提示词:")
    for msg in messages:
        print(f"  [{msg.type}]: {msg.content[:50]}...")


def main():
    """主函数"""
    print("=" * 50)
    print("提示词模板示例")
    print("=" * 50)
    
    basic_template_demo()
    chat_template_demo()
    few_shot_demo()
    placeholder_demo()
    
    print("\n✅ 所有示例完成！")


if __name__ == "__main__":
    main()
