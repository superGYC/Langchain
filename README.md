# LangChain 完全使用指导手册

## 项目简介

本项目是一个全面的 LangChain 学习资源库，包含六大核心组件详解、完整 Demo 示例和工程级项目代码。

## 项目结构

```
.
├── docs/                          # 核心组件文档
│   ├── 01-model-io.md            # Model I/O 组件
│   ├── 02-retrieval.md           # Retrieval 组件
│   ├── 03-agents.md              # Agents 组件
│   ├── 04-tools.md               # Tools 组件
│   ├── 05-memory.md              # Memory 组件
│   └── 06-chains.md              # Chains 组件
│
├── examples/                      # 示例代码
│   ├── 01_quickstart/            # 快速入门
│   ├── 02_rag/                   # RAG 示例
│   ├── 03_agents/                # Agent 示例
│   ├── 04_memory/                # Memory 示例
│   └── 05_advanced/              # 高级用法
│
├── projects/                      # 完整工程
│   ├── rag-chatbot/              # RAG 知识库问答
│   ├── ai-assistant/             # AI Agent 助手
│   └── multi-agent/              # 多 Agent 协作
│
├── README.md                      # 项目说明
├── requirements.txt               # 依赖列表
└── .env.example                   # 环境变量模板
```

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Keys
```

3. 运行示例
```bash
python examples/01_quickstart/hello_langchain.py
```

## 学习路径

1. **第1周**：阅读 docs/ 目录下的组件文档
2. **第2周**：运行和学习 examples/ 目录下的示例
3. **第3周**：研究 projects/ 目录下的完整工程

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain Hub](https://smith.langchain.com/hub)
