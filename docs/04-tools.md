# 4. Tools（工具）

Tools 是扩展 LLM 能力边界的外部功能接口，让 Agent 能够与现实世界交互。

## 4.1 工具基础

### 4.1.1 工具定义

```python
from langchain.agents import tool

# 方式1: 使用装饰器（推荐）
@tool
def search(query: str) -> str:
    """搜索最新信息。
    
    Args:
        query: 搜索关键词
        
    Returns:
        搜索结果摘要
    """
    # 实现搜索逻辑
    return f"搜索 '{query}' 的结果..."

# 方式2: 使用 Tool 类
from langchain.agents import Tool

calculator_tool = Tool(
    name="Calculator",
    func=lambda x: eval(x),
    description="用于数学计算，输入应为有效的数学表达式"
)
```

### 4.1.2 工具结构

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    limit: int = Field(default=5, description="返回结果数量")

@tool(args_schema=SearchInput)
def advanced_search(query: str, limit: int = 5) -> str:
    """执行高级搜索。
    
    支持多引擎搜索，返回结构化结果。
    """
    return f"搜索 '{query}'，返回 {limit} 条结果"
```

## 4.2 常用内置工具

### 4.2.1 搜索工具

```python
from langchain_community.tools import DuckDuckGoSearchRun, TavilySearchResults

# DuckDuckGo 搜索（免费）
search = DuckDuckGoSearchRun()
result = search.run("LangChain 最新版本")

# Tavily 搜索（推荐，结果质量高）
tavily_search = TavilySearchResults(max_results=5)
results = tavily_search.invoke({"query": "AI 最新发展"})

# Google 搜索
from langchain_community.tools import GoogleSearchResults
google_search = GoogleSearchResults(
    google_api_key="...",
    google_cse_id="..."
)
```

### 4.2.2 计算工具

```python
from langchain.chains import LLMMathChain

# 数学计算链
llm_math = LLMMathChain.from_llm(llm, verbose=True)
result = llm_math.run("123 * 456 + 789")

# 或使用工具
@tool
def calculator(expression: str) -> str:
    """执行数学计算。
    
    Args:
        expression: 数学表达式，如 "123 * 456"
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"
```

### 4.2.3 数据库工具

```python
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase

# 连接数据库
db = SQLDatabase.from_uri("sqlite:///chinook.db")

# SQL 查询工具
sql_tool = QuerySQLDataBaseTool(db=db)

@tool
def query_database(query: str) -> str:
    """执行 SQL 查询。
    
    Args:
        query: SQL 查询语句
    """
    return sql_tool.invoke(query)
```

### 4.2.4 API 工具

```python
import requests

@tool
def weather_api(city: str) -> str:
    """获取城市天气信息。
    
    Args:
        city: 城市名称，如 "北京"
    """
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.weather.com/v1/current?city={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return f"{city} 当前温度: {data['temp']}°C, 天气: {data['condition']}"
```

## 4.3 自定义工具开发

### 4.3.1 文件操作工具

```python
@tool
def read_file(file_path: str) -> str:
    """读取文件内容。
    
    Args:
        file_path: 文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """写入文件内容。
    
    Args:
        file_path: 文件路径
        content: 文件内容
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入 {file_path}"
    except Exception as e:
        return f"写入失败: {str(e)}"

@tool
def list_directory(dir_path: str = ".") -> str:
    """列出目录内容。
    
    Args:
        dir_path: 目录路径，默认为当前目录
    """
    try:
        files = os.listdir(dir_path)
        return "\n".join(files)
    except Exception as e:
        return f"列出目录失败: {str(e)}"
```

### 4.3.2 代码执行工具

```python
@tool
def execute_python(code: str) -> str:
    """执行 Python 代码。
    
    Args:
        code: Python 代码字符串
        
    警告: 在生产环境中使用需要严格的安全控制
    """
    import io
    import sys
    
    # 捕获输出
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        exec(code, {"__builtins__": __builtins__})
        output = buffer.getvalue()
        return output if output else "代码执行成功，无输出"
    except Exception as e:
        return f"执行错误: {str(e)}"
    finally:
        sys.stdout = old_stdout
```

### 4.3.3 浏览器工具

```python
from langchain_community.tools import PlayWrightBrowserTool

# 使用 Playwright 浏览器工具
browser_tool = PlayWrightBrowserTool()

@tool
def web_scrape(url: str) -> str:
    """抓取网页内容。
    
    Args:
        url: 网页 URL
    """
    try:
        response = requests.get(url, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # 提取主要内容
        paragraphs = soup.find_all('p')
        content = '\n'.join([p.get_text() for p in paragraphs[:10]])
        return content[:2000]  # 限制长度
    except Exception as e:
        return f"抓取失败: {str(e)}"
```

## 4.4 工具组合与管理

### 4.4.1 工具列表

```python
# 组合多个工具
tools = [
    search,
    calculator,
    weather_api,
    read_file,
    write_file,
]

# 传递给 Agent
agent = create_react_agent(llm, tools, prompt)
```

### 4.4.2 工具套件（Toolkits）

```python
from langchain_community.agent_toolkits import FileManagementToolkit

# 文件管理工具套件
file_toolkit = FileManagementToolkit(
    root_dir="./data",
    selected_tools=["read", "write", "list_directory"]
)
file_tools = file_toolkit.get_tools()

# 组合到工具列表
all_tools = tools + file_tools
```

### 4.4.3 动态工具选择

```python
from langchain.agents import create_openai_functions_agent

# 根据场景动态选择工具
def get_tools_for_task(task_type: str):
    tool_sets = {
        "research": [search, web_scrape, read_file],
        "coding": [execute_python, read_file, write_file],
        "analysis": [calculator, query_database, search],
        "general": [search, calculator, weather_api]
    }
    return tool_sets.get(task_type, tool_sets["general"])

# 使用
tools = get_tools_for_task("coding")
agent = create_openai_functions_agent(llm, tools, prompt)
```

## 4.5 工具最佳实践

### 4.5.1 工具描述优化

```python
# ❌ 不好的描述
@tool
def search(q: str):
    """搜索"""
    pass

# ✅ 好的描述
@tool  
def search(query: str) -> str:
    """使用搜索引擎获取最新信息。
    
    当你需要了解：
    - 当前事件或新闻
    - 特定主题的最新信息
    - 实时数据（天气、股价等）
    - 你不知道的事实性信息
    
    Args:
        query: 具体的搜索查询语句，应该清晰、具体
        
    Returns:
        搜索结果的摘要，包含关键信息
        
    示例:
        query: "2024年Python最新版本特性"
    """
    pass
```

### 4.5.2 错误处理

```python
@tool
def robust_api_call(endpoint: str, params: dict = None) -> str:
    """调用 API。
    
    Args:
        endpoint: API 端点
        params: 请求参数
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            if attempt == max_retries - 1:
                return "错误: API 请求超时"
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                return f"错误: API 请求失败 - {str(e)}"
            time.sleep(2 ** attempt)  # 指数退避
```

---

## 最佳实践

### ✅ Do

- 为工具编写详细、清晰的描述
- 使用类型注解和 args_schema
- 添加输入验证和错误处理
- 限制工具的访问权限（特别是执行类工具）
- 为工具返回值添加清晰的说明

### ❌ Don't

- 不要创建过于通用的工具
- 不要让工具处理敏感操作而不加验证
- 不要在工具描述中泄露内部实现细节
- 不要创建功能重叠的工具

---

## 参考资源

- [LangChain Tools 文档](https://python.langchain.com/docs/modules/agents/tools/)
- [Custom Tools Guide](https://python.langchain.com/docs/modules/agents/tools/custom_tools/)
- [Toolkits 文档](https://python.langchain.com/docs/modules/agents/toolkits/)
