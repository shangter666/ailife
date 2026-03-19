import json
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import custom configurations and memory setup
from config_loader import config
from memory_manager import UserMemory

# 1. State 定义
class AgentState(TypedDict):
    """
    包含了消息历史，以及用户当前的记忆快照。
    使用 add_messages 允许新消息自动追加至原有会话列表末尾。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    memory_snapshot: UserMemory

# 实例化基于配置文件的 LLM 模型
llm = ChatOpenAI(
    api_key=config.llm_settings.api_key,
    base_url=config.llm_settings.base_url,
    model=config.llm_settings.model_name,
    temperature=config.llm_settings.temperature,
)

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

# 定义可用工具列表
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """Get the current time and date in the specific timezone."""
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

tools = [get_current_time]
# 绑定工具至核心对话模型
llm_with_tools = llm.bind_tools(tools)

# 2. Node 1: chat_node
def chat_node(state: AgentState):
    """
    调用绑定了 tools 的 LLM 生成回复。
    """
    messages = state["messages"]
    memory = state["memory_snapshot"]
    
    # 注入 memory_snapshot 保持人设
    memory_prompt = f"""
你是一个数字化分身 AI Agent。请根据以下记忆信息来保持你的设定并回答用户。
[基本信息]: {memory.basic_info}
[性格特征]: {memory.personality_traits}
[重要事件]: {memory.significant_events}
[说话风格]: {memory.speaking_style}

请充分运用以上记忆和用户交流。如果用户询问时间，请调用时间工具。
"""
    system_msg = SystemMessage(content=memory_prompt)
    
    # 组合 SystemMessage 和对话历史发送给挂载了工具的大模型
    response = llm_with_tools.invoke([system_msg] + messages)
    
    return {"messages": [response]}

# 3. Node 2: reflect_node
def reflect_node(state: AgentState):
    """
    对话结束后获取新特征或事实。使用系统级提示词进行全息画像提取和更新。
    """
    messages = state["messages"]
    memory = state["memory_snapshot"]
    
    # 将会话历史拼接成文本
    conversation_history = "\n".join(
        [f"{m.type}: {m.content}" for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    )
    
    # 获取现有全息画像（排除自动更新的时间戳）
    current_profile = json.dumps(memory.model_dump(exclude={"last_updated"}), ensure_ascii=False)
    
    # 动态读取外部提取提示词文件，实现代码与提示词分离
    import os
    soul_prompt_path = os.path.join(os.path.dirname(__file__), "soul.md")
    with open(soul_prompt_path, "r", encoding="utf-8") as f:
        soul_template = f.read()
        
    # 基于常规占位符进行替换，避免硬编码 f-string 破坏含有真实 JSON 的模板
    reflect_prompt = soul_template.replace("{current_profile}", current_profile).replace("{conversation_history}", conversation_history)
    
    human_msg = HumanMessage(content=reflect_prompt)
    
    try:
        # 使用 LangChain 的原生机制，利用 Pydantic 约束大模型输出
        structured_llm = llm.with_structured_output(UserMemory, method="json_mode")
        updated_memory = structured_llm.invoke([human_msg])
        
        print("\n[Reflect Node] Updated Profile (Structured Output):")
        print(updated_memory.model_dump())
        print("-" * 40)
        
        # 结构化输出会直接返回校验好的 UserMemory 对象实例，我们直接将其应用替换状态
        return {"memory_snapshot": updated_memory}
    except Exception as e:
        print(f"Error generating or parsing Structured Output from Reflect Node: {e}")
        return {}

# 构建附加工具节点的 Graph
tool_node = ToolNode(tools)

# 4. Node 3: compress_memory_node
def compress_memory_node(state: AgentState):
    """
    当对话记录过长时，将历史记录浓缩为摘要以节省上下文 Token。
    """
    messages = state["messages"]
    
    # 留下最后两次交互（通常是人与 AI 的一问一答），压缩之前的记录
    messages_to_compress = messages[:-2]
    
    if not messages_to_compress:
        return {}
        
    conversation_history = "\n".join(
        [f"{m.type}: {m.content}" for m in messages_to_compress if isinstance(m, (HumanMessage, AIMessage))]
    )
    
    summary_prompt = f"请高度概括并压缩以下对话历史，提取核心意图，作为后续对话的上下文简述：\n{conversation_history}"
    human_msg = HumanMessage(content=summary_prompt)
    summary_response = llm.invoke([human_msg])
    
    try:
        from langchain_core.messages import RemoveMessage
        # 利用 RemoveMessage 来精确删除长序列历史
        delete_actions = [RemoveMessage(id=m.id) for m in messages_to_compress if m.id]
        
        # 将被删除的数据固化为一个系统层级的摘要消息
        summary_msg = SystemMessage(content=f"[系统自动折叠的历史摘要]:\n{summary_response.content}")
        
        print(f"\n[Compress Node] Compressed {len(messages_to_compress)} messages into summary.")
        return {"messages": delete_actions + [summary_msg]}
    except Exception as e:
        print(f"Error compressing memory: {e}")
        return {}
        
def should_compress(state: AgentState):
    messages = state["messages"]
    # 阈值配置：当历史堆叠超过 6 条消息后开始触发滑动窗口压缩
    if len(messages) > 6:
        return "compress_memory_node"
    return "chat_node"

builder = StateGraph(AgentState)

builder.add_node("compress_memory_node", compress_memory_node)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)
builder.add_node("reflect_node", reflect_node)

# 定义 Edge逻辑：START -> (条件判断是否压缩) -> chat_node <-> tools
# 结束聊天循环后走向 -> reflect_node -> END
builder.add_conditional_edges(START, should_compress, path_map={"compress_memory_node": "compress_memory_node", "chat_node": "chat_node"})
builder.add_edge("compress_memory_node", "chat_node")
builder.add_conditional_edges("chat_node", tools_condition, path_map={"tools": "tools", "__end__": "reflect_node"})
builder.add_edge("tools", "chat_node")
builder.add_edge("reflect_node", END)

# 6. 导出 app 实例
app = builder.compile()

if __name__ == "__main__":
    from memory_manager import MemoryManager
    
    # 快速测试下工作流（注意：请确保 Qwen 模型配置的 api_key 等信息真实有效）
    manager = MemoryManager(config.storage_settings.memory_path)
    memory = manager.load_memory()
    
    initial_state = {
        "messages": [HumanMessage(content="你好！昨天我开始学习弹吉他了，手指尖磨得好疼。")],
        "memory_snapshot": memory
    }
    
    print(">>> Triggering App >>>\n")
    try:
        for event in app.stream(initial_state):
            for key, value in event.items():
                print(f"--- Completed Node: {key} ---")
                if key == "chat_node" and "messages" in value:
                    for msg in value["messages"]:
                        print(f"Reply: {msg.content}")
                        print()
    except Exception as e:
        print(f"Workflow execution failed: {e}")
