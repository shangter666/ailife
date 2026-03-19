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

# 2. Node 1: chat_node
def chat_node(state: AgentState):
    """
    调用 LLM 生成回复。
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

请充分运用以上记忆和用户交流。
"""
    system_msg = SystemMessage(content=memory_prompt)
    
    # 组合 SystemMessage 和对话历史发送给大模型
    response = llm.invoke([system_msg] + messages)
    
    # 返回新的 LLM 消息回复
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
    response = llm.invoke([human_msg])
    
    print("\n[Reflect Node] Updated Profile JSON:")
    print(response.content)
    print("-" * 40)
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        json_data = json.loads(content)
        
        # 将 LLM 输出的完整新画像直接覆盖替换当前内存的各项参数
        if "basic_info" in json_data and isinstance(json_data["basic_info"], dict):
            memory.basic_info = json_data["basic_info"]
            
        if "personality_traits" in json_data and isinstance(json_data["personality_traits"], list):
            memory.personality_traits = json_data["personality_traits"]
            
        if "significant_events" in json_data and isinstance(json_data["significant_events"], list):
            memory.significant_events = json_data["significant_events"]
            
        if "speaking_style" in json_data and isinstance(json_data["speaking_style"], list):
            memory.speaking_style = json_data["speaking_style"]
            
        return {"memory_snapshot": memory}
    except Exception as e:
        print(f"Error parsing JSON from Reflect Node: {e}")
        return {}

# 4. 构建 Graph
builder = StateGraph(AgentState)

builder.add_node("chat_node", chat_node)
builder.add_node("reflect_node", reflect_node)

# 5. 定义 Edge: START -> chat_node -> reflect_node -> END
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", "reflect_node")
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
