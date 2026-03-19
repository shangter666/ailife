import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from config_loader import config
from memory_manager import MemoryManager
from agent_workflow import app as agent_app
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

def get_user_memory_manager(user_id: str) -> MemoryManager:
    """
    Generates a MemoryManager for a unique user_id.
    """
    storage_path = config.storage_settings.memory_path
    if storage_path.endswith(".json"):
        dir_name = os.path.dirname(storage_path)
        path = os.path.join(dir_name, f"{user_id}.json") if dir_name else f"{user_id}.json"
    else:
        path = os.path.join(storage_path, f"{user_id}.json")
    return MemoryManager(path)

@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    manager = get_user_memory_manager(request.user_id)
    # 根据 user_id 加载记忆
    memory = manager.load_memory()
    
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "memory_snapshot": memory
    }
    
    print(f"Processing chat for {request.user_id}...")
    try:
        final_state = agent_app.invoke(initial_state)
        
        # workflow 完成，自动触发的 reflect_node 已更新 memory_snapshot 引用
        updated_memory = final_state.get("memory_snapshot", memory)
        
        # 自动执行落盘保存最新 JSON
        manager.save_memory(updated_memory)
        
        # 从消息序列尾部提取最新的回复
        messages = final_state.get("messages", [])
        ai_reply = ""
        if messages and isinstance(messages[-1], AIMessage):
            ai_reply = messages[-1].content
            
        return ChatResponse(reply=ai_reply)
        
    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/memory/{user_id}")
async def get_memory(user_id: str):
    """
    查询指定用户的记忆实体。
    """
    manager = get_user_memory_manager(user_id)
    memory = manager.load_memory()
    return memory.model_dump()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
