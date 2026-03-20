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

from fastapi.responses import StreamingResponse

@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    manager = get_user_memory_manager(request.user_id)
    # 根据 user_id 加载记忆
    memory = manager.load_memory()
    
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "memory_snapshot": memory,
        "user_id": request.user_id
    }
    
    print(f"Processing streaming chat for {request.user_id}...")
    
    async def generate_response():
        final_state = None
        try:
            # stream_mode=["messages", "values"] 将流式发射字级别块，同时捕捉节点结束时刻的全景状态
            async for output in agent_app.astream(initial_state, stream_mode=["messages", "values"]):
                mode, data = output
                if mode == "messages":
                    msg, metadata = data
                    # 如果这句消息是 LLM(对话节点) 的逐字流块，返回给客户端
                    if metadata.get("langgraph_node") == "chat_node" and msg.content:
                        if isinstance(msg.content, str):
                            yield msg.content
                elif mode == "values":
                    # 会在每一个 Node 执行完成后抓取到最新的整体 Graph State
                    final_state = data
            
            # 当所有流程走完（如后台的 reflect 思考结束），进行自动状态落盘保存
            if final_state:
                updated_memory = final_state.get("memory_snapshot", memory)
                manager.save_memory(updated_memory)
                print(f"[{request.user_id}] Memory saved successfully post-stream.")
                
                # Phase 4: 后挂情境记忆入库
                messages = final_state.get("messages", [])
                if len(messages) >= 2:
                    ai_msg = messages[-1].content
                    # 找到触发这次对话的人类消息
                    human_msg = next((m.content for m in reversed(messages[:-1]) if isinstance(m, HumanMessage)), None)
                    if human_msg and ai_msg:
                        from vector_memory import EpisodicMemoryManager
                        v_manager = EpisodicMemoryManager(request.user_id, base_dir=os.path.join(os.path.dirname(config.storage_settings.memory_path), "chroma_db"))
                        v_manager.add_memory(human_msg, ai_msg)
                        print(f"[{request.user_id}] Vector episodic memory synced.")
        except Exception as e:
            print(f"Stream error: {e}")
            yield f"\n[Backend Error: {e}]"

    return StreamingResponse(generate_response(), media_type="text/plain")

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
