import os
import uuid
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class EpisodicMemoryManager:
    """
    负责将细粒度的日常聊天记录编码为向量，存于 ChromaDB 提供长期情境记忆搜索能力。
    """
    def __init__(self, user_id: str, base_dir: str = "./memory/chroma_db"):
        self.user_id = user_id
        # 为每个用户隔离本地持久化目录
        self.persist_directory = os.path.join(base_dir, user_id)
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 使用本地开源、且极其轻量快速的句子嵌入模型，无需额外 API Key 成本
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 初始化 ChromaDB
        self.vector_store = Chroma(
            collection_name=f"episodic_memory_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def add_memory(self, user_msg: str, ai_reply: str):
        """将一轮交互追加至长期回溯库中"""
        content = f"User: {user_msg}\nAgent: {ai_reply}"
        doc = Document(
            page_content=content,
            metadata={
                "user_id": self.user_id,
                "timestamp": time.time()
            }
        )
        # Langchain-Chroma 默认底层为本地客户端，添加时会自动刷盘
        self.vector_store.add_documents([doc], ids=[str(uuid.uuid4())])
        
    def search_memory(self, query: str, top_k: int = 3) -> str:
        """基于用户最新的提问，语义检索曾经可能讨论过的相关细节"""
        try:
            # 执行相似度搜索
            results = self.vector_store.similarity_search(query, k=top_k)
            if not results:
                return ""
            
            # 将检索出的 N 条历史情境合并为纯文本上下文
            context = "\n---\n".join([doc.page_content for doc in results])
            return context
        except Exception as e:
            print(f"Chroma Semantic Search Error: {e}")
            return ""

    def get_all_history(self) -> list:
        """获取并按时间戳返回用户所有的历史对话记录"""
        try:
            results = self.vector_store.get(include=["documents", "metadatas"])
            if not results or not results.get("documents"):
                return []
            
            history = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                history.append({
                    "content": doc,
                    "timestamp": meta.get("timestamp", 0) if meta else 0
                })
            
            # 按时间戳增序排列（旧的在前，新的在后）
            history.sort(key=lambda x: x["timestamp"])
            return history
        except Exception as e:
            print(f"Fetch History Error: {e}")
            return []
