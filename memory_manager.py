import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from pydantic import BaseModel, Field

class UserMemory(BaseModel):
    """Data structure for a digital AI Agent's memory."""
    basic_info: Dict[str, Any] = Field(default_factory=dict)
    personality_traits: List[str] = Field(default_factory=list)
    significant_events: List[str] = Field(default_factory=list)
    speaking_style: List[str] = Field(default_factory=list)
    last_updated: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )

class MemoryManager:
    """Manages loading and saving the UserMemory JSON file locally."""
    def __init__(self, storage_path: str):
        """
        Initialize the MemoryManager.
        If storage_path points to a directory, user_memory.json is appended.
        """
        if storage_path.endswith(".json"):
            self.file_path = storage_path
        else:
            self.file_path = os.path.join(storage_path, "user_memory.json")

    def load_memory(self) -> UserMemory:
        """Loads memory from local JSON file. If missing/invalid, creates a default memory instance."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return UserMemory(**data)
            except Exception as e:
                print(f"Failed to load memory from {self.file_path}: {e}")
                
        # Return empty memory if file doesn't exist or loading failed
        return UserMemory()

    def save_memory(self, memory: UserMemory) -> None:
        """Saves current memory to the local JSON file, updating the last_updated timestamp."""
        # Update timestamp to the current UTC epoch time 
        memory.last_updated = datetime.now(timezone.utc).timestamp()
        
        # Ensure that the directory exists
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(self.file_path, "w", encoding="utf-8") as f:
            # Using Pydantic V2 model_dump method
            json.dump(memory.model_dump(), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    from config_loader import config
    
    manager = MemoryManager(config.storage_settings.memory_path)
    
    # Check if we can load memory
    memory = manager.load_memory()
    print("Initial Memory:", memory.model_dump())
    
    # Modify memory and save
    memory.basic_info["name"] = "Alice"
    memory.personality_traits.append("Friendly")
    manager.save_memory(memory)
    print(f"Memory successfully saved to {manager.file_path}")
