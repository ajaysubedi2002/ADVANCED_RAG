# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "API"
    
    # Models    
    CHAT_MODEL: str 
    EMBED_MODEL: str 
    MAX_RAG_CONTEXT_CHARS: int 
    RAG_CONTEXT_PLACEHOLDER: str  
    RERANK_MODEL:str
    
    # Vector Store
    CHROMA_DB_PATH: str 
    
    # Vector store collections
    COLLECTIONS_NAME: str
    
    # Ollama configuration
    OLLAMA_HOST: str 
    
    VECTOR_JSON_PATH: str  
     
    # Pydantic V2 configuration management
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # Gracefully ignores extra .env lines
    )

settings = Settings()
