import os

class Config:
    # PostgreSQL配置
    PG_HOST = "localhost"
    PG_PORT = 5432
    PG_DB = "policy_search_db"
    PG_USER = "postgres"
    PG_PASSWORD = "wh20040427"

    # Milvus配置
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "document_chunks"

    # Embedding模型
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

    # 混合检索参数
    EXACT_TOP_N = 5    # 标题精确匹配返回条数
    SEMANTIC_TOP_N = 10 # 语义检索返回条数

    # LLM 大模型配置
    LLM_BASE_URL = "https://api.deepseek.com/v1"
    LLM_API_KEY = "sk-3c86a16be7bd4598890741a4d60e54ba"
    LLM_MODEL = "deepseek-chat"


