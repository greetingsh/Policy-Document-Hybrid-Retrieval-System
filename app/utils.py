import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sqlalchemy import create_engine
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from flask import jsonify
from app.config import Config

# 1. PostgreSQL连接引擎
pg_engine = create_engine(
    f"postgresql+psycopg2://{Config.PG_USER}:{Config.PG_PASSWORD}@{Config.PG_HOST}:{Config.PG_PORT}/{Config.PG_DB}"
)

# 2. Milvus客户端
milvus_client = MilvusClient(Config.MILVUS_URI)

# 3. Embedding模型 - 启动时全局预加载
print("正在加载Embedding模型...")
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
print("✅ Embedding模型加载完成")

# 4. 统一响应封装
def success_response(data=None, msg="success"):
    return jsonify({"code": 200, "msg": msg, "data": data})

def error_response(msg="error", code=400):
    return jsonify({"code": code, "msg": msg, "data": None})

# 5. 大模型客户端
import httpx
from openai import OpenAI

llm_client = OpenAI(
    base_url=Config.LLM_BASE_URL,
    api_key=Config.LLM_API_KEY,
    http_client=httpx.Client()  # 显式创建客户端，绕过自动代理配置
)

def llm_chat(system_prompt, user_prompt):
    """统一大模型调用封装"""
    try:
        response = llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI service error: {str(e)}"

