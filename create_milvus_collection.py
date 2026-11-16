from pymilvus import MilvusClient, DataType

# ===================== 1. 建立连接（最新标准写法） =====================
client = MilvusClient(uri="http://localhost:19530")

# ===================== 2. 配置项 =====================
COLLECTION_NAME = "document_chunks"  # 集合名
VECTOR_DIM = 512                     # 向量维度，与 bge-small-zh 对应

# ===================== 3. 清理旧集合（可选，全新创建时执行） =====================
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"已移除旧集合: {COLLECTION_NAME}")

# ===================== 4. 定义表结构（最新 Schema 写法） =====================
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=False,  # 关闭动态字段，严格按定义字段存储
    description="香港政策文档分片向量库"
)

# 主键：分片唯一ID，自增
schema.add_field(
    field_name="chunk_id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

# 关联 PostgreSQL 文档主键
schema.add_field(
    field_name="doc_id",
    datatype=DataType.INT64
)

# 分片原文，检索时可直接返回
schema.add_field(
    field_name="chunk_text",
    datatype=DataType.VARCHAR,
    max_length=2000
)

# 文本向量字段
schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=VECTOR_DIM
)

# ===================== 5. 创建集合 =====================
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema
)
print(f"✅ 集合 {COLLECTION_NAME} 创建成功")

# ===================== 6. 创建向量索引（最新索引 API） =====================
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="HNSW",        # 目前主流高性能向量索引
    metric_type="COSINE",     # 余弦相似度，语义检索标准选型
    params={
        "M": 16,
        "efConstruction": 200
    }
)

client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)
print("✅ HNSW 向量索引创建完成")

# ===================== 7. 加载集合到内存（检索前置条件） =====================
client.load_collection(COLLECTION_NAME)
print("✅ 集合已加载到内存，可正常写入和检索")

