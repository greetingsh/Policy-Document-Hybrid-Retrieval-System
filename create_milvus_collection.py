from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

COLLECTION_NAME = "document_chunks"
VECTOR_DIM = 512

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"已移除旧集合: {COLLECTION_NAME}")

schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
    description="香港政策文档分片向量库"
)

# 主键
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

# 分片原文
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


client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema
)
print(f"✅ 集合 {COLLECTION_NAME} 创建成功")


index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
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

client.load_collection(COLLECTION_NAME)
print("✅ 集合已加载到内存，可正常写入和检索")

