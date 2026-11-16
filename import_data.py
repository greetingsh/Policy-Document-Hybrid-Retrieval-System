import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# ====================== 1. 配置信息（修改这里即可） ======================
# PostgreSQL 配置
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "policy_search_db",  # 你的数据库名
    "user": "postgres",
    "password": "wh20040427"
}

# Milvus 配置
MILVUS_CONFIG = {
    "host": "localhost",
    "port": 19530,
    "collection_name": "document_chunks"  # 向量集合名
}

# 向量模型配置（用中文轻量模型，本地直接跑，无需API）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 512维，中文效果好，速度快
VECTOR_DIM = 512  # 和模型维度一致
CHUNK_SIZE = 500  # 每个文本分片的字符数
CHUNK_OVERLAP = 50  # 分片重叠字符数，避免语义断裂


# ====================== 2. 工具函数 ======================
def clean_text(text):
    """清洗文本：去除多余换行、空格、特殊字符"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按字符数切割文本，带重叠"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= text_len:
            break
    return chunks


# ====================== 3. PostgreSQL 操作 ======================
def init_pg_connection(config):
    """初始化PostgreSQL连接"""
    url = f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    engine = create_engine(url)
    return engine


def get_org_cat_id(engine, org_name, cat_name):
    """根据名称获取机构ID和分类ID"""
    with engine.connect() as conn:
        # 查询机构ID
        org_result = conn.execute(text("SELECT org_id FROM orgs WHERE org_name = :name"), {"name": org_name}).fetchone()
        if not org_result:
            raise ValueError(f"机构不存在: {org_name}，请先在orgs表添加")
        org_id = org_result[0]

        # 查询分类ID
        cat_result = conn.execute(text("SELECT cat_id FROM categories WHERE cat_name = :name"),
                                  {"name": cat_name}).fetchone()
        if not cat_result:
            raise ValueError(f"分类不存在: {cat_name}，请先在categories表添加")
        cat_id = cat_result[0]

    return org_id, cat_id


def insert_document(engine, doc_data):
    """插入单篇文档到PostgreSQL，返回doc_id"""
    sql = """
    INSERT INTO documents (title, org_id, cat_id, publish_date, source_url, summary, content)
    VALUES (:title, :org_id, :cat_id, :publish_date, :source_url, :summary, :content)
    ON CONFLICT (title) DO NOTHING
    RETURNING doc_id;
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), doc_data)
        conn.commit()
        doc_id = result.fetchone()
        return doc_id[0] if doc_id else None


# ====================== 4. Milvus 操作 ======================
def init_milvus(config):
    """连接Milvus并创建集合（如果不存在）"""
    connections.connect(host=config["host"], port=config["port"])
    collection_name = config["collection_name"]

    # 如果集合已存在直接返回
    if utility.has_collection(collection_name):
        print(f"Milvus集合 {collection_name} 已存在")
        return Collection(collection_name)

    # 定义集合字段
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.INT64),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    schema = CollectionSchema(fields, description="政策文档分片向量")
    collection = Collection(collection_name, schema)

    # 创建向量索引（HNSW + 余弦相似度）
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"Milvus集合 {collection_name} 创建成功")
    return collection


def insert_vectors(collection, doc_id, chunks, vectors):
    """批量插入分片向量到Milvus"""
    data = [
        [doc_id] * len(chunks),  # doc_id列
        chunks,  # 分片文本列
        vectors.tolist()  # 向量列
    ]
    collection.insert(data)
    collection.flush()


# ====================== 5. 主导入流程 ======================
def main():
    # 1. 初始化连接
    print("正在连接数据库...")
    pg_engine = init_pg_connection(PG_CONFIG)
    milvus_col = init_milvus(MILVUS_CONFIG)

    # 2. 加载向量模型
    print(f"正在加载Embedding模型: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 3. 读取元数据
    metadata_path = "data/metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("找不到data/metadata.csv，请先整理元数据")
    df = pd.read_csv(metadata_path)
    print(f"读取到 {len(df)} 条文档元数据")

    # 4. 遍历导入
    success_count = 0
    for idx, row in df.iterrows():
        print(f"\n正在处理第 {idx + 1}/{len(df)} 篇: {row['title']}")

        try:
            # 4.1 获取机构和分类ID
            org_id, cat_id = get_org_cat_id(pg_engine, row["org_name"], row["cat_name"])

            # 4.2 读取正文
            doc_path = os.path.join(r"D:\5003\poject\data\docs", row["filename"])
            if not os.path.exists(doc_path):
                print(f"  跳过：找不到正文文件 {doc_path}")
                continue
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = clean_text(content)

            # 4.3 写入PostgreSQL
            doc_data = {
                "title": row["title"],
                "org_id": org_id,
                "cat_id": cat_id,
                "publish_date": row["publish_date"],
                "source_url": row.get("source_url", ""),
                "summary": row.get("summary", ""),
                "content": content
            }
            doc_id = insert_document(pg_engine, doc_data)
            if not doc_id:
                print(f"  跳过：标题已存在")
                continue

            # 4.4 文本分片
            chunks = split_text(content)
            if not chunks:
                print(f"  跳过：正文为空")
                continue

            # 4.5 生成向量
            vectors = model.encode(chunks, normalize_embeddings=True)

            # 4.6 写入Milvus
            insert_vectors(milvus_col, doc_id, chunks, vectors)

            success_count += 1
            print(f"  导入成功！doc_id={doc_id}，分片数={len(chunks)}")

        except Exception as e:
            print(f"  导入失败: {str(e)}")
            continue

    # 5. 完成统计
    print("\n" + "=" * 50)
    print(f"导入完成！成功 {success_count} 篇，失败 {len(df) - success_count} 篇")

    # 加载集合到内存，方便后续检索
    milvus_col.load()
    print("Milvus集合已加载到内存，可直接检索")


if __name__ == "__main__":
    main()
