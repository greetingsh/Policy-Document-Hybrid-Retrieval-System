from flask import Blueprint, request
from sqlalchemy import text
from app.utils import pg_engine, milvus_client, embedding_model, success_response, error_response
from app.config import Config

doc_bp = Blueprint("doc", __name__)

# ===================== 工具函数 =====================
def get_filtered_doc_ids(org_id, cat_id, start_date, end_date):
    """根据过滤条件获取符合要求的doc_id列表"""
    sql = "SELECT doc_id FROM documents WHERE 1=1"
    params = {}

    if org_id:
        sql += " AND org_id = :org_id"
        params["org_id"] = org_id
    if cat_id:
        sql += " AND cat_id = :cat_id"
        params["cat_id"] = cat_id
    if start_date:
        sql += " AND publish_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        sql += " AND publish_date <= :end_date"
        params["end_date"] = end_date

    try:
        with pg_engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        print(f"获取过滤doc_id失败: {str(e)}")
        return []

def exact_search(keyword, org_id, cat_id, start_date, end_date, limit):
    """标题模糊匹配检索，小数据量下ILIKE比全文检索更适配中文"""
    sql = """
    SELECT 
        d.doc_id, d.title, d.summary, d.publish_date, 
        o.org_name, c.cat_name, d.source_url
    FROM documents d
    JOIN orgs o ON d.org_id = o.org_id
    JOIN categories c ON d.cat_id = c.cat_id
    WHERE d.title ILIKE :keyword
    """
    params = {"keyword": f"%{keyword}%"}

    if org_id:
        sql += " AND d.org_id = :org_id"
        params["org_id"] = org_id
    if cat_id:
        sql += " AND d.cat_id = :cat_id"
        params["cat_id"] = cat_id
    if start_date:
        sql += " AND d.publish_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        sql += " AND d.publish_date <= :end_date"
        params["end_date"] = end_date

    sql += " ORDER BY d.publish_date DESC LIMIT :limit"
    params["limit"] = limit

    try:
        with pg_engine.connect() as conn:
            result = conn.execute(text(sql), params).mappings().all()
            return [dict(row) for row in result]
    except Exception as e:
        print(f"标题检索失败: {str(e)}")
        return []

def semantic_search(keyword, doc_id_list, limit):
    """Milvus语义向量检索"""
    if not doc_id_list:
        return []

    try:
        # 生成查询向量
        query_vec = embedding_model.encode([keyword], normalize_embeddings=True)[0].tolist()

        # 构建Milvus过滤表达式
        if len(doc_id_list) == 1:
            filter_expr = f"doc_id == {doc_id_list[0]}"
        else:
            id_str = ", ".join(map(str, doc_id_list))
            filter_expr = f"doc_id in [{id_str}]"

        # 执行向量检索
        res = milvus_client.search(
            collection_name=Config.COLLECTION_NAME,
            data=[query_vec],
            limit=limit * 3,
            output_fields=["doc_id", "chunk_text"],
            search_params={"metric_type": "COSINE", "params": {"ef": 32}},
            filter=filter_expr
        )

        # 按doc_id聚合，取最高相似度
        doc_score_map = {}
        doc_chunk_map = {}
        for hit in res[0]:
            entity = hit.get("entity", {})
            doc_id = entity.get("doc_id")
            score = hit.get("score", 0.0)
            chunk_text = entity.get("chunk_text", "")
            if doc_id is None:
                continue
            if doc_id not in doc_score_map or score > doc_score_map[doc_id]:
                doc_score_map[doc_id] = score
                doc_chunk_map[doc_id] = chunk_text

        # 按相似度降序取TopN
        sorted_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
        if not sorted_docs:
            return []

        # 批量查询文档元数据
        doc_ids = [d[0] for d in sorted_docs]
        meta_sql = """
        SELECT d.doc_id, d.title, d.summary, d.publish_date, o.org_name, c.cat_name, d.source_url
        FROM documents d
        JOIN orgs o ON d.org_id = o.org_id
        JOIN categories c ON d.cat_id = c.cat_id
        WHERE d.doc_id = ANY(:ids)
        """
        with pg_engine.connect() as conn:
            meta_result = conn.execute(text(meta_sql), {"ids": doc_ids}).mappings().all()
            doc_meta = {row["doc_id"]: dict(row) for row in meta_result}

        # 组装结果
        result_list = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_meta:
                item = doc_meta[doc_id]
                item["similarity"] = round(score, 4)
                highlight = doc_chunk_map[doc_id]
                item["highlight"] = highlight[:150] + "..." if len(highlight) > 150 else highlight
                result_list.append(item)
        return result_list

    except Exception as e:
        print(f"语义检索失败: {str(e)}")
        return []

# ===================== 接口 =====================
@doc_bp.route("/search", methods=["POST"])
def hybrid_search():
    """混合检索接口：标题匹配 + 语义检索，支持条件过滤，全局异常兜底"""
    try:
        data = request.get_json() or {}
        keyword = data.get("keyword", "").strip()
        org_id = data.get("org_id")
        cat_id = data.get("cat_id")
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if not keyword:
            return error_response("Search keyword cannot be empty")

        # 1. 获取范围内的所有doc_id
        filtered_doc_ids = get_filtered_doc_ids(org_id, cat_id, start_date, end_date)
        if not filtered_doc_ids:
            return success_response({"exact": [], "semantic": [], "total": 0})

        # 2. 第一路：标题模糊匹配
        exact_results = exact_search(keyword, org_id, cat_id, start_date, end_date, Config.EXACT_TOP_N)
        exact_doc_ids = {d["doc_id"] for d in exact_results}

        # 3. 第二路：语义检索（排除已精确匹配的文档）
        semantic_doc_ids = [d for d in filtered_doc_ids if d not in exact_doc_ids]
        semantic_results = semantic_search(keyword, semantic_doc_ids, Config.SEMANTIC_TOP_N)

        # 4. 标记匹配类型
        for item in exact_results:
            item["match_type"] = "exact"
            item["similarity"] = 1.0
        for item in semantic_results:
            item["match_type"] = "semantic"

        return success_response({
            "exact": exact_results,
            "semantic": semantic_results,
            "total": len(exact_results) + len(semantic_results)
        })

    except Exception as e:
        print(f"检索接口异常: {str(e)}")
        return error_response(f"Search failed: {str(e)}", 500)

@doc_bp.route("/orgs", methods=["GET"])
def get_orgs():
    """获取所有机构列表"""
    try:
        with pg_engine.connect() as conn:
            result = conn.execute(text("SELECT org_id, org_name FROM orgs ORDER BY org_id")).mappings().all()
            return success_response([dict(row) for row in result])
    except Exception as e:
        return error_response(f"Failed to get authority list: {str(e)}", 500)

@doc_bp.route("/categories", methods=["GET"])
def get_categories():
    """获取所有分类列表"""
    try:
        with pg_engine.connect() as conn:
            result = conn.execute(text("SELECT cat_id, cat_name FROM categories ORDER BY cat_id")).mappings().all()
            return success_response([dict(row) for row in result])
    except Exception as e:
        return error_response(f"Failed to get category list: {str(e)}", 500)

@doc_bp.route("/detail/<int:doc_id>", methods=["GET"])
def get_doc_detail(doc_id):
    """获取单篇文档详情"""
    try:
        sql = """
        SELECT d.doc_id, d.title, d.content, d.summary, d.publish_date, 
               o.org_name, c.cat_name, d.source_url
        FROM documents d
        JOIN orgs o ON d.org_id = o.org_id
        JOIN categories c ON d.cat_id = c.cat_id
        WHERE d.doc_id = :doc_id
        """
        with pg_engine.connect() as conn:
            result = conn.execute(text(sql), {"doc_id": doc_id}).mappings().first()
            if not result:
                return error_response("Document not found", 404)
            return success_response(dict(result))
    except Exception as e:
        return error_response(f"Failed to get document detail: {str(e)}", 500)
