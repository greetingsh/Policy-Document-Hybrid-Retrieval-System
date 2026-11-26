from flask import Blueprint, request
from sqlalchemy import text
from app.utils import pg_engine, llm_client, success_response, error_response
from app.utils import embedding_model, milvus_client
from app.config import Config

ai_bp = Blueprint("ai", __name__)

SYSTEM_PROMPT = """
You are a professional Hong Kong policy assistant. Answer questions strictly based on the provided document context.
If the answer cannot be found in the context, clearly state "The relevant information is not found in the current documents".
Keep answers concise, structured and accurate. Do not make up content.
"""


def get_doc_content_by_id(doc_id):
    """根据doc_id获取文档全文"""
    sql = """
    SELECT d.doc_id, d.title, d.content, o.org_name, d.publish_date
    FROM documents d
    JOIN orgs o ON d.org_id = o.org_id
    WHERE d.doc_id = :doc_id
    """
    with pg_engine.connect() as conn:
        row = conn.execute(text(sql), {"doc_id": doc_id}).mappings().first()
        return dict(row) if row else None


def retrieve_relevant_chunks(question, doc_ids, top_k=3):
    """从指定文档中召回最相关的文本片段，doc_ids为空时全库检索"""
    query_vec = embedding_model.encode([question], normalize_embeddings=True)[0].tolist()

    search_kwargs = {
        "collection_name": Config.COLLECTION_NAME,
        "data": [query_vec],
        "limit": top_k,
        "output_fields": ["doc_id", "chunk_text"],
        "search_params": {"metric_type": "COSINE", "params": {"ef": 32}}
    }

    if doc_ids:
        if len(doc_ids) == 1:
            filter_expr = f"doc_id == {doc_ids[0]}"
        else:
            id_str = ", ".join(map(str, doc_ids))
            filter_expr = f"doc_id in [{id_str}]"
        search_kwargs["filter"] = filter_expr

    res = milvus_client.search(**search_kwargs)

    chunks = []
    for hit in res[0]:
        entity = hit.get("entity", {})
        chunks.append({
            "doc_id": entity.get("doc_id"),
            "content": entity.get("chunk_text", ""),
            "score": hit.get("score", 0.0)
        })
    return chunks



@ai_bp.route("/summarize", methods=["POST"])
def summarize_doc():
    """总结单篇文档全文"""
    try:
        data = request.get_json() or {}
        doc_id = data.get("doc_id")

        if not doc_id:
            return error_response("doc_id is required")

        doc = get_doc_content_by_id(doc_id)
        if not doc:
            return error_response("Document not found")

        prompt = f"""
        Document Title: {doc['title']}
        Issuing Authority: {doc['org_name']}
        Publication Date: {doc['publish_date']}

        Full Content:
        {doc['content']}

        Please provide a structured summary of this document, including:
        1. Core theme
        2. Key points
        3. Target audience / applicable scope
        """

        summary = llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        return success_response({
            "doc_id": doc_id,
            "title": doc["title"],
            "summary": summary
        })

    except Exception as e:
        return error_response(f"Summarize failed: {str(e)}", 500)


@ai_bp.route("/chat", methods=["POST"])
def chat_with_docs():
    """基于指定文档范围的RAG问答"""
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()
        doc_ids = data.get("doc_ids", [])

        if not question:
            return error_response("Question cannot be empty")

        # 召回相关片段
        chunks = retrieve_relevant_chunks(question, doc_ids, top_k=4)

        if not chunks:
            return success_response({
                "answer": "No relevant document content found for your question.",
                "sources": []
            })

        # 拼接上下文
        context = "\n\n".join([
            f"[Fragment {i + 1}] (doc_id: {c['doc_id']})\n{c['content']}"
            for i, c in enumerate(chunks)
        ])

        prompt = f"""
        Reference Context:
        {context}

        User Question: {question}

        Answer based only on the reference context above. Cite the corresponding document source where appropriate.
        """

        answer = llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        return success_response({
            "answer": answer,
            "sources": [{"doc_id": c["doc_id"], "score": c["score"]} for c in chunks]
        })

    except Exception as e:
        return error_response(f"Chat failed: {str(e)}", 500)
