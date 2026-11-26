import streamlit as st
import requests
from datetime import date

BACKEND_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="HK Policy Document Search",
    page_icon="📄",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_doc_ids" not in st.session_state:
    st.session_state.current_doc_ids = []

if "search_keyword" not in st.session_state:
    st.session_state.search_keyword = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = {"exact": [], "semantic": [], "total": 0}
if "has_searched" not in st.session_state:
    st.session_state.has_searched = False

st.title("📄 Hong Kong Policy Document Hybrid Search System")
st.caption("Supports exact title matching + semantic vector search, filterable by authority, category and date range")

col_main, col_ai = st.columns([3, 2])

with col_main:
    with st.sidebar:
        st.header("Filter Conditions")


        @st.cache_data
        def get_orgs():
            try:
                resp = requests.get(f"{BACKEND_URL}/api/doc/orgs", timeout=5)
                if resp.status_code == 200:
                    return resp.json()["data"]
            except Exception:
                pass
            return []


        @st.cache_data
        def get_categories():
            try:
                resp = requests.get(f"{BACKEND_URL}/api/doc/categories", timeout=5)
                if resp.status_code == 200:
                    return resp.json()["data"]
            except Exception:
                pass
            return []


        org_list = get_orgs()
        cat_list = get_categories()

        org_options = {item["org_name"]: item["org_id"] for item in org_list} if org_list else {}
        cat_options = {item["cat_name"]: item["cat_id"] for item in cat_list} if cat_list else {}

        selected_org = st.selectbox("Issuing Authority", options=["All"] + list(org_options.keys()))
        selected_cat = st.selectbox("Policy Category", options=["All"] + list(cat_options.keys()))

        date_range = st.date_input(
            "Publication Date Range",
            value=(date(2020, 1, 1), date(2026, 12, 31)),
            format="YYYY-MM-DD"
        )

    keyword = st.text_input("🔍 Enter search keyword", placeholder="e.g. visa, curriculum, passport")
    search_btn = st.button("Start Search", type="primary", use_container_width=True)

    if search_btn and keyword.strip():
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")
        else:
            start_date = date_range.strftime("%Y-%m-%d")
            end_date = date_range.strftime("%Y-%m-%d")

        org_id = org_options[selected_org] if selected_org != "All" and selected_org in org_options else None
        cat_id = cat_options[selected_cat] if selected_cat != "All" and selected_cat in cat_options else None

        payload = {
            "keyword": keyword.strip(),
            "org_id": org_id,
            "cat_id": cat_id,
            "start_date": start_date,
            "end_date": end_date
        }

        try:
            with st.spinner("Retrieving documents..."):
                resp = requests.post(f"{BACKEND_URL}/api/doc/search", json=payload, timeout=30)

            if resp.status_code != 200:
                st.error(f"Search failed: {resp.json().get('msg', 'Unknown error')}")
            else:
                result = resp.json()["data"]
                # 存入session_state持久化
                st.session_state.search_keyword = keyword.strip()
                st.session_state.search_results = result
                st.session_state.has_searched = True
                # 更新AI可用的文档范围
                all_docs = result["exact"] + result["semantic"]
                st.session_state.current_doc_ids = [d["doc_id"] for d in all_docs]
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

    if st.session_state.has_searched:
        exact_list = st.session_state.search_results["exact"]
        semantic_list = st.session_state.search_results["semantic"]
        total = st.session_state.search_results["total"]

        st.success(f"Found {total} relevant documents in total")
        st.divider()

        st.subheader("🎯 Exact Title Match")
        if not exact_list:
            st.info("No exact title match results")
        else:
            for idx, doc in enumerate(exact_list, 1):
                with st.expander(f"**{idx}. {doc['title']}**", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.markdown(f"**Authority**: {doc['org_name']}")
                    col2.markdown(f"**Category**: {doc['cat_name']}")
                    col3.markdown(f"**Published**: {doc['publish_date']}")
                    st.markdown(f"**Summary**: {doc['summary'] if doc['summary'] else 'No summary available'}")
                    if doc['source_url']:
                        st.markdown(f"[Original Link]({doc['source_url']})")

                    if st.button("📝 Summarize with AI", key=f"sum_exact_{doc['doc_id']}"):
                        sum_resp = requests.post(
                            f"{BACKEND_URL}/api/ai/summarize",
                            json={"doc_id": doc["doc_id"]}
                        )
                        if sum_resp.status_code == 200:
                            st.markdown("**AI Summary:**")
                            st.write(sum_resp.json()["data"]["summary"])
                        else:
                            st.error("Summary failed")

                    if st.button("View Full Text", key=f"exact_{doc['doc_id']}"):
                        detail_resp = requests.get(f"{BACKEND_URL}/api/doc/detail/{doc['doc_id']}", timeout=10)
                        if detail_resp.status_code == 200:
                            content = detail_resp.json()["data"]["content"]
                            st.text_area("Full Document", content, height=300)

        st.divider()

        st.subheader("🧠 Semantically Relevant Content")
        if not semantic_list:
            st.info("No semantically relevant results")
        else:
            for idx, doc in enumerate(semantic_list, 1):
                with st.expander(f"**{idx}. {doc['title']}**  ·  Similarity {doc['similarity']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.markdown(f"**Authority**: {doc['org_name']}")
                    col2.markdown(f"**Category**: {doc['cat_name']}")
                    col3.markdown(f"**Published**: {doc['publish_date']}")
                    st.markdown(f"**Relevant Snippet**: {doc['highlight']}")
                    st.markdown(f"**Summary**: {doc['summary'] if doc['summary'] else 'No summary available'}")
                    if doc['source_url']:
                        st.markdown(f"[Original Link]({doc['source_url']})")

                    if st.button("📝 Summarize with AI", key=f"sum_sem_{doc['doc_id']}"):
                        sum_resp = requests.post(
                            f"{BACKEND_URL}/api/ai/summarize",
                            json={"doc_id": doc["doc_id"]}
                        )
                        if sum_resp.status_code == 200:
                            st.markdown("**AI Summary:**")
                            st.write(sum_resp.json()["data"]["summary"])
                        else:
                            st.error("Summary failed")

                    if st.button("View Full Text", key=f"semantic_{doc['doc_id']}"):
                        detail_resp = requests.get(f"{BACKEND_URL}/api/doc/detail/{doc['doc_id']}", timeout=10)
                        if detail_resp.status_code == 200:
                            content = detail_resp.json()["data"]["content"]
                            st.text_area("Full Document", content, height=300)

with col_ai:
    st.subheader("🤖 AI Policy Assistant")

    with st.expander("⚙️ Settings", expanded=False):
        ai_enabled = st.checkbox("Enable AI Assistant", value=True)
        use_search_scope = st.checkbox("Answer only based on current search results", value=True)
        st.caption(f"Current document scope: {len(st.session_state.current_doc_ids)} documents")

    if ai_enabled:
        chat_container = st.container(height=400, border=True)

        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        user_input = st.text_input("Ask about policies...", placeholder="Type your question here", key="ai_input")
        send_btn = st.button("Send", type="primary", use_container_width=True)

        if send_btn and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        payload = {
                            "question": st.session_state.chat_history[-1]["content"],
                            "doc_ids": st.session_state.current_doc_ids if use_search_scope else []
                        }
                        try:
                            resp = requests.post(f"{BACKEND_URL}/api/ai/chat", json=payload, timeout=60)
                            if resp.status_code != 200:
                                try:
                                    err_msg = resp.json().get("msg", "AI service error")
                                except Exception:
                                    err_msg = f"AI service error (status {resp.status_code})"
                                st.error(err_msg)
                            else:
                                try:
                                    resp_json = resp.json()
                                    resp_data = resp_json.get("data")
                                    if not resp_data or not isinstance(resp_data, dict):
                                        err_msg = resp_json.get("msg", "No valid response from AI")
                                        st.error(err_msg)
                                    else:
                                        answer = resp_data.get("answer", "No answer available")
                                        st.write(answer)
                                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                                except Exception as e:
                                    st.error(f"Failed to parse AI response: {str(e)}")
                        except Exception as e:
                            st.error(f"Failed to connect to AI service: {str(e)}")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("AI assistant is disabled. Enable in settings to use summary and Q&A features.")
