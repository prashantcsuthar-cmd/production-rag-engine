import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# Page Configuration
st.set_page_config(
    page_title="Enterprise RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e364f;
    }
    .source-box {
        background-color: #161b22;
        border-left: 4px solid #4F46E5;
        padding: 10px;
        margin-top: 5px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Enterprise Knowledge Engine")
st.caption("Powered by Gemini 2.5 Flash, Qdrant Vector DB & FastAPI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=60)
    st.title("Control Panel")
    
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader("Drop a PDF here", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Ingest & Index PDF", type="primary", use_container_width=True):
            with st.spinner("Indexing vector database..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("✅ File embedded & temporary copy deleted!")
                        st.info(res.json().get("message"))
                    else:
                        st.error(f"Error: {res.json().get('detail')}")
                except Exception as e:
                    st.error(f"Backend connection error: {e}")

    st.divider()
    
    # System Status Card
    st.subheader("⚙️ System Status")
    try:
        health = requests.get(API_URL, timeout=2).json()
        st.success("🟢 REST API: ONLINE")
    except Exception:
        st.error("🔴 REST API: OFFLINE")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- CHAT ENGINE UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Context Sources & Confidence Scores"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <b>Source {idx}:</b> <code>{src['file_name']}</code> (Page {src['page_label']})<br/>
                        <b>Relevance Score:</b> <code>{src['score']}</code><br/>
                        <small><i>"{src['text_snippet']}"</i></small>
                    </div>
                    """, unsafe_allow_html=True)

if prompt := st.chat_input("Ask any question about your ingested documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vector space & synthesizing answer..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={"message": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("🔍 View Context Sources & Confidence Scores"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>Source {idx}:</b> <code>{src['file_name']}</code> (Page {src['page_label']})<br/>
                                    <b>Relevance Score:</b> <code>{src['score']}</code><br/>
                                    <small><i>"{src['text_snippet']}"</i></small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Could not reach FastAPI server: {e}")