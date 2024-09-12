import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.together import TogetherLLM

import chromadb

# 保留原有的 initialize_index 函數
def initialize_index():
    # ... (保持不變) ...

# 修改 create_prompt 函數
def create_prompt(user_input, result):
    prompt = f"""
    任務：根據提供的上下文，對用戶的問題進行簡明且具信息量的回應。

上下文：{result}

用戶問題：{user_input}

指導方針：

相關性：直接關注用戶的問題。
簡潔性：避免不必要的細節。
準確性：確保事實正確。
清晰性：使用清楚的語言。
上下文意識：如果上下文不充分，使用一般知識作答。
誠實性：若缺乏信息，請明確說明。
回應格式：

直接回答
簡短解釋（如有必要）
引用（如有必要）
結論
    """
    return prompt

# 初始化索引和查詢引擎
@st.cache_resource
def load_index_and_engine():
    index = initialize_index()
    query_engine = index.as_query_engine()
    return index, query_engine

# Streamlit 應用主體
def main():
    st.title("RAG 對話機器人")

    # 初始化會話狀態
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 加載索引和查詢引擎
    index, query_engine = load_index_and_engine()

    # 顯示聊天歷史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用戶輸入
    if prompt := st.chat_input("請輸入您的問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            result = query_engine.query(prompt)
            full_prompt = create_prompt(prompt, result)
            
            llm = TogetherLLM(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                api_key="78aec3e2080904fc08729c407db1931e5851e8f03ff0fd0c4b29941340fcc8cc"
            )
            response = llm.complete(full_prompt)
            
            message_placeholder.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

if __name__ == "__main__":
    main()