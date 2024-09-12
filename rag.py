import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
import os
from pdf2image import convert_from_path
from PIL import Image
import io
import base64
import requests
import json

# 保留原有的 initialize_index 函數
def initialize_index():
    # 初始化 Chroma 数据库客户端
    db = chromadb.PersistentClient(path="./chroma_db")
    # 获取或创建一个名为 "my-docs" 的集合
    chroma_collection = db.get_or_create_collection("my-docs")
    # 创建 ChromaVectorStore 实例
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # 创建存储上下文
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 使用 BAAI/bge-large-en-v1.5 嵌入模型
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    # 設置全局嵌入模型和 LLM
    Settings.embed_model = embed_model
    Settings.llm = OllamaLLM()  # 使用自定義的 OllamaLLM

    # 移除 TogetherLLM 的设置
    
    # 检查集合是否已存在数据
    if chroma_collection.count() > 0:
        print("Loading existing index...")
        # 如果存在，从向量存储加载索引
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("Creating new index...")
        # 如果不存在，从文档目录加载数据并创建新索引
        documents = SimpleDirectoryReader("./documents").load_data()
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

# 保留原有的 create_prompt 函數
def create_prompt(user_input, result):
    prompt = f"""
    任務：根據提供的上下文，對用戶的問題進行簡明且具信息量的回應。

上下文：{result.response}

用戶問題：{user_input}

指導方針：

相關性：直接關注用戶的問題。
簡潔性：避免不必要的細節。
準確性：確保事實正確。
清晰性：使用清楚的語言。
上下文意識：如果上下文不充分，使用一般知識作答。
誠實性：若缺乏信息，請明確說明。
引用：在回答中引用來源文件名稱和頁碼（如果有）。
回應格式：

直接回答
簡短解釋（如有必要）
引用（格式：[文件名, 頁碼]）
結論
    """
    return prompt

# 修改文件上傳處理函數
def handle_file_upload(uploaded_files):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("./documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("文件上傳成功！")

# 保留重新索引函數
def reindex():
    # 清除現有的 Chroma 數據庫
    db = chromadb.PersistentClient(path="./chroma_db")
    db.delete_collection("my-docs")
    
    # 重新初始化索引
    index = initialize_index()
    query_engine = index.as_query_engine()
    st.session_state.index = index
    st.session_state.query_engine = query_engine
    st.success("重新索引完成！")

# 初始化索引和查詢引擎
@st.cache_resource
def load_index_and_engine():
    index = initialize_index()
    query_engine = index.as_query_engine()
    return index, query_engine

# 修改生成 PDF 頁面縮圖函數
def generate_pdf_thumbnail(file_path, page_number):
    images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
    if images:
        img = images[0]
        img.thumbnail((600, 600))  # 調整縮圖大小為 600x600
        return img
    return None

# 新增函數：將圖片轉換為 base64 編碼
def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# 修改生成可縮放的圖片 HTML 函數
def get_zoomable_image_html(img, caption):
    img_base64 = img_to_base64(img)
    return f"""
    <figure>
        <a href="data:image/png;base64,{img_base64}" target="_blank">
            <img src="data:image/png;base64,{img_base64}" 
                 alt="{caption}" 
                 style="cursor: zoom-in; max-width: 600px; max-height: 600px;">
        </a>
        <figcaption>{caption}</figcaption>
    </figure>
    """

# 新增函數：使用 Ollama LLM 生成回答
def generate_ollama_response(prompt):
    url = "http://ollama.webtw.xyz:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "reflection",  # 或者您想使用的其他模型
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"錯誤：無法從 Ollama 獲取回應。狀態碼：{response.status_code}"

# 修改 handle_ai_response 函數
def handle_ai_response(prompt):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = st.session_state.query_engine.query(prompt)
        
        # 提取引用信息並生成縮圖
        source_nodes = result.source_nodes
        sources = []
        for node in source_nodes:
            metadata = node.node.metadata
            file_name = metadata.get('file_name', '未知文件')
            page_label = metadata.get('page_label', '未知頁碼')
            file_path = os.path.join("./documents", file_name)
            
            if file_name.lower().endswith('.pdf') and os.path.exists(file_path):
                try:
                    page_number = int(page_label)
                    thumbnail = generate_pdf_thumbnail(file_path, page_number)
                    if thumbnail:
                        caption = f"{file_name}, 頁碼: {page_label}"
                        st.markdown(get_zoomable_image_html(thumbnail, caption), unsafe_allow_html=True)
                except ValueError:
                    st.write(f"無法生成縮圖：{file_name}, 頁碼: {page_label}")
            
            sources.append(f"[{file_name}, 頁碼: {page_label}]")
        
        # 組合回答和引用
        full_response = f"{result.response}\n\n引用來源：\n" + "\n".join(sources)
        
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 修改 Streamlit 應用主體
def main():
    st.set_page_config(page_title="檢修助手", layout="wide")
    
    # 頁面標題
    st.title("檢修助手")
    
    # 創建一個模擬模態框的 expander
    with st.expander("文件上傳和重新索引", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader("上傳文件", accept_multiple_files=True, key="file_uploader")
            if uploaded_files:
                handle_file_upload(uploaded_files)
        
        with col2:
            if st.button("重新索引", key="reindex_button"):
                reindex()
    
    # 初始化會話狀態
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 加載索引和查詢引擎
    if "index" not in st.session_state or "query_engine" not in st.session_state:
        index, query_engine = load_index_and_engine()
        st.session_state.index = index
        st.session_state.query_engine = query_engine

    # 顯示歡迎信息和建議查詢內容
    if not st.session_state.messages:
        st.markdown("""
        ## 歡迎使用檢修助手！
        
        您可以詢問任何關於檢修的問題。以下是一些建議的查詢內容：
        """)
        suggestions = [
            "這些文檔主要討論了哪些主題？",
            "文檔中有哪些關鍵概念？",
            "能總結一下文檔的主要觀點嗎？"
        ]
        for suggestion in suggestions:
            if st.button(suggestion):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                handle_ai_response(suggestion)

    # 顯示聊天歷史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用戶輸入
    if prompt := st.chat_input("請輸入您的問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_ai_response(prompt)

if __name__ == "__main__":
    main()