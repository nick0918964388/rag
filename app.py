from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.together import TogetherLLM

import chromadb
import autogen
from autogen import ConversableAgent

app = Flask(__name__)

# Load the data

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

    # 设置全局嵌入模型
    Settings.embed_model = embed_model
    Settings.llm = TogetherLLM( model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", api_key="78aec3e2080904fc08729c407db1931e5851e8f03ff0fd0c4b29941340fcc8cc" )

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

def create_prompt(user_input):
    result = query_engine.query(user_input)

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
index = initialize_index()
query_engine = index.as_query_engine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    prompt = create_prompt(user_input)
    
    # 配置LLM(语言模型)
    llm_config = {
        "config_list": [
            # {
            #     "model": "llama-3.1-8b-instant",
            #     "api_key": os.getenv("GROQ_API_KEY"),
            #     "api_type": "groq",
            # }
            {
              "model": "mattshumer/reflection-70b:free",
              "base_url": "https://openrouter.ai/api/v1",
              "api_key": "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",
              "cache_seed": 42
            },
        ]
    }

    # 创建RAG机器人代理
    rag_agent = ConversableAgent(
        name="RAGbot",
        system_message="你是一個認真的RAG聊天機器人 , 請用繁體中文回答問題",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
  
    reply = rag_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
    
    return jsonify({'answer': reply.message})

if __name__ == '__main__':
    app.run(debug=True)