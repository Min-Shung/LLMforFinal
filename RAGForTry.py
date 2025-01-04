from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma.vectorstores import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
import os

# Step 1: 加載文本
file_path = 'commodity.txt'
loader = TextLoader(file_path=file_path, encoding='utf-8')
pages = loader.load()

# Step 2: 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ".", ","]
)
chunks = text_splitter.split_documents(pages)
print(f"分割後的文本段落數量: {len(chunks)}")

# Step 3: 嵌入模型初始化
embeddings_model = OllamaEmbeddings(model="llama3.2")

# 將文本嵌入為向量
documents = [chunk.page_content for chunk in chunks]
vectors = embeddings_model.embed_documents(documents)
print(f"完成嵌入的文本數量: {len(vectors)}")

# Step 4: 建立向量檢索數據庫
db_directory = 'db'
if not os.path.exists(db_directory):
    db = Chroma.from_documents(
        collection_name='product_recommendation',
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=db_directory
    )
else:
    db = Chroma(collection_name='product_recommendation', persist_directory=db_directory, embedding_function=embeddings_model)

# Step 5: 初始化語言模型
chat_model = ChatOllama(model="taide_model", base_url="http://localhost:11434")

# Step 6: 查詢流程
def recommend_product(query):
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    
    prompt = f"""根據以下產品資訊推薦產品:
    {context}
    問題: {query}"""
    
    response = chat_model.invoke(prompt).content
    return response

# Step 7: 初始化聊天記憶
memory = SQLChatMessageHistory(
    session_id="product_chat",
    connection="sqlite:///chat_memory.db"
)

# Step 8: 模擬對話與記憶
from langchain_core.messages import HumanMessage, AIMessage
def chat_with_memory(user_query):
    # 將用戶輸入轉為 HumanMessage
    memory.add_message(HumanMessage(content=user_query))
    # 執行產品推薦
    response = recommend_product(user_query)
    # 將 AI 回覆轉為 AIMessage
    memory.add_message(AIMessage(content=response))
    return response

# 測試查詢
query = "推薦一款適合拍照的手機"
response = chat_with_memory(query)
print("AI 回覆:", response)

# 模擬對話
user_query = "我想學游泳"
response = chat_with_memory(user_query)
print("AI 回覆:", response)
