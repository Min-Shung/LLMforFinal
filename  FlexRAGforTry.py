from flexrag.models import HFEncoderConfig, OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Step 3: 初始化檢索器
# 將文本分割為可檢索的向量
# 首先將每個文本片段轉換為向量形式
documents = [chunk.page_content for chunk in chunks]

# 設置檢索器
retriever_cfg = DenseRetrieverConfig(database_path="path_to_database", top_k=3)
retriever_cfg.query_encoder_config.encoder_type = "hf"
retriever_cfg.query_encoder_config.hf_config = HFEncoderConfig(
    model_path="facebook/contriever"  # 使用 Hugging Face 的 Contriever 模型
)
retriever = DenseRetriever(retriever_cfg)

# Step 4: 初始化生成器
generator = OpenAIGenerator(
    OpenAIGeneratorConfig(
        model_name="gpt-4o-mini", api_key="your_openai_key", do_sample=False
    )
)

# Step 5: 查詢流程 - 用於生成回答
def recommend_product(query):
    # 使用檢索器來找到最相關的上下文
    context = retriever.search(query)
    prompt_str = ""
    for ctx in context:
        prompt_str += f"Question: {query}\nContext: {ctx.data['text']}"
    
    # 使用生成器生成回答
    prompt = ChatPrompt()
    prompt.update(ChatTurn(role="user", content=prompt_str))
    response = generator.chat(prompt)
    
    return response

# Step 6: 模擬對話與記憶
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

# 設置聊天記錄
memory = SQLChatMessageHistory(
    session_id="product_chat",
    connection="sqlite:///chat_memory.db"
)

# Step 7: 模擬對話過程
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
