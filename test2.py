import streamlit as st
import ollama
import json

# 商品資料庫
products = [
    {"category": "手機", "name": "iPhone 14", "price": 799, "features": ["128GB", "5G", "OLED 屏幕"]},
    {"category": "手機", "name": "Samsung Galaxy S22", "price": 699, "features": ["128GB", "5G", "AMOLED 屏幕"]},
    {"category": "筆電", "name": "MacBook Air", "price": 999, "features": ["M2", "256GB", "Retina 屏幕"]}
]

# 根據需求推薦商品
def recommend_products(category, budget, feature=None):
    return [
        product for product in products
        if product["category"] == category and product["price"] <= budget and (feature in product["features"] if feature else True)
    ]

# 與 Ollama 聯動的主程式
def chatbot():
    print("歡迎使用商品推薦機器人！")
    user_input = input("請輸入您的需求：")

    # 向 Ollama 發送請求
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "你是一個商品推薦助手，幫助用戶找到符合需求的商品。"},
            {"role": "user", "content": user_input}
        ]
    )

    # 解析 Ollama 回應
    response_text = response["content"]
    print(f"Ollama 回應: {response_text}")

    # 模擬需求解析（實際應由模型返回結構化數據）
    # 範例格式: {"category": "手機", "budget": 500, "feature": "128GB"}
    try:
        extracted_data = json.loads(response_text)
    except json.JSONDecodeError:
        print("無法解析需求，請重新輸入具體需求！")
        return

    category = extracted_data.get("category")
    budget = extracted_data.get("budget")
    feature = extracted_data.get("feature")

    # 商品推薦
    recommendations = recommend_products(category, budget, feature)
    if not recommendations:
        print("抱歉，沒有找到符合您需求的商品。")
        return

    print("以下是推薦的商品：")
    prices = []
    for product in recommendations:
        print(f"{product['name']} - ${product['price']} ({', '.join(product['features'])})")
        prices.append(product["price"])

    # 計算金額
    total = sum(prices)
    print(f"推薦商品的總金額約為：${total}")

if __name__ == "__main__":
    chatbot()
