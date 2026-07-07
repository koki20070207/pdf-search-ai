import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

@st.cache_resource
def init_system():
    """初期化"""
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="modular_rag_db")
    return model, collection

#分割
def create_hierarchical_chunks(text, parent_size=1000, child_size=200):
    """チャンクの親子化"""
    hierarchical_data = []
    for i in range(0, len(text), parent_size):
        parent_text = text[i : i + parent_size]
        for j in range(0, len(parent_text), child_size):
            child_text = parent_text[j : j + child_size]
            if len(child_text) >= 20: # 短すぎるデータを弾く
                hierarchical_data.append({"parent": parent_text, "child": child_text})
    return hierarchical_data

#DBに登録
def register_pdf(uploaded_file, model, collection):
    """PDF読み込み、保存"""
    reader = PdfReader(uploaded_file)
    full_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    
    data_list = create_hierarchical_chunks(full_text)
    
    embeddings, documents, metadatas, ids = [], [], [], []
    for idx, item in enumerate(data_list):
        embeddings.append(model.encode(item["child"]).tolist())
        documents.append(item["child"])
        # 親は付属情報として保存
        metadatas.append({"parent_text": item["parent"], "source": uploaded_file.name})
        ids.append(f"{uploaded_file.name}_idx{idx}")

    #upsertで重複防止
    collection.upsert(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)

#回答
def generate_rag_response(prompt, model, collection):
    """Geminiに回答を作らせる"""
    query_vector = model.encode(prompt).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=1)
    
    #if hit
    if results["metadatas"] and len(results["metadatas"][0]) > 0:
        parent_context = results["metadatas"][0][0]["parent_text"]
        child_context = results["documents"][0][0]
        
        llm = genai.GenerativeModel('gemini-2.5-flash')
        # 親をGeminiに渡す
        response = llm.generate_content(
            f"以下の【参考資料】に基づいて【質問】に答えてください。\n"
            f"【参考資料】\n{parent_context}\n"
            f"【質問】\n{prompt}"
        )
        return response.text, child_context, parent_context
    
    #else
    return "資料が見つかりませんでした。", None, None

#UI
def main():
    """Streamlitの画面描画とユーザー操作の受付"""
    st.set_page_config(page_title="PDF AIチャット", layout="wide")
    st.title("PDF AIチャット")
    
    model, collection = init_system()
    
    #ファイルアップロード
    with st.sidebar:
        st.header("📁 資料追加")
        uploaded_file = st.file_uploader("PDFを選択", type=["pdf"])
        if uploaded_file and st.button("記憶させる"):
            with st.spinner("登録中..."):
                register_pdf(uploaded_file, model, collection)
                st.success(f"「{uploaded_file.name}」を登録しました。")

    #履歴
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    #入力時の処理
    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("検索中...", expanded=True) as status:
                st.write("回答を作成しています...")

        with st.chat_message("assistant"):
            # 検索・生成関数を呼び出す
            answer, child, parent = generate_rag_response(prompt, model, collection)
            st.markdown(answer)
            
            #アコーディオン
            if parent:
                with st.expander("内部データを確認"):
                    st.write("**ヒットしたチャンク:**", child)
                    st.write("**AIへ渡したチャンク:**", parent)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

#開始地点
if __name__ == "__main__":
    main()