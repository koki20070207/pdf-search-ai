import os
import base64
import streamlit as st
import chromadb
import time
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
    
    #PDF棚
    pdf_collection = chroma_client.get_or_create_collection(name="modular_rag_db")
    #長期記憶棚
    chat_collection = chroma_client.get_or_create_collection(name="chat_memory_db")
    
    return model, pdf_collection, chat_collection

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

def register_chat_memory(prompt, answer, model, chat_collection):
    """会話履歴を保存"""
    #親データ
    conversation_context = f"ユーザー: {prompt}\nAI: {answer}"
    
    #質問を保存
    q_embedding = model.encode(prompt).tolist()
    q_id = f"chat_q_{int(time.time_ns())}"
    chat_collection.upsert(
        embeddings=[q_embedding],
        documents=[prompt],
        metadatas=[{"parent_text": conversation_context, "type": "question"}],
        ids=[q_id]
    )
    
    #回答を保存
    a_embedding = model.encode(answer).tolist()
    a_id = f"chat_a_{int(time.time_ns())}"
    chat_collection.upsert(
        embeddings=[a_embedding],
        documents=[answer],
        metadatas=[{"parent_text": conversation_context, "type": "answer"}],
        ids=[a_id]
    )

#回答
def generate_rag_response(prompt, chat_history, model, pdf_collection, chat_collection, threshold):
    """PDFと長期記憶の両方から検索してGeminiに回答を作らせる"""
    query_vector = model.encode(prompt).tolist()
    
    # 1. PDF本棚から検索
    pdf_results = pdf_collection.query(query_embeddings=[query_vector], n_results=1)
    pdf_context = "なし"
    child_context = "なし"
    #データがない場合は999にする
    pdf_score = pdf_results["distances"][0][0] if pdf_results["distances"] and len(pdf_results["distances"][0]) > 0 else 999.0
    
    #足切
    if pdf_score < threshold and pdf_results["metadatas"] and len(pdf_results["metadatas"][0]) > 0:
        pdf_context = pdf_results["metadatas"][0][0]["parent_text"]
        child_context = pdf_results["documents"][0][0]
        
    #長期記憶棚から検索
    memory_results = chat_collection.query(query_embeddings=[query_vector], n_results=1)
    memory_context = "なし"

    mem_score = memory_results["distances"][0][0] if memory_results["distances"] and len(memory_results["distances"][0]) > 0 else 999.0
    
    #足切り
    if mem_score < threshold and memory_results["metadatas"] and len(memory_results["metadatas"][0]) > 0:
        memory_context = memory_results["metadatas"][0][0]["parent_text"]
        
    #直近の会話履歴をまとめる
    history_text = ""
    for msg in chat_history[-6:]:
        role = "ユーザー" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content']}\n"
    
    #プロンプト
    llm = genai.GenerativeModel('gemini-2.5-flash')
    response = llm.generate_content(
        f"あなたはユーザーの過去の会話をすべて記憶している専属アシスタントです。\n"
        f"「AIなので個別の情報を記憶できません」といった定型文は絶対に言わないでください。\n"
        f"以下の【履歴】、【過去の記憶】、【参考資料】のみを参考にして回答してください。\n"
        f"※ただし、資料や記憶の中に「なし」と書かれている場合は、その情報は存在しないものとして扱い、知ったかぶりをせず会話してください。\n\n"
        f"【直近の会話履歴】\n{history_text if history_text else 'なし'}\n"
        f"【過去の長期記憶】\n{memory_context}\n"
        f"【参考資料（PDF）】\n{pdf_context}\n\n"
        f"【質問】\n{prompt}"
    )
    
    #6つのデータを返す
    return response.text, child_context, pdf_context, memory_context, pdf_score, mem_score
#UI
def main():
    """Streamlitの画面描画とユーザー操作の受付"""
    # 画面を広く使う設定
    st.set_page_config(page_title="PDF AIチャット", layout="wide")
    st.title("PDF AIチャット")
    
    model, collection, chat_collection = init_system()
    
    #ファイルアップロードと設定
    with st.sidebar:
        st.header("📁 資料追加")
        uploaded_file = st.file_uploader("PDFを選択", type=["pdf"])
        if uploaded_file and st.button("記憶させる"):
            with st.spinner("登録中..."):
                register_pdf(uploaded_file, model, collection)
                st.success(f"「{uploaded_file.name}」を登録しました。")
        
        st.divider()
        st.header("⚙️ 設定")
        threshold_label = st.sidebar.segmented_control(
            "🔍 検索の厳しさ",
            options=["厳重 (0.2)", "標準 (0.3)", "緩め (0.5)"],
            selection_mode="single",
            default="標準 (0.3)"
        )
        if not threshold_label:
            threshold_label = "標準 (0.3)"
        threshold_map = {"厳重 (0.2)": 0.2, "標準 (0.3)": 0.3, "緩め (0.5)": 0.5}
        threshold = threshold_map[threshold_label]
        #pdf配置設定
        preview_pos = st.segmented_control(
            "📄 PDFの表示位置",
            options=["左", "非表示", "右"],
            default="非表示" # デフォルトの選択
            )

    #画面分割
    # 最初は画面全体をチャット用にしておく（デフォルト設定）
    chat_container = st.container()

    current_pos = preview_pos if preview_pos else "非表示"

    # 表示位置が指定されていて、PDFがある場合だけ画面を割る
    if current_pos != "非表示" and uploaded_file:
        col1, col2 = st.columns([1, 1])
        preview_col, chat_container = (col2, col1) if current_pos == "右" else (col1, col2)
        
        # PDFプレビュー画面の描画
        with preview_col:
            st.markdown(f"**プレビュー:** {uploaded_file.name}")
            bytes_data = uploaded_file.getvalue()
            base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    # 以降のチャット画面は chat_container の中に配置する
    with chat_container:
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
                    
                    history_for_ai = st.session_state.messages[:-1]
                
                    answer, child, parent, mem_context, pdf_score, mem_score = generate_rag_response(
                        prompt, history_for_ai, model, collection, chat_collection, threshold
                    )
                    
                    status.update(label="完了", state="complete", expanded=False)
                
                st.markdown(answer)
                
                #アコーディオンの中身
                with st.expander("内部データを確認"):
                    st.write(f"**長期記憶スコア (距離: {mem_score:.2f} / {threshold}未満で採用):**")
                    st.write(mem_context)
                    st.write("---")
                    st.write(f"**PDFスコア (距離: {pdf_score:.2f} / {threshold}未満で採用):**")
                    st.write("**ヒットした子:**", child)
                    st.write("**AIへ渡した親:**", parent)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            register_chat_memory(prompt, answer, model, chat_collection) #会話記憶

#開始地点
if __name__ == "__main__":
    main()