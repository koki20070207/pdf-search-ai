import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

#初期設定
load_dotenv()
debug_key = os.getenv("GEMINI_API_KEY")
print(f"デバッグ確認: 読み込んだキーの長さ = {len(debug_key) if debug_key else 0}")
print(f"キーの最初の4文字 = {debug_key[:4] if debug_key else 'なし'}")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PDF_FOLDER = "pdf_files"

print("ChromaDB起動中")

#保存場所を指定
chroma_client = chromadb.PersistentClient(path="./chroma_db")
#記憶の引き出し:=「my_pdf_rag」
collection = chroma_client.get_or_create_collection(name="my_pdf_rag")

#モデル準備
model = SentenceTransformer('intfloat/multilingual-e5-small')

#DB登録
all_pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

for file_name in all_pdf_files:
    #登録済みかチェック
    existing = collection.get(where={"source": file_name})
    
    if len(existing["ids"]) > 0:
        continue
        
    print(f"新しいファイルなのでDBに記憶: {file_name}")
    pdf_path = os.path.join(PDF_FOLDER, file_name)
    reader = PdfReader(pdf_path)
    
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() or ""
        
    #300文字ずつ分割
    chunk_size = 300
    chunks = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]
    
    if not chunks:
        continue
        
    #AIでベクトル化
    vectors = model.encode(chunks).tolist()
    
    #IDとファイル名作成
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name} for _ in range(len(chunks))]
    
    #DBに追加
    collection.add(
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"{file_name} ：完了")

print("DB登録完了")

#質問
query = input("\n質問を入力してください: ")
query_vector = model.encode(query).tolist()

print(f"検索中")
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1  #一番似ている答え引っ張り出し
)

#検索結果の取り出し
best_match_text = results["documents"][0][0]
score = results["distances"][0][0] #0に近いほど優秀

print("\n検索結果")
print(f"距離スコア: {score:.4f} ※0に近いほど高精度:")
print("\n")
print(best_match_text)
print("\n")

#Geminiによる最終回答の生成
print("\nAIが回答を作成中")
llm = genai.GenerativeModel('gemini-2.5-flash')
prompt = f"""
あなたは優秀なアシスタントです。以下の【参考資料】だけを使って、【質問】に自然な日本語で答えてください。
参考資料に答えがない場合は「提供された資料には記載がありません」と正直に答えてください。

【参考資料】
{best_match_text}

【質問】
{query}
"""
response = llm.generate_content(prompt)

print("\n最終回答")
print(response.text)
print("\n")