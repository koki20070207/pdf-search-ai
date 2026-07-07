from sentence_transformers import SentenceTransformer

#テキスト再現
chunks = [
    "RAG（検索拡張生成）は、大規模言語モデルに外部の知識ベースを組み合わせる技術です。",
    "これにより、AIが最新の情報や、社内の独自文書に基づいた正確な回答を作成できるようになります。"
]

print("読み込んでいます...")

#AIモデル呼び出し
model = SentenceTransformer('intfloat/multilingual-e5-small')

print("ベクトル変換中")

#テキスト数字変換！
vectors = model.encode(chunks)

#結果
print("\n変換成功")
print(f"1つ目の文章のベクトルの数: {len(vectors[0])}個の数字に変換されました。")
print(f"実際の数字: {vectors[0][:5]}")