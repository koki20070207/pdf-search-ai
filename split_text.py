from pypdf import PdfReader

pdf_path = "pdf_files/demo.pdf"
reader = PdfReader(pdf_path)
extracted_text = ""
for page in reader.pages:
    extracted_text += page.extract_text()

#300文字ずつチャンク化
chunk_size = 300
chunks = []

for i in range(0, len(extracted_text), chunk_size):
    chunk = extracted_text[i:i + chunk_size]
    chunks.append(chunk)

#分割された結果を表示して確認する
print(f"✅ 成功！ 全体を {len(chunks)} 個の塊に分割しました。")
print("\n--- 1つ目の塊（最初の300文字） ---")
print(chunks[0])
print("---------------------------------")

if len(chunks) > 1:
    print("\n--- 2つ目の塊（次の300文字） ---")
    print(chunks[1])
    print("---------------------------------")