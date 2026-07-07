from pypdf import PdfReader

#pdf_filesフォルダpdf読み込み
pdf_path = "pdf_files/demo.pdf"
reader = PdfReader(pdf_path)

#テキストを抽出して結合する
extracted_text = ""
for page in reader.pages:
    extracted_text += page.extract_text()

#結果をターミナルに表示して確認
print(f"読み込んだ文字数: {len(extracted_text)}文字")
print("--- 抽出したテキストの最初の200文字 ---")
print(extracted_text[:200])
print("---------------------------------------")