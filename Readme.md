# Sake Master

スマートフォンでお酒のラベルを撮影し、  
バーのマスターと会話するような感覚で  
お酒の方向性・根拠・合わせ方を知るための Web アプリです。

---

## 概要

- 成分表・表ラベル・バーコードなどの写真を送信
- Google Cloud Vision API で OCR
- 抽出した情報のみを根拠として LLM が回答
- 断定しすぎず、押し付けない「マスター口調」

**正解を当てるアプリではなく、会話体験を重視しています。**

---

## 主な特徴

- 📸 写真は複数枚送信可能（最初は1枚、あとから追加もOK）
- 🍶 同一のお酒はセッションとして扱い、情報を蓄積
- 🔄 「次のお酒を相談する」ボタンで完全リセット
- 🧠 LLMは OCR由来の情報のみを根拠に使用（捏造防止）
- 🔐 静的解析・SBOM を含むセキュリティ証跡を作成

---

## 技術スタック

- Backend: FastAPI
- Frontend: Jinja2 + Vanilla JavaScript
- OCR: Google Cloud Vision API
- LLM: OpenAI API
- Infra: Google Cloud Run

---

## ローカル実行

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn main:app --reload
