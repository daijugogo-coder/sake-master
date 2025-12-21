import os
import re
import uuid
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import traceback
import requests

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Google Vision
from google.cloud import vision

# ----------------------------
# App & Templates
# ----------------------------
app = FastAPI(title="Sake Master (OCR + LLM)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# Settings (ENV)
# ----------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Session store (MVP: in-memory)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# UX Fixed copy
# ----------------------------
WELCOME_MESSAGE = "いらっしゃいませ。今日はどんなお酒を召し上がりますか？"
UPLOAD_HINT = "成分表など（原材料・度数などが載っている箇所）が写る写真を投稿してください。"


@dataclass
class ExtractedInfo:
    category: str  # "wine" | "sake" | "shochu" | "whisky" | "beer" | "unknown"
    abv: Optional[str] = None
    maker: Optional[str] = None          # 蔵 / winery / distillery
    brand: Optional[str] = None          # 銘柄候補
    region: Optional[str] = None         # 産地候補
    grapes_or_rice: Optional[str] = None # 品種 / 原料米など
    keywords: List[str] = None
    raw_text_snippet: str = ""


# ----------------------------
# Utilities: OCR
# ----------------------------
def run_ocr(image_bytes: bytes) -> str:
    """
    Google Cloud Vision OCR.
    Uses DOCUMENT_TEXT_DETECTION for better layout handling.
    Returns OCR text (str). Raises on API error.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")
    annotation = response.full_text_annotation
    return annotation.text or ""


def is_readable_enough(ocr_text: str) -> bool:
    t = (ocr_text or "").strip()
    if len(t) < 40:
        return False
    alpha_num = sum(c.isalnum() for c in t)
    if alpha_num < 15:
        return False
    return True


# ----------------------------
# Utilities: Extraction (lightweight, no scoring)
# ----------------------------
WINE_GRAPES = [
    "Cabernet Sauvignon", "Cabernet", "Merlot", "Pinot Noir", "Syrah", "Shiraz",
    "Grenache", "Tempranillo", "Sangiovese", "Chardonnay", "Sauvignon Blanc",
    "Riesling", "Pinot Grigio", "Viognier", "Malbec", "Zinfandel",
    "カベルネ", "メルロー", "ピノ", "シラー", "シャルドネ", "ソーヴィニヨン", "リースリング"
]

SAKE_KEYWORDS = ["純米", "吟醸", "大吟醸", "本醸造", "特別純米", "純米吟醸", "純米大吟醸", "生酒", "無濾過", "原酒", "火入れ"]
WINE_KEYWORDS = ["ワイン", "Vin", "Wine", "AOC", "DOC", "IGT", "DOP", "IGP", "Appellation", "辛口", "甘口", "Dry", "Sec", "Doux"]
WHISKY_KEYWORDS = ["ウイスキー", "Whisky", "Whiskey", "Single Malt", "Blended", "Bourbon"]
SHOCHU_KEYWORDS = ["焼酎", "芋", "麦", "米", "黒糖", "泡盛"]


def guess_category(text: str) -> str:
    t = text or ""
    if any(k in t for k in SAKE_KEYWORDS) or "清酒" in t or "日本酒" in t:
        return "sake"
    if any(k in t for k in WINE_KEYWORDS) or any(g in t for g in WINE_GRAPES):
        return "wine"
    if any(k in t for k in WHISKY_KEYWORDS):
        return "whisky"
    if any(k in t for k in SHOCHU_KEYWORDS):
        return "shochu"
    return "unknown"


def extract_abv(text: str) -> Optional[str]:
    patterns = [
        r"(?:アルコール分|Alc\.?|ABV)\s*[:：]?\s*(\d{1,2}(?:\.\d)?)\s*%?",
        r"(\d{1,2}(?:\.\d)?)\s*%?\s*(?:vol|VOL|Vol)"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1)
            return f"{val}%"
    return None


def extract_region(text: str) -> Optional[str]:
    prefs = ["北海道","青森","岩手","宮城","秋田","山形","福島","茨城","栃木","群馬","埼玉","千葉","東京","神奈川",
             "新潟","富山","石川","福井","山梨","長野","岐阜","静岡","愛知","三重","滋賀","京都","大阪","兵庫","奈良","和歌山",
             "鳥取","島根","岡山","広島","山口","徳島","香川","愛媛","高知","福岡","佐賀","長崎","熊本","大分","宮崎","鹿児島","沖縄"]
    for p in prefs:
        if p in text:
            return p
    for token in ["Bordeaux", "Bourgogne", "Burgundy", "Mosel", "Napa", "Toscana", "Rioja", "Barossa"]:
        if token in text:
            return token
    return None


def extract_maker_candidate(text: str) -> Optional[str]:
    patterns = [
        r"(?:製造者|製造元|醸造元|販売者)\s*[:：]?\s*([^\n]{2,30})",
        r"([^\n]{2,30})(?:酒造|酒造株式会社|株式会社|ワイナリー|Winery|Distillery|蒸溜所)"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"[ 　\t]+", " ", cand)
            return cand[:40]
    return None


def extract_grapes_or_rice(text: str) -> Optional[str]:
    found = []
    for g in WINE_GRAPES:
        if g in text:
            found.append(g)
    if found:
        return ", ".join(sorted(set(found)))[:120]

    m = re.search(r"(?:原材料名|原材料)\s*[:：]?\s*([^\n]{2,60})", text)
    if m:
        return m.group(1).strip()[:120]
    return None


def extract_brand_candidate(text: str, maker: Optional[str]) -> Optional[str]:
    bad = ["原材料", "アルコール", "内容量", "保存", "注意", "飲酒", "製造", "販売", "お問い合わせ", "住所", "電話", "〒"]
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines[:20]:
        if any(b in ln for b in bad):
            continue
        if maker and maker in ln:
            continue
        if 2 <= len(ln) <= 18:
            return ln
    return None


def extract_keywords(text: str) -> List[str]:
    kws = []
    for k in SAKE_KEYWORDS + WINE_KEYWORDS + WHISKY_KEYWORDS + SHOCHU_KEYWORDS:
        if k in text:
            kws.append(k)
    out = []
    for k in kws:
        if k not in out:
            out.append(k)
    return out[:12]


def extract_info(ocr_text: str) -> ExtractedInfo:
    cat = guess_category(ocr_text)
    abv = extract_abv(ocr_text)
    region = extract_region(ocr_text)
    maker = extract_maker_candidate(ocr_text)
    grapes_or_rice = extract_grapes_or_rice(ocr_text)
    brand = extract_brand_candidate(ocr_text, maker)
    keywords = extract_keywords(ocr_text)
    snippet = (ocr_text.strip().replace("\r", "")[:300] + "…") if len(ocr_text.strip()) > 300 else ocr_text.strip()

    return ExtractedInfo(
        category=cat,
        abv=abv,
        maker=maker,
        brand=brand,
        region=region,
        grapes_or_rice=grapes_or_rice,
        keywords=keywords or [],
        raw_text_snippet=snippet
    )


# ----------------------------
# LLM client (OpenAI minimal)
# ----------------------------
def call_llm(messages: List[Dict[str, str]], temperature: float = 0.6) -> str:
    if LLM_PROVIDER != "openai":
        raise RuntimeError("LLM_PROVIDER is not supported in this MVP. Set LLM_PROVIDER=openai.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def build_system_prompt() -> str:
    return """
あなたはバーの「マスター」です。相手は「お客様」です。
口調は丁寧だが堅すぎない。説教しない。質問攻めにしない。押し付けない。

重要ルール：
- 事実として断定して良いのは、与えられた「OCR文字」と「抽出JSON」だけ。
- それ以外は捏造禁止。知らないことは無理に言わない。
- ただし「定性的な推定」はOK（例：酸味が輪郭、香りが立つ、キレが良い等）。
- ペアリングは「料理名」ではなく、料理の性質（脂・塩・酸・旨味・香草・辛味など）と理由で説明する。
- 一般論で終わらず、必ず「この一本」に引き戻す。
- 回答は肯定文中心。ただし誇張しない。
- テンポ重視。うんちくは出しすぎない。聞かれたら答える。

出力フォーマット（必ず守る）：
①このお酒の方向性（2〜3文）
②根拠（箇条書きで短く、OCRから読めたことだけ）
③合わせ方の提案（理由つきで最大2つ）
④最後に一言だけ質問（0〜1個。聞かない選択もOK）
"""


def master_reply(
    extracted: ExtractedInfo,
    customer_text: str,
    chat_history: List[Dict[str, str]],
    ocr_snippets: List[str],
) -> str:
    facts = {
        "extracted_latest": asdict(extracted),
        "ocr_text_snippets": ocr_snippets[-3:],  # 直近3枚分だけ渡す（長すぎ防止）
    }

    context_block = (
        "【利用可能な事実（ここだけが根拠）】\n"
        f"{facts}\n\n"
        "【お客様の発話】\n"
        f"{customer_text}\n\n"
        "補足：もし『表ラベル（銘柄）』や『バーコード』の写真があるなら、追加で伝えてもらって構いません。"
    )

    messages = [{"role": "system", "content": build_system_prompt()}]

    # last few turns only
    for m in chat_history[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": context_block})
    return call_llm(messages, temperature=0.65)


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "welcome": WELCOME_MESSAGE,
        "hint": UPLOAD_HINT
    })

@app.post("/api/analyze")
async def analyze(
    customer_text: str = Form(""),
    photos: List[UploadFile] = File(...),   # ★複数枚
    session_id: str = Form(""),            # ★既存セッションに追加するため（任意）
):
    """
    Upload photo(s) -> OCR -> master reply
    - session_idが空なら新規セッション開始
    - session_idがあれば、既存セッションに画像を追加して再回答
    - photos は複数枚可（1枚でもOK）
    """

    # If user didn't type anything, treat as "photo only"
    customer_text = (customer_text or "").strip() or "（写真を投稿しました）"

    # 既存セッションか新規か
    if session_id and session_id in SESSIONS:
        sess = SESSIONS[session_id]
    else:
        session_id = str(uuid.uuid4())
        sess = {
            "created_at": time.time(),
            "history": [{"role": "assistant", "content": f"マスター：{WELCOME_MESSAGE}"}],
            "images": [],  # OCRの蓄積
        }
        SESSIONS[session_id] = sess

    # 写真が無い場合（フロントが required/disabled でも念のため）
    if not photos or len(photos) == 0:
        return {
            "status": "need_retake",
            "master": "写真が選択されていないようです。写真を選んで送ってください。",
            "session_id": session_id,
        }

    # 今回の複数枚のOCR結果をまとめる（統合抽出用）
    ocr_texts_this_request: List[str] = []

    # 複数枚を順にOCR
    for idx, photo in enumerate(photos, start=1):
        try:
            image_bytes = await photo.read()
        except Exception:
            image_bytes = b""

        if not image_bytes or len(image_bytes) < 1000:
            # 1枚でも壊れてたら全体を止めるより「その写真はダメ」と返すほうが親切だが、
            # MVPでは簡潔に「撮り直し」でまとめて返す
            return {
                "status": "need_retake",
                "master": "写真の一部が受け取れませんでした。もう一度、写真を選び直して送ってください。",
                "session_id": session_id,
            }

        try:
            ocr_text = run_ocr(image_bytes)
        except Exception as e:
            print("=== OCR ERROR ===")
            traceback.print_exc()
            return JSONResponse(
                {"status": "error", "message": f"OCRに失敗しました: {type(e).__name__}: {str(e)}"},
                status_code=500
            )

        # 読めない写真が混じるケース：ここは「その写真だけ無視」もできるが、
        # UX的には“ちゃんと読めた情報だけで返す”のが自然。
        # ただし全部読めないなら撮り直し。
        if is_readable_enough(ocr_text):
            ocr_texts_this_request.append(ocr_text)

            extracted_each = extract_info(ocr_text)
            sess["images"].append({
                "added_at": time.time(),
                "ocr_text": ocr_text,
                "extracted": asdict(extracted_each),
            })

    if not ocr_texts_this_request:
        master_msg = (
            "成分表などの文字がうまく読み取れませんでした。\n"
            "瓶への直接印刷や反射の影響かもしれません。\n"
            "文字が大きく写るように撮って、もう一度お願いできますか？"
        )
        return {
            "status": "need_retake",
            "master": master_msg,
            "session_id": session_id,
        }

    # 会話履歴
    sess["history"].append({"role": "user", "content": f"お客様：{customer_text}"})

    # ★今回分を統合して “この一本” の抽出を作る（最新1枚より自然）
    combined_for_extract = "\n\n".join(ocr_texts_this_request)
    extracted = extract_info(combined_for_extract)

    # LLMへ渡すOCRスニペット（直近複数枚）
    ocr_snippets = []
    for it in sess["images"]:
        sn = (it["ocr_text"] or "").strip().replace("\r", "")
        if len(sn) > 300:
            sn = sn[:300] + "…"
        ocr_snippets.append(sn)

    try:
        master_msg = master_reply(
            extracted=extracted,
            customer_text=customer_text,
            chat_history=sess["history"],
            ocr_snippets=ocr_snippets,
        )
    except Exception as e:
        print("=== LLM ERROR ===")
        traceback.print_exc()
        return JSONResponse(
            {"status": "error", "message": f"LLMに失敗しました: {type(e).__name__}: {str(e)}"},
            status_code=500
        )

    sess["history"].append({"role": "assistant", "content": f"マスター：{master_msg}"})

    return {
        "status": "ok",
        "session_id": session_id,
        "master": master_msg,
        "extracted": asdict(extracted),
        "ocr_text_snippet": extracted.raw_text_snippet,
        "images_count": len(sess["images"]),
        "uploaded_count": len(photos),
        "accepted_count": len(ocr_texts_this_request),
    }


@app.post("/api/chat")
async def chat(
    session_id: str = Form(...),
    customer_text: str = Form(...),
):
    """
    Chat continuation (no story mode)
    """
    if session_id not in SESSIONS:
        return JSONResponse({"status": "error", "message": "セッションが見つかりません。"}, status_code=404)

    sess = SESSIONS[session_id]
    customer_text = (customer_text or "").strip()
    if not customer_text:
        return JSONResponse({"status": "error", "message": "お客様の入力が空です。"}, status_code=400)

    sess["history"].append({"role": "user", "content": f"お客様：{customer_text}"})

    # 最新の抽出を使う（画像がゼロなら unknown）
    if not sess["images"]:
        extracted = ExtractedInfo(category="unknown", keywords=[])
        ocr_snippets = []
    else:
        latest = sess["images"][-1]
        extracted = ExtractedInfo(**latest["extracted"])
        ocr_snippets = []
        for it in sess["images"]:
            sn = (it["ocr_text"] or "").strip().replace("\r", "")
            if len(sn) > 300:
                sn = sn[:300] + "…"
            ocr_snippets.append(sn)

    try:
        master_msg = master_reply(
            extracted=extracted,
            customer_text=customer_text,
            chat_history=sess["history"],
            ocr_snippets=ocr_snippets,
        )
    except Exception as e:
        print("=== LLM ERROR ===")
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": f"LLMに失敗しました: {type(e).__name__}: {str(e)}"}, status_code=500)

    sess["history"].append({"role": "assistant", "content": f"マスター：{master_msg}"})

    return {
        "status": "ok",
        "master": master_msg,
        "images_count": len(sess["images"]),
    }

from pydantic import BaseModel

class ResetReq(BaseModel):
    session_id: str

@app.post("/api/reset")
def reset(req: ResetReq):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}
