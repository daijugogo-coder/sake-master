import os
import re
import uuid
import time
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import traceback
import requests

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Google Vision
from google.cloud import vision

from pydantic import BaseModel

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
UPLOAD_HINT = (
    "商品のバーコード画像、もしくは成分表（原材料、度数など）、商品名の写る写真を送ってください。\n"
    "PCの場合は複数の画像を一度に送ってください。"
)

# ----------------------------
# Data model
# ----------------------------
@dataclass
class ExtractedInfo:
    category: str  # "wine" | "sake" | "shochu" | "whisky" | "beer" | "unknown"
    abv: Optional[str] = None
    maker: Optional[str] = None
    brand: Optional[str] = None
    region: Optional[str] = None
    grapes_or_rice: Optional[str] = None
    # Sonar typing対応：Noneにしない
    keywords: List[str] = field(default_factory=list)
    raw_text_snippet: str = ""


# ----------------------------
# Utilities: OCR
# ----------------------------
def run_ocr(image_bytes: bytes) -> str:
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
# Utilities: Extraction
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
        keywords=keywords,
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
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code >= 400:
        print("=== OPENAI ERROR ===")
        print("status:", r.status_code)
        print("body:", r.text)
        print("payload(model):", payload.get("model"))
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


# ----------------------------
# Prompts
# ----------------------------
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


def build_humor_prompt() -> str:
    return """
あなたはバーの「マスター」です。相手は「お客様」です。

目的：
お客様が送ってきた対象が「酒ではない（可能性が高い）」場合でも、冷たく突き放さず、軽いノリツッコミで楽しく返す。

絶対ルール：
- 1〜2行の軽いボケ＋ツッコミ（短く）
- 相手を貶さない（バカにしない）
- 商品自体を否定しない（価値を認める）
- 最後は必ず「役に立つ一言」で締める（1行）
- 大阪弁固定はしない。デフォルトは標準語（東京寄り）で「〜じゃん」を自然に使う
- ただし、お客様の口調が方言っぽい場合は「語尾を少し寄せる程度」にする（やりすぎ禁止）

禁止：
- 説教、冷笑、人格否定、過度な方言の誇張、断定しすぎ

出力フォーマット（必ず守る）：
（ボケ＋ツッコミ：1〜2行）
（役に立つ一言：1行。一般的な選び方・使い方・次に撮ると良い写真など）
"""


# ----------------------------
# JSON parsing helpers (S5857対策：reluctant quantifier を使わない)
# ----------------------------
def _strip_code_fence(s: str) -> str:
    """
    ```json ... ``` みたいなコードフェンスがあれば中身だけ返す。
    正規表現は使わない（S5857対策）。
    """
    if not s:
        return ""
    t = s.strip()
    if "```" not in t:
        return t

    first = t.find("```")
    if first < 0:
        return t
    second = t.find("```", first + 3)
    if second < 0:
        return t

    inner = t[first + 3:second].strip()

    # 先頭が "json" の場合を剥ぐ
    if inner.lower().startswith("json"):
        inner = inner[4:].lstrip()
    return inner


def _extract_first_json_object(s: str) -> str:
    """
    文字列から最初の { ... } を雑に抜く。
    正規表現で .*? をやらない（S5857対策）。
    """
    if not s:
        return ""
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        return s[start:end + 1]
    return ""


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None

    t = s.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    t2 = _strip_code_fence(t)
    j = _extract_first_json_object(t2)
    if not j:
        # フェンス無しでも再トライ
        j = _extract_first_json_object(t)
    if not j:
        return None

    try:
        return json.loads(j)
    except Exception:
        return None


# ----------------------------
# LLM judge + replies
# ----------------------------
def classify_alcohol_or_not(
    customer_text: str,
    ocr_snippets: List[str],
    extracted: ExtractedInfo,
) -> Dict[str, Any]:
    sys = (
        "You are a strict classifier. Do NOT write prose. Output JSON only.\n"
        "Decide whether the item is alcoholic beverage or not, based on given hints.\n"
        "If unsure, choose unknown.\n"
    )
    user = {
        "customer_text": customer_text,
        "extracted": asdict(extracted),
        "ocr_text_snippets": ocr_snippets[-3:],
        "output_json_schema": {
            "category": "alcohol | non_alcohol | unknown",
            "confidence": "0.0-1.0",
            "reason": "max 1 short sentence",
            "dialect_hint": "standard | kansai | tohoku | kyushu | other"
        }
    }
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
    raw = call_llm(messages, temperature=0.0)
    obj = _safe_json_load(raw) or {}

    category = obj.get("category", "unknown")
    confidence = obj.get("confidence", 0.0)
    reason = obj.get("reason", "")
    dialect_hint = obj.get("dialect_hint", "standard")

    if category not in ("alcohol", "non_alcohol", "unknown"):
        category = "unknown"
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if dialect_hint not in ("standard", "kansai", "tohoku", "kyushu", "other"):
        dialect_hint = "standard"

    return {
        "category": category,
        "confidence": confidence,
        "reason": str(reason)[:120],
        "dialect_hint": dialect_hint,
        "raw": str(raw)[:500],
    }


def humor_reply(
    customer_text: str,
    ocr_snippets: List[str],
    extracted: ExtractedInfo,
    dialect_hint: str,
) -> str:
    dialect_instruction = ""
    if dialect_hint == "kansai":
        dialect_instruction = "関西っぽい口調なら語尾を少しだけ寄せてOK（例：〜じゃん→〜やん）。やりすぎ禁止。"
    elif dialect_hint == "tohoku":
        dialect_instruction = "東北っぽい口調なら語尾を少しだけ柔らかく寄せてOK。やりすぎ禁止。"
    elif dialect_hint == "kyushu":
        dialect_instruction = "九州っぽい口調なら語尾を少しだけ寄せてOK。やりすぎ禁止。"
    else:
        dialect_instruction = "基本は標準語（東京寄り）。"

    facts = {
        "extracted_latest": asdict(extracted),
        "ocr_text_snippets": ocr_snippets[-3:],
        "customer_text": customer_text,
        "dialect_hint": dialect_hint,
        "dialect_instruction": dialect_instruction,
    }

    messages = [
        {"role": "system", "content": build_humor_prompt()},
        {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
    ]
    return call_llm(messages, temperature=0.75)


def master_reply(
    extracted: ExtractedInfo,
    customer_text: str,
    chat_history: List[Dict[str, str]],
    ocr_snippets: List[str],
) -> str:
    facts = {
        "extracted_latest": asdict(extracted),
        "ocr_text_snippets": ocr_snippets[-3:],
    }
    context_block = (
        "【利用可能な事実（ここだけが根拠）】\n"
        f"{facts}\n\n"
        "【お客様の発話】\n"
        f"{customer_text}\n\n"
        "補足：もし『表ラベル（銘柄）』や『バーコード』の写真があるなら、追加で伝えてもらって構いません。"
    )

    messages = [{"role": "system", "content": build_system_prompt()}]
    for m in chat_history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": context_block})

    return call_llm(messages, temperature=0.65)


# ----------------------------
# Session helpers (Cognitive Complexity対策：analyzeを薄くする)
# ----------------------------
def get_or_create_session(session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]

    new_id = str(uuid.uuid4())
    sess = {
        "created_at": time.time(),
        "history": [{"role": "assistant", "content": f"マスター：{WELCOME_MESSAGE}"}],
        "images": [],
    }
    SESSIONS[new_id] = sess
    return new_id, sess


async def read_photo_bytes(photo: UploadFile) -> bytes:
    try:
        return await photo.read()
    except Exception:
        return b""


def add_image_record(sess: Dict[str, Any], ocr_text: str) -> None:
    extracted_each = extract_info(ocr_text)
    sess["images"].append({
        "added_at": time.time(),
        "ocr_text": ocr_text,
        "extracted": asdict(extracted_each),
    })


def build_ocr_snippets(sess: Dict[str, Any], limit: int = 6) -> List[str]:
    snippets: List[str] = []
    for it in sess["images"][-limit:]:
        sn = (it.get("ocr_text") or "").strip().replace("\r", "")
        if len(sn) > 300:
            sn = sn[:300] + "…"
        snippets.append(sn)
    return snippets


def build_combined_extract_text(ocr_texts_this_request: List[str], sess: Dict[str, Any]) -> str:
    if ocr_texts_this_request:
        return "\n\n".join(ocr_texts_this_request)
    last_texts = [it["ocr_text"] for it in sess["images"][-2:]]
    return "\n\n".join(last_texts) if last_texts else ""


def normalize_customer_text(customer_text: str) -> Tuple[str, str]:
    raw = (customer_text or "").strip()
    return raw, (raw if raw else "（写真を投稿しました）")


def need_more_info_response(session_id: Optional[str]) -> Dict[str, Any]:
    return {
        "status": "need_more_info",
        "session_id": session_id,
        "master": "写真かメッセージのどちらかを送ってください。例えば『辛口が好き』だけでもOKです。"
    }


def need_retake_response(session_id: str, msg: str) -> Dict[str, Any]:
    return {"status": "need_retake", "master": msg, "session_id": session_id}


async def ocr_photos_and_store(
    sess: Dict[str, Any],
    photos: List[UploadFile],
    session_id: str,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    ocr_texts_this_request: List[str] = []

    for photo in photos:
        image_bytes = await read_photo_bytes(photo)
        if not image_bytes or len(image_bytes) < 1000:
            return [], need_retake_response(
                session_id,
                "写真の一部が受け取れませんでした。もう一度、写真を選び直して送ってください。"
            )

        try:
            ocr_text = run_ocr(image_bytes)
        except Exception as e:
            print("=== OCR ERROR ===")
            traceback.print_exc()
            return [], {
                "status": "error",
                "message": f"OCRに失敗しました: {type(e).__name__}: {str(e)}",
                "http_status": 500
            }

        if is_readable_enough(ocr_text):
            ocr_texts_this_request.append(ocr_text)
            add_image_record(sess, ocr_text)

    if photos and not ocr_texts_this_request:
        return [], need_retake_response(
            session_id,
            "成分表などの文字がうまく読み取れませんでした。\n"
            "反射やピンぼけの可能性があります。\n"
            "文字が大きく写るように撮って、もう一度お願いできますか？"
        )

    return ocr_texts_this_request, None


def choose_master_message(
    customer_text_for_llm: str,
    ocr_snippets: List[str],
    extracted: ExtractedInfo,
    history: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    classification = {"category": "unknown", "confidence": 0.0, "dialect_hint": "standard", "reason": ""}

    try:
        classification = classify_alcohol_or_not(
            customer_text=customer_text_for_llm,
            ocr_snippets=ocr_snippets,
            extracted=extracted,
        )
    except Exception:
        print("=== CLASSIFIER ERROR (ignored) ===")
        traceback.print_exc()

    should_humor = (
        classification.get("category") == "non_alcohol"
        and float(classification.get("confidence", 0.0)) >= 0.70
    )

    if should_humor:
        msg = humor_reply(
            customer_text=customer_text_for_llm,
            ocr_snippets=ocr_snippets,
            extracted=extracted,
            dialect_hint=str(classification.get("dialect_hint", "standard")),
        )
        return msg, classification

    msg = master_reply(
        extracted=extracted,
        customer_text=customer_text_for_llm,
        chat_history=history,
        ocr_snippets=ocr_snippets,
    )
    return msg, classification


async def analyze_impl(
    customer_text: str,
    session_id: Optional[str],
    photos: Optional[List[UploadFile]],
) -> Any:
    customer_text_raw, customer_text_for_llm = normalize_customer_text(customer_text)
    photo_count = len(photos) if photos else 0

    if photo_count == 0 and customer_text_raw == "":
        return need_more_info_response(session_id)

    session_id, sess = get_or_create_session(session_id)

    ocr_texts_this_request: List[str] = []
    if photos:
        ocr_texts_this_request, err = await ocr_photos_and_store(sess, photos, session_id)
        if err:
            if err.get("status") == "error" and err.get("http_status"):
                return JSONResponse({"status": "error", "message": err["message"]}, status_code=err["http_status"])
            return err

    sess["history"].append({"role": "user", "content": f"お客様：{customer_text_for_llm}"})

    combined_for_extract = build_combined_extract_text(ocr_texts_this_request, sess)
    extracted = extract_info(combined_for_extract) if combined_for_extract else ExtractedInfo(category="unknown")

    ocr_snippets = build_ocr_snippets(sess, limit=6)

    try:
        master_msg, classification = choose_master_message(
            customer_text_for_llm=customer_text_for_llm,
            ocr_snippets=ocr_snippets,
            extracted=extracted,
            history=sess["history"],
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
        "uploaded_count": photo_count,
        "accepted_count": len(ocr_texts_this_request),
        "classification": {
            "category": classification.get("category"),
            "confidence": classification.get("confidence"),
            "reason": classification.get("reason"),
            "dialect_hint": classification.get("dialect_hint"),
        }
    }


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
    session_id: Optional[str] = Form(None),
    photos: Optional[List[UploadFile]] = File(None),
):
    # Sonarの複雑度をここでゼロにする（実処理は別関数）
    return await analyze_impl(customer_text=customer_text, session_id=session_id, photos=photos)


@app.post("/api/chat")
async def chat(
    session_id: str = Form(...),
    customer_text: str = Form(...),
):
    if session_id not in SESSIONS:
        return JSONResponse({"status": "error", "message": "セッションが見つかりません。"}, status_code=404)

    sess = SESSIONS[session_id]
    customer_text = (customer_text or "").strip()
    if not customer_text:
        return JSONResponse({"status": "error", "message": "お客様の入力が空です。"}, status_code=400)

    sess["history"].append({"role": "user", "content": f"お客様：{customer_text}"})

    if not sess["images"]:
        extracted = ExtractedInfo(category="unknown")
        ocr_snippets: List[str] = []
    else:
        latest = sess["images"][-1]
        extracted = ExtractedInfo(**latest["extracted"])
        ocr_snippets = build_ocr_snippets(sess, limit=6)

    try:
        master_msg, classification = choose_master_message(
            customer_text_for_llm=customer_text,
            ocr_snippets=ocr_snippets,
            extracted=extracted,
            history=sess["history"],
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
        "master": master_msg,
        "images_count": len(sess["images"]),
        "classification": {
            "category": classification.get("category"),
            "confidence": classification.get("confidence"),
            "reason": classification.get("reason"),
            "dialect_hint": classification.get("dialect_hint"),
        }
    }


# ----------------------------
# Reset
# ----------------------------
class ResetReq(BaseModel):
    session_id: str


@app.post("/api/reset")
def reset(req: ResetReq):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}
