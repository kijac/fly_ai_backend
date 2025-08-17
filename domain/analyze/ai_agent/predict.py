# ai_pricer_sameitem_price.py
import os, io, csv, re, json, base64
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageEnhance

import torch
from transformers import CLIPProcessor, CLIPModel

from dotenv import load_dotenv
from anthropic import Anthropic

# ======================= 사용자 설정 =======================
IMAGE_PATH   = "old_toy/img_0006_front.png"

# 기본값 (외부에서 모델 설정이 전달되지 않을 때 사용)
DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14"
DEFAULT_FEATURES_NPY = "toys_index/used_features_large_patch14.npy"
DEFAULT_PATHS_NPY = "toys_index/used_paths_large_patch14.npy"

# 동적 설정을 위한 변수들 (외부에서 설정 가능)
MODEL_NAME = DEFAULT_MODEL_NAME
FEATURES_NPY = DEFAULT_FEATURES_NPY
PATHS_NPY = DEFAULT_PATHS_NPY

TOPK         = 6   # 넉넉히 뽑고 Top3만 사용

# paths.npy 에서 basename만 사용해, 실제 이미지는 이 폴더에서 찾습니다.
IMAGE_BASE_DIR = "used/used"

# CSV에서 파일명→가격 매핑
PRODUCTS_INFO_CSV  = "products_info.csv"
FILENAME_COL = "thumbnail_filename"
PRICE_COL    = "price"
PRODUCT_COL  = "title"   # (선택: 없으면 자동 건너뜀)

NORMALIZE_FILENAMES = True
CLAUDE_MODEL = "claude-sonnet-4-20250514"
# ==========================================================
load_dotenv()

# ----------------------- 유틸 -----------------------
def _parse_price_to_int(text: str) -> Optional[int]:
    if text is None: return None
    t = str(text).strip().replace(",", "").replace("원", "").replace("KRW", "")
    if not re.search(r"[0-9]", t): return None
    try:
        return int(float(t))
    except ValueError:
        digits = "".join(ch for ch in t if ch.isdigit())
        return int(digits) if digits else None

def _norm_name(path_or_name: str) -> str:
    name = os.path.basename(str(path_or_name)).strip()
    return name.lower() if NORMALIZE_FILENAMES else name

def load_meta_maps(csv_path, filename_col, price_col, product_col=None) -> Tuple[Dict[str,int], Optional[Dict[str,str]]]:
    if not os.path.isfile(csv_path):
        return {}, None
    encodings = ("utf-8-sig", "utf-8", "cp949")
    last_err = None
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                if filename_col not in fields or price_col not in fields:
                    raise KeyError(f"CSV에 '{filename_col}', '{price_col}' 컬럼이 필요. 현재: {fields}")
                price_map, title_map = {}, {}
                has_title = product_col in fields if product_col else False
                for row in reader:
                    raw = row.get(filename_col, "")
                    fname = _norm_name(raw)
                    if not fname: continue
                    price = _parse_price_to_int(row.get(price_col))
                    if price is not None: price_map[fname] = price
                    if has_title: title_map[fname] = str(row.get(product_col, "")).strip()
                return price_map, (title_map if has_title else None)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV 인코딩 실패(시도: {encodings}) 원인: {last_err}")

# ----------------------- CLIP 임베딩 -----------------------
def load_model(model_name: str, device: torch.device):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

def embed_image(model, processor, img_path, device) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32").flatten()

# ----------------------- 이미지 → Base64 -----------------------
def _resize_to_jpeg_bytes(img: Image.Image, max_side=1024, enhance=False) -> bytes:
    w,h = img.size
    scale = min(1.0, max_side / max(w,h))
    if scale < 1.0: img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    if enhance:
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Sharpness(img).enhance(1.6)
        img = ImageEnhance.Brightness(img).enhance(1.05)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92); return buf.getvalue()

def _path_to_b64(path: str, enhance=False) -> str:
    img = Image.open(path).convert("RGB")
    return base64.b64encode(_resize_to_jpeg_bytes(img, enhance=enhance)).decode("utf-8")

# ----------------------- 동일품 판정 프롬프트 -----------------------
SAME_ITEM_PROMPT = """
You are a product matcher. For each CANDIDATE photo, decide if it is the SAME PRODUCT/MODEL as the QUERY photo.
Focus on brand/series/character, mold/shape, printed patterns, colorway, scale/size cues, accessories/parts, packaging text or set ID.
Do NOT be fooled by pose/angle/lighting. If unsure, answer false.

Return STRICT JSON ONLY, exactly this schema:
{"same": [true, true, true]}

Rules:
- The array order MUST match the order of the CANDIDATE blocks you receive.
- "same" means same model/edition (not just same category/character).
- Variant/limited/colorway/set-ID mismatch => false.
- STRICT JSON only. No extra text.
"""

# ----------------------- Claude 호출 래퍼 -----------------------
class SameItemJudgeClaude:
    def __init__(self, client: Anthropic):
        self.client = client

    def judge(self, query_path: str, candidates: List[dict]) -> List[bool]:
        contents = [{"type":"text","text": SAME_ITEM_PROMPT}]
        contents.append({"type":"text","text": f"QUERY: {os.path.basename(query_path)}"})
        contents.append({"type":"image","source":{
            "type":"base64","media_type":"image/jpeg","data": _path_to_b64(query_path)
        }})

        for c in candidates:
            meta_line = f"CANDIDATE rank={c['rank']} | file={c['filename']} | sim={c['sim']:.4f}"
            if c.get("title"): meta_line += f" | title={c['title']}"
            contents.append({"type":"text","text": meta_line})
            contents.append({"type":"image","source":{
                "type":"base64","media_type":"image/jpeg","data": _path_to_b64(c['resolved_path'])
            }})

        msg = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,  # 출력이 매우 짧음
            temperature=0.1,
            system="Return STRICT JSON. No extra text.",
            messages=[{"role":"user","content": contents}]
        )

        raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json","",1).strip()

        try:
            data = json.loads(raw)
            same = list(data.get("same", []))
            same = [bool(x) for x in same][:len(candidates)]
            if len(same) < len(candidates):
                same += [False] * (len(candidates) - len(same))
        except Exception:
            same = [False] * len(candidates)
        return same

# ----------------------- 유사도 검색 -----------------------
def similar_search(query_img: str, feats_npy: str, paths_npy: str,
                   model_name: str, topk: int=6):
    if not os.path.isfile(query_img): raise FileNotFoundError(f"IMAGE_PATH not found: {query_img}")
    if not os.path.isfile(feats_npy) or not os.path.isfile(paths_npy):
        raise FileNotFoundError("features/paths npy 필요")

    feats = np.load(feats_npy).astype("float32")
    paths = np.load(paths_npy, allow_pickle=True)
    if feats.ndim!=2 or feats.shape[0]!=len(paths): raise ValueError("features/paths 크기 불일치")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_name, device)

    q = embed_image(model, processor, query_img, device)
    q = q / (np.linalg.norm(q)+1e-9)
    feats = feats / (np.linalg.norm(feats,axis=1,keepdims=True)+1e-9)

    sims = feats @ q
    top_idx = np.argsort(sims)[::-1][:topk]

    entries=[]
    for rank,i in enumerate(top_idx, start=1):
        p = str(paths[i])
        entries.append({
            "rank": rank,
            "path": p,                                   # 원래 경로 문자열
            "filename": _norm_name(os.path.basename(p)), # basename만 추출(매핑/재해결 키)
            "sim": float(sims[i])
        })
    return entries

# ----------------------- 경로 재해결 -----------------------
def resolve_to_base_dir(filename: str, base_dir: str) -> str:
    resolved = os.path.join(base_dir, os.path.basename(filename))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {resolved}")
    return resolved

# ----------------------- 가격 조회 유틸 -----------------------
def _try_extension_variants(fname: str, price_map: Dict[str,int]) -> Optional[int]:
    stem,_ = os.path.splitext(fname)
    for e in [".jpg",".jpeg",".png",".webp",".bmp",".jfif"]:
        key = _norm_name(f"{stem}{e}")
        if key in price_map: return price_map[key]
    return None

def lookup_price_for_filename(filename: str, price_map: Dict[str,int]) -> Optional[int]:
    key = _norm_name(filename)
    if key in price_map: return price_map[key]
    return _try_extension_variants(key, price_map)

# ----------------------- 중앙값/평균 계산 -----------------------
def median_of(values: List[int]) -> Optional[float]:
    if not values: return None
    xs = sorted(values)
    n = len(xs)
    if n % 2 == 1:
        return float(xs[n//2])
    else:
        return (xs[n//2 - 1] + xs[n//2]) / 2.0

# ----------------------- 기준 이미지 선택 -----------------------
def choose_reference_image(matched_rows: List[dict]) -> Tuple[Optional[str], str]:
    """
    matched_rows: [{filename, price:int, sim:float, resolved_path:str}, ...]  // same=True만
    규칙:
      - 1개/3개(홀수): 가격 오름차순으로 중앙값에 해당하는 항목의 이미지
        * 동일 가격이 여러 개면 그 안에서 sim 최댓값으로 타이브레이크
      - 2개(짝수): 유사도(sim) 최댓값의 이미지
    return: (ref_path:str|None, reason:str)
    """
    n = len(matched_rows)
    if n == 0:
        return None, "no_same_item"

    rows = [r for r in matched_rows if isinstance(r.get("price"), int)]
    if not rows:
        return None, "no_price_in_matches"

    if n in (1, 3):
        rows.sort(key=lambda r: (r["price"], -r["sim"]))  # 가격↑, 같은 가격이면 sim↓ 우선
        mid_idx = len(rows) // 2
        ref = rows[mid_idx]
        reason = f"odd-{n} median price={ref['price']:,}"
    elif n == 2:
        ref = max(rows, key=lambda r: r["sim"])
        reason = f"even-2 highest-sim={ref['sim']:.4f}"
    else:
        rows.sort(key=lambda r: (r["price"], -r["sim"]))
        ref = rows[len(rows)//2]
        reason = "fallback median"
    return ref["resolved_path"], reason

# ----------------------- 메인 파이프라인 -----------------------
def run_sameitem_price(image_path: str = IMAGE_PATH,
                       feats_npy: str = FEATURES_NPY,
                       paths_npy: str = PATHS_NPY,
                       csv_path: str = PRODUCTS_INFO_CSV,
                       topk: int = TOPK,
                       base_dir: str = IMAGE_BASE_DIR):

    # 0) CSV 로드 (가격/제목 맵)
    price_map, title_map = load_meta_maps(csv_path, FILENAME_COL, PRICE_COL, PRODUCT_COL)

    # 1) 유사도 검색
    entries = similar_search(image_path, feats_npy, paths_npy, MODEL_NAME, topk)

    # 2) Top3 후보 + 경로 재해결 + (제목 보강)
    candidates = []
    for c in entries[:3]:
        resolved = resolve_to_base_dir(c["filename"], base_dir)
        nc = dict(c)
        nc["resolved_path"] = resolved
        nc["title"] = title_map.get(c["filename"]) if title_map else None
        candidates.append(nc)

    print("\n=== 🔍 유사도 Top3(판정 대상) ===")
    for c in candidates:
        print(f"{c['rank']:>2}) sim={c['sim']:.4f} | file={c['filename']} | resolved={c['resolved_path']} | title={c.get('title') or '-'}")

    # 3) LLM 동일품 판정 (true/false 배열)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("환경변수 ANTHROPIC_API_KEY 없음")
    client = Anthropic(api_key=api_key)
    judge = SameItemJudgeClaude(client)
    same_flags = judge.judge(image_path, candidates)

    print("\n=== ✅ 동일품 판정(JSON) ===")
    print(json.dumps({"same": same_flags}, ensure_ascii=False, indent=2))

    # 4) same=True 후보들의 가격/유사도/경로를 모아두기
    matched_rows: List[dict] = []
    matched_prices: List[int] = []
    matched_files: List[str] = []

    for c, is_same in zip(candidates, same_flags):
        if is_same:
            fn = _norm_name(c["filename"])
            price = lookup_price_for_filename(fn, price_map)
            if price is not None:
                matched_rows.append({
                    "filename": fn,
                    "price": int(price),
                    "sim": float(c["sim"]),
                    "resolved_path": c["resolved_path"],
                })
                matched_prices.append(int(price))
                matched_files.append(fn)

    count = len(matched_prices)

    # 5) 규칙 적용: 1/3개 → 중앙값, 2개 → 평균, 0개 → 판단 불가
    print("\n=== 💰 기준가 산출 ===")
    if count == 0:
        print("결과: 판단 불가 (same=True가 없거나, CSV에 가격이 없음)")
        return {"status":"unavailable","reason":"no_same_item_or_price","matched":[]}
    elif count == 1 or count == 3:
        base_price = median_of(matched_prices)
        print(f"same 매칭 {count}개 → 중앙값 기준가 = {int(base_price):,}원")
    elif count == 2:
        base_price = sum(matched_prices)/2.0
        print(f"same 매칭 2개 → 평균 기준가 = {int(round(base_price)):,}원")
    else:
        base_price = median_of(matched_prices)

    print(f"- 매칭 파일: {matched_files}")
    print(f"- 매칭 가격: {[f'{p:,}원' for p in matched_prices]}")

    # 6) ★ 기준 이미지 1장 선택 (홀수=중앙값 대응, 짝수=유사도 1위)
    ref_path, ref_reason = choose_reference_image(matched_rows)
    ref_bytes = None
    if ref_path and os.path.isfile(ref_path):
        with open(ref_path, "rb") as f:
            ref_bytes = f.read()

    print("\n=== 🎯 기준 이미지 선택 ===")
    if ref_path:
        print(f"- ref_path: {ref_path}")
        print(f"- reason  : {ref_reason}")
    else:
        print(f"- 기준 이미지 선택 실패: {ref_reason}")

    return {
        "status":"ok",
        "matched_count": count,
        "matched_files": matched_files,
        "matched_prices": matched_prices,
        "baseline_price": int(round(base_price)),
        "ref_image_path": ref_path,
        "ref_image_bytes": ref_bytes,        # supervisor/agents가 바로 사용 가능
        "ref_selection_reason": ref_reason,
        "same_flags": same_flags,            # 디버그용
        "candidates": candidates             # 디버그/표시용
    }

if __name__ == "__main__":
    run_sameitem_price()
