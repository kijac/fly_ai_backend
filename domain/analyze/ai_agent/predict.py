# predict.py  (Top-1 유사도만 사용 / 마지막 2단계 경로로 신품 이미지 탐색 / CSV: product_name, retail_price, used_price_avg, retail_link)
import os
import csv
import re
import json
from typing import Dict, Optional, List

import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

from dotenv import load_dotenv

load_dotenv()

# ======================= 경로 설정 (상대경로 우선) =======================
# 프로젝트 루트(이 파일 기준) 추정
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _env_or_join(env_key: str, default_rel: str) -> str:
    v = os.getenv(env_key)
    if v and os.path.exists(v):
        return v
    p = os.path.join(ROOT_DIR, default_rel)
    return p

# 테스트 입력 이미지는 test/ 아래 경로 (사용자 입력 전용)
IMAGE_PATH   = _env_or_join("IMAGE_PATH",   os.path.join("test", "헬로카봇_로드세이버", "thunder_0074.webp"))

# 인덱스는 "train" 데이터로 생성됨
FEATURES_NPY = _env_or_join("FEATURES_NPY", os.path.join("toys_index", "train_features_large_patch14-336.npy"))
PATHS_NPY    = _env_or_join("PATHS_NPY",    os.path.join("toys_index", "train_paths_large_patch14-336.npy"))

# CLIP 거대(336) 모델
MODEL_NAME   = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14-336")

# 신품(비교 기준) 이미지가 존재하는 베이스 폴더 (train)
# paths.npy의 원본 경로에서 마지막 두 파트 <폴더>/<파일>을 추출하여 여기 아래에서 찾음
IMAGE_BASE_DIR = _env_or_join("IMAGE_BASE_DIR", "train")

# CSV(신규 스키마): product_name, retail_price(신품가), used_price_avg(중고 평균가), retail_link(옵션)
PRODUCTS_INFO_CSV  = _env_or_join("PRODUCTS_INFO_CSV", "carbot_data_final.csv")
COL_PRODUCT_NAME   = "product_name"
COL_RETAIL_PRICE   = "retail_price"
COL_USED_AVG       = "used_price_avg"
COL_RETAIL_LINK    = "retail_link"

# 기타
NORMALIZE_FILENAMES = True
TOPK = 1  # 호환성용 상수 (외부에서 import)
# =======================================================================


# ----------------------- 문자열 정규화 -----------------------
def _norm_name(path_or_name: str) -> str:
    name = os.path.basename(str(path_or_name)).strip()
    return name.lower() if NORMALIZE_FILENAMES else name

def _norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = os.path.splitext(s)[0]  # 확장자 제거
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def _last2_relpath(p: str) -> str:
    """경로의 마지막 두 파트를 '<dir>/<file>' 형태로 반환. 파트가 1개면 파일명만."""
    p = os.path.normpath(str(p))
    parts = re.split(r"[\\/]+", p)
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1])
    return parts[-1]

def _parent_dir_name(p: str) -> str:
    p = os.path.normpath(str(p))
    return os.path.basename(os.path.dirname(p))


# ----------------------- CSV 카탈로그 로드 -----------------------
def _to_int_price(x) -> Optional[int]:
    x = str(x or "").replace(",", "").replace("원", "").strip()
    if not re.search(r"[0-9]", x):
        return None
    try:
        return int(float(x))
    except ValueError:
        digits = "".join(ch for ch in x if x and ch.isdigit())
        return int(digits) if digits else None

def _build_catalog(csv_path: str):
    """CSV: product_name, retail_price, used_price_avg, (retail_link)"""
    if not os.path.isfile(csv_path):
        return {}, {}, {}, {}

    encodings = ("utf-8-sig", "utf-8", "cp949")
    last_err = None
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                need = {COL_PRODUCT_NAME, COL_RETAIL_PRICE, COL_USED_AVG}
                if not need.issubset(fields):
                    raise KeyError(f"CSV columns missing: need {need}, got {fields}")

                retail_by_key: Dict[str, int] = {}
                usedavg_by_key: Dict[str, int] = {}
                name_by_key: Dict[str, str] = {}
                link_by_key: Dict[str, str] = {}

                for row in reader:
                    name = (row.get(COL_PRODUCT_NAME) or "").strip()
                    if not name:
                        continue
                    key = _norm_key(name)
                    rp = _to_int_price(row.get(COL_RETAIL_PRICE))
                    up = _to_int_price(row.get(COL_USED_AVG))
                    if rp is not None:
                        retail_by_key[key] = rp
                    if up is not None:
                        usedavg_by_key[key] = up
                    name_by_key[key] = name
                    if COL_RETAIL_LINK in fields:
                        link_by_key[key] = (row.get(COL_RETAIL_LINK) or "").strip()
                return retail_by_key, usedavg_by_key, name_by_key, link_by_key
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV 인코딩 실패: {last_err}")


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

def embed_texts(model, processor, texts: List[str], device) -> np.ndarray:
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


# ----------------------- 경로/IO -----------------------
def resolve_from_last2(original_path: str, base_dir: Optional[str]) -> Optional[str]:
    """
    base_dir/<last-folder>/<filename> 우선 탐색, 없으면 base_dir/<filename> 폴백.
    """
    if not base_dir:
        return None
    last2 = _last2_relpath(original_path)                 # <dir>/<file> 또는 <file>
    cand = os.path.join(base_dir, last2)
    if os.path.isfile(cand):
        return cand
    just_file = os.path.join(base_dir, os.path.basename(original_path))
    if os.path.isfile(just_file):
        return just_file
    return None

def _read_bytes_or_none(path: Optional[str]) -> Optional[bytes]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


# ----------------------- 유사도 검색 (Top-1) -----------------------
def similar_search_top1(query_img: str, feats_npy: str, paths_npy: str,
                        model_name: str) -> dict:
    if not os.path.isfile(query_img):
        raise FileNotFoundError(f"IMAGE_PATH not found: {query_img}")
    if not os.path.isfile(feats_npy) or not os.path.isfile(paths_npy):
        raise FileNotFoundError("features/paths npy 필요")

    feats = np.load(feats_npy).astype("float32")
    paths = np.load(paths_npy, allow_pickle=True)
    if feats.ndim != 2 or feats.shape[0] != len(paths):
        raise ValueError("features/paths 크기 불일치")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_name, device)

    q = embed_image(model, processor, query_img, device)
    q = q / (np.linalg.norm(q) + 1e-9)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

    sims = feats @ q
    i = int(np.argmax(sims))
    return {
        "rank": 1,
        "path": str(paths[i]),                                  # 원본 경로(인덱스 생성 시)
        "filename": _norm_name(os.path.basename(str(paths[i]))),
        "sim": float(sims[i]),
    }


# ----------------------- 매칭 유틸 -----------------------
def _best_match_key_by_tokens(tokens: List[str], keys: List[str]) -> Optional[str]:
    """여러 토큰(파일명 stem, 상위 폴더명 등)으로 product_name 매칭 시도."""
    if not keys:
        return None
    # 1) 완전 일치 우선
    n_tokens = [_norm_key(t) for t in tokens if t]
    for t in n_tokens:
        if t in keys:
            return t
    # 2) 부분 포함(긴 키 우선)
    cands = []
    for t in n_tokens:
        cands += [k for k in keys if (t in k or k in t)]
    if cands:
        return sorted(set(cands), key=len, reverse=True)[0]
    return None

def _best_match_key_by_clip(query_img_path: str, name_by_key: Dict[str, str]) -> Optional[str]:
    """파일명/폴더명 매칭 실패 시: CLIP 이미지↔텍스트 매칭으로 product_name 찾기"""
    if not name_by_key:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(MODEL_NAME, device)
    qvec = embed_image(model, processor, query_img_path, device)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-9)

    keys = list(name_by_key.keys())
    names = [name_by_key[k] for k in keys]
    tfeats = embed_texts(model, processor, names, device)   # (N, D), 이미 정규화됨
    sims = tfeats @ qvec                                    # (N,)
    idx = int(np.argmax(sims))
    return keys[idx] if len(keys) else None


# ----------------------- 메인 파이프라인 -----------------------
def run_sameitem_price(
    image_path: str = IMAGE_PATH,
    feats_npy: str = FEATURES_NPY,
    paths_npy: str = PATHS_NPY,
    csv_path: str = PRODUCTS_INFO_CSV,
    base_dir: str = IMAGE_BASE_DIR,
    topk: int = TOPK,    # 호환성 파라미터(내부에서는 Top-1 고정)
) -> dict:
    # 0) 카탈로그 로드
    retail_by_key, usedavg_by_key, name_by_key, link_by_key = _build_catalog(csv_path)

    # 1) 유사도 Top-1
    top1 = similar_search_top1(image_path, feats_npy, paths_npy, MODEL_NAME)
    orig_path = top1["path"]
    fname     = top1["filename"]
    stem      = os.path.splitext(fname)[0]
    parent    = _parent_dir_name(orig_path)

    # 2) ★ 신품(비교 기준) 이미지 경로: base_dir/<last-folder>/<file> → 없으면 base_dir/<file>
    ref_new_path = resolve_from_last2(orig_path, base_dir)

    # 3) product_name 매칭
    all_keys = list(name_by_key.keys())
    match_key = _best_match_key_by_tokens([stem, parent], all_keys)
    reason = "tokens-match"
    if not match_key:
        match_key = _best_match_key_by_clip(image_path, name_by_key)
        reason = "clip-text-match" if match_key else "no-catalog-match"

    if not match_key:
        return {
            "status": "unavailable",
            "matched_count": 0,
            "matched_files": [],
            "matched_prices": [],
            "baseline_price": None,
            "retail_price": None,
            "toy_name": None,
            "retail_link": None,
            "ref_new_image_path": ref_new_path,
            "ref_new_image_bytes": _read_bytes_or_none(ref_new_path),
            # 중고 이미지는 비교에 사용하지 않음(명시적으로 None)
            "ref_image_path": None,
            "ref_image_bytes": None,
            "ref_selection_reason": f"top1-{reason}",
            "same_flags": [],
            "candidates": [{ **top1, "resolved_path": ref_new_path or "", "title": None }],
        }

    toy_name = name_by_key.get(match_key)
    used_avg = usedavg_by_key.get(match_key)
    retail   = retail_by_key.get(match_key)

    if used_avg is None and retail is None:
        return {
            "status": "unavailable",
            "matched_count": 0,
            "matched_files": [],
            "matched_prices": [],
            "baseline_price": None,
            "retail_price": None,
            "toy_name": toy_name,
            "retail_link": link_by_key.get(match_key),
            "ref_new_image_path": ref_new_path,
            "ref_new_image_bytes": _read_bytes_or_none(ref_new_path),
            "ref_image_path": None,
            "ref_image_bytes": None,
            "ref_selection_reason": f"top1-{reason}-no-prices",
            "same_flags": [],
            "candidates": [{ **top1, "resolved_path": ref_new_path or "", "title": toy_name }],
        }

    matched_count = 1 if used_avg is not None else 0
    matched_files = [fname] if matched_count == 1 else []
    matched_prices = [int(used_avg)] if matched_count == 1 else []

    return {
        "status": "ok",
        "matched_count": matched_count,
        "matched_files": matched_files,
        "matched_prices": matched_prices,
        "baseline_price": int(used_avg) if used_avg is not None else None,  # 중고 기준가
        "retail_price": int(retail) if retail is not None else None,        # 신제품가
        "toy_name": toy_name,
        "retail_link": link_by_key.get(match_key),
        # ★ 비교 기준: 신품 이미지(train에서 last-2 경로로 탐색)
        "ref_new_image_path": ref_new_path,
        "ref_new_image_bytes": _read_bytes_or_none(ref_new_path),
        # 중고 참조는 사용하지 않음
        "ref_image_path": None,
        "ref_image_bytes": None,
        "ref_selection_reason": f"top1-{reason}-last2",
        "same_flags": [],
        "candidates": [{ **top1, "resolved_path": ref_new_path or "", "title": toy_name }],
    }


# ----------------------- 스크립트 단독 실행 -----------------------
if __name__ == "__main__":
    out = run_sameitem_price()
    summary = {
        "status": out.get("status"),
        "toy_name": out.get("toy_name"),
        "baseline_price": out.get("baseline_price"),
        "retail_price": out.get("retail_price"),
        "matched_files": out.get("matched_files"),
        "ref_new_image_path": out.get("ref_new_image_path"),
        "ref_selection_reason": out.get("ref_selection_reason"),
        "top1_sim": (out.get("candidates") or [{}])[0].get("sim"),
    }
    print(json.dumps(summary, ensure_ascii=False))
