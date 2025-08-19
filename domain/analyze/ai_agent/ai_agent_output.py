# ai_agent_output.py  (최종 JSON: toy_name, retail_price, purchase_price, soil, damage, toy_type, material)
import os
import re
import json
import base64
import argparse
import tempfile
import atexit
from typing import List, Optional

import pandas as pd
import ntpath  # for Windows-style basename

# === (1) 기준가 파이프라인 ===
from predict import (
    run_sameitem_price,
    IMAGE_BASE_DIR, FEATURES_NPY, PATHS_NPY, PRODUCTS_INFO_CSV, TOPK, IMAGE_PATH
)

# === (2) 상태 판별 에이전트 ===
from ai_agent.supervisor_agent import SupervisorAgent


# -----------------------
# 유틸
# -----------------------
def _read_bytes_or_none(p: Optional[str]) -> Optional[bytes]:
    if not p:
        return None
    with open(p, "rb") as f:
        return f.read()

def get_title_from_csv(csv_path: Optional[str], ref_image_path: Optional[str]) -> str:
    """
    (백업용) 과거 CSV(title, price, thumbnail_filename...) 스키마 대비용.
    신규 스키마에서는 predict가 toy_name을 직접 주므로 보통은 사용되지 않음.
    """
    if not csv_path or not ref_image_path or not os.path.exists(csv_path):
        return "불명"
    try:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return "불명"

    if df is None or df.empty:
        return "불명"

    # 신규 스키마일 경우 product_name이 있으면 파일명으로 매칭 시도하지 않고 불명 처리
    if "product_name" in df.columns:
        return "불명"

    if "thumbnail_filename" not in df.columns or "title" not in df.columns:
        return "불명"

    ref_base = ntpath.basename(str(ref_image_path)).strip().lower()
    ser = df["thumbnail_filename"].astype(str).str.strip().str.lower()
    mask = ser == ref_base
    if mask.any():
        return str(df.loc[mask, "title"].iloc[0]).strip()
    # 확장자 제거 매칭 보조
    ref_stem = os.path.splitext(ref_base)[0]
    mask2 = ser.apply(lambda x: os.path.splitext(x)[0]) == ref_stem
    if mask2.any():
        return str(df.loc[mask2, "title"].iloc[0]).strip()
    return "불명"


def _pick_ref_bytes(baseline: dict) -> Optional[bytes]:
    """새 predict가 'ref_new_image_bytes'를 제공하면 그걸 사용."""
    return baseline.get("ref_new_image_bytes") or baseline.get("ref_image_bytes")

def _pick_ref_path(baseline: dict) -> Optional[str]:
    """경로도 동일한 우선순위로 선택."""
    return baseline.get("ref_new_image_path") or baseline.get("ref_image_path")


# -----------------------
# 상태 판별
# -----------------------
def run_condition_agents(
    front_path: str,
    left_path: Optional[str] = None,
    rear_path: Optional[str] = None,
    right_path: Optional[str] = None,
    ref_image_bytes: Optional[bytes] = None,  # 기준 이미지 1장 (신상품)
) -> dict:
    front_b = _read_bytes_or_none(front_path)
    left_b  = _read_bytes_or_none(left_path)  if left_path  else front_b
    rear_b  = _read_bytes_or_none(rear_path)  if rear_path  else front_b
    right_b = _read_bytes_or_none(right_path) if right_path else front_b

    if ref_image_bytes is None:
        return {
            "장난감 종류": "others",
            "재료": "unknown",
            "오염도": "불명",
            "파손": "불명",
            "크기": None,
            "건전지 여부": None,
            "토큰 사용량": {"total": 0},
        }

    sup = SupervisorAgent()
    result = sup.process(front_b, left_b, rear_b, right_b, ref_image_bytes)

    out = {
        "장난감 종류": result.get("장난감 종류", "others"),
        "재료": result.get("재료", "unknown"),
        "오염도": result.get("오염도", "불명"),
        "파손": result.get("파손", "불명"),
        "크기": result.get("크기"),
        "건전지 여부": result.get("건전지 여부"),
        "토큰 사용량": result.get("토큰 사용량", {}),
    }
    # DamageAgent가 주는 등급/보조 필드 그대로 전달
    if "파손 상대등급" in result:
        out["파손 상대등급"] = result["파손 상대등급"]   # 'A'..'E'
    if "relative_steps" in result:
        out["파손 단계차"] = result["relative_steps"]    # 0..4 (있다면)
    if "파손 점수(중고)" in result:
        out["파손 점수(중고)"] = result["파손 점수(중고)"]  # 0..4 (있다면)
    if "파손 점수(신품)" in result:
        out["파손 점수(신품)"] = result["파손 점수(신품)"]  # 0..4 (있다면)

    # SoilAgent: 등급/점수 필드 반영 (신규 포맷)
    if "오염 상대등급" in result:
        out["오염 상대등급"] = result["오염 상대등급"]       # 'A'..'E'
    if "오염 점수(중고)" in result:
        out["오염 점수(중고)"] = result["오염 점수(중고)"]   # 0..4
    if "오염 점수(신품)" in result:
        out["오염 점수(신품)"] = result["오염 점수(신품)"]   # 0..4

    # (구버전 호환) -2..+2가 있으면 같이 전달
    if "기준 대비(오염)" in result:
        out["기준 대비(오염)"] = result["기준 대비(오염)"]
    return out


# -----------------------
# 보조: -2..+2 → A~E (호환용)
# -----------------------
def _level_to_grade(level: int) -> str:
    """
    2→A, 1→B, 0→C, -1→D, -2→E
    """
    try:
        lv = int(level)
    except Exception:
        lv = 0
    if lv >= 2: return "A"
    if lv == 1: return "B"
    if lv == 0: return "C"
    if lv == -1: return "D"
    return "E"  # lv <= -2


# -----------------------
# 등급(A~E) → 가격보정 계수
# -----------------------
def _grade_to_multiplier(grade: Optional[str]) -> float:
    """
    A(신품 수준)→1.00, B→0.95, C→0.90, D→0.80, E→0.65
    """
    g = (grade or "").strip().upper()
    table = {"A": 1.00, "B": 0.95, "C": 0.90, "D": 0.80, "E": 0.65}
    return table.get(g, 0.90)  # 알 수 없는 경우 보수적으로 C 수준


# -----------------------
# 통합 파이프라인
# -----------------------
def run_full_pipeline(
    query_image_path: str,
    feats_npy: str = FEATURES_NPY,
    paths_npy: str = PATHS_NPY,
    csv_path: str = PRODUCTS_INFO_CSV,
    base_dir: str = IMAGE_BASE_DIR,
    topk: int = TOPK,  # predict가 Top-1로 강제하더라도 옵션은 남겨둠(호환)
    left: Optional[str] = None,
    rear: Optional[str] = None,
    right: Optional[str] = None,
) -> dict:
    baseline = run_sameitem_price(
        image_path=query_image_path,
        feats_npy=feats_npy,
        paths_npy=paths_npy,
        csv_path=csv_path,
        base_dir=base_dir,
        topk=topk,
    )

    ref_bytes = _pick_ref_bytes(baseline)

    if baseline.get("status") != "ok" or not ref_bytes:
        attrs = run_condition_agents(
            front_path=query_image_path,
            left_path=left, rear_path=rear, right_path=right,
            ref_image_bytes=None
        )
        return {
            "기준가_status": baseline.get("status"),
            "baseline_price": baseline.get("baseline_price"),
            "retail_price": baseline.get("retail_price"),
            "toy_name": baseline.get("toy_name"),
            "기준가_매칭개수": baseline.get("matched_count", 0),
            "매칭파일": baseline.get("matched_files", []),
            "매칭가격": baseline.get("matched_prices", []),
            "종류": attrs.get("장난감 종류"),
            "소재": attrs.get("재료"),
            "오염도": attrs.get("오염도"),
            "파손도": attrs.get("파손"),
            "파손 상대등급": attrs.get("파손 상대등급"),
            "파손 단계차": attrs.get("파손 단계차"),
            "파손 점수(중고)": attrs.get("파손 점수(중고)"),
            "파손 점수(신품)": attrs.get("파손 점수(신품)"),
            "오염 상대등급": attrs.get("오염 상대등급"),
            "오염 점수(중고)": attrs.get("오염 점수(중고)"),
            "오염 점수(신품)": attrs.get("오염 점수(신품)"),
            "크기": attrs.get("크기"),
            "건전지": attrs.get("건전지 여부"),
            "기준 대비(오염)": attrs.get("기준 대비(오염)"),
            "에이전트_토큰합": attrs.get("토큰 사용량", {}).get("total", 0),
            "ref_image_path": _pick_ref_path(baseline),          # 그대로 전달
            "ref_selection_reason": baseline.get("ref_selection_reason"),
            "csv_path": csv_path,
        }

    attrs = run_condition_agents(
        front_path=query_image_path,
        left_path=left, rear_path=rear, right_path=right,
        ref_image_bytes=ref_bytes
    )

    out = {
        "기준가_status": baseline.get("status"),
        "baseline_price": baseline.get("baseline_price"),
        "retail_price": baseline.get("retail_price"),
        "toy_name": baseline.get("toy_name"),
        "기준가_매칭개수": baseline.get("matched_count", 0),

        "종류": attrs.get("장난감 종류"),
        "소재": attrs.get("재료"),
        "오염도": attrs.get("오염도"),
        "파손도": attrs.get("파손"),
        "파손 상대등급": attrs.get("파손 상대등급"),
        "파손 단계차": attrs.get("파손 단계차"),
        "파손 점수(중고)": attrs.get("파손 점수(중고)"),
        "파손 점수(신품)": attrs.get("파손 점수(신품)"),
        "오염 상대등급": attrs.get("오염 상대등급"),
        "오염 점수(중고)": attrs.get("오염 점수(중고)"),
        "오염 점수(신품)": attrs.get("오염 점수(신품)"),
        "크기": attrs.get("크기"),
        "건전지": attrs.get("건전지 여부"),

        "기준 대비(오염)": attrs.get("기준 대비(오염)"),

        "매칭파일": baseline.get("matched_files", []),
        "매칭가격": baseline.get("matched_prices", []),

        "에이전트_토큰합": attrs.get("토큰 사용량", {}).get("total", 0),
        "ref_image_path": _pick_ref_path(baseline),
        "ref_selection_reason": baseline.get("ref_selection_reason"),
        "csv_path": csv_path,
    }
    return out


# -----------------------
# 추천가(매입가) 산출 — 파손/오염 모두 A~E 등급 사용
# -----------------------
def _round_to_unit(x: float, unit: int = 100) -> int:
    return int(round(x / unit) * unit)

def estimate_purchase_price_by_grade(
    baseline_price: int,
    *,
    damage_grade: Optional[str],        # 'A'..'E'
    soil_grade: Optional[str],          # 'A'..'E'
    base_bias: float = 0.7
) -> dict:
    dmg_mult  = _grade_to_multiplier(damage_grade)
    soil_mult = _grade_to_multiplier(soil_grade)

    biased_baseline = baseline_price * float(base_bias)
    total_mult = dmg_mult * soil_mult
    raw_price = biased_baseline * total_mult
    recommended = _round_to_unit(raw_price, 100)

    return {
        "recommended_price": int(recommended),
        "effective_multiplier_vs_baseline": round(base_bias * total_mult, 4),
        "damage_multiplier": dmg_mult,
        "soil_multiplier": soil_mult,
    }


# -----------------------
# data URL 지원 (CLI)
# -----------------------
_DATAURL_RE = re.compile(r"^data:(?P<mime>[\w\-/+\.]+)?;base64,(?P<b64>[A-Za-z0-9+/=]+)$")
_TEMP_FILES: List[str] = []

def dataurl_to_bytes(dataurl: Optional[str]) -> Optional[bytes]:
    if not dataurl:
        return None
    m = _DATAURL_RE.match(dataurl.strip())
    if not m:
        return None
    try:
        return base64.b64decode(m.group("b64"))
    except Exception:
        return None

def bytes_to_tempfile(b: bytes, suffix: str = ".jpg") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(b)
    f.flush()
    f.close()
    _TEMP_FILES.append(f.name)
    return f.name

def _cleanup_tempfiles():
    for p in _TEMP_FILES:
        try:
            os.remove(p)
        except Exception:
            pass
atexit.register(_cleanup_tempfiles)


# -----------------------
# CLI
# -----------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="기준가 + 상태판별 통합 파이프라인(JSON 출력)")
    # 파일 경로 입력
    parser.add_argument("--query", type=str, default=IMAGE_PATH, help="메인 이미지(앞면) - 파일 경로")
    parser.add_argument("--left", type=str, default=None, help="왼쪽 이미지 - 파일 경로")
    parser.add_argument("--rear", type=str, default=None, help="뒷면 이미지 - 파일 경로")
    parser.add_argument("--right", type=str, default=None, help="오른쪽 이미지 - 파일 경로")

    # data URL 입력 (웹/앱에서 바로 전달)
    parser.add_argument("--query-dataurl", type=str, default=None, help="메인 이미지 data URL (data:image/jpeg;base64,...)")
    parser.add_argument("--left-dataurl", type=str, default=None, help="왼쪽 이미지 data URL")
    parser.add_argument("--rear-dataurl", type=str, default=None, help="뒷면 이미지 data URL")
    parser.add_argument("--right-dataurl", type=str, default=None, help="오른쪽 이미지 data URL")

    parser.add_argument("--feats", type=str, default=FEATURES_NPY, help="features.npy")
    parser.add_argument("--paths", type=str, default=PATHS_NPY, help="paths.npy")
    parser.add_argument("--csv",   type=str, default=PRODUCTS_INFO_CSV, help="가격 CSV 경로")
    parser.add_argument("--base",  type=str, default=IMAGE_BASE_DIR, help="후보 실이미지 폴더")
    parser.add_argument("--topk",  type=int, default=TOPK, help="유사도 검색 상위 K (predict 내부에서 Top-1 강제 가능)")
    parser.add_argument("--base-bias", type=float, default=0.7, help="매입가 기준가 바이어스 (기본 0.7)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # data URL 우선 처리 → 임시 경로 변환
    query_path = args.query
    left_path  = args.left
    rear_path  = args.rear
    right_path = args.right

    if args.query_dataurl:
        qb = dataurl_to_bytes(args.query_dataurl)
        if qb: query_path = bytes_to_tempfile(qb, ".jpg")
    if args.left_dataurl:
        lb = dataurl_to_bytes(args.left_dataurl)
        if lb: left_path = bytes_to_tempfile(lb, ".jpg")
    if args.rear_dataurl:
        rb = dataurl_to_bytes(args.rear_dataurl)
        if rb: rear_path = bytes_to_tempfile(rb, ".jpg")
    if args.right_dataurl:
        rb = dataurl_to_bytes(args.right_dataurl)
        if rb: right_path = bytes_to_tempfile(rb, ".jpg")

    final_json = run_full_pipeline_and_get_final_json(
        query_image_path=query_path,
        feats_npy=args.feats,
        paths_npy=args.paths,
        csv_path=args.csv,
        base_dir=args.base,
        topk=args.topk,
        left=left_path, rear=rear_path, right=right_path,
        base_bias=args.base_bias,
    )
    
    # CLI에서는 JSON 출력, 함수 호출에서는 반환
    if __name__ == "__main__":
        print(json.dumps(final_json, ensure_ascii=False))
    
    return final_json


def run_full_pipeline_and_get_final_json(
    query_image_path: str,
    feats_npy: str = FEATURES_NPY,
    paths_npy: str = PATHS_NPY,
    csv_path: str = PRODUCTS_INFO_CSV,
    base_dir: str = IMAGE_BASE_DIR,
    topk: int = TOPK,
    left: Optional[str] = None,
    rear: Optional[str] = None,
    right: Optional[str] = None,
    base_bias: float = 0.7,
) -> dict:
    """
    run_full_pipeline을 실행하고 최종 JSON 형태로 반환하는 함수
    """
    out = run_full_pipeline(
        query_image_path=query_image_path,
        feats_npy=feats_npy,
        paths_npy=paths_npy,
        csv_path=csv_path,
        base_dir=base_dir,
        topk=topk,
        left=left, rear=rear, right=right,
    )

    # ---- 최종 JSON 스키마 구성 ----
    baseline_price = out.get("baseline_price")  # 중고 기준가(used avg)
    retail_price   = out.get("retail_price")    # 신제품가
    toy_name       = out.get("toy_name")

    # Fallback: predict가 toy_name을 못 준 경우 (구형 CSV 대비)
    if not toy_name:
        ref_path_for_title = out.get("ref_image_path")
        toy_name = get_title_from_csv(out.get("csv_path"), ref_path_for_title)

    # 에이전트 결과 (등급 우선)
    damage_grade = out.get("파손 상대등급")                      # 'A'..'E'
    # Soil: 우선 등급, 없으면 -2..+2 → 등급으로 변환, 그래도 없으면 텍스트
    soil_grade = out.get("오염 상대등급")
    if not soil_grade:
        lvl = out.get("기준 대비(오염)")
        if lvl is not None:
            try:
                soil_grade = _level_to_grade(int(lvl))
            except Exception:
                soil_grade = None

    # 매입가 계산 (baseline_price가 있을 때만), 파손/오염 모두 등급 기반
    purchase_price = None
    if baseline_price is not None:
        rec = estimate_purchase_price_by_grade(
            baseline_price=int(baseline_price),
            damage_grade=damage_grade,
            soil_grade=soil_grade,
            base_bias=base_bias,
        )
        purchase_price = int(rec["recommended_price"])

    final_json = {
        "toy_name": toy_name,
        "retail_price": int(retail_price) if retail_price is not None else None,
        "purchase_price": purchase_price,
        "soil": soil_grade if soil_grade else out.get("오염도"),
        "damage": damage_grade if damage_grade else out.get("파손도"),
        "toy_type": out.get("종류"),
        "material": out.get("소재"),
    }
    
    return final_json


if __name__ == "__main__":
    main()
