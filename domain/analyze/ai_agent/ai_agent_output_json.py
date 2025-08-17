# ai_agent_output_json.py
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

# ======================= 모델 설정 =======================
# 사용법: 아래 변수들만 수정하면 모델이 자동으로 변경됩니다
CLIP_MODEL = "openai/clip-vit-large-patch14"  # CLIP 모델 변경 시
CLAUDE_MODEL = "claude-sonnet-4-20250514"     # Claude 모델 변경 시

# 모델별 특징 벡터 매핑 (새로운 CLIP 모델 추가 시 여기에 추가)
CLIP_FEATURES_MAP = {
    "openai/clip-vit-large-patch14": {
        "features": "toys_index/used_features_large_patch14.npy",
        "paths": "toys_index/used_paths_large_patch14.npy"
    },
    "openai/clip-vit-base-patch16": {
        "features": "toys_index/used_features_base_patch16.npy",
        "paths": "toys_index/used_paths_base_patch16.npy"
    }
}

# 사용 가능한 모델 예시:
# CLIP_MODEL = "openai/clip-vit-base-patch16"  # 더 빠른 모델
# CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # 최신 모델
# ==========================================================

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
    CSV 스키마(title, price, thumbnail_filename, page, url) 기준으로
    reference 이미지의 basename과 thumbnail_filename이 일치하는 행의 title을 반환.
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


# -----------------------
# 상태 판별
# -----------------------
def run_condition_agents(
    front_path: str,
    left_path: Optional[str] = None,
    rear_path: Optional[str] = None,
    right_path: Optional[str] = None,
    ref_image_bytes: Optional[bytes] = None,  # 기준 이미지 1장
) -> dict:
    front_b = _read_bytes_or_none(front_path)
    left_b  = _read_bytes_or_none(left_path)  if left_path  else front_b
    rear_b  = _read_bytes_or_none(rear_path)  if rear_path  else front_b
    right_b = _read_bytes_or_none(right_path) if right_path else front_b

    if ref_image_bytes is None:
        return {
            "장난감 종류": "불명",
            "재료": "불명",
            "오염도": "불명",
            "파손": "불명",
            "크기": None,
            "건전지 여부": None,
            "토큰 사용량": {"total": 0},
        }

    sup = SupervisorAgent(claude_model=CLAUDE_MODEL)
    result = sup.process(front_b, left_b, rear_b, right_b, ref_image_bytes)

    out = {
        "장난감 종류": result.get("장난감 종류", "불명"),
        "재료": result.get("재료", "불명"),
        "오염도": result.get("오염도", "불명"),
        "파손": result.get("파손", "불명"),
        "크기": result.get("크기"),
        "건전지 여부": result.get("건전지 여부"),
        "토큰 사용량": result.get("토큰 사용량", {}),
    }
    if "기준 대비(파손)" in result:
        out["기준 대비(파손)"] = result["기준 대비(파손)"]
    if "기준 대비(오염)" in result:
        out["기준 대비(오염)"] = result["기준 대비(오염)"]
    return out


# -----------------------
# 통합 파이프라인
# -----------------------
def run_full_pipeline(
    query_image_path: str,
    feats_npy: str = FEATURES_NPY,
    paths_npy: str = PATHS_NPY,
    csv_path: str = PRODUCTS_INFO_CSV,
    base_dir: str = IMAGE_BASE_DIR,
    topk: int = TOPK,
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

    if baseline.get("status") != "ok" or not baseline.get("ref_image_bytes"):
        attrs = run_condition_agents(
            front_path=query_image_path,
            left_path=left, rear_path=rear, right_path=right,
            ref_image_bytes=None
        )
        return {
            "기준가_status": baseline.get("status"),
            "기준가": None,
            "기준가_매칭개수": baseline.get("matched_count", 0),
            "매칭파일": baseline.get("matched_files", []),
            "매칭가격": baseline.get("matched_prices", []),
            "종류": attrs.get("장난감 종류"),
            "소재": attrs.get("재료"),
            "오염도": attrs.get("오염도"),
            "파손도": attrs.get("파손"),
            "크기": attrs.get("크기"),
            "건전지": attrs.get("건전지 여부"),
            "기준 대비(파손)": attrs.get("기준 대비(파손)"),
            "기준 대비(오염)": attrs.get("기준 대비(오염)"),
            "에이전트_토큰합": attrs.get("토큰 사용량", {}).get("total", 0),
            "ref_image_path": None,
            "ref_selection_reason": baseline.get("ref_selection_reason"),
            "csv_path": csv_path,
        }

    attrs = run_condition_agents(
        front_path=query_image_path,
        left_path=left, rear_path=rear, right_path=right,
        ref_image_bytes=baseline["ref_image_bytes"]
    )

    out = {
        "기준가_status": baseline.get("status"),
        "기준가": baseline.get("baseline_price"),
        "기준가_매칭개수": baseline.get("matched_count", 0),

        "종류": attrs.get("장난감 종류"),
        "소재": attrs.get("재료"),
        "오염도": attrs.get("오염도"),
        "파손도": attrs.get("파손"),
        "크기": attrs.get("크기"),
        "건전지": attrs.get("건전지 여부"),

        "기준 대비(파손)": attrs.get("기준 대비(파손)"),
        "기준 대비(오염)": attrs.get("기준 대비(오염)"),

        "매칭파일": baseline.get("matched_files", []),
        "매칭가격": baseline.get("matched_prices", []),

        "에이전트_토큰합": attrs.get("토큰 사용량", {}).get("total", 0),
        "ref_image_path": baseline.get("ref_image_path"),
        "ref_selection_reason": baseline.get("ref_selection_reason"),
        "csv_path": csv_path,
    }
    return out


# -----------------------
# 추천가 산출
# -----------------------
def _round_to_unit(x: float, unit: int = 100) -> int:
    return int(round(x / unit) * unit)

def estimate_sale_price(
    baseline_price: int,
    damage_delta_level: int,   # -2..+2 (클수록 양호)
    soil_delta_level: int,     # -2..+2 (클수록 깨끗)
    matched_prices: Optional[List[int]] = None,  # 미사용
    base_bias: float = 0.8
) -> dict:
    damage_mult = {
        -2: 0.70, -1: 0.85, 0: 1.00, 1: 1.10, 2: 1.20
    }.get(int(damage_delta_level), 1.0)
    soil_mult = {
        -2: 0.85, -1: 0.93, 0: 1.00, 1: 1.04, 2: 1.08
    }.get(int(soil_delta_level), 1.0)

    biased_baseline = baseline_price * float(base_bias)
    total_mult = damage_mult * soil_mult
    raw_price = biased_baseline * total_mult
    recommended = _round_to_unit(raw_price, 100)

    return {
        "recommended_price": int(recommended),
        "effective_multiplier_vs_baseline": round(base_bias * total_mult, 4),
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
# ToyAnalysisResult 생성 헬퍼
# -----------------------
def _decide_donation(possible: Optional[str], damage: Optional[str], soil: Optional[str]) -> str:
    """
    매우 단순한 휴리스틱(원하면 서버 쪽 정책으로 대체 가능):
    - 파손 '심함'/'크다' 또는 오염도 '높음'/'심함'이면 '불가'
    - 위가 아니면 '가능'
    - 정보 부족 시 '평가 필요'
    """
    txt = lambda x: (x or "").strip().lower()
    d = txt(damage)
    s = txt(soil)
    if any(k in d for k in ["심", "크", "파손 심"]):  # 심함/크다 등
        return "불가"
    if any(k in s for k in ["높", "심"]):  # 높음/심함
        return "불가"
    if not damage and not soil:
        return "평가 필요"
    return "가능"

def _format_token_usage(total: Optional[int]) -> dict:
    # 명세상 세부 구조는 구현에 따름 → 최소 total만 채워서 반환
    return {"total": int(total or 0)}


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
    parser.add_argument("--topk",  type=int, default=TOPK, help="유사도 검색 상위 K")
    parser.add_argument("--base-bias", type=float, default=0.8, help="기본 기준가 바이어스")
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

    # 모델 설정 적용
    clip_files = CLIP_FEATURES_MAP.get(CLIP_MODEL, {
        "features": "toys_index/used_features_large_patch14.npy",
        "paths": "toys_index/used_paths_large_patch14.npy"
    })
    
    # predict.py의 모델 설정 업데이트
    import predict
    predict.MODEL_NAME = CLIP_MODEL
    predict.FEATURES_NPY = clip_files["features"]
    predict.PATHS_NPY = clip_files["paths"]
    
    out = run_full_pipeline(
        query_image_path=query_path,
        feats_npy=clip_files["features"],
        paths_npy=clip_files["paths"],
        csv_path=args.csv,
        base_dir=args.base,
        topk=args.topk,
        left=left_path, rear=rear_path, right=right_path,
    )

    # -----------------------
    # 필요 시 참고용(상품명/가격): classify 명세의 result에는 포함되지 않음
    # -----------------------
    baseline_price = out.get("기준가")
    ref_image_path = out.get("ref_image_path")
    csv_path_used  = out.get("csv_path")
    product_name = get_title_from_csv(csv_path_used, ref_image_path)

    # -----------------------
    # ToyAnalysisResult 구성 (한글 키)
    # -----------------------
    # 파손/오염 상대 레벨(기준 대비) → 기부 가능 여부 간단 휴리스틱
    damage_level = out.get("기준 대비(파손)")
    soil_level   = out.get("기준 대비(오염)")

    toy_result = {
        "장난감_종류": out.get("종류") or "불명",
        "건전지_여부": out.get("건전지"),          # 예: "있음"/"없음"/None
        "재료": out.get("소재") or "불명",
        "파손": out.get("파손도") or "불명",
        "오염도": out.get("오염도") or "불명",
        "크기": out.get("크기"),                  # 문자열 또는 None
        # 서버 정책으로 대체 가능: 여기선 간단 휴리스틱
        "기부_가능_여부": _decide_donation(None, out.get("파손도"), out.get("오염도")),
        "기부_불가_사유": None,                   # 정책상 불가 사유 산출 시 여기에 기입
        "수리_분해": "불필요" if str(out.get("파손도") or "").strip() in ["없음", "경미"] else "검토 필요",
        "관찰사항": None,                         # 에이전트가 텍스트 메모를 줄 수 있으면 채우기
        "토큰_사용량": _format_token_usage(out.get("에이전트_토큰합")),
    }

    # (선택) 내부적으로 활용할 수 있는 추천가 계산 — classify result에는 포함하지 않음
    # baseline_price가 있고 내부 로깅/추가 저장이 필요하면 separate channel로 활용
    # if baseline_price:
    #     rec = estimate_sale_price(
    #         baseline_price=baseline_price,
    #         damage_delta_level=int(damage_level or 0),
    #         soil_delta_level=int(soil_level or 0),
    #         matched_prices=out.get("매칭가격", []),
    #         base_bias=args.base_bias,
    #     )
    #     # 필요하면 toy_result에 포함하지 말고, 서버 내부 저장/로그만 수행

    # -----------------------
    # 전체 분석 결과 JSON 구성
    # -----------------------
    complete_result = {
        # 기본 분석 결과
        "analysis_result": toy_result,
        
        # 기준가 정보
        "baseline_info": {
            "status": out.get("기준가_status"),
            "baseline_price": out.get("기준가"),
            "matched_count": out.get("기준가_매칭개수"),
            "matched_files": out.get("매칭파일", []),
            "matched_prices": out.get("매칭가격", []),
            "ref_image_path": out.get("ref_image_path"),
            "ref_selection_reason": out.get("ref_selection_reason"),
            "product_name": product_name
        },
        
        # 상세 분석 정보
        "detailed_analysis": {
            "toy_type": out.get("종류"),
            "material": out.get("소재"),
            "damage": out.get("파손도"),
            "soil": out.get("오염도"),
            "size": out.get("크기"),
            "battery": out.get("건전지"),
            "damage_vs_reference": out.get("기준 대비(파손)"),
            "soil_vs_reference": out.get("기준 대비(오염)")
        },
        
        # 시스템 정보
        "system_info": {
            "total_tokens": out.get("에이전트_토큰합"),
            "csv_path": out.get("csv_path"),
            "analysis_timestamp": None,  # 필요시 datetime.now().isoformat() 추가
            "models_used": {
                "clip_model": CLIP_MODEL,
                "claude_model": CLAUDE_MODEL
            }
        },
        
        # 추천가 계산 (기준가가 있을 때만)
        "price_recommendation": None
    }
    
    # 추천가 계산 (기준가가 있을 때만)
    if baseline_price:
        rec = estimate_sale_price(
            baseline_price=baseline_price,
            damage_delta_level=int(damage_level or 0),
            soil_delta_level=int(soil_level or 0),
            matched_prices=out.get("매칭가격", []),
            base_bias=args.base_bias,
        )
        complete_result["price_recommendation"] = {
            "recommended_price": rec["recommended_price"],
            "effective_multiplier": rec["effective_multiplier_vs_baseline"],
            "baseline_price": baseline_price,
            "damage_level": damage_level,
            "soil_level": soil_level
        }

    # 최종 출력: 전체 분석 결과 (JSON 한 객체)
    print(json.dumps(complete_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
