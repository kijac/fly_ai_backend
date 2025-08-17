# ai_agent_output.py
import os
import json
import argparse

# === (1) 기준가 파이프라인 ===
from predict import (
    run_sameitem_price,
    IMAGE_BASE_DIR, FEATURES_NPY, PATHS_NPY, PRODUCTS_INFO_CSV, TOPK, IMAGE_PATH
)

# === (2) 상태 판별 에이전트 ===
from ai_agent.supervisor_agent import SupervisorAgent


def _read_bytes_or_none(p: str):
    if not p:
        return None
    with open(p, "rb") as f:
        return f.read()


def run_condition_agents(
    front_path: str,
    left_path: str = None,
    rear_path: str = None,
    right_path: str = None,
    ref_image_bytes: bytes = None,  # ✅ 기준 이미지 1장
):
    # 한 장만 주어지면 그 한 장을 4번 복제해서 사용
    front_b = _read_bytes_or_none(front_path)
    left_b  = _read_bytes_or_none(left_path)  if left_path  else front_b
    rear_b  = _read_bytes_or_none(rear_path)  if rear_path  else front_b
    right_b = _read_bytes_or_none(right_path) if right_path else front_b

    if ref_image_bytes is None:
        raise RuntimeError("ref_image_bytes 가 없습니다. (predict 단계에서 기준 이미지가 선택되지 않았습니다.)")

    sup = SupervisorAgent()
    result = sup.process(front_b, left_b, rear_b, right_b, ref_image_bytes)

    out = {
        "장난감 종류": result.get("장난감 종류", "불명"),
        "재료": result.get("재료", "불명"),
        "오염도": result.get("오염도", "불명"),
        "파손": result.get("파손", "불명"),
        "크기": result.get("크기"),
        "건전지 여부": result.get("건전지 여부"),
        "토큰 사용량(agents 합계)": result.get("토큰 사용량", {}).get("total", 0),
    }
    if "기준 대비(파손)" in result:
        out["기준 대비(파손)"] = result["기준 대비(파손)"]
    if "기준 대비(오염)" in result:
        out["기준 대비(오염)"] = result["기준 대비(오염)"]
    return out


def run_full_pipeline(
    query_image_path: str,
    feats_npy: str = FEATURES_NPY,
    paths_npy: str = PATHS_NPY,
    csv_path: str = PRODUCTS_INFO_CSV,
    base_dir: str = IMAGE_BASE_DIR,
    topk: int = TOPK,
    left: str = None,
    rear: str = None,
    right: str = None,
):
    # --- (1) 기준가 ---
    baseline = run_sameitem_price(
        image_path=query_image_path,
        feats_npy=feats_npy,
        paths_npy=paths_npy,
        csv_path=csv_path,
        base_dir=base_dir,
        topk=topk,
    )

    if baseline.get("status") != "ok" or not baseline.get("ref_image_bytes"):
        return {
            "기준가_status": baseline.get("status"),
            "기준가": baseline.get("baseline_price") if baseline.get("status") == "ok" else None,
            "기준가_매칭개수": baseline.get("matched_count", 0),
            "매칭파일": baseline.get("matched_files", []),
            "매칭가격": baseline.get("matched_prices", []),
            "오류": "기준(ref) 이미지가 없어 상태 비교 불가"
        }

    # --- (2) 상태 판별 (기준 1장 전달) ---
    attrs = run_condition_agents(
        front_path=query_image_path,
        left_path=left, rear_path=rear, right_path=right,
        ref_image_bytes=baseline["ref_image_bytes"]
    )

    # --- (3) 통합 ---
    out = {
        "기준가_status": baseline.get("status"),
        "기준가": baseline.get("baseline_price") if baseline.get("status") == "ok" else None,
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

        "에이전트_토큰합": attrs.get("토큰 사용량(agents 합계)"),
        "ref_image_path": baseline.get("ref_image_path"),
        "ref_selection_reason": baseline.get("ref_selection_reason"),
    }
    return out


# ====== ✅ 추천가 산출 유틸(0.8x 기본 바이어스) ======
def _round_to_unit(x: float, unit: int = 100) -> int:
    return int(round(x / unit) * unit)


def estimate_sale_price(
    baseline_price: int,
    damage_delta_level: int,   # -2..+2 (클수록 양호)
    soil_delta_level: int,     # -2..+2 (클수록 깨끗)
    matched_prices: list[int] | None = None,  # 호환성 유지용 파라미터 (사용 안 함)
    base_bias: float = 0.8     # 기본 0.8배
) -> dict:
    """
    매칭가 기반 상·하한(최소×0.8, 최대×1.1) 클램핑 로직을 제거한 버전.
    순수하게 baseline × base_bias × damage_mult × soil_mult 만 적용.
    """
    # 레벨 → 배수
    damage_mult = {
        -2: 0.70, -1: 0.85, 0: 1.00, 1: 1.10, 2: 1.20
    }.get(int(damage_delta_level), 1.0)
    soil_mult = {
        -2: 0.85, -1: 0.93, 0: 1.00, 1: 1.04, 2: 1.08
    }.get(int(soil_delta_level), 1.0)

    biased_baseline = baseline_price * float(base_bias)
    total_mult = damage_mult * soil_mult
    raw_price = biased_baseline * total_mult

    # ✅ 클램핑 제거됨

    recommended = _round_to_unit(raw_price, 100)

    return {
        "recommended_price": int(recommended),
        "effective_multiplier_vs_baseline": round(base_bias * total_mult, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="기준가 + 상태판별 통합 파이프라인")
    parser.add_argument("--query", type=str, default=IMAGE_PATH, help="기준가/상태판별에 사용할 메인 이미지(앞면)")
    parser.add_argument("--left", type=str, default=None, help="왼쪽 이미지 (없으면 query 복제)")
    parser.add_argument("--rear", type=str, default=None, help="뒷면 이미지 (없으면 query 복제)")
    parser.add_argument("--right", type=str, default=None, help="오른쪽 이미지 (없으면 query 복제)")

    parser.add_argument("--feats", type=str, default=FEATURES_NPY, help="features.npy")
    parser.add_argument("--paths", type=str, default=PATHS_NPY, help="paths.npy")
    parser.add_argument("--csv",   type=str, default=PRODUCTS_INFO_CSV, help="가격 CSV 경로")
    parser.add_argument("--base",  type=str, default=IMAGE_BASE_DIR, help="후보 실이미지 폴더(TopK basename을 여기서 찾음)")
    parser.add_argument("--topk",  type=int, default=TOPK, help="유사도 검색 상위 K")

    # ✅ 0.8배 바이어스 조절 옵션
    parser.add_argument("--base-bias", type=float, default=0.8, help="기본 기준가에 곱할 바이어스(기본 0.8)")

    args = parser.parse_args()

    out = run_full_pipeline(
        query_image_path=args.query,
        feats_npy=args.feats,
        paths_npy=args.paths,
        csv_path=args.csv,
        base_dir=args.base,
        topk=args.topk,
        left=args.left, rear=args.rear, right=args.right,
    )

    # 기준가가 없으면 추천가 산출 불가 → 빈 값 출력
    baseline_price = out.get("기준가")
    if not baseline_price:
        print(0)
        return

    damage_level = int(out.get("기준 대비(파손)") or 0)
    soil_level   = int(out.get("기준 대비(오염)") or 0)
    matched_prices = out.get("매칭가격", [])

    rec = estimate_sale_price(
        baseline_price=baseline_price,
        damage_delta_level=damage_level,
        soil_delta_level=soil_level,
        matched_prices=matched_prices,   # 현재는 미사용(호환성 유지)
        base_bias=args.base_bias,
    )

    # ✅ 최종적으로 "추천 가격만" 터미널에 출력
    print(rec["recommended_price"])


if __name__ == "__main__":
    main()
