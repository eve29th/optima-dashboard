# 실행: streamlit run dashboard/streamlit70.py
# dashboard 안에서 data를 찾도록 되어 있음
# 사용: diag_common_sentB, wordclouds_DARKGRAY, top5, top20, product_wordclouds_all, product_model_input_TRAIN_sentB, train_with_rem4_sentA
# =============================
# REM4 (Category-level) widget: FIXED VERSION
# - median=50 정규화 (최근 12개월 대비 상대적 악화/개선이 보이게)
# - 불만/집중도 0,5 고정처럼 보이던 현상 해결(clip+floor 제거)
# - 상품-월 집계 시 first 대신 median 사용(첫 값 NaN으로 날아가는 문제 방지)
# - 다이아 라벨 겹침 해결(타이틀/패딩/오프셋 재조정)
# - Δ(전월대비) 색: 악화=빨강 / 개선=파랑 (표 + 다이아 동일)
# =============================

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import date, datetime


import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Altair는 환경에 따라 미설치일 수 있어 fallback을 둡니다.
try:
    import altair as alt  # type: ignore
    ALT_AVAILABLE = True
except Exception:
    alt = None  # type: ignore
    ALT_AVAILABLE = False

st.set_page_config(page_title='판매 리스크 조기경보', layout='wide')



# =============================
# UI Styling (REM4 cards etc.)
# =============================
st.markdown(
    """
<style>
  /* REM4 KPI cards */
  .rem4-kpi-card {
    background: white;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 10px;
  }
  .rem4-kpi-label {
    font-size: 12px;
    font-weight: 800;
    color: rgba(0,0,0,0.60);
    letter-spacing: -0.2px;
  }
  .rem4-kpi-value {
    font-size: 28px;
    font-weight: 900;
    line-height: 1.08;
    margin-top: 3px;
    letter-spacing: -0.6px;
  }
  .rem4-kpi-delta {
    font-size: 12px;
    font-weight: 900;
    margin-top: 6px;
  }
  .rem4-note {
    font-size: 12px;
    color: rgba(0,0,0,0.60);
    line-height: 1.45;
  }
  .rem4-subtitle {
    font-size: 20px;
    font-weight: 900;
    margin: 2px 0 6px 0;
    letter-spacing: -0.4px;
  }
  .rem4-line-legend {
    display: flex;
    gap: 18px;
    align-items: center;
    font-size: 12px;
    color: rgba(0,0,0,0.55);
    margin-top: -8px;
  }
  .rem4-line-legend .legend-item {display:flex; align-items:center; gap:10px;}
.rem4-line-legend .legend-line {
  width: 34px;
  height: 0;
  border-top: 4px solid #111111;   /* 현재 달: 검정 실선 */
  transform: translateY(-1px);
}
.rem4-line-legend .legend-line.prev {
  border-top-color: #6a6a6a;       /* 이전 달: 회색 */
  border-top-style: dotted;        /* 점선 */
  opacity: 0.8;
}

/* Streamlit layout tightening */
  .block-container {padding-top: 4.0rem; padding-bottom: 5.0rem;}

  /* Fix top/bottom clipping when printing/exporting */
  @media print {
    @page { margin: 16mm; }
    html, body { margin: 0 !important; padding: 0 !important; }
    .block-container {padding-top: 4.0rem; padding-bottom: 5.0rem;}
    header, footer, .stToolbar, [data-testid="stStatusWidget"] { display: none !important; }
  }

  /* Home summary metric cards */
    /* Home summary metric cards */
  .month-summary-metric {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 10px;
  }
  .msm-label {
    font-size: 13px;
    font-weight: 500;
    color: #111827;
    letter-spacing: -0.2px;
  }
  .msm-emoji {
    font-size: 16px;
    line-height: 1;
    vertical-align: -1px;
    /* keep emoji colors but make them look more vivid */
    filter: saturate(1.85) contrast(1.28);
    text-shadow: 0 0 1px rgba(0,0,0,0.10);
  }
  .msm-value {
  font-weight: 620; /* slightly thinner */
  line-height: 1.02;
  margin-top: 6px;
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: nowrap;
  white-space: nowrap;
}
.msm-main {
  font-size: clamp(32px, 2.6vw, 48px); /* slightly smaller */
  letter-spacing: -0.6px;
  white-space: nowrap;
}
.msm-sub {
  font-size: clamp(28px, 2.2vw, 42px); /* slightly smaller than main */
  font-weight: inherit;
  color: rgba(17,24,39,0.86);
  margin-left: 0;
  letter-spacing: -0.15px;
  white-space: nowrap;
}

/* Sidebar help (용어 도움말) */
  .help-note {font-size: 13px; line-height: 1.5;}
  .help-block {
    margin-bottom: 10px;
    padding: 10px 12px;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    background: #ffffff;
  }
  .help-head {font-weight: 900; color: rgba(0,0,0,0.70);}
  .help-sub {margin-top: 6px; font-size: 12px; color: rgba(0,0,0,0.60); line-height: 1.45;}

  /* REM4 panels / tables */
  .rem4-panel {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 14px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 10px;
  }
  table.rem4-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    overflow: hidden;
  }
  table.rem4-table th {
    text-align: left;
    font-size: 12px;
    color: rgba(0,0,0,0.65);
    padding: 10px 12px;
    background: #f7f8fa;
    border-bottom: 1px solid rgba(0,0,0,0.06);
  }
  table.rem4-table td {
    padding: 10px 12px;
    font-size: 13px;
    vertical-align: top;
    border-bottom: 1px solid rgba(0,0,0,0.06);
  }
  table.rem4-table tr:last-child td {border-bottom: none;}
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit 기본 여백을 줄여 화면을 꽉 채우도록 조정

# =============================
# Constants
# =============================
COUPANG_URL_FMT = "https://www.coupang.com/vp/products/{product_id}"


# Policy rates (알림 정책)
POLICY_RISK_RATE = 0.05
POLICY_TOTAL_ALERT_RATE = 0.20
# 주의는 '총 알림량(위험+주의)'을 20%로 맞추기 위해 15%로 운영 (위험 제외 후 잔여군 상위 15%)
POLICY_CAUTION_RATE = max(POLICY_TOTAL_ALERT_RATE - POLICY_RISK_RATE, 0.0)

# =============================
# Tooltip / Help copy (to avoid misunderstanding)
# =============================
HELP_SCORE = "우선순위점수(score): 0~2 범위 운영 점수(정렬용)"
HELP_RISK = f"🔴위험=급격 악화 신호(p2) 상위 {int(POLICY_RISK_RATE*100)}%"
HELP_CAUTION = f"🟡주의=(위험 제외) 우선순위점수(score) 상위 {int(POLICY_CAUTION_RATE*100)}%"
HELP_GRADE = "알림등급: 🔴위험 / 🟡주의"
HELP_REM_SCORE = "리뷰 신호 종합점수(S): 수요/평판/불만/집중도를 종합한 상태 요약(설명용). 위험/주의 판정에는 사용하지 않습니다."

# 카테고리 순서(표/필터 표시용)
CATEGORY_ORDER_KO = ["가전", "반려동물", "뷰티", "완구/취미", "출산/유아동"]

# 영문/폴더명 → 한글 카테고리
CAT_EN_TO_KO: Dict[str, str] = {
    "Appliances": "가전",
    "Pet": "반려동물",
    "Beauty": "뷰티",
    "Toy": "완구/취미",
    "Toys": "완구/취미",
    "Baby": "출산/유아동",
    "Baby_Products": "출산/유아동",
    "Baby Products": "출산/유아동",
}

# 한글 카테고리 → 파일/폴더 힌트(기본값)
# (wordcloud 폴더/파일 찾기는 내부에서 fallback 목록으로 보강합니다.)
CAT_KO_TO_EN_HINT: Dict[str, str] = {
    "가전": "Appliances",
    "반려동물": "Pet",
    "뷰티": "Beauty",
    "완구/취미": "Toys",
    "출산/유아동": "Baby",
}

# 대표문장/리뷰에서 부정확률 컬럼 후보(1개만 사용)
NEG_SCORE_CANDIDATES = ("neg_prob_koroberta", "neg_prob", "p_neg", "sent_neg_prob")


# =============================
# Matplotlib font (Korean) setup
# =============================
def _mpl_setup_korean_font() -> bool:
    """가능하면 한글 폰트 적용(그래프 한글 깨짐 방지). 없으면 영어 라벨로 폴백."""
    try:
        import matplotlib
        from matplotlib import font_manager as fm
        from matplotlib.ft2font import FT2Font

        candidates = [
            "Malgun Gothic",
            "AppleGothic",
            "NanumGothic",
            "Noto Sans KR",
            "Noto Sans CJK KR",
            "Noto Sans CJK",
        ]
        available = {f.name for f in fm.fontManager.ttflist}

        for c in candidates:
            if c not in available:
                continue
            try:
                fp = fm.findfont(c, fallback_to_default=False)
                ft = FT2Font(fp)
                if ft.get_char_index(ord("가")) == 0:
                    continue
                matplotlib.rcParams["font.family"] = c
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


MPL_HAS_KR = _mpl_setup_korean_font()

def _badge(text: str) -> str:
    return (
        "<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        "background:#f2f4f7;font-size:12px;'>"
        f"{text}</span>"
    )

def _stars(rating) -> str:
    try:
        r = int(float(rating))
        r = max(1, min(5, r))
        return "⭐" * r
    except Exception:
        return "—"

def _safe_num(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def _fmt_int_or_dash(x) -> str:
    v = _safe_num(x)
    if v is None:
        return "—"
    try:
        return f"{int(round(v)):,}"
    except Exception:
        return "—"

def _fmt_float_or_dash(x, digits=2) -> str:
    v = _safe_num(x)
    if v is None:
        return "—"
    return f"{v:.{digits}f}"

def _score_bar_text(score, maxv=2.0, width=14) -> str:
    """
    ✅ 퍼센트(progress) 대신: 막대 + '1.25 / 2.0'
    """
    s = _safe_num(score)
    if s is None:
        return "—"
    s = max(0.0, min(float(s), float(maxv)))
    p = s / maxv if maxv > 0 else 0.0
    filled = int(round(p * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {s:.2f} / {maxv:.1f}"



def _progress_col_cfg(label: str, minv: float = 0.0, maxv: float = 2.0, fmt: str = "%.2f") -> dict:
    """Streamlit column_config helper (safe for older Streamlit).
    - 우선순위점수(score) 컬럼: ProgressColumn + tooltip(help)
    - 알림등급 컬럼: TextColumn + tooltip(help)
    """
    cfg: dict = {}

    # 1) score bar tooltip
    try:
        cfg[label] = st.column_config.ProgressColumn(
            label,
            min_value=minv,
            max_value=maxv,
            format=fmt,
            help=HELP_SCORE,
        )
    except Exception:
        pass

    # 2) risk/caution label tooltip
    try:
        cfg["알림등급"] = st.column_config.TextColumn(
            "알림등급",
            help=HELP_GRADE,
        )
    except Exception:
        pass

    return cfg

def _with_row_numbers(df: pd.DataFrame, col_name: str = "번호") -> pd.DataFrame:
    """표 왼쪽 index를 1부터 보이게(0 제거). '번호' 컬럼은 만들지 않습니다."""
    d = df.copy()
    d.index = np.arange(1, len(d) + 1)
    d.index.name = col_name if col_name is not None else ""
    return d


def _st_dataframe(d, width="stretch", height=None, **kwargs):
    """streamlit 버전별 API 차이를 흡수:
    - 최신: width='stretch' | 'content' | int
    - 구버전: use_container_width=True/False

    NOTE: 최신 Streamlit에서는 height=None 전달 시 에러(StreamlitInvalidHeightError)가 날 수 있어
    height가 None이면 아예 인자를 생략합니다.
    """
    try:
        if height is None:
            return st.dataframe(d, width=width, **kwargs)
        return st.dataframe(d, width=width, height=height, **kwargs)
    except TypeError:
        # 구버전 호환
        use_cw = (str(width) == "stretch")
        if height is None:
            return st.dataframe(d, use_container_width=use_cw, **kwargs)
        return st.dataframe(d, use_container_width=use_cw, height=height, **kwargs)

def _st_image(img, width="stretch", **kwargs):
    """streamlit 버전별 API 차이를 흡수:
    - 최신: width='stretch' | 'content' | int
    - 구버전: use_container_width=True/False
    """
    try:
        return st.image(img, width=width, **kwargs)
    except TypeError:
        return st.image(img, use_container_width=(str(width) == "stretch"), **kwargs)


def _st_altair_chart(chart, width="stretch", **kwargs):
    """streamlit 버전별 API 차이를 흡수:
    - 최신: width='stretch' | 'content' | int
    - 구버전: use_container_width=True/False
    """
    try:
        return st.altair_chart(chart, width=width, **kwargs)
    except TypeError:
        return st.altair_chart(chart, use_container_width=(str(width) == "stretch"), **kwargs)


def _st_pyplot(fig, width="stretch", **kwargs):
    """streamlit 버전별 API 차이를 흡수:
    - 최신: width='stretch' | 'content' | int
    - 구버전: use_container_width=True/False
    """
    try:
        return st.pyplot(fig, width=width, **kwargs)
    except TypeError:
        return st.pyplot(fig, use_container_width=(str(width) == "stretch"), **kwargs)


def _resolve_data_dir() -> Path:
    """
    현재 파일 위치 기준으로 ./data 우선, 없으면 CWD/data
    """
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    cand1 = here / "data"
    cand2 = Path.cwd() / "data"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return Path.cwd()

def _resolve_diag_base_dir(data_dir: Path) -> Path:
    """
    data/diag_common_sentB 하위에 진단 산출물 존재 가정
    """
    for name in ["diag_common_sentB", "diag_common_sentB_CORE", "diag", "diagnostic"]:
        p = data_dir / name
        if p.exists():
            return p
    return data_dir

def _extract_product_id(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    m = re.search(r"/products/(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6,})", s)
    if m:
        return m.group(1)
    return s

def _cat_ko_from_any(x: str) -> str:
    x = str(x)
    if x in CAT_EN_TO_KO:
        return CAT_EN_TO_KO[x]
    if x in CATEGORY_ORDER_KO:
        return x
    for ko in CATEGORY_ORDER_KO:
        if ko in x:
            return ko
    return x


# =============================
# File finders
# =============================
def _find_first(base: Path, pattern: str) -> Optional[Path]:
    hits = list(base.rglob(pattern))
    if not hits:
        return None
    hits = sorted(hits, key=lambda p: p.name)
    return hits[-1]

def _find_queue_file(diag_base: Path, kind: str, ym: str) -> Optional[Path]:
    if kind == "top5":
        pats = [f"*QUEUE_TOP5*{ym}*.csv", f"*TOP5*{ym}*.csv"]
    else:
        pats = [f"*QUEUE_TOP20*{ym}*.csv", f"*TOP20*{ym}*.csv"]
    for pat in pats:
        p = _find_first(diag_base, pat)
        if p:
            return p
    return None

def _find_history_row_pred(diag_base: Path) -> Optional[Path]:
    return _find_first(diag_base, "*HISTORY_ROW_PRED*.csv") or _find_first(diag_base, "*ROW_PRED*.csv")

def _find_review_evidence_file(data_dir: Path, cat_en_hint: str) -> Optional[Path]:
    """
    대표 리뷰/토픽 근거 파일: coupang_{Cat}_train_with_rem4_sentA.csv 계열
    """
    pats = [
        f"*{cat_en_hint}*train_with_rem4*sentA*.csv",
        f"*{cat_en_hint}*train_with_rem4*.csv",
        f"*coupang*{cat_en_hint}*train_with_rem4*.csv",
    ]
    for pat in pats:
        p = _find_first(data_dir, pat)
        if p:
            return p
    return _find_first(data_dir, "*train_with_rem4*sentA*.csv") or _find_first(data_dir, "*train_with_rem4*.csv")


def _find_product_model_input_file(data_dir: Path, cat_en_hint: str) -> Optional[Path]:
    """product_model_input_TRAIN_sentB 파일(카테고리별) 찾기."""
    pats = [
        f"*{cat_en_hint}*product_model_input*TRAIN*sentB*.csv",
        f"*coupang*{cat_en_hint}*product_model_input*TRAIN*sentB*.csv",
        f"*{cat_en_hint}*product_model_input*.csv",
    ]
    for pat in pats:
        p = _find_first(data_dir, pat)
        if p:
            return p
    return _find_first(data_dir, "*product_model_input*TRAIN*sentB*.csv")


def _find_index_all_products_file(data_dir: Path) -> Optional[Path]:
    """index_all_products.csv(전체 상품 목록) 위치는 환경마다 달라 폴백 후보를 여러 개 둠."""
    cands = [
        data_dir / "index_all_products.csv",
        data_dir.parent / "index_all_products.csv",
        Path.cwd() / "index_all_products.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def _count_total_products_from_index(path: str) -> Optional[int]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # 가능한 상품ID 컬럼 후보
    for c in ["상품ID", "product_id", "product", "상품", "prod", "pid"]:
        if c in df.columns:
            s = df[c].astype(str).map(_extract_product_id)
            s = s[s.astype(str).str.len() > 0]
            return int(s.nunique()) if len(s) else None
    return None


@st.cache_data(show_spinner=False)
def _n_valid_by_category_from_train_inputs(data_dir_str: str, ym: str) -> pd.DataFrame:
    """CATEGORY_QUEUE가 없을 때 product_model_input_TRAIN_sentB로 카테고리별 n_valid를 대체 계산."""
    data_dir = Path(data_dir_str)
    rows = []
    for cat_ko in CATEGORY_ORDER_KO:
        cat_hint = CAT_KO_TO_EN_HINT.get(cat_ko)
        if not cat_hint:
            continue
        p = _find_product_model_input_file(data_dir, cat_hint)
        if not p:
            continue

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        # month 필터 (가능하면)
        try:
            df = _ensure_month(df, out_col="month")
            df = df[df["month"].astype(str) == str(ym)].copy()
        except Exception:
            pass

        # 상품ID 추출
        prod_col = "product" if "product" in df.columns else ("상품" if "상품" in df.columns else None)
        if not prod_col:
            continue
        s = df[prod_col].astype(str).map(_extract_product_id)
        s = s[s.astype(str).str.len() > 0]
        n_valid = int(s.nunique()) if len(s) else 0
        rows.append({"카테고리": cat_ko, "전체상품(n_valid)": n_valid})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out[f"risk_cut({int(POLICY_RISK_RATE*100)}%)"] = out["전체상품(n_valid)"].apply(lambda n: _round_half_up_int(float(n) * POLICY_RISK_RATE))
    out[f"caution_cut({int(POLICY_CAUTION_RATE*100)}%)"] = out["전체상품(n_valid)"].apply(lambda n: _round_half_up_int(float(n) * POLICY_CAUTION_RATE))
    out[f"alert_cut({int(POLICY_TOTAL_ALERT_RATE*100)}%)"] = out["전체상품(n_valid)"].apply(lambda n: _round_half_up_int(float(n) * POLICY_TOTAL_ALERT_RATE))
    out = out.sort_values("전체상품(n_valid)", ascending=False).reset_index(drop=True)
    return out


# =============================
# Cached loaders
# =============================
@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# =============================
# Queue file loader (FIX: load_queue undefined)
# =============================
@st.cache_data(show_spinner=False)
def _load_queue(path: str, kind: str) -> pd.DataFrame:
    """QUEUE_TOP5/TOP20 파일을 읽고 대시보드 공통 컬럼(카테고리/등급/상태 등)으로 정규화."""
    df = _read_csv(path)
    assume = "위험" if kind == "top5" else "주의"
    return _prepare_queue(df, assume_level=assume)

# =============================
# Wordcloud image finders (streamlit11.py pattern + your actual folder structure)
# =============================
# ✅ 카테고리 전체 워드클라우드:
#   data/wordclouds_DARKGRAY 내에서
#   **bgFFFFFF_textGrad2038F8to2F3136_ranked.png** 만 사용
#   (월(YYYY-MM) 폴더가 있으면 해당 월 우선)
#
# ✅ 카테고리 부정 워드클라우드(카테고리별 1장):
#   data/negative_wordclouds/wordcloud_negative_<CategoryEN>_*.png
#   (전체 카테고리 합본이면 ALL5)
#
# ✅ 상품 워드클라우드(상품번호 파일명):
#   data/product_wordclouds_all/whitebg/<CategoryEN>/<product_id>.png
#
# ※ 위험/주의 상품 워드클라우드가 추후 따로 만들어질 수 있으므로
#   mode('risk'/'caution')는 해당 키워드가 경로/파일명에 포함된 경우 우선 탐색하고,
#   없으면 'all'로 폴백합니다.

WC_CAT_FOLDER_FALLBACK = {
    "Appliances": ["Appliances", "HomeAppliances", "home_appliances"],
    "Baby": ["Baby_Products", "Baby", "BabyKids", "baby"],
    "Beauty": ["Beauty", "BeautyCare", "beauty"],
    "Pet": ["Pet", "PetSupplies", "pet"],
    "Toys": ["Toys", "Toy", "ToysKids", "toys"],
}
WC_PRODUCT_FOLDER_FALLBACK = WC_CAT_FOLDER_FALLBACK

WC_CAT_TARGET_BASENAME = "bgFFFFFF_textGrad2038F8to2F3136_ranked.png"
WC_CAT_FALLBACK_PATTERNS = [
    WC_CAT_TARGET_BASENAME,
    "*bgFFFFFF_textGrad2038F8to2F3136_ranked*.png",
    "*textGrad2038F8to2F3136*ranked*.png",
]


WC_CAT_TARGET_KEY = "bgffffff_textgrad2038f8to2f3136_ranked"
def _pick_best_wc(paths: List[Path], ym: str) -> Optional[Path]:
    if not paths:
        return None
    ym = str(ym)
    ym_hits = [p for p in paths if (ym in p.as_posix()) or (ym in p.name)]
    cand = ym_hits if ym_hits else paths
    # 월 매칭이 여러 개면 최신 수정 파일 우선
    cand = sorted(cand, key=lambda p: (p.stat().st_mtime, p.as_posix()))
    return cand[-1]

# -----------------------------------------------------------------------------
# Wordcloud image background preference (white background only, when available)
# -----------------------------------------------------------------------------
_WHITE_BG_HINTS = ("whitebg", "bgffffff", "bgfff", "bg_white", "white_background", "bg-white")
_NON_WHITE_BG_HINTS = ("bluebg", "darkbg", "bg000", "bg111", "bg1e")

def _is_white_bg_path(p: Path) -> bool:
    s = p.as_posix().lower()
    if any(bad in s for bad in _NON_WHITE_BG_HINTS):
        return False
    return any(good in s for good in _WHITE_BG_HINTS)

def _ym_tokens(ym: str) -> list[str]:
    """Return multiple date tokens for matching filenames (2025-12, 2025_12, 202512)."""
    if not ym:
        return []
    ym = str(ym).strip()
    y, m = None, None
    m1 = re.match(r"^(\d{4})[-_](\d{1,2})$", ym)
    if m1:
        y, m = m1.group(1), f"{int(m1.group(2)):02d}"
    else:
        m2 = re.match(r"^(\d{4})(\d{2})$", ym)
        if m2:
            y, m = m2.group(1), m2.group(2)
    if not (y and m):
        return [ym]
    return [f"{y}-{m}", f"{y}_{m}", f"{y}{m}", f"{y}.{m}"]


def _score_wc_path(p: Path, ym: str | None = None, tokens: list[str] | None = None) -> float:
    s = 0.0
    name = p.name.lower()
    pstr = str(p).lower()

    if "whitebg" in pstr:
        s += 50
    if "bluebg" in pstr:
        s -= 50

    if "/month/" in pstr or "\\month\\" in pstr:
        s += 8
    if "/all/" in pstr or "\\all\\" in pstr:
        s += 4
    if "/year/" in pstr or "\\year\\" in pstr:
        s -= 2

    if ym:
        for tok in _ym_tokens(ym):
            if tok.lower() in name:
                s += 40
                break
        compact = re.sub(r"[^0-9]", "", ym)
        if compact and compact in re.sub(r"[^0-9]", "", name):
            s += 35

    if tokens:
        for t in tokens:
            tl = str(t).lower().strip()
            if not tl:
                continue
            if tl in name:
                s += 10
            elif tl in pstr:
                s += 6

    s -= len(pstr) / 4000.0
    return s


def _pick_best_wc_scored(
    candidates: list[Path],
    ym: str | None = None,
    tokens: list[str] | None = None,
) -> Path | None:
    """Pick best candidate with scoring. Safe even if ym/tokens are omitted."""
    if not candidates:
        return None
    return max(candidates, key=lambda p: _score_wc_path(p, ym=ym, tokens=tokens))

def find_category_wordcloud_all(data_dir: Path, cat_hint: str, ym: str | None = None) -> Path | None:
    """
    카테고리 '전체' 워드클라우드(전체 리뷰 기준):
    - 우선: data/wordclouds_DARKGRAY/*/bgFFFFFF_textGrad2038F8to2F3136_ranked.png (하위폴더 1~N단계)
    - 파일명이 조금 달라도(textGrad/2038F8to2F3136/ranked 포함) 최대한 찾아서 표시합니다.
    - cat_hint가 특정 카테고리면(예: Appliances) 해당 토큰/한글명과 매칭되는 경로를 우선합니다.
    - cat_hint가 ALL_CATEGORIES면(all/overall/total/전체 등) '전체용'으로 보이는 경로를 우선합니다.
    """
    base = data_dir / "wordclouds_DARKGRAY"
    if not base.exists():
        return None

    # 1) 가장 구체적인 파일명부터 탐색
    candidates: list[Path] = []
    candidates.extend(list(base.rglob(WC_CAT_TARGET_BASENAME)))

    # 2) 파일명이 조금 달라도(textGrad/색상/ranked) 잡아내기
    if not candidates:
        candidates.extend(list(base.rglob("*textGrad2038F8to2F3136*ranked*.png")))
    if not candidates:
        candidates.extend(list(base.rglob("*2038F8to2F3136*ranked*.png")))
    if not candidates:
        # 최후의 fallback: ranked 키워드만이라도
        candidates.extend([p for p in base.rglob("*.png") if "ranked" in p.name.lower()])

    if not candidates:
        return None

    # 3) 카테고리 매칭 토큰 준비(영문/한글 모두)
    hint = str(cat_hint or "")
    hint_l = hint.lower()

    if hint == "ALL_CATEGORIES":
        tokens = ["all", "overall", "total", "allcategories", "all_categories", "전체"]
        # '전체용' 후보를 우선 필터링 (없으면 전체 후보 사용)
        prefer = [p for p in candidates if any(t in p.as_posix().lower() for t in tokens)]
        if prefer:
            candidates = prefer
        return _pick_best_wc_scored(candidates, ym=ym, tokens=tokens)

    tokens = [hint]
    ko = CAT_EN_TO_KO.get(hint)
    if ko:
        tokens.append(ko)

    # 일부 카테고리 폴더/파일명 변형 대응
    if hint_l in ("toys", "toy"):
        tokens += ["Toys", "Toy", "완구", "취미"]
    if hint_l in ("pet", "pets"):
        tokens += ["Pet", "Pets", "반려", "반려동물"]
    if hint_l in ("beauty",):
        tokens += ["Beauty", "뷰티", "미용"]
    if hint_l in ("baby",):
        tokens += ["Baby", "출산", "유아", "유아동"]
    if hint_l in ("appliances",):
        tokens += ["Appliances", "가전"]

    # 4) 직접 경로가 있으면 최우선
    direct = base / hint / WC_CAT_TARGET_BASENAME
    if direct.exists():
        return direct

    # 5) 토큰이 경로/파일명에 포함된 후보 우선(없으면 전체 후보 유지)
    toks_l = [t.lower() for t in tokens if t]
    prefer = [p for p in candidates if any(t in p.as_posix().lower() for t in toks_l)]
    if prefer:
        candidates = prefer

    # 6) 최종 선택(ym/토큰/수정시각 등을 점수화)
    return _pick_best_wc_scored(candidates, ym=ym, tokens=tokens)


def find_category_wordcloud_top(data_dir: Path, cat_hint: str, ym: str, mode: str = "risk") -> Path | None:
    """
    위험/주의 카테고리 워드클라우드(whitebg만):
    - 위험(risk): data/top5/**/whitebg/*.png
    - 주의(caution): data/top20/**/whitebg/*.png

    파일명 날짜 포맷은 Baby_2025-12_whitebg.png / Baby_2025_12_whitebg.png 등 모두 허용합니다.
    """
    mode = (mode or "").strip().lower()
    if mode in ("risk", "위험", "top5"):
        level = "위험"
    elif mode in ("caution", "주의", "top20"):
        level = "주의"
    else:
        # unknown mode
        return None

    if level == "위험":
        roots = [data_dir / "top5" / "CT_top5_wc", data_dir / "top5" / "ALL_top5_wc"]
    else:
        roots = [data_dir / "top20" / "CT_top20_wc", data_dir / "top20" / "ALL_top20_wc"]

    cat_hint = str(cat_hint)
    if cat_hint == "ALL_CATEGORIES":
        cat_tokens = ["ALL", "all", "ALL_CATEGORIES", "allcategories"]
    else:
        cat_tokens = [cat_hint]
        ko = CAT_EN_TO_KO.get(cat_hint)
        if ko:
            cat_tokens.append(ko)
        if cat_hint == "Toy":
            cat_tokens += ["Toys"]
        if cat_hint == "Baby":
            cat_tokens += ["Baby_Products", "BabyProducts", "Baby-Products"]

    ym_toks = _ym_tokens(ym)

    candidates: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*.png"):
            pstr = str(p).lower()
            if "whitebg" not in pstr:
                continue
            # 혹시 섞여 들어온 결과 제거(필요 시)
            if "bluebg" in pstr:
                continue
            candidates.append(p)

    if not candidates:
        return None

    def _has_cat(p: Path) -> bool:
        name = p.name.lower()
        pstr = str(p).lower()
        return any(t.lower() in name or t.lower() in pstr for t in cat_tokens)

    by_cat = [p for p in candidates if _has_cat(p)]
    if by_cat:
        candidates = by_cat

    def _has_ym(p: Path) -> bool:
        n = p.name.lower()
        return any(t.lower() in n for t in ym_toks) or (re.sub(r"[^0-9]", "", ym) in re.sub(r"[^0-9]", "", n))

    by_ym = [p for p in candidates if _has_ym(p)]
    if by_ym:
        candidates = by_ym

    return _pick_best_wc_scored(candidates, ym=ym, tokens=cat_tokens)

def find_product_wordcloud_image(
    data_dir: Path,
    cat_en_hint: str,
    product_id: str,
    mode: str,
    allow_fallback_to_all: bool = True,
) -> Optional[Path]:
    """상품 워드클라우드 이미지 찾기.
    - 전체: product_wordclouds_all/whitebg 우선
    - 위험: top5(PR, whitebg) 우선
    - 주의: top20(PR, whitebg) 우선
    파일명은 {product_id}.png 또는 {product_id}_whitebg.png 형태를 지원합니다.

    ※ mode가 위험/주의인데 이미지가 없을 때,
      allow_fallback_to_all=False면 '전체' 이미지로 폴백하지 않습니다(헷갈림 방지).
    """
    pid = str(product_id)

    names = [f"{pid}_whitebg.png", f"{pid}-whitebg.png", f"{pid}whitebg.png", f"{pid}.png"]
    folder_candidates = WC_PRODUCT_FOLDER_FALLBACK.get(cat_en_hint, [cat_en_hint])
    root_dir = data_dir.parent

    base_priority: List[Path] = []
    if mode == "risk":
        base_priority += [
            root_dir / "top5" / "PR_top5_wc",
            root_dir / "top5",
            data_dir / "top5" / "PR_top5_wc",
            data_dir / "top5",
        ]
    elif mode == "caution":
        base_priority += [
            root_dir / "top20" / "PR_top20_wc",
            root_dir / "top20",
            data_dir / "top20" / "PR_top20_wc",
            data_dir / "top20",
        ]
    elif mode == "all":
        pass
    else:
        return None

    if (mode == "all") or allow_fallback_to_all:
        base_priority += [
            data_dir / "product_wordclouds_all" / "whitebg",
            root_dir / "product_wordclouds_all" / "whitebg",
        ]

    def _pick_from_base(base: Path) -> Optional[Path]:
        if not base.exists():
            return None

        for folder in folder_candidates:
            for nm in names:
                p = base / folder / nm
                if p.exists():
                    return p

        # 파일명이 꼭 product_id로 시작하지 않는 케이스(접두어/접미어 포함)를 대비해
        # '*{pid}*' 와일드카드로 넓게 찾는다.
        hits = list(base.rglob(f"*{pid}*whitebg*.png"))
        if not hits:
            hits = list(base.rglob(f"*{pid}*.png"))
        if not hits:
            return None

        # 1) 카테고리 폴더 힌트가 경로에 포함된 파일 우선
        pri = [h for h in hits if any(f"/{f}/" in h.as_posix() for f in folder_candidates)]
        cand = pri or hits

        # 2) 'white background' 후보가 있으면 그것만 사용 (bluebg는 최후순위)
        white_cand = [p for p in cand if _is_white_bg_path(p)]
        if white_cand:
            cand = white_cand
        else:
            non_blue = [p for p in cand if "bluebg" not in p.as_posix().lower()]
            if non_blue:
                cand = non_blue

        # 3) 파일명(확장자 제외)이 pid로 정확히 시작하는 파일 우선
        def _score(p: Path) -> tuple:
            stem = p.stem
            s = p.as_posix().lower()
            exact_prefix = 0 if stem.startswith(pid) else 1
            white_pref = 0 if _is_white_bg_path(p) else 1
            blue_penalty = 1 if "bluebg" in s else 0
            return (exact_prefix, blue_penalty, white_pref, -p.stat().st_mtime, len(s))

        cand = sorted(cand, key=_score)
        return cand[0]

    for base in base_priority:
        picked = _pick_from_base(base)
        if picked:
            return picked

    return None

def _ensure_month(df: pd.DataFrame, out_col="month") -> pd.DataFrame:
    if "time_bucket" in df.columns:
        df[out_col] = df["time_bucket"].astype(str)
        return df
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df[out_col] = dt.dt.strftime("%Y-%m")
        return df
    raise ValueError("month를 만들 수 있는 컬럼(time_bucket 또는 date)이 없습니다.")


# =============================
# Month listing / prev month
# =============================
def _extract_ym_from_name(name: str) -> Optional[str]:
    m = re.search(r"(20\d{2}-\d{2})", name)
    return m.group(1) if m else None

def _list_months(diag_base: Path) -> List[str]:
    months = set()
    for p in diag_base.rglob("*.csv"):
        if ("QUEUE_TOP5" in p.name) or ("QUEUE_TOP20" in p.name):
            ym = _extract_ym_from_name(p.name)
            if ym:
                months.add(ym)
    if not months:
        for p in diag_base.rglob("*.csv"):
            ym = _extract_ym_from_name(p.name)
            if ym:
                months.add(ym)
    return sorted(list(months))

def _prev_available_month(months_asc: List[str], ym: str) -> Optional[str]:
    if ym not in months_asc:
        return None
    idx = months_asc.index(ym)
    if idx <= 0:
        return None
    return months_asc[idx - 1]


# =============================
# Queue normalize / enrich
# =============================
def _prepare_queue(df: pd.DataFrame, assume_level: str) -> pd.DataFrame:
    d = df.copy()

    if "product" not in d.columns and "상품" in d.columns:
        d["product"] = d["상품"]
    if "category" not in d.columns and "카테고리" in d.columns:
        d["category"] = d["카테고리"]
    if "time_bucket" not in d.columns and "month" in d.columns:
        d["time_bucket"] = d["month"]

    d["상품ID"] = d["product"].astype(str).map(_extract_product_id)
    def _to_coupang_url(x):
        if x is None or pd.isna(x):
            return ""
        s = str(x).strip()
        if (not s) or (s.lower() == "nan"):
            return ""
        # 숫자가 float 형태로 들어오는 경우(예: 123.0) 처리
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]
        return COUPANG_URL_FMT.format(product_id=s)

    d["상품URL"] = d["상품ID"].map(_to_coupang_url)
    d["카테고리"] = d.get("category", "").astype(str).map(_cat_ko_from_any)

    if "state" in d.columns:
        d["state"] = pd.to_numeric(d["state"], errors="coerce")
        d["등급"] = np.where(d["state"] >= 2, "위험", np.where(d["state"] == 1, "주의", "일반"))
    else:
        d["등급"] = assume_level

    score_col = None
    for c in ["score", "Risk0_2", "Risk_0_2", "Risk_0_2_u", "Risk_0_2_s"]:
        if c in d.columns:
            score_col = c
            break
    d["우선순위점수"] = pd.to_numeric(d[score_col], errors="coerce") if score_col else np.nan

    rf = "reliability_flag_u" if "reliability_flag_u" in d.columns else ("reliability_flag" if "reliability_flag" in d.columns else None)
    if rf:
        d[rf] = d[rf].astype(str)

        def _status(x: str) -> str:
            if x == "low_conf":
                return "🔒 잠금(수동점검)"
            if x in ("no_data", "cat_fallback"):
                return "⚪ 정보부족(Gray)"
            return "✅ 기준 통과"

        d["상태"] = d[rf].map(_status)
        d["신뢰도"] = d[rf]
    else:
        d["상태"] = "—"
        d["신뢰도"] = "—"

    d["알림등급"] = np.where(d["등급"] == "위험", "🔴위험",
                      np.where(d["등급"] == "주의", "🟡주의", "—"))

    d["월"] = d["time_bucket"].astype(str) if "time_bucket" in d.columns else ""
    return d

def _dedup_keep_highest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["_score_for_sort"] = pd.to_numeric(d["우선순위점수"], errors="coerce").fillna(-1)
    d = d.sort_values(["상품ID", "_score_for_sort"], ascending=[True, False]).drop(columns=["_score_for_sort"])
    d = d.drop_duplicates(subset=["상품ID"], keep="first")
    return d

def _load_history(diag_base: Path) -> pd.DataFrame:
    p = _find_history_row_pred(diag_base)
    if not p:
        return pd.DataFrame()
    h = _read_csv(str(p))
    if "product" in h.columns:
        h["상품ID"] = h["product"].astype(str).map(_extract_product_id)
    if "time_bucket" in h.columns:
        h["월"] = h["time_bucket"].astype(str)
    return h

def _enrich_queue_with_history(q: pd.DataFrame, history: pd.DataFrame, ym: str) -> pd.DataFrame:
    """
    HISTORY_ROW_PRED(원장)에서 리뷰수/평점 1차 보강
    """
    d = q.copy()
    if "리뷰수" not in d.columns:
        d["리뷰수"] = np.nan
    if "평점" not in d.columns:
        d["평점"] = np.nan

    if history is None or history.empty:
        return d

    if "월" not in history.columns:
        if "time_bucket" in history.columns:
            history = history.copy()
            history["월"] = history["time_bucket"].astype(str)
        else:
            return d

    hh = history[history["월"].astype(str) == str(ym)].copy()
    if hh.empty:
        return d

    review_cands = ["n_reviews", "n_reviews_u", "n_reviews_s", "n_reviews_log1p", "n_reviews_log"]
    rating_cands = ["mean_rating", "rating_mean", "avg_rating", "mean_star", "star_mean",
                    "mean_rating_s", "rating_mean_s"]

    review_col = next((c for c in review_cands if c in hh.columns), None)
    rating_col = next((c for c in rating_cands if c in hh.columns), None)

    keep = ["상품ID"]
    if review_col:
        keep.append(review_col)
    if rating_col:
        keep.append(rating_col)

    hh = hh[keep].drop_duplicates(subset=["상품ID"]).copy()

    rename_map = {}
    if review_col:
        rename_map[review_col] = "__H_reviews_raw"
    if rating_col:
        rename_map[rating_col] = "__H_rating_raw"
    hh = hh.rename(columns=rename_map)

    d = d.merge(hh, on="상품ID", how="left")

    if "__H_reviews_raw" in d.columns:
        raw = pd.to_numeric(d["__H_reviews_raw"], errors="coerce")
        if review_col in ("n_reviews_log1p", "n_reviews_log"):
            raw = np.expm1(raw)
        d["리뷰수"] = pd.to_numeric(d["리뷰수"], errors="coerce").fillna(raw)

    if "__H_rating_raw" in d.columns:
        rawr = pd.to_numeric(d["평점"], errors="coerce")
        fillr = pd.to_numeric(d["__H_rating_raw"], errors="coerce")
        d["평점"] = rawr.fillna(fillr)

    for c in ["__H_reviews_raw", "__H_rating_raw"]:
        if c in d.columns:
            d = d.drop(columns=[c])

    return d


# =============================
# Review / Topic (Evidence) + 2차 보강
# =============================
# =============================
# REM4 (Category-level) widget: Diamond + current/prev table
# =============================

REM4_LABELS_KO = {
    "M": "수요",
    "Q": "평판",
    "C": "불만",
    "CT_shift": "집중도",
    "S": "리뷰 신호 종합점수",
}


REM4_AXES_KO = ["수요", "평판", "불만", "집중도"]
AUBERGINE = "#561689"  # theme accent color


@st.cache_data(show_spinner=False)
def _load_rem4_category_month_all(data_dir_str: str) -> pd.DataFrame:
    """
    카테고리 레벨 REM4(M/Q/C/CT_shift) + S를 (카테고리×월)로 집계해 반환합니다.
    - 소스: 카테고리별 coupang_{Cat}_train_with_rem4_sentA.csv (리뷰 행 단위)
    - 처리:
        (1) (상품ID×월)로 묶고 n_reviews(리뷰행수) 계산
        (2) 카테고리×월로 리뷰수 가중평균 집계
    FIX:
      - 상품-월의 M/Q/C/CT/S는 'first'가 아니라 'median' 사용 (첫 행이 NaN이면 통째로 NaN 되던 케이스 방지)
    """
    from pathlib import Path

    data_dir = Path(data_dir_str)
    out_chunks: List[pd.DataFrame] = []

    need_cols = {"M", "Q", "C", "CT_shift"}
    id_cols = ["product", "상품ID", "상품"]
    time_cols = ["month", "time_bucket", "date"]
    extra_cols = ["is_aug", "S"]

    for cat_ko in CATEGORY_ORDER_KO:
        cat_hint = CAT_KO_TO_EN_HINT.get(cat_ko)
        if not cat_hint:
            continue

        p = _find_review_evidence_file(data_dir, cat_hint)
        if not p:
            continue

        # 메모리 절약 로드 (가능하면)
        try:
            df = pd.read_csv(
                p,
                low_memory=False,
                usecols=lambda c: (c in need_cols) or (c in id_cols) or (c in time_cols) or (c in extra_cols),
            )
        except Exception:
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                continue

        # month 통일
        if "month" not in df.columns:
            try:
                df = _ensure_month(df, out_col="month")
            except Exception:
                continue
        df["month"] = df["month"].astype(str)

        # 증강 제거
        if "is_aug" in df.columns:
            try:
                df = df[df["is_aug"] == 0].copy()
            except Exception:
                pass

        # 상품ID 통일
        if "product" in df.columns:
            df["상품ID"] = df["product"].astype(str).map(_extract_product_id)
        elif "상품ID" in df.columns:
            df["상품ID"] = df["상품ID"].astype(str).map(_extract_product_id)
        elif "상품" in df.columns:
            df["상품ID"] = df["상품"].astype(str).map(_extract_product_id)
        else:
            continue

        # 필수 컬럼 체크
        if not need_cols.issubset(set(df.columns)):
            continue

        keep = ["month", "상품ID", "M", "Q", "C", "CT_shift"]
        if "S" in df.columns:
            keep.append("S")
        df = df[keep].copy()

        for c in ["M", "Q", "C", "CT_shift"] + (["S"] if "S" in df.columns else []):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # (1) 상품-월 집계: n_reviews(리뷰행수) + 지표값(median)
        agg_map = {
            "n_reviews": ("상품ID", "size"),
            "M": ("M", "median"),
            "Q": ("Q", "median"),
            "C": ("C", "median"),
            "CT_shift": ("CT_shift", "median"),
        }
        if "S" in df.columns:
            agg_map["S"] = ("S", "median")

        pm = df.groupby(["month", "상품ID"], as_index=False).agg(**agg_map)

        # 가중평균 함수
        def _wavg(x: np.ndarray, w: np.ndarray) -> float:
            m = np.isfinite(x) & np.isfinite(w) & (w > 0)
            if m.sum() == 0:
                return float("nan")
            return float(np.average(x[m], weights=w[m]))

        # (2) 카테고리-월 집계(리뷰수 가중평균)
        rows = []
        for mth, g in pm.groupby("month"):
            w = g["n_reviews"].to_numpy(dtype=float)
            out = {
                "카테고리": cat_ko,
                "month": str(mth),
                "n_products": int(g["상품ID"].nunique()),
                "n_reviews": int(np.nansum(w)),
                "M": _wavg(g["M"].to_numpy(dtype=float), w),
                "Q": _wavg(g["Q"].to_numpy(dtype=float), w),
                "C": _wavg(g["C"].to_numpy(dtype=float), w),
                "CT_shift": _wavg(g["CT_shift"].to_numpy(dtype=float), w),
            }
            if "S" in g.columns:
                out["S"] = _wavg(g["S"].to_numpy(dtype=float), w)
            rows.append(out)

        cm = pd.DataFrame(rows)
        if not cm.empty:
            out_chunks.append(cm)

    if not out_chunks:
        return pd.DataFrame()

    out = pd.concat(out_chunks, ignore_index=True)
    out["month"] = out["month"].astype(str)
    out = out.sort_values(["카테고리", "month"]).reset_index(drop=True)
    return out

def _percentile_rank_0_100(arr: np.ndarray, v: float) -> float:
    arr = arr[np.isfinite(arr)]
    if (arr is None) or (len(arr) == 0) or (not np.isfinite(v)):
        return float("nan")
    return float((arr <= v).mean() * 100.0)

def _rem4_severity_scores(
    df_cat: pd.DataFrame,
    ym: str,
    prev_row: Optional[pd.Series],
    window: int = 12,
    k: float = 0.8,
    clip_z: float = 2.75,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    ✅ (핵심 FIX) 중앙값=50 정규화 스코어
    - 목적: "최근 12개월 대비 상대적 악화/개선"을 0~100으로 표시하되,
            '좋은 쪽'도 바닥에 붙어 5.0 고정처럼 보이지 않게.
    - 값↑ = 리스크↑ 로 통일:
        * 수요(M), 평판(Q) : 원값이 낮을수록 악화 → 내부에서 -M, -Q로 변환(=worse space)
        * 불만(C), 집중도(CT_shift): 원값↑가 악화 → 그대로
    - 매핑: 최근 window개월(worse space) 분포에서
        median -> 50
        위로 갈수록(악화) 50~100
        아래로 갈수록(개선) 50~0
      score = 50 + 50*tanh(k*z), z=(v-median)/IQR
    - debug_stats: 축별 median/IQR/cur_raw 등 확인용
    """
    if df_cat is None or df_cat.empty:
        return {}, None, {}

    d = df_cat.copy()
    d["month"] = d["month"].astype(str)
    d = d.sort_values("month")

    cur_df = d[d["month"] == str(ym)]
    if cur_df.empty:
        return {}, None, {}
    cur = cur_df.iloc[0]

    base = d[d["month"] <= str(ym)].tail(window)
    if base.empty:
        base = d.tail(window)

    def _to_worse(col: str, x: float) -> float:
        if not np.isfinite(x):
            return float("nan")
        return float(-x) if col in ["M", "Q"] else float(x)

    def _score_from_series(col: str, series: pd.Series, v_raw: float) -> Tuple[float, Dict[str, float]]:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        arr = np.array([_to_worse(col, a) for a in arr], dtype=float)
        arr = arr[np.isfinite(arr)]

        v = _to_worse(col, float(v_raw))

        stats = {"median": np.nan, "iqr": np.nan, "v_raw": float(v_raw) if np.isfinite(v_raw) else np.nan, "v_worse": v}

        if (len(arr) < 3) or (not np.isfinite(v)):
            return float("nan"), stats

        q25, med, q75 = np.nanpercentile(arr, [25, 50, 75])
        iqr = float(q75 - q25)
        if (not np.isfinite(iqr)) or (iqr < 1e-12):
            sd = float(np.nanstd(arr))
            iqr = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0

        z = float((v - med) / iqr)
        z = float(np.clip(z, -clip_z, clip_z))

        score = 50.0 + 50.0 * float(np.tanh(k * z))
        score = float(np.clip(score, 0.0, 100.0))

        stats.update({"median": float(med), "iqr": float(iqr), "z": z, "score": score})
        return score, stats

    # cur
    cur_scores = {}
    debug = {}
    col_map = {"수요": "M", "평판": "Q", "불만": "C", "집중도": "CT_shift"}

    for axis_ko, col in col_map.items():
        v_raw = float(pd.to_numeric(cur.get(col, np.nan), errors="coerce"))
        sc, stt = _score_from_series(col, base[col] if col in base.columns else pd.Series(dtype=float), v_raw)
        cur_scores[axis_ko] = sc
        debug[axis_ko] = stt

    # prev
    prev_scores = None
    if prev_row is not None:
        prev_scores = {}
        for axis_ko, col in col_map.items():
            v_raw = float(pd.to_numeric(prev_row.get(col, np.nan), errors="coerce"))
            sc, _ = _score_from_series(col, base[col] if col in base.columns else pd.Series(dtype=float), v_raw)
            prev_scores[axis_ko] = sc

    return cur_scores, prev_scores, debug



def _score_0_100_median50(
    series,
    v_raw: float,
    invert: bool = False,
    k: float = 0.8,
    clip_z: float = 2.75,
) -> tuple[float, dict]:
    """0~100 정규화 스코어(중앙값=50).

    - series: 기준 분포(최근 12개월 등)
    - invert: True면 값이 낮을수록 악화(=리스크↑)인 지표에 사용
    - mapping: score = 50 + 50*tanh(k*z), z=(v-median)/IQR
    """
    import numpy as np
    import pandas as pd

    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if invert:
        arr = -arr

    v = float(pd.to_numeric(v_raw, errors='coerce')) if v_raw is not None else float('nan')
    if invert and np.isfinite(v):
        v = -v

    stats = {'median': float('nan'), 'iqr': float('nan'), 'v_raw': float(v_raw) if np.isfinite(v) else float('nan'), 'invert': bool(invert)}

    if (len(arr) < 3) or (not np.isfinite(v)):
        return float('nan'), stats

    q25, med, q75 = np.nanpercentile(arr, [25, 50, 75])
    iqr = float(q75 - q25)
    if (not np.isfinite(iqr)) or (iqr < 1e-12):
        sd = float(np.nanstd(arr))
        iqr = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0

    z = float((v - med) / iqr)
    z = float(np.clip(z, -clip_z, clip_z))

    score = 50.0 + 50.0 * float(np.tanh(k * z))
    score = float(np.clip(score, 0.0, 100.0))

    stats.update({'median': float(med), 'iqr': float(iqr), 'z': z, 'score': score})
    return score, stats


def _score_from_window(series, v_raw: float, invert: bool = False, k: float = 0.8, clip_z: float = 2.75):
    """호환성 래퍼: 예전 코드에서 invert= 인자 때문에 터지는 문제 방지."""
    return _score_0_100_median50(series, v_raw, invert=invert, k=k, clip_z=clip_z)

def _plot_rem4_diamond(
    cur: Dict[str, float],
    prev: Optional[Dict[str, float]] = None,
    title: str = "리뷰 지표 요약(레이더 차트)",
    anno: Optional[Dict[str, Tuple[str, str, str]]] = None,  # {"수요":(val_str, delta_str, color), ...}
) -> plt.Figure:
    """
    FIX:
    - 타이틀/상단 라벨 겹침 방지(axes box 재배치 + top 라벨 y 낮춤)
    - 축별 변화(전월→이번달)를 선분으로 강조(악화=빨강, 개선=파랑)
    - 텍스트는 axes 내부에 두어 잘림 방지
    """
    def _z(x):
        return 0.0 if (x is None or (isinstance(x, float) and (not np.isfinite(x)))) else float(x)

    def _pts(d: Dict[str, float]) -> np.ndarray:
        top = _z(d.get("평판", np.nan))
        right = _z(d.get("불만", np.nan))
        bottom = _z(d.get("집중도", np.nan))
        left = _z(d.get("수요", np.nan))
        return np.array([[0.0, top], [right, 0.0], [0.0, -bottom], [-left, 0.0], [0.0, top]])

    pts_cur = _pts(cur)
    pts_prev = _pts(prev) if prev else None

    fig = plt.figure(figsize=(3.05, 2.85), dpi=185)
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.76])  # 여백 확보(라벨 겹침/잘림 방지)

    if title:
        fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=11.0, fontweight="bold")

    for sp in ax.spines.values():
        sp.set_visible(False)

    # grid (0~100)
    for g in [20, 40, 60, 80, 100]:
        grid = np.array([[0, g], [g, 0], [0, -g], [-g, 0], [0, g]])
        lw = 1.15 if g == 60 else 0.9
        a = 0.32 if g == 60 else 0.16
        ax.plot(grid[:, 0], grid[:, 1], linewidth=lw, alpha=a, color="#8a8a8a")

    ax.axhline(0, linewidth=0.8, alpha=0.14, color="#8a8a8a")
    ax.axvline(0, linewidth=0.8, alpha=0.14, color="#8a8a8a")

    # prev polygon
    if pts_prev is not None:
        ax.plot(pts_prev[:, 0], pts_prev[:, 1], linewidth=2.2, linestyle=":", alpha=0.65, color="#6a6a6a")

    # cur polygon (다이아몬드 색상: 검정)
    diamond_col = "#111111"
    ax.plot(pts_cur[:, 0], pts_cur[:, 1], linewidth=3.0, alpha=0.95, color=diamond_col)
    ax.fill(pts_cur[:, 0], pts_cur[:, 1], alpha=0.06, color=diamond_col)
    ax.scatter(pts_cur[:-1, 0], pts_cur[:-1, 1], s=18, color=diamond_col, alpha=0.95)

    # axis delta 강조 (prev->cur)
    if prev is not None:
        axes = [
            ("평판", (0.0, +1.0)),
            ("불만", (+1.0, 0.0)),
            ("집중도", (0.0, -1.0)),
            ("수요", (-1.0, 0.0)),
        ]
        for name, (dx, dy) in axes:
            v0 = _z(prev.get(name, np.nan))
            v1 = _z(cur.get(name, np.nan))
            if not (np.isfinite(v0) and np.isfinite(v1)):
                continue
            delta = v1 - v0
            col = "red" if delta > 0 else ("blue" if delta < 0 else "#666666")
            x0, y0 = dx * v0, dy * v0
            x1, y1 = dx * v1, dy * v1
            ax.plot([x0, x1], [y0, y1], linewidth=4.0, alpha=0.45, color=col)
            ax.scatter([x0, x1], [y0, y1], s=14, color=col, alpha=0.85)

    ax.set_xlim(-135, 135)
    ax.set_ylim(-125, 125)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    def _draw(name: str, x: float, y: float, ha: str, va: str, d_yoff: float):
        # 숫자를 라벨 오른쪽에 붙여 표기 (예: '평판 32.3')
        if anno and name in anno:
            v, d, c = anno[name]
            label = f"{name} {v}" if v else name
            ax.text(x, y, label, ha=ha, va=va, fontsize=10.6, fontweight="bold", color="#111111")
            if d:
                ax.text(x, y + d_yoff, f"({d})", ha=ha, va=va, fontsize=9.2, color=c, fontweight="bold")
        else:
            ax.text(x, y, name, ha=ha, va=va, fontsize=10.6, fontweight="bold", color="#111111")

    # 라벨 위치(상단은 제목과 겹치지 않게 y 낮춤)
    _draw("평판",   0,   84, "center", "bottom", d_yoff=-16)
    _draw("집중도", 0,  -90, "center", "top",    d_yoff=-16)
    _draw("수요",  -96,  0,  "right",  "center", d_yoff=-18)
    _draw("불만",   96,  0,  "left",   "center", d_yoff=-18)

    return fig

def _render_rem4_category_widget(rem4_cm_all: pd.DataFrame, cat_ko: str, ym: str) -> None:
    """Category-level REM4 (리뷰 신호 지표) widget.

    디자인/가독성 개선:
    - '리뷰 신호 지표' 중복 타이틀 제거(상위 섹션에서 1회만 노출)
    - 요약(한눈에) / 해석 / 디버그를 탭으로 분리
    - 주요 지표를 KPI 카드로 표시(값 + Δ)
    """

    if rem4_cm_all is None or rem4_cm_all.empty:
        st.info("리뷰 신호 지표(REM4) 집계 파일을 찾지 못했습니다. (*train_with_rem4_sentA* 확인)")
        return

    df_cat = rem4_cm_all[rem4_cm_all["카테고리"].astype(str) == str(cat_ko)].copy()
    if df_cat.empty:
        st.info(f"'{cat_ko}' 카테고리의 리뷰 신호 지표 데이터를 찾지 못했습니다.")
        return

    df_cat["month"] = df_cat["month"].astype(str)
    df_cat = df_cat.sort_values("month")

    cur_df = df_cat[df_cat["month"] == str(ym)]
    if cur_df.empty:
        st.info(f"{ym} 월 리뷰 신호 지표 데이터가 없습니다.")
        return
    cur = cur_df.iloc[0]

    months_cat = sorted(df_cat["month"].dropna().astype(str).unique().tolist())
    prev_ym = _prev_available_month(months_cat, str(ym))
    prev = df_cat[df_cat["month"] == str(prev_ym)].iloc[0] if prev_ym else None

    # ✅ 중앙값=50 정규화 스코어 (0~100)
    cur_scores, prev_scores, debug_stats = _rem4_severity_scores(
        df_cat,
        ym=str(ym),
        prev_row=prev,
        window=12,
        k=0.8,       # 민감도(↑ = 더 민감/과장)
        clip_z=2.75,
    )

    # Δ 색 규칙(공통): Δ>0 악화(빨강) / Δ<0 개선(파랑)
    def _delta_style(d: float) -> tuple[str, str]:
        if not np.isfinite(d):
            return ("#6b7280", "•")
        if d > 0:
            return ("#d11a2a", "▲")  # 악화
        if d < 0:
            return ("#1d4ed8", "▼")  # 개선
        return ("#6b7280", "•")

    def _fmt_score(v: float) -> str:
        if not np.isfinite(v):
            return "—"
        return f"{v:.1f}"

    # 다이아몬드 annotation
    anno = {}
    kpis = []
    for k in REM4_AXES_KO:
        v_cur = float(cur_scores.get(k, np.nan))
        v_prev = float(prev_scores.get(k, np.nan)) if prev_scores else np.nan
        dlt = (v_cur - v_prev) if (np.isfinite(v_cur) and np.isfinite(v_prev)) else np.nan
        color, _ = _delta_style(dlt)

        d_str = f"{dlt:+.1f}" if np.isfinite(dlt) else ""
        anno[k] = (_fmt_score(v_cur), d_str, color)
        kpis.append((k, v_cur, dlt))

    # --- REM 종합점수(S) ---
    s_cur_sc = np.nan
    s_prev_sc = np.nan
    dlt_s = np.nan
    if "S" in df_cat.columns:
        base_s = df_cat[df_cat["month"].astype(str) <= str(ym)].tail(12)
        s_series = base_s["S"] if ("S" in base_s.columns) else df_cat["S"]

        s_cur_raw = float(pd.to_numeric(cur.get("S", np.nan), errors="coerce"))
        s_prev_raw = float(pd.to_numeric(prev.get("S", np.nan), errors="coerce")) if prev is not None else np.nan

        s_cur_sc, _ = _score_0_100_median50(s_series, s_cur_raw, invert=False)
        if np.isfinite(s_prev_raw):
            s_prev_sc, _ = _score_0_100_median50(s_series, s_prev_raw, invert=False)
        dlt_s = (s_cur_sc - s_prev_sc) if (np.isfinite(s_cur_sc) and np.isfinite(s_prev_sc)) else np.nan

    # --- UI helpers ---
    def _kpi_card(label: str, value: float, delta: float, sub: str = "") -> None:
        v_txt = _fmt_score(float(value))
        if np.isfinite(delta):
            color, arrow = _delta_style(float(delta))
            d_txt = f"{float(delta):+,.1f}"
            delta_html = f'<span class="rem4-kpi-delta" style="color:{color};">{arrow} {d_txt} (전월 대비)</span>'
        else:
            delta_html = '<span class="rem4-kpi-delta" style="color:#6b7280;">• —</span>'

        is_s = label.strip().startswith("종합점수")
        card_style = (f' style="border: 1px solid {AUBERGINE}33; max-width: 520px; margin: 0 auto; text-align: center;"' if is_s else "")
        label_style = f' style="color:{AUBERGINE};"' if is_s else ""
        value_style = f' style="color:{AUBERGINE};"' if is_s else ""

        st.markdown(
            f"""
<div class="rem4-kpi-card"{card_style}>
  <div class="rem4-kpi-label"{label_style}>{label}</div>
  <div class="rem4-kpi-value"{value_style}>{v_txt}</div>
  {delta_html}
  <div class="rem4-note">{sub}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    # --- Header (중복 타이틀 방지: '리뷰 신호 지표'는 상위 섹션에서 1회만) ---
    st.markdown(f"#### {cat_ko} | {ym}")
    st.markdown(
        '<div class="rem4-note">최근 12개월 대비 상대적 위치를 <b>0~100</b>으로 정규화했습니다(중앙값=50). 값↑ = 리스크↑, Δ는 전월 대비 변화(악화=빨강 / 개선=파랑)입니다.</div>',
        unsafe_allow_html=True,
    )

    tab_sum, tab_help, tab_dbg = st.tabs(["요약", "지표 해석", "원값/정규화"])

    # =========================
    # TAB 1) 요약
    # =========================
    with tab_sum:
        # 1) KPI 카드: 4개 축을 한 줄에 배치
        kpi_map = {k: (v, d) for (k, v, d) in kpis}
        cols_kpi = st.columns(4, gap="small")
        for col, axis in zip(cols_kpi, REM4_AXES_KO):
            with col:
                v, d = kpi_map.get(axis, (np.nan, np.nan))
                _kpi_card(axis, v, d)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # 2) 종합점수(S) (설명용) — 가운데 정렬
        if np.isfinite(s_cur_sc):
            _l, _m, _r = st.columns([1, 2, 1])
            with _m:
                _kpi_card("종합점수(S)", s_cur_sc, dlt_s, sub="설명용(판정 미사용)")


        # 3) 요약(레이더 차트) + 종합점수(S) 추이(최근 6개월) — 좌/우 배치
        st.markdown('<div style=\"height:96px\"></div>', unsafe_allow_html=True)

        col_dia, col_tr = st.columns([1.0, 1.35], gap="large")

        with col_dia:
            st.markdown('<div class="rem4-subtitle">🎯요약 (레이더 차트)</div>', unsafe_allow_html=True)
            fig = _plot_rem4_diamond(
                cur_scores,
                prev_scores,
                title="",
                anno=anno,
            )
            _st_pyplot(fig, width="stretch", clear_figure=True)
            st.markdown(
                """
<div class="rem4-line-legend">
  <div class="legend-item"><span class="legend-line"></span><span>현재 달</span></div>
  <div class="legend-item"><span class="legend-line prev"></span><span>이전 달</span></div>
</div>
                """,
                unsafe_allow_html=True,
            )

        with col_tr:
            if "S" in df_cat.columns:
                st.markdown('<div class="rem4-subtitle">📈 종합점수(S) 추이 (최근 6개월)</div>', unsafe_allow_html=True)
                st.markdown("<div style=\"height:14px\"></div>", unsafe_allow_html=True)

                months_up_to = months_cat[: months_cat.index(str(ym)) + 1] if str(ym) in months_cat else months_cat
                tr_months = months_up_to[-6:]
                tr_scores = []
                for mth in tr_months:
                    row_m = df_cat[df_cat["month"].astype(str) == str(mth)]
                    v_raw = float(pd.to_numeric(row_m["S"].iloc[0], errors="coerce")) if (not row_m.empty) else np.nan
                    base_s = df_cat[df_cat["month"].astype(str) <= str(mth)].tail(12)
                    sc, _ = _score_0_100_median50(base_s["S"], v_raw, invert=False)
                    tr_scores.append(sc)

                if len(tr_months) >= 2 and any([np.isfinite(x) for x in tr_scores]):
                    fig2 = plt.figure(figsize=(7.8, 2.6), dpi=180)
                    ax2 = plt.gca()
                    ax2.plot(tr_months, tr_scores, marker="o", color=AUBERGINE)
                    ax2.set_ylim(0, 100)
                    ax2.set_xlabel("월" if MPL_HAS_KR else "Month")
                    ax2.set_ylabel("S(0~100)" if MPL_HAS_KR else "S (0-100)")
                    ax2.grid(alpha=0.25)
                    for label in ax2.get_xticklabels():
                        label.set_rotation(45)
                        label.set_horizontalalignment("right")
                    fig2.tight_layout(pad=0.4)
                    _st_pyplot(fig2, width="stretch", clear_figure=True)
                else:
                    st.caption("S 추이를 그리기엔 월 데이터가 부족합니다.")
            else:
                st.caption("S 추이 데이터를 찾지 못했습니다.")

    # =========================
    # TAB 2) 지표 해석
    # =========================
    with tab_help:
        st.markdown("**지표 해석(값↑ = 리스크↑)**")

        def _help_kpi(title: str, lines: List[str]) -> None:
            html = f"""<div class="rem4-kpi-card">
  <div class="rem4-kpi-label">{title}</div>
  <div class="rem4-note">{'<br/>'.join(lines)}</div>
</div>"""
            st.markdown(html, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            _help_kpi("수요 (유입 / 관심 약화)", ["• 리뷰 유입 추세 ↓ → 리스크 ↑", "• 신규 리뷰 비중 ↓ → 리스크 ↑"])
            _help_kpi("불만 (원인 구조 / 강도)", ["• 불만 토픽 비중 ↑ → 리스크 ↑", "• 부정 강도 ↑ → 리스크 ↑"])
        with c2:
            _help_kpi("평판 (만족도 하락)", ["• 평점 ↓ → 리스크 ↑", "• 저평점 비중 ↑ → 리스크 ↑"])
            _help_kpi("집중도 (쏠림 / 구조 리스크)", ["• 상위 상품 비중 ↑ → 리스크 ↑", "• 집중도 지수 ↑ → 리스크 ↑"])

        st.info("지표는 '상대적 신호'입니다. 실제 액션은 **상품 레벨 증거(리뷰/토픽/변화)**와 함께 확인하는 것을 권장합니다.")

    # =========================
    # TAB 3) 원값/정규화
    # =========================
    with tab_dbg:
        raw_row = {
            "M(수요 원값)": float(pd.to_numeric(cur.get("M", np.nan), errors="coerce")),
            "Q(평판 원값)": float(pd.to_numeric(cur.get("Q", np.nan), errors="coerce")),
            "C(불만 원값)": float(pd.to_numeric(cur.get("C", np.nan), errors="coerce")),
            "CT(집중도 원값)": float(pd.to_numeric(cur.get("CT_shift", np.nan), errors="coerce")),
            "S(종합점수 원값)": float(pd.to_numeric(cur.get("S", np.nan), errors="coerce")) if "S" in cur else np.nan,
            "n_products": int(pd.to_numeric(cur.get("n_products", np.nan), errors="coerce")) if "n_products" in cur else np.nan,
            "n_reviews": int(pd.to_numeric(cur.get("n_reviews", np.nan), errors="coerce")) if "n_reviews" in cur else np.nan,
            "prev_month": prev_ym,
        }

        cL, cR = st.columns([0.45, 0.55], gap="large")

        with cL:
            st.markdown("**이번달 원값(집계)**")
            key_map = {
                "n_products": "n_products(상품수)",
                "n_reviews": "n_reviews(리뷰수)",
                "prev_month": "prev_month(전월)",
            }
            raw_items = []
            for k in raw_row.keys():
                label = key_map.get(k, k)
                v = raw_row.get(k)
                if k in ("n_products", "n_reviews"):
                    v_str = f"{int(v):,}" if pd.notna(v) else "—"
                elif k == "prev_month":
                    v_str = str(v) if v is not None else "—"
                else:
                    v_str = f"{float(v):.4f}" if pd.notna(v) else "—"
                raw_items.append({"항목": label, "값": v_str})
            raw_df = pd.DataFrame(raw_items)
            _st_dataframe(raw_df, width="stretch", hide_index=True)

        with cR:
            st.markdown("**정규화**")
            dbg_df = pd.DataFrame.from_dict(debug_stats, orient="index").reset_index().rename(columns={"index": "지표"})
            for c in ["median", "iqr", "v_raw", "v_worse", "z", "score"]:
                if c in dbg_df.columns:
                    dbg_df[c] = pd.to_numeric(dbg_df[c], errors="coerce")
            if "score" in dbg_df.columns:
                dbg_df["score"] = dbg_df["score"].round(2)
            for c in ["median", "iqr", "v_raw", "v_worse", "z"]:
                if c in dbg_df.columns:
                    dbg_df[c] = dbg_df[c].round(4)

            # 컬럼 표시 순서
            show_cols = [c for c in ["지표", "score", "z", "median", "iqr", "v_raw", "v_worse"] if c in dbg_df.columns]
            rename_cols = {
                "score": "score(정규화점수)",
                "z": "z(Z점수)",
                "median": "median(중앙값)",
                "iqr": "iqr(IQR,사분위범위)",
                "v_raw": "v_raw(원값)",
                "v_worse": "v_worse(위험방향)",
            }
            dbg_show = dbg_df[show_cols].rename(columns=rename_cols)
            _st_dataframe(dbg_show, width="stretch", hide_index=True)



def _load_review_evidence(path: str) -> pd.DataFrame:
    df = _read_csv(path).copy()
    if "product" in df.columns:
        df["상품ID"] = df["product"].astype(str).map(_extract_product_id)
    df = _ensure_month(df, out_col="month")

    keep = []
    for c in ["platform", "category", "product", "상품ID", "time_bucket", "month",
              "date", "rating", "text_for_nlp", "topic"]:
        if c in df.columns:
            keep.append(c)

    # neg score 후보 1개만
    for c in NEG_SCORE_CANDIDATES:
        if c in df.columns and c not in keep:
            keep.append(c)
            break

    if "is_aug" in df.columns:
        keep.append("is_aug")

    df = df[keep].copy()

    if "text_for_nlp" in df.columns:
        df["text_for_nlp"] = (
            df["text_for_nlp"].astype(str)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
        )
    return df

def _fill_reviews_and_rating_from_review_files(queue_all: pd.DataFrame, ym: str, data_dir: Path) -> pd.DataFrame:
    """
    HISTORY_ROW_PRED로 못 채운 리뷰수/평점을
    카테고리별 train_with_rem4_sentA에서 (상품ID×월)로 집계해 2차 보강
    """
    if queue_all.empty:
        return queue_all

    d = queue_all.copy()
    d["리뷰수"] = pd.to_numeric(d.get("리뷰수", np.nan), errors="coerce")
    d["평점"] = pd.to_numeric(d.get("평점", np.nan), errors="coerce")

    need_reviews = d["리뷰수"].isna()
    need_rating = d["평점"].isna()
    if not (need_reviews.any() or need_rating.any()):
        d["리뷰수"] = d["리뷰수"].round().astype("Int64")
        d["평점"] = d["평점"].round(2)
        return d

    chunks = []
    for cat_ko in sorted(set(d["카테고리"].dropna().astype(str).tolist())):
        cat_hint = CAT_KO_TO_EN_HINT.get(cat_ko)
        if not cat_hint:
            continue
        rev_p = _find_review_evidence_file(data_dir, cat_hint)
        if not rev_p:
            continue
        rev = _load_review_evidence(str(rev_p))
        if "month" not in rev.columns:
            continue

        rv = rev[rev["month"].astype(str) == str(ym)].copy()
        if rv.empty:
            continue
        if "is_aug" in rv.columns:
            rv = rv[rv["is_aug"] == 0].copy()

        # 리뷰수: 해당 상품/월 리뷰 행수
        # 평점: rating 평균(없으면 NaN)
        agg = rv.groupby("상품ID").agg(
            __calc_reviews=("상품ID", "size"),
            __calc_rating=("rating", "mean") if "rating" in rv.columns else ("상품ID", lambda x: np.nan),
        ).reset_index()
        agg["카테고리"] = cat_ko
        chunks.append(agg)

    if chunks:
        calc = pd.concat(chunks, ignore_index=True)
        d = d.merge(calc, on=["카테고리", "상품ID"], how="left")

        if "__calc_reviews" in d.columns:
            d["리뷰수"] = d["리뷰수"].fillna(pd.to_numeric(d["__calc_reviews"], errors="coerce"))
        if "__calc_rating" in d.columns:
            d["평점"] = d["평점"].fillna(pd.to_numeric(d["__calc_rating"], errors="coerce"))

        for c in ["__calc_reviews", "__calc_rating"]:
            if c in d.columns:
                d = d.drop(columns=[c])

    d["리뷰수"] = pd.to_numeric(d["리뷰수"], errors="coerce").round().astype("Int64")
    d["평점"] = pd.to_numeric(d["평점"], errors="coerce").round(2)
    return d

def _product_reviews_topn(rev: pd.DataFrame, product_id: str, ym: str, topn=5, rating_max: Optional[int] = None, date_filter: Optional[date] = None) -> pd.DataFrame:
    d = rev[rev["상품ID"].astype(str) == str(product_id)].copy()
    if "month" in d.columns:
        d = d[d["month"].astype(str) == str(ym)].copy()
    # Optional day filter (month 기준 정책은 유지, 화면 표시만 날짜로 좁힘)
    if date_filter is not None and "date" in d.columns:
        dd = pd.to_datetime(d["date"], errors="coerce")
        d = d[dd.dt.date == date_filter].copy()

    if rating_max is not None and "rating" in d.columns:
        d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
        d = d[d["rating"] <= rating_max].copy()

    if "is_aug" in d.columns:
        d = d[d["is_aug"] == 0].copy()

    score_col = None
    for c in NEG_SCORE_CANDIDATES:
        if c in d.columns:
            score_col = c
            d[c] = pd.to_numeric(d[c], errors="coerce")
            break

    if score_col:
        d = d.sort_values(score_col, ascending=False)
    else:
        if "text_for_nlp" in d.columns:
            d["len_"] = d["text_for_nlp"].astype(str).str.len()
            d = d.sort_values("len_", ascending=False)

    return d.head(topn).reset_index(drop=True)

def _topic_shift_top2(rev: pd.DataFrame, product_id: str, ym: str, top_k=2) -> Tuple[pd.DataFrame, str]:
    """
    전월 대비 토픽 share 변화 Top2
    - 전월 데이터 없으면: 이번달 Top 토픽(share)로 폴백
    """
    d = rev[rev["상품ID"].astype(str) == str(product_id)].copy()
    if "topic" not in d.columns:
        return pd.DataFrame(), "topic 컬럼이 없어 계산할 수 없습니다."
    if "month" not in d.columns:
        return pd.DataFrame(), "month/time_bucket가 없어 계산할 수 없습니다."

    cur = str(ym)
    try:
        prev = (pd.Period(cur) - 1).strftime("%Y-%m")
    except Exception:
        prev = None

    if prev is None:
        return pd.DataFrame(), "전월을 계산할 수 없습니다."

    d_cur = d[d["month"].astype(str) == cur].copy()
    d_prev = d[d["month"].astype(str) == prev].copy()

    if len(d_cur) == 0:
        return pd.DataFrame(), "이번달 토픽 데이터가 없습니다."

    def _topic_share(dd: pd.DataFrame) -> pd.Series:
        c = dd.groupby("topic").size()
        return (c / c.sum()).sort_values(ascending=False)

    share_cur = _topic_share(d_cur)

    if len(d_prev) == 0:
        out = share_cur.head(top_k).reset_index()
        out.columns = ["topic", f"{cur}_share"]
        out["note"] = "전월 데이터 없음 → 이번달 Top 토픽"
        return out.round(4), "전월 데이터가 없어 ‘급변’ 대신 ‘이번달 Top 토픽’으로 표시합니다."

    share_prev = _topic_share(d_prev)

    all_topics = sorted(set(share_cur.index) | set(share_prev.index))
    dfp = pd.DataFrame({
        prev: [share_prev.get(t, 0.0) for t in all_topics],
        cur:  [share_cur.get(t, 0.0) for t in all_topics],
    }, index=all_topics)

    dfp["Δ_share"] = dfp[cur] - dfp[prev]
    dfp["absΔ"] = dfp["Δ_share"].abs()
    top = dfp.sort_values("absΔ", ascending=False).head(top_k).copy()
    top["dir"] = np.where(top["Δ_share"] >= 0, "▲", "▼")
    top = top[[prev, cur, "Δ_share", "dir"]].reset_index().rename(columns={"index": "topic"})
    return top.round(4), ""


def _format_topic_shift_display(top2: pd.DataFrame, ym: str) -> pd.DataFrame:
    """상품상세 '토픽 급변' 표를 요청 형식으로 변환:
    토픽 / 전월 / 이번달(변화량)  (share/방향 컬럼 제거)
    """
    if top2 is None or top2.empty:
        return pd.DataFrame()

    d = top2.copy()

    if "topic" in d.columns:
        d = d.rename(columns={"topic": "토픽"})
    elif "토픽" not in d.columns and d.columns.tolist():
        d = d.rename(columns={d.columns[0]: "토픽"})

    month_cols = [c for c in d.columns if re.match(r"^\d{4}-\d{2}$", str(c))]
    prev_col, cur_col = (month_cols[0], month_cols[1]) if len(month_cols) >= 2 else (None, None)

    share_cols = [
        c for c in d.columns
        if str(c).endswith("_share") and re.match(r"^\d{4}-\d{2}_share$", str(c))
    ]
    cur_share_col = share_cols[0] if share_cols else None

    def _fmt_float(x):
        return "-" if pd.isna(x) else f"{float(x):.4f}"

    def _fmt_delta(x):
        return "" if pd.isna(x) else f"{float(x):+.4f}"

    if prev_col and cur_col:
        prev_v = pd.to_numeric(d[prev_col], errors="coerce")
        cur_v = pd.to_numeric(d[cur_col], errors="coerce")
        delta_v = (
            pd.to_numeric(d["Δ_share"], errors="coerce") if "Δ_share" in d.columns
            else pd.to_numeric(d.get("delta", np.nan), errors="coerce")
        )

        return pd.DataFrame({
            "토픽": d["토픽"].astype(str),
            "전월": [_fmt_float(x) for x in prev_v],
            "이번달(변화량)": [
                (f"{_fmt_float(c)} ({_fmt_delta(dd)})" if _fmt_delta(dd) else _fmt_float(c))
                for c, dd in zip(cur_v, delta_v)
            ],
        })

    if cur_share_col:
        cur_v = pd.to_numeric(d[cur_share_col], errors="coerce")
        return pd.DataFrame({
            "토픽": d["토픽"].astype(str),
            "전월": ["-"] * len(d),
            "이번달(변화량)": [_fmt_float(x) for x in cur_v],
        })

    cols = [c for c in ["토픽", "전월", "이번달(변화량)"] if c in d.columns]
    return d[cols].copy()

def _extract_keywords(texts: List[str], topn=20) -> List[str]:
    """
    간단 키워드(형태소 분석 없이):
    - 한글 2자 이상 토큰 빈도 상위
    - 숫자/단위성 토큰(예: 개, 번 등) 및 너무 일반적인 불용어 제거
    """
    joined = " ".join([t for t in texts if isinstance(t, str)])
    # 숫자는 제거(예: 3개 -> 개)
    joined = re.sub(r"\d+", " ", joined)

    toks = re.findall(r"[가-힣]{2,}", joined)
    if not toks:
        return []

    stop = {
        "정도","그냥","진짜","너무","조금","일단","근데","그리고","그래서",
        "사용","제품","상품","구매","배송","포장","가격","정말",
        "개","번","때","것","수","점","좀","많이","바로"
    }
    toks = [t for t in toks if t not in stop]

    from collections import Counter
    return [w for w, _ in Counter(toks).most_common(topn)]


def _sidebar_help():
    with st.sidebar.expander("용어 도움말", expanded=False):
        st.markdown(
            """
<div class="help-note">
  <div class="help-block">
    <span class="help-head">알림등급:</span> 🔴위험 / 🟡주의
    <div class="help-sub">
      ⓘ 위험=급격 악화 신호(p2) 상위 5%<br/>
      주의=우선순위점수(score) 상위 15%에서 위험 제외 (총 알림량=20%)
    </div>
  </div>

  <div class="help-block">
    <span class="help-head">우선순위점수(score):</span>
    <div class="help-sub">0~2 범위 운영 점수(정렬용)</div>
  </div>

  <div class="help-block">
    <span class="help-head">🔒 잠금(low_conf):</span>
    <div class="help-sub">→ 자동 액션 금지(수동 점검)</div>
  </div>

  <div class="help-block" style="margin-bottom: 0;">
    <span class="help-head">⚪ Gray(no_data/cat_fallback):</span>
    <div class="help-sub">→ 점수 미산출, 큐 제외</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


# =============================
# 0212 추가 함수
# =============================
def _apply_valid_for_cut(df: pd.DataFrame) -> pd.DataFrame:
    # model_used 없으면 p0 존재로 대체
    if "model_used" in df.columns:
        cond_model = (df["model_used"] == 1)
    else:
        cond_model = df["p0"].notna() if "p0" in df.columns else True

    if "reliability_flag_u" in df.columns:
        cond_rel = df["reliability_flag_u"].isin(["ok", "low_conf"])
    else:
        cond_rel = True

    return df[cond_model & cond_rel].copy()

def _month_grade_counts_from_history(history: pd.DataFrame, ym: str) -> dict:
    if history is None or history.empty:
        return {"n_valid": None, "n_state0": 0, "n_state1": 0, "n_state2": 0}

    h = history.copy()
    # month 컬럼 통일
    if "월" in h.columns:
        h["_month"] = h["월"].astype(str)
    elif "time_bucket" in h.columns:
        h["_month"] = h["time_bucket"].astype(str)
    elif "month" in h.columns:
        h["_month"] = h["month"].astype(str)
    else:
        return {"n_valid": None, "n_state0": 0, "n_state1": 0, "n_state2": 0}

    h = h[h["_month"] == str(ym)].copy()
    if h.empty:
        return {"n_valid": 0, "n_state0": 0, "n_state1": 0, "n_state2": 0}

    h = _apply_valid_for_cut(h)

    # state 통일
    state_col = "state" if "state" in h.columns else ("state_u" if "state_u" in h.columns else None)
    if state_col is None:
        return {"n_valid": int(len(h)), "n_state0": 0, "n_state1": 0, "n_state2": 0}

    s = pd.to_numeric(h[state_col], errors="coerce").fillna(0).astype(int)
    return {
        "n_valid": int(len(h)),
        "n_state0": int((s == 0).sum()),
        "n_state1": int((s == 1).sum()),
        "n_state2": int((s >= 2).sum()),
    }



# =============================
# Cutoff helpers (Home drilldown)
# =============================
def _round_half_up_int(x: float) -> int:
    """Half-up rounding: 0.5 -> 1 (policy counts)."""
    try:
        x = float(x)
    except Exception:
        return 0
    return int(np.floor(x + 0.5))


def _rep_rows_from_history_month(history: pd.DataFrame, ym: str) -> pd.DataFrame:
    """One row per product for a given month (consistent representative row).

    - filter month
    - apply valid_for_cut (same denominator)
    - pick representative per 상품ID by: highest state -> highest p2 -> highest score
    """
    if history is None or history.empty:
        return pd.DataFrame()

    d = history.copy()

    # month column normalize
    if "월" in d.columns:
        d = d[d["월"].astype(str) == str(ym)].copy()
    elif "time_bucket" in d.columns:
        d = d[d["time_bucket"].astype(str) == str(ym)].copy()
    elif "month" in d.columns:
        d = d[d["month"].astype(str) == str(ym)].copy()
    else:
        return pd.DataFrame()

    d = _apply_valid_for_cut(d)
    if d.empty:
        return d

    # product
    if "product" not in d.columns:
        if "상품" in d.columns:
            d["product"] = d["상품"]
        else:
            return pd.DataFrame()
    d["상품ID"] = d["product"].astype(str).map(_extract_product_id)

    # category
    if "category" not in d.columns:
        if "Category" in d.columns:
            d["category"] = d["Category"]
        elif "카테고리" in d.columns:
            d["category"] = d["카테고리"]
        else:
            d["category"] = ""
    d["카테고리"] = d["category"].astype(str).map(_cat_ko_from_any)

    # state
    if "state" not in d.columns:
        if "state_u" in d.columns:
            d["state"] = d["state_u"]
        else:
            d["state"] = 0
    d["_state"] = pd.to_numeric(d["state"], errors="coerce").fillna(0).astype(int)

    # score (0~2)
    score_col = next((c for c in ["score", "Risk0_2", "Risk_0_2", "Risk_0_2_u", "Risk_0_2_s"] if c in d.columns), None)
    if score_col is None:
        d["_score"] = np.nan
    else:
        d["_score"] = pd.to_numeric(d[score_col], errors="coerce")

    # p2 (risk probability for top5)
    if "p2" in d.columns:
        d["_p2"] = pd.to_numeric(d["p2"], errors="coerce")
    else:
        # fallback: if p2 missing, use score as a weak proxy (still lets UI render)
        d["_p2"] = d["_score"].copy()

    # representative row per product
    d = d.sort_values(
        ["상품ID", "_state", "_p2", "_score"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    rep = d.drop_duplicates("상품ID", keep="first").copy()
    return rep


def _cutoffs_from_rep(rep: pd.DataFrame, risk_rate: float = POLICY_RISK_RATE, caution_rate: float = POLICY_CAUTION_RATE) -> dict:
    """Compute cutoff values from representative rows (one row per product).

    정책 분모(n_valid)는 rep의 상품 유니크 수(= valid_for_cut 적용 이후)로 고정하고,
    컷오프 값 계산은 실제로 p2/score가 유효한 표본에서만 수행합니다.
    (이렇게 해야 KPI 분모/비율과 컷오프 캡션 숫자가 서로 어긋나지 않습니다.)
    """
    if rep is None or rep.empty:
        return {
            "n_valid": 0,
            "risk_n": 0,
            "risk_n_eff": 0,
            "risk_p2_cut": np.nan,
            "caution_n": 0,
            "caution_n_eff": 0,
            "caution_score_cut": np.nan,
        }

    base = rep.copy()
    n_valid = int(base["상품ID"].astype(str).nunique()) if "상품ID" in base.columns else int(len(base))

    risk_n = _round_half_up_int(n_valid * float(risk_rate))
    caution_n = _round_half_up_int(n_valid * float(caution_rate))

    # risk: top by p2
    risk_p2_cut = np.nan
    risk_ids: set[str] = set()
    risk_n_eff = 0
    if risk_n > 0 and "_p2" in base.columns:
        p2 = pd.to_numeric(base["_p2"], errors="coerce")
        r_sorted = base[np.isfinite(p2)].copy()
        r_sorted["_p2_num"] = pd.to_numeric(r_sorted["_p2"], errors="coerce")
        r_sorted = r_sorted.sort_values("_p2_num", ascending=False, kind="mergesort").reset_index(drop=True)

        risk_n_eff = min(risk_n, len(r_sorted))
        if risk_n_eff > 0:
            risk_p2_cut = float(r_sorted.loc[risk_n_eff - 1, "_p2_num"])
            if "상품ID" in r_sorted.columns:
                risk_ids = set(r_sorted.head(risk_n_eff)["상품ID"].astype(str).tolist())

    # caution: next by score among remaining (exclude risk)
    caution_score_cut = np.nan
    caution_n_eff = 0
    if caution_n > 0 and "_score" in base.columns:
        rem = base[~base["상품ID"].astype(str).isin(risk_ids)].copy() if "상품ID" in base.columns else base.copy()
        sc = pd.to_numeric(rem["_score"], errors="coerce")
        rem = rem[np.isfinite(sc)].copy()
        rem["_score_num"] = pd.to_numeric(rem["_score"], errors="coerce")
        rem = rem.sort_values("_score_num", ascending=False, kind="mergesort").reset_index(drop=True)

        caution_n_eff = min(caution_n, len(rem))
        if caution_n_eff > 0:
            caution_score_cut = float(rem.loc[caution_n_eff - 1, "_score_num"])

    return {
        "n_valid": int(n_valid),
        "risk_n": int(risk_n),
        "risk_n_eff": int(risk_n_eff),
        "risk_p2_cut": risk_p2_cut,
        "caution_n": int(caution_n),
        "caution_n_eff": int(caution_n_eff),
        "caution_score_cut": caution_score_cut,
    }


def _cutoff_table_by_category(rep: pd.DataFrame, risk_rate: float = POLICY_RISK_RATE, caution_rate: float = POLICY_CAUTION_RATE) -> pd.DataFrame:
    if rep is None or rep.empty or "카테고리" not in rep.columns:
        return pd.DataFrame()

    rows = []
    for cat, g in rep.groupby("카테고리"):
        info = _cutoffs_from_rep(g, risk_rate=risk_rate, caution_rate=caution_rate)
        rows.append({
            "카테고리": str(cat),
            "유효대상(n_valid)": int(info["n_valid"]),
            "🔴위험N(5%)": int(info["risk_n"]),
            "🔴p2 컷오프": info["risk_p2_cut"],
            "🟡주의N(15%)": int(info["caution_n"]),
            "🟡score 컷오프": info["caution_score_cut"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["카테고리"] = out["카테고리"].astype(str).map(_cat_ko_from_any)
    out["카테고리"] = pd.Categorical(out["카테고리"], categories=CATEGORY_ORDER_KO, ordered=True)
    out = out.sort_values("카테고리").reset_index(drop=True)
    return out
# =============================
# Views
# =============================
def _render_home(
    queue_all: pd.DataFrame,
    data_dir: Path,
    diag_base: Path,
    ym: str,
    history: pd.DataFrame,
):
    st.markdown("## 🏠 홈 (요약)")

    # =========================
    # 0) KPI 집계(코랩 정책과 정합)
    # - 분모: 이번달 유효대상 n_valid
    # - 위험: round_half_up(n_valid * 5%)
    # - 주의: round_half_up(n_valid * 15%)
    # =========================

    # ✅ cnt는 항상 정의(분기마다 UnboundLocalError 방지)
    cnt = _month_grade_counts_from_history(history, ym)  # dict 반환
    cnt_valid = isinstance(cnt, dict) and (cnt.get("n_valid") not in (None, 0))

    # 0-1) 1순위: CATEGORY_QUEUE (월별 유효대상 n_valid가 가장 명시적)
    cq_path = _find_first(diag_base, f"*CATEGORY_QUEUE*{ym}*.csv")
    cq_df = None
    total_products = None
    if cq_path:
        try:
            cq_df = pd.read_csv(cq_path, low_memory=False)
            if "n_valid" in cq_df.columns:
                total_products = int(pd.to_numeric(cq_df["n_valid"], errors="coerce").fillna(0).sum())
        except Exception:
            cq_df = None
            total_products = None

    # 0-2) 2순위: HISTORY_ROW_PRED 기반 valid_for_cut 유니크 상품 수
    if (total_products is None) or (total_products <= 0):
        cnt = _month_grade_counts_from_history(history, ym)
        total_products = cnt.get("n_valid", None)

    # 0-3) 최후 폴백: 실제 큐 크기(Top5+Top20) - 정확한 '전체'는 아님
    if (total_products is None) or (int(total_products) <= 0):
        total_products = int(queue_all["상품ID"].astype(str).nunique()) if (queue_all is not None and not queue_all.empty and "상품ID" in queue_all.columns) else int(len(queue_all))

    denom = int(total_products) if int(total_products) > 0 else max(int(len(queue_all)), 1)

    # denom은 가능하면 history(valid_for_cut) 기준으로 통일 (5% 정책과 가장 정합)
    if cnt_valid:
        denom = int(cnt["n_valid"])
    else:
        denom = int(total_products) if (total_products is not None and int(total_products) > 0) else max(int(len(queue_all)), 1)

    # ✅ 위험/주의/일반 카운트: HISTORY의 product-level state를 우선 사용(월별 점검 등급 카운트와 동일)
    if cnt_valid:
        red_n = int(cnt.get("n_state2", 0))
        yellow_n = int(cnt.get("n_state1", 0))
        normal_n = int(cnt.get("n_state0", max(denom - red_n - yellow_n, 0)))
    else:
        # fallback: 정책(5%/15%) 기반
        red_n = _round_half_up_int(denom * POLICY_RISK_RATE)
        yellow_n = _round_half_up_int(denom * POLICY_CAUTION_RATE)
        normal_n = max(denom - red_n - yellow_n, 0)


    # ✅ 잠금(low_conf): HISTORY 기준 + '상품 단위 유니크'로 집계(중복행 방지)
    lock_n = 0
    try:
        if history is not None and not history.empty:
            h = history.copy()

            # month 컬럼 통일
            if "월" in h.columns:
                h["_month"] = h["월"].astype(str)
            elif "time_bucket" in h.columns:
                h["_month"] = h["time_bucket"].astype(str)
            elif "month" in h.columns:
                h["_month"] = h["month"].astype(str)
            else:
                h["_month"] = ""

            h = h[h["_month"] == str(ym)].copy()
            h = _apply_valid_for_cut(h)

            # 상품ID 통일
            if "상품ID" not in h.columns:
                if "product" in h.columns:
                    h["상품ID"] = h["product"].astype(str).map(_extract_product_id)
                elif "상품" in h.columns:
                    h["상품ID"] = h["상품"].astype(str).map(_extract_product_id)

            if ("reliability_flag_u" in h.columns) and ("상품ID" in h.columns):
                is_lock = (h["reliability_flag_u"].astype(str) == "low_conf").astype(int)
                lock_by_prod = is_lock.groupby(h["상품ID"].astype(str)).max()
                lock_n = int(lock_by_prod.sum())
            elif (queue_all is not None) and (not queue_all.empty) and ("상태" in queue_all.columns):
                lock_n = int(queue_all["상태"].astype(str).str.contains("잠금").sum())
    except Exception:
        lock_n = int(queue_all["상태"].astype(str).str.contains("잠금").sum()) if (queue_all is not None and not queue_all.empty and "상태" in queue_all.columns) else 0

    def pct(n: int) -> str:
        return f"{(int(n)/denom)*100:.1f}%"
    # =========================
    # 2) 이번 달 요약 KPI
    # =========================
    st.markdown("### 📌 이번 달 요약")

    

    def _msm(label: str, value: str) -> str:
        # emoji는 알파/투명도 영향을 받지 않도록 분리 렌더
        label_html = (
            label.replace("🔴", "<span class='msm-emoji'>🔴</span>")
                 .replace("🟡", "<span class='msm-emoji'>🟡</span>")
                 .replace("🔒", "<span class='msm-emoji'>🔒</span>")
        )

        # 값이 "N (p%)" 형태면 N은 크게, (p%)는 보조 크기로 표기
        main, sub = value, ""
        m = re.match(r"^(.*?)(\s*\(.*\))\s*$", str(value))
        if m:
            main, sub = m.group(1).strip(), m.group(2).strip()

        if sub:
            value_html = f"<span class='msm-main'>{main}</span><span class='msm-sub'>{sub}</span>"
        else:
            value_html = f"<span class='msm-main'>{value}</span>"

        return f"""<div class='month-summary-metric'>
  <div class='msm-label'>{label_html}</div>
  <div class='msm-value'>{value_html}</div>
</div>"""


    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _msm("전체 데이터", f"{int(denom):,}"),
        unsafe_allow_html=True,
    )
    c2.markdown(_msm("🔴 위험", f"{int(red_n):,}  ({pct(int(red_n))})"), unsafe_allow_html=True)
    c3.markdown(_msm("🟡 주의", f"{int(yellow_n):,}  ({pct(int(yellow_n))})"), unsafe_allow_html=True)
    c4.markdown(_msm("잠금(🔒)", f"{int(lock_n):,}  ({pct(int(lock_n))})"), unsafe_allow_html=True)


    # =========================
    # 3) 이번 달 컷 기준(드릴다운)
    # - 🔴위험: p2 상위 5%
    # - 🟡주의: (위험 제외) score 상위 15%  → 총 알림량(위험+주의)=20% 유지
    # =========================
    st.markdown("### 🎯 이번 달 컷 기준")

    rep_month = _rep_rows_from_history_month(history, ym)
    info_cut = _cutoffs_from_rep(rep_month, risk_rate=POLICY_RISK_RATE, caution_rate=POLICY_CAUTION_RATE)

    def _fmt_cut(x, digits=4) -> str:
        try:
            v = float(x)
            if not np.isfinite(v):
                return "—"
            return f"{v:.{digits}f}"
        except Exception:
            return "—"

    cc1, cc2 = st.columns(2, gap="large")
    cc1.markdown(_msm("🔴위험 컷오프 (p2 상위 5%)", _fmt_cut(info_cut.get("risk_p2_cut", np.nan), 4)), unsafe_allow_html=True)

    cc2.markdown(_msm("🟡주의 컷오프 (score 상위 15%, 위험 제외)", _fmt_cut(info_cut.get("caution_score_cut", np.nan), 2)), unsafe_allow_html=True)

    # =========================
    # 4) 개수 한눈에 보기(그래프)
    # =========================
    st.markdown("### 📊 개수 한눈에 보기")

    # Altair가 있으면 Altair로, 없으면 Matplotlib로 폴백
    if ALT_AVAILABLE:
        chart_df = pd.DataFrame(
            {
                "구분": ["🔴위험", "🟡주의", "🔒잠금"],
                "개수": [int(red_n), int(yellow_n), int(lock_n)],
                "정렬": [3, 2, 1],
            }
        )

        max_val = int(chart_df["개수"].max()) if not chart_df.empty else 0
        if max_val <= 50:
            x_axis = alt.Axis(values=list(range(0, max_val + 1)), format="d", labelFontSize=12, titleFontSize=12)
        else:
            x_axis = alt.Axis(format=",.0f", tickCount=6, labelFontSize=12, titleFontSize=12)

        # NOTE: 기존 코드 스타일 유지 (색 지정은 이미 파일에 존재하므로 그대로 둠)
        color_scale = alt.Scale(
            domain=["🔴위험", "🟡주의", "🔒잠금"],
            range=["#E74C3C", "#F4D03F", "#B0B0B0"],
        )

        base = (
            alt.Chart(chart_df)
            .encode(
                y=alt.Y(
                    "구분:N",
                    sort=alt.SortField(field="정렬", order="descending"),
                    title=None,
                    axis=alt.Axis(labelFontSize=13),
                    scale=alt.Scale(paddingInner=0.4),
                ),
                x=alt.X("개수:Q", title="개수", axis=x_axis),
                color=alt.Color("구분:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("구분:N", title="구분"),
                    alt.Tooltip("개수:Q", title="개수", format=",.0f"),
                ],
            )
        )

        bars = base.mark_bar(cornerRadiusEnd=8, height=35)
        labels = base.mark_text(align="left", dx=8, fontSize=14, fontWeight="bold").encode(
            text=alt.Text("개수:Q", format=",.0f")
        )

        chart = (
            (bars + labels)
            .properties(height=220)
            .configure_axis(grid=True, gridOpacity=0.15)
            .configure_view(strokeWidth=0)
        )
        _st_altair_chart(chart, width="stretch")

    else:
        fig = plt.figure(figsize=(9, 2.6), dpi=160)
        ax = plt.gca()
        labels = ["위험", "주의", "잠금"] if MPL_HAS_KR else ["Risk", "Caution", "Lock"]
        values = [int(red_n), int(yellow_n), int(lock_n)]
        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_xlabel("개수" if MPL_HAS_KR else "Count")
        st.pyplot(fig, clear_figure=True)

    # =========================
    # 5) 내 점검 리스트(요약) - queue_all 기반(Top5/Top20 실 리스트)
    # =========================
    st.markdown("---")
    st.markdown("### ✅ 내 점검 리스트(요약)")

    if queue_all.empty:
        st.info("이번 달 점검 대상이 없습니다.")
        return

    d = queue_all.copy()
    d["_score"] = pd.to_numeric(d["우선순위점수"], errors="coerce")
    # 우선순위점수 높은 순으로 정렬(요청사항)
    d = d.sort_values("_score", ascending=False, na_position="last").reset_index(drop=True)
    d["우선순위점수(0~2)"] = d["_score"].round(2)

    d["리뷰수"] = d["리뷰수"].apply(_fmt_int_or_dash)
    d["평점"] = d["평점"].apply(lambda x: _fmt_float_or_dash(x, 2))

    disp = d[["알림등급", "카테고리", "상품ID", "우선순위점수(0~2)", "리뷰수", "평점", "상태"]].copy()
    _st_dataframe(
        _with_row_numbers(disp),
        height=420,
        column_config=_progress_col_cfg("우선순위점수(0~2)"),
    )


def _build_policy_queue_from_history(history: pd.DataFrame, ym: str) -> pd.DataFrame:
    """Build a **product-level** queue list from HISTORY_ROW_PRED.

    Why this exists:
    - QUEUE_TOP5/TOP20 files can contain **multiple rows per product** (e.g., multiple topics/explanations),
      so "unique product" counts on the dashboard can collapse (449 rows -> 180 products).
    - Policy is defined on the **product unit** (5% red, 15% yellow) after `valid_for_cut` filtering.

    This function:
    1) filters to month `ym`
    2) applies `valid_for_cut` (same as KPI denominator)
    3) selects ONE representative row per product (highest state, then p2/score)
    4) keeps only products with state>=1 (yellow/red)

    Returns a dataframe that still contains the original prediction fields (product/category/score/state/...),
    but at **one row per product**.
    """
    if history is None or history.empty:
        return pd.DataFrame()

    d = history.copy()
    if "월" in d.columns:
        d = d[d["월"].astype(str) == str(ym)].copy()
    elif "time_bucket" in d.columns:
        d = d[d["time_bucket"].astype(str) == str(ym)].copy()
    else:
        return pd.DataFrame()

    # Apply the same denominator rule
    d = _apply_valid_for_cut(d)
    if d.empty:
        return d

    # Normalize key columns
    if "product" not in d.columns:
        return pd.DataFrame()

    # category column variants
    if "category" not in d.columns:
        if "Category" in d.columns:
            d["category"] = d["Category"]
        elif "카테고리" in d.columns:
            d["category"] = d["카테고리"]
        else:
            d["category"] = ""

    # state column variants
    if "state" not in d.columns:
        if "state_u" in d.columns:
            d["state"] = d["state_u"]
        else:
            d["state"] = 0

    # score/p2 column variants
    score_col = "score" if "score" in d.columns else ("Risk0_2" if "Risk0_2" in d.columns else None)
    if score_col is None:
        d["score"] = 0.0
        score_col = "score"

    p2_col = "p2" if "p2" in d.columns else None
    if p2_col is None:
        # not available -> fallback to score
        d["_p2"] = pd.to_numeric(d[score_col], errors="coerce")
        p2_col = "_p2"

    # Build product id
    d["상품ID"] = d["product"].astype(str).apply(_extract_product_id)

    # Ranking key: pick representative row per product
    d["_state"] = pd.to_numeric(d["state"], errors="coerce").fillna(0).astype(int)
    d["_score"] = pd.to_numeric(d[score_col], errors="coerce").fillna(-1.0)
    d["_p2"] = pd.to_numeric(d[p2_col], errors="coerce").fillna(-1.0)
    d["_rank2"] = np.where(d["_state"] >= 2, d["_p2"], d["_score"])

    # stable representative: highest state, then (p2 for red / score for yellow)
    d = d.sort_values(
        ["상품ID", "_state", "_rank2", "_score"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    rep = d.drop_duplicates("상품ID", keep="first").copy()

    # Keep only yellow/red
    rep = rep[rep["_state"] >= 1].copy()

    # Clean temp cols
    rep.drop(columns=[c for c in ["_state", "_score", "_p2", "_rank2"] if c in rep.columns], inplace=True)

    return rep

def _build_category_summary(queue_all: pd.DataFrame, queue_prev: pd.DataFrame) -> pd.DataFrame:
    if queue_all.empty:
        return pd.DataFrame()

    cur = queue_all.copy()
    prev = queue_prev.copy() if not queue_prev.empty else pd.DataFrame(columns=cur.columns)

    def _agg(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return pd.DataFrame({"카테고리": CATEGORY_ORDER_KO, "위험": 0, "주의": 0, "잠금비중": 0.0})

        out = (
            d.groupby("카테고리")
            .agg(
                위험=("등급", lambda s: int((s == "위험").sum())),
                주의=("등급", lambda s: int((s == "주의").sum())),
                잠금비중=("상태", lambda s: float(s.astype(str).str.contains("잠금").mean()) if len(s) > 0 else 0.0),
            )
            .reset_index()
        )
        return out

    a_cur = _agg(cur)
    a_prev = _agg(prev).rename(columns={"위험": "위험_prev", "주의": "주의_prev", "잠금비중": "잠금비중_prev"})

    out = a_cur.merge(a_prev, on="카테고리", how="left")
    out["위험_prev"] = out["위험_prev"].fillna(0).astype(int)
    out["주의_prev"] = out["주의_prev"].fillna(0).astype(int)

    out["Δ위험"] = out["위험"] - out["위험_prev"]
    out["Δ주의"] = out["주의"] - out["주의_prev"]

    out["최근 악화"] = np.where((out["Δ위험"] + out["Δ주의"]) > 0, "▲", "—")
    out["잠금비중"] = out["잠금비중"].fillna(0.0)

    out["카테고리"] = pd.Categorical(out["카테고리"], categories=CATEGORY_ORDER_KO, ordered=True)
    out = out.sort_values("카테고리")

    return out[["카테고리", "위험", "주의", "최근 악화", "잠금비중", "Δ위험", "Δ주의"]]

@st.cache_data(show_spinner=False)
def _category_trend_counts(diag_base: Path, cat_ko: str, months_asc: List[str], last_n: int = 6, ref_ym: str | None = None) -> pd.DataFrame:
    """최근 N개월 카테고리별 위험/주의/잠금 개수 추이(표시용)."""
    if not months_asc:
        return pd.DataFrame()
    months_list = months_asc
    if ref_ym and (ref_ym in months_asc):
        months_list = months_asc[: months_asc.index(ref_ym) + 1]
    months = months_list[-last_n:]
    rows = []
    for mth in months:
        p5 = _find_queue_file(diag_base, "top5", mth)
        p20 = _find_queue_file(diag_base, "top20", mth)
        if not p5 or not p20:
            continue
        try:
            df5 = _load_queue(str(p5), kind="top5")
            df20 = _load_queue(str(p20), kind="top20")
            q = _dedup_keep_highest(pd.concat([df5, df20], ignore_index=True))
            q = q[q["카테고리"].astype(str) == str(cat_ko)].copy()
            red_n = int((q["등급"] == "위험").sum()) if not q.empty else 0
            yel_n = int((q["등급"] == "주의").sum()) if not q.empty else 0
            lock_n = int(q["상태"].astype(str).str.contains("잠금").sum()) if not q.empty else 0
            rows.append({"month": mth, "risk": red_n, "caution": yel_n, "lock": lock_n})
        except Exception:
            continue
    return pd.DataFrame(rows)


def _render_category(queue_all: pd.DataFrame, queue_prev: pd.DataFrame, data_dir: Path, diag_base: Path, ym: str, months_asc: List[str]):
    st.markdown("## 🗂️ 카테고리 뷰")

    summ = _build_category_summary(queue_all, queue_prev)
    if summ.empty:
        st.info("카테고리 요약을 만들 수 없습니다.")
        return

    # ---- 테이블: 최근 악화 컬럼 삭제 + 위험/주의에 전월 대비량 포함 ----
    show = summ.copy()
    show["잠금 비중"] = show["잠금비중"].map(lambda x: f"{x*100:.0f}%")
    show["위험"] = show.apply(lambda r: f"{int(r['위험'])}({int(r['Δ위험']):+d})" if pd.notna(r.get("Δ위험")) else f"{int(r['위험'])}", axis=1)
    show["주의"] = show.apply(lambda r: f"{int(r['주의'])}({int(r['Δ주의']):+d})" if pd.notna(r.get("Δ주의")) else f"{int(r['주의'])}", axis=1)

    # ✅ 상세 진단 카테고리(REM4/추세/워드클라우드 공용)
    cat_options = show["카테고리"].astype(str).tolist()
    if not cat_options:
        st.info("카테고리 목록을 만들 수 없습니다.")
        return

    cat_focus = st.selectbox(
        "카테고리(상세 진단)",
        options=cat_options,
        index=0,
        key="cat_focus",
    )


    # 여백을 그래프로 활용: 좌(표) / 우(추세선)
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        show_tbl = show[["카테고리", "위험", "주의", "잠금 비중"]].rename(
            columns={"위험": "🔴위험(전월 대비)", "주의": "🟡주의(전월 대비)"}
        )

        # ✅ 5행만 표시
        show_tbl = show_tbl.head(5)

        # ✅ 행번호 부여(표시용)
        show_tbl = _with_row_numbers(show_tbl)

        # ✅ 높이 자동 계산 (5행에 딱 맞춤)
        row_h = 35
        header_h = 35
        table_h = header_h + row_h * min(5, len(show_tbl))

        _st_dataframe(show_tbl, height=table_h)

    with right:
        st.markdown("### 📈 추세(최근 6개월)")
        tr = _category_trend_counts(diag_base, cat_focus, months_asc, last_n=6, ref_ym=ym)
        if not tr.empty:
            fig = plt.figure(figsize=(6.2, 3.2), dpi=180)
            ax = plt.gca()
            ax.plot(tr["month"], tr["risk"], marker="o", label=("위험" if MPL_HAS_KR else "Risk"), color="red", linewidth=2)
            ax.plot(tr["month"], tr["caution"], marker="o", label=("주의" if MPL_HAS_KR else "Caution"), color="gold", linewidth=2)
            ax.set_xlabel("월" if MPL_HAS_KR else "Month")
            ax.set_ylabel("개수" if MPL_HAS_KR else "Count")
            ax.tick_params(axis="both", labelsize=10)
            ax.legend(fontsize=10, loc="best")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    # =========================
    # ✅ REM4 진단(카테고리 레벨 전용)
    # =========================
    st.markdown("---")
    st.markdown("### 🔷 리뷰 신호 지표")
    rem4_cm_all = _load_rem4_category_month_all(str(data_dir))
    _render_rem4_category_widget(rem4_cm_all, cat_focus, ym)

    # =========================
    # ✅ 워드클라우드: cat_focus와 자동 동기화 (경고 제거 버전)
    # =========================
    st.markdown("---")
    st.markdown("### ☁️ 카테고리 워드클라우드")

    wc_options = ["전체"] + CATEGORY_ORDER_KO

    # 1) 초기값 세팅(위젯 생성 전에만!)
    if "cat_wc_sel" not in st.session_state:
        st.session_state["cat_wc_sel"] = cat_focus if cat_focus in wc_options else "전체"

    # 2) cat_focus 바뀌면 워드클라우드도 동기화
    prev_focus = st.session_state.get("_cat_focus_prev")
    if prev_focus != cat_focus:
        # 사용자가 '전체'를 고른 경우는 존중(고정)
        if st.session_state.get("cat_wc_sel") != "전체":
            st.session_state["cat_wc_sel"] = cat_focus
        st.session_state["_cat_focus_prev"] = cat_focus

    # 3) 값이 옵션 밖이면 보정
    if st.session_state.get("cat_wc_sel") not in wc_options:
        st.session_state["cat_wc_sel"] = "전체"

    # 4) 위젯 생성: index 주지 말고 key만!
    cat_sel = st.selectbox(
        "카테고리 선택",
        options=wc_options,
        key="cat_wc_sel",
    )

    if cat_sel == "전체":
        cat_hint = "ALL_CATEGORIES"
    else:
        cat_hint = CAT_KO_TO_EN_HINT.get(cat_sel)
        if not cat_hint:
            st.info("카테고리 힌트를 찾을 수 없습니다.")
            return

    wc_all = find_category_wordcloud_all(data_dir, cat_hint, ym)
    wc_risk = find_category_wordcloud_top(data_dir, cat_hint, ym, mode="risk")
    wc_caution = find_category_wordcloud_top(data_dir, cat_hint, ym, mode="caution")

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("#### 전체")
        if wc_all:
            _st_image(str(wc_all), width="stretch")
        else:
            st.info("전체 워드클라우드 이미지가 없습니다. (wordclouds_DARKGRAY 경로 확인)")

    with c2:
        st.markdown("#### 🔴위험")
        if wc_risk:
            _st_image(str(wc_risk), width="stretch")
        else:
            st.info("위험 워드클라우드 이미지가 없습니다. (top5/CT_top5_wc whitebg 경로 확인)")

    with c3:
        st.markdown("#### 🟡주의")
        if wc_caution:
            _st_image(str(wc_caution), width="stretch")
        else:
            st.info("주의 워드클라우드 이미지가 없습니다. (top20/CT_top20_wc whitebg 경로 확인)")

    # (2) 폴백:
    if (cat_sel != "전체") and (wc_all is None) and (wc_risk is None) and (wc_caution is None):
        st.markdown("##### (폴백) 키워드 텍스트")
        rev_p = _find_review_evidence_file(data_dir, cat_hint)
        if not rev_p:
            st.info("해당 카테고리 리뷰 파일(*train_with_rem4*)을 찾지 못했습니다.")
            return

        rev = _load_review_evidence(str(rev_p))
        if "month" in rev.columns:
            rev = rev[rev["month"].astype(str) == str(ym)].copy()
        if "is_aug" in rev.columns:
            rev = rev[rev["is_aug"] == 0].copy()

        if "text_for_nlp" not in rev.columns or rev.empty:
            st.info("키워드를 만들 텍스트가 부족합니다.")
            return

        neg_col = None
        for c in NEG_SCORE_CANDIDATES:
            if c in rev.columns:
                neg_col = c
                break

        rev_all_text = rev["text_for_nlp"].dropna().astype(str).tolist()
        rev_neg = rev.copy()
        if neg_col:
            rev_neg[neg_col] = pd.to_numeric(rev_neg[neg_col], errors="coerce")
            rev_neg = rev_neg[rev_neg[neg_col] >= 0.8]
        elif "rating" in rev_neg.columns:
            rev_neg["rating"] = pd.to_numeric(rev_neg["rating"], errors="coerce")
            rev_neg = rev_neg[rev_neg["rating"] <= 2]
        else:
            rev_neg = rev_neg.iloc[0:0]

        rev_neg_text = rev_neg["text_for_nlp"].dropna().astype(str).tolist()

        cL2, cR2 = st.columns(2, gap="large")
        with cL2:
            st.markdown("**전체 키워드(텍스트)**")
            kws_all = _extract_keywords(rev_all_text, topn=30)
            st.write(", ".join(kws_all) if kws_all else "키워드를 추출할 수 없습니다.")
        with cR2:
            st.markdown("**부정 리뷰 키워드(텍스트)**")
            kws_neg = _extract_keywords(rev_neg_text, topn=30)
            st.write(", ".join(kws_neg) if kws_neg else "부정 리뷰가 부족해 키워드를 추출할 수 없습니다.")

def _render_product(queue_all: pd.DataFrame, ym: str, product_hint: str, category_filter: str, data_dir: Path, selected_date: Optional[date] = None):
    st.markdown("## ▶️ 상품 뷰")

    if queue_all.empty:
        st.info("이번 달 점검 대상이 없습니다.")
        return

    # -------------------------
    # 상단 필터바: 카테고리/등급/상태/정렬
    # -------------------------
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

    cat_opt = ["전체"] + CATEGORY_ORDER_KO
    cat_pick = f1.selectbox("카테고리", options=cat_opt, index=cat_opt.index(category_filter) if category_filter in cat_opt else 0)
    grade_pick = f2.selectbox("알림등급", options=["전체", "🔴위험", "🟡주의"], index=0)
    status_pick = f3.selectbox("상태", options=["전체", "✅ 기준 통과", "🔒 잠금(수동점검)", "⚪ 정보부족(Gray)"], index=0)
    sort_pick = f4.selectbox("정렬", options=["우선순위 높은 순", "리뷰수 많은 순"], index=0)

    d = queue_all.copy()

    if product_hint:
        d = d[d["상품ID"].astype(str).str.contains(product_hint, na=False)].copy()

    if cat_pick != "전체":
        d = d[d["카테고리"] == cat_pick].copy()

    if grade_pick != "전체":
        g = "위험" if "위험" in grade_pick else "주의"
        d = d[d["등급"] == g].copy()

    if status_pick != "전체":
        d = d[d["상태"] == status_pick].copy()

    d["_score"] = pd.to_numeric(d["우선순위점수"], errors="coerce").fillna(-1)
    d["_reviews"] = pd.to_numeric(d["리뷰수"], errors="coerce").fillna(-1)

    if sort_pick == "리뷰수 많은 순":
        d = d.sort_values("_reviews", ascending=False)
    else:
        d = d.sort_values("_score", ascending=False)

    d["우선순위점수(0~2)"] = d["_score"].round(2).replace(-1, np.nan)
    d["리뷰수"] = d["리뷰수"].apply(_fmt_int_or_dash)
    d["평점"] = d["평점"].apply(lambda x: _fmt_float_or_dash(x, 2))

    # -------------------------
    # 점검 리스트 표
    # -------------------------
    list_df = d[["알림등급", "카테고리", "상품ID", "우선순위점수(0~2)", "리뷰수", "평점", "상태"]].copy()
    _st_dataframe(_with_row_numbers(list_df), height=420, column_config=_progress_col_cfg("우선순위점수(0~2)"))

    st.markdown("---")
    st.markdown("### ▶️ 상품 상세")

    pid_list = list_df["상품ID"].dropna().astype(str).unique().tolist()
    if not pid_list:
        st.info("필터 결과가 없습니다.")
        return

    pid_sel = st.selectbox("상품 선택(상품ID)", options=pid_list, index=0)
    url = COUPANG_URL_FMT.format(product_id=pid_sel)
    st.markdown(f"- 상품 URL: {url}")

    row = d[d["상품ID"].astype(str) == str(pid_sel)].head(1)
    grade = row["등급"].iloc[0] if not row.empty else "—"
    status = row["상태"].iloc[0] if not row.empty else "—"
    cat_ko = row["카테고리"].iloc[0] if not row.empty else "출산/유아동"
    cat_hint = CAT_KO_TO_EN_HINT.get(_cat_ko_from_any(cat_ko), "Baby")

    # 해당 상품 카테고리의 리뷰/토픽 파일 로드 (없으면 None)
    rev = None
    rev_p = _find_review_evidence_file(data_dir, cat_hint)
    if rev_p:
        rev = _load_review_evidence(str(rev_p))

    # -------------------------
    # ✅ 상품 URL 아래 "2단(좌/우)" 구성
    # -------------------------
    left, right = st.columns([1.05, 0.95])

    # ===== LEFT =====
    with left:
        # 1) 추천 액션 + 상태 배지
        # ✅ 추천 액션을 '진단→근거→실행' 3단으로 구체화합니다.
        # - 현재 단계에서는 grade/status만으로도 의미 있는 체크리스트를 제공합니다.
        # - (선택) 아래에서 토픽 급변/부정리뷰/REM4 등을 계산했다면, 그 결과를 '근거'에 추가해도 됩니다.

        def _grade_badge(g: str) -> str:
            g = str(g or "").strip()
            if g == "위험":
                return _badge("🔴위험")
            if g == "주의":
                return _badge("🟡주의")
            return _badge("일반")

        def _action_pack(g: str, topics: list[str] | None = None, keywords: list[str] | None = None) -> dict:
            """추천 액션 패키지(진단→근거→실행).

            - grade(위험/주의/일반) 1차 분기
            - topics(토픽 급변 Top2 등) / keywords(부정 리뷰 키워드 등)로 2차 세분화
            - topics/keywords가 없어도 동작하도록 설계
            """
            g = str(g or "").strip()
            topics = [str(t).strip() for t in (topics or []) if str(t).strip()]
            keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]

            # -------------------------
            # 1) 토픽/키워드 → '이슈 타입' 추정(간단 룰 기반)
            # -------------------------
            def _infer_issue_types(tokens: list[str]) -> list[str]:
                s = " ".join(tokens).lower()
                types: list[str] = []

                def has(*ws: str) -> bool:
                    return any(w.lower() in s for w in ws)

                if has("배송", "지연", "늦", "로켓", "택배", "물류", "파손", "포장", "박스"):
                    types.append("logistics")
                if has("불량", "고장", "품질", "내구", "하자", "작동", "오작동", "누전", "발열"):
                    types.append("quality")
                if has("구성", "구성품", "누락", "빠짐", "부품", "설명서", "설명", "가이드"):
                    types.append("missing_or_manual")
                if has("사용", "설치", "조립", "연결", "설정", "호환", "연동", "인식"):
                    types.append("usability_compat")
                if has("사이즈", "크기", "치수", "작다", "크다", "핏"):
                    types.append("size")
                if has("냄새", "소음", "소리", "진동", "먼지"):
                    types.append("sensory")
                if has("배터리", "충전", "전원", "전기", "전압"):
                    types.append("battery_power")
                if has("피부", "자극", "알레르기", "성분", "가려", "두드러기", "독", "안전", "위생"):
                    types.append("safety_ingredient")

                return types or ["generic"]

            issue_types = _infer_issue_types(topics + keywords)

            # -------------------------
            # 2) 이슈 타입별 액션(grade별로 톤/우선순위 조정)
            # -------------------------
            def _topic_actions_for_grade(gt: str, itypes: list[str]) -> tuple[list[str], list[str]]:
                """(cause_add, do_add)"""
                cause_add: list[str] = []
                do_add: list[str] = []

                # 공통: 토픽/키워드가 있으면 근거에 '특정 이슈'를 명시
                if topics:
                    cause_add.append(f"토픽 급변(Top): {', '.join(topics[:2])}")
                if keywords:
                    cause_add.append(f"부정 키워드: {', '.join(keywords[:5])}")

                # 타입별 플레이북
                for t in itypes:
                    if t == "logistics":
                        do_add += [
                            "**배송/포장/파손**: 출고 리드타임·택배사 이슈·포장 상태 점검(동일 월 리뷰에서 반복되는지)",
                            "배송지연/파손 시 **사전 안내 문구**(배송 일정/보상 기준) 업데이트",
                        ]
                        if gt == "위험":
                            do_add += ["반품/교환 급증 여부 확인 후 **CS 우선 대응(스크립트/FAQ)** 적용"]
                    elif t == "quality":
                        do_add += [
                            "**품질/불량**: LOT/제조일/공급사 변경 여부 확인(특정 기간 집중 발생 여부)",
                            "불량 유형이 명확하면 **교환/환불 기준**과 검수 프로세스(입고/출고) 점검",
                        ]
                        if gt == "위험":
                            do_add += ["동일 불량 다발 시 **판매 일시중지/리콜 검토(내부 기준)**"]
                    elif t == "missing_or_manual":
                        do_add += [
                            "**구성품/누락/설명**: 구성품 표기·옵션 매칭·패키징 체크리스트 점검",
                            "설명서/가이드 부족이면 **이미지/동영상 가이드** 추가(오해 포인트 우선)",
                        ]
                    elif t == "usability_compat":
                        do_add += [
                            "**사용/설치/호환**: 호환 기기/환경/설치 조건을 상세페이지에 명확히 표기",
                            "초기 불만이 많으면 **설치 가이드(3~5 step)**를 상단에 고정",
                        ]
                    elif t == "size":
                        do_add += [
                            "**사이즈/크기**: 실측/착용/비교 이미지 보강 + 오차 범위 안내",
                            "‘작다/크다’ 반복 시 사이즈 선택 가이드(추천 기준) 추가",
                        ]
                    elif t == "sensory":
                        do_add += [
                            "**냄새/소음/진동**: 사용 초기/환경 조건(환기/세척/시간 경과) 안내 강화",
                            "원인 추정이 가능하면 **해결 팁(FAQ)**로 즉시 연결",
                        ]
                    elif t == "battery_power":
                        do_add += [
                            "**배터리/전원**: 사용 시간/충전 조건/호환 어댑터 스펙을 재검증(과장 표기 방지)",
                            "펌웨어/설정 이슈가 있으면 초기 세팅 가이드 제공",
                        ]
                    elif t == "safety_ingredient":
                        do_add += [
                            "**안전/성분/자극**: 성분표/주의사항/연령·피부 타입 가이드를 명확히 표기",
                            "자극/알레르기 언급 증가 시 **경고 문구 + 고객 케어 프로세스** 점검",
                        ]
                        if gt == "위험":
                            do_add += ["민감 이슈 다발 시 **리스크 커뮤니케이션(공지/FAQ) 우선 적용**"]
                    else:
                        # generic
                        do_add += ["핵심 이슈가 모호하면 최근 부정 리뷰 5개를 읽고 **원인 1~2개로 수렴**시키기"]

                # 중복 제거(순서 유지)
                def dedup(xs: list[str]) -> list[str]:
                    out=[]
                    seen=set()
                    for x in xs:
                        if x not in seen:
                            out.append(x); seen.add(x)
                    return out

                return dedup(cause_add), dedup(do_add)

            # 1차 기본 pack
            if g == "위험":
                base = {
                    "summary": "급격 악화 신호(🔴위험) → **즉시 원인 분리 + 고객영향 최소화**가 우선입니다.",
                    "cause": [
                        "최근 1~2주 **부정 리뷰/CS 이슈**가 특정 원인(배송/품질/구성/AS 등)으로 수렴하는지 확인",
                        "**평점 하락**이 동반되는지(저평점 비중↑) 확인",
                        "전월 대비 **토픽/키워드 급변** 여부(급증 토픽이 있으면 우선 점검)",
                    ],
                    "do": [
                        "**상품페이지/옵션/스펙 표기** 즉시 점검(오해 소지/누락/조건 미기재)",
                        "**CS 우선순위 상향**(동일 이슈 반복 시 템플릿 답변 적용)",
                        "다음 달(또는 2주 후) 동일 지표/토픽이 **재악화**되는지 모니터링(재발 시 escal.)",
                    ],
                }
            elif g == "주의":
                base = {
                    "summary": "완만한 악화 신호(🟡주의) → **사전 안내/가설 검증 + 모니터링 강화**가 적합합니다.",
                    "cause": [
                        "부정 리뷰에서 반복되는 **핵심 키워드 3~5개** 추출(예: 배터리/소음/누락/냄새)",
                        "토픽 share 변화가 있으면 **증가 토픽 1~2개** 원인 점검",
                        "리뷰수 증가/감소에 따른 **노이즈(표본 변화)** 여부 확인",
                    ],
                    "do": [
                        "상품페이지 내 **주의사항/사용조건/사이즈/구성품** 안내 보강",
                        "반품·교환 사유 Top을 확인하고 **사전 차단 문구** 추가",
                        "다음 달 동일 토픽/키워드가 재증가하면 🔴위험 대응으로 전환",
                    ],
                }
            else:
                base = {
                    "summary": "현재는 일반 상태 → **가벼운 모니터링**으로 충분합니다.",
                    "cause": [
                        "리뷰수/평점이 급변하는지 월 단위로 확인",
                        "주요 키워드 변화가 생기면 원인 점검(주의로 전환)",
                    ],
                    "do": [
                        "월 1회 기준으로 지표/토픽 변화만 확인",
                    ],
                }

            # 2차 세분화(토픽/키워드 기반)
            cause_add, do_add = _topic_actions_for_grade(g, issue_types)
            if cause_add:
                base["cause"] = cause_add + base["cause"]
            if do_add:
                # grade별로 너무 길어지는 것 방지: 위험은 상단 6개, 주의는 상단 5개 정도만 노출
                max_n = 6 if g == "위험" else (5 if g == "주의" else 4)
                base["do"] = (do_add + base["do"])[:max_n]

            return base

        # 토픽/키워드 준비(가능하면)
        topics_for_action: list[str] = []
        keywords_for_action: list[str] = []
        try:
            if rev is not None and isinstance(rev, pd.DataFrame) and (not rev.empty):
                top2_df, _note = _topic_shift_top2(rev, pid_sel, ym, top_k=2)
                if top2_df is not None and not top2_df.empty:
                    if "topic" in top2_df.columns:
                        topics_for_action = top2_df["topic"].astype(str).head(2).tolist()
                    elif "토픽" in top2_df.columns:
                        topics_for_action = top2_df["토픽"].astype(str).head(2).tolist()

                # 부정 리뷰에서 키워드 뽑기(간단 룰) — 토픽이 없거나 보조 근거로 사용
                neg_df = _product_reviews_topn(rev, pid_sel, ym, topn=8, rating_max=2, date_filter=selected_date)
                if neg_df is not None and (not neg_df.empty) and ("text_for_nlp" in neg_df.columns):
                    keywords_for_action = _extract_keywords(neg_df["text_for_nlp"].dropna().astype(str).tolist(), topn=10)
        except Exception:
            pass

        pack = _action_pack(grade, topics=topics_for_action, keywords=keywords_for_action)

        st.markdown("#### 1) 추천 액션")
        st.markdown(
            f"{_grade_badge(grade)} &nbsp;&nbsp; {_badge(status)}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**상태 요약:** {pack['summary']}")
        st.markdown("**원인 체크(근거):**")
        st.markdown("\n".join([f"- {x}" for x in pack["cause"]]))
        st.markdown("**실행 체크리스트:**")
        st.markdown("\n".join([f"- {x}" for x in pack["do"]]))


        # 2) 토픽 급변 Top2(전월 대비)
        st.markdown("#### 2) 토픽 급변 Top2(전월 대비)")
        if rev is None or rev.empty:
            st.info("토픽/리뷰 파일(*train_with_rem4*)을 찾지 못했습니다.")
        else:
            top2, note = _topic_shift_top2(rev, pid_sel, ym, top_k=2)
            if top2.empty:
                st.info("토픽 급변을 계산할 수 없습니다(전월/토픽 데이터 부족).")
            else:
                # 요청 형식: 토픽 / 전월 / 이번달(변화량)  (share/방향 제거)
                top2_disp = _format_topic_shift_display(top2, ym)
                top2_disp = top2_disp.head(2)  # 안전 가드 (혹시 모를 초과 방지)

                # 🔹 행 수에 맞춰 높이 자동 계산
                row_h = 35      # 한 행 높이(대략)
                header_h = 35   # 헤더 높이
                table_h = header_h + row_h * len(top2_disp)

                _st_dataframe(_with_row_numbers(top2_disp), height=table_h)

        # 3) 상품 워드클라우드
        st.markdown("#### 3) 상품 워드클라우드")
        mode = st.radio(
                label="",
                options=["전체", "위험", "주의"],
                horizontal=True,
                index=0,
                key="prod_wc_mode",
                label_visibility="collapsed"   # ← 이게 제일 깔끔
            )

        mode_key = "all" if mode == "전체" else ("risk" if mode == "위험" else "caution")
        wc_prod = find_product_wordcloud_image(data_dir, cat_hint, str(pid_sel), mode=mode_key, allow_fallback_to_all=(mode_key == "all"))

        if wc_prod:
            _st_image(str(wc_prod), width='stretch')
        else:
            st.info("상품 워드클라우드 이미지가 없습니다. (product_wordclouds_all/whitebg 또는 top5/top20 whitebg 경로 확인)")
            # 폴백: 텍스트 키워드
            if rev is None or rev.empty:
                st.info("리뷰 데이터를 불러올 수 없어 키워드를 생성할 수 없습니다.")
            else:
                rr = rev[(rev["상품ID"].astype(str) == str(pid_sel))].copy()
                if "month" in rr.columns:
                    rr = rr[rr["month"].astype(str) == str(ym)].copy()
                if "is_aug" in rr.columns:
                    rr = rr[rr["is_aug"] == 0].copy()

                if "text_for_nlp" not in rr.columns or rr.empty:
                    st.info("키워드를 만들 텍스트가 부족합니다.")
                else:
                    neg_col = None
                    for c in NEG_SCORE_CANDIDATES:
                        if c in rr.columns:
                            neg_col = c
                            break

                    rr_use = rr.copy()
                    if mode != "전체":
                        if neg_col:
                            rr_use[neg_col] = pd.to_numeric(rr_use[neg_col], errors="coerce")
                            if mode == "위험":
                                rr_use = rr_use[rr_use[neg_col] >= 0.8]
                            else:
                                rr_use = rr_use[(rr_use[neg_col] >= 0.6) & (rr_use[neg_col] < 0.8)]
                        elif "rating" in rr_use.columns:
                            rr_use["rating"] = pd.to_numeric(rr_use["rating"], errors="coerce")
                            if mode == "위험":
                                rr_use = rr_use[rr_use["rating"] <= 2]
                            else:
                                rr_use = rr_use[rr_use["rating"] == 3]
                        else:
                            rr_use = rr_use.iloc[0:0]

                    kws = _extract_keywords(rr_use["text_for_nlp"].dropna().astype(str).tolist(), topn=30)
                    if not kws:
                        st.info("선택한 조건에서 키워드를 만들 텍스트가 부족합니다.")
                    else:
                        st.write(", ".join(kws[:30]))

    # ===== RIGHT =====
    with right:

        st.markdown("#### 4) 대표 리뷰(근거)")
        if rev is None or rev.empty:
            st.info("리뷰 원문 파일을 찾지 못했습니다. (예: *train_with_rem4*sentA*.csv 를 data/에 넣어주세요)")
        else:
            use_low_rating = st.checkbox("저평점(≤3)만 보기", value=False)
            rating_max = 3 if use_low_rating else None

            if selected_date is not None:
                st.caption(f"※ {selected_date} 날짜의 리뷰만 표시 중 (컷오프/정책은 월 기준 유지)")

            top = _product_reviews_topn(rev, str(pid_sel), ym, topn=5, rating_max=rating_max, date_filter=selected_date)
            if top.empty:
                st.info("해당 상품/월의 리뷰가 없거나 필터 조건에 맞는 데이터가 없습니다.")
            else:
                score_col = next((c for c in NEG_SCORE_CANDIDATES if c in top.columns), None)

                for i in range(len(top)):
                    r = top.loc[i].to_dict()
                    rating = r.get("rating", None)
                    date = r.get("date", "")
                    topic = r.get("topic", "")
                    txt = str(r.get("text_for_nlp", ""))

                    header = f"**{i+1}.** {_stars(rating)} | {date} | 토픽: `{topic}`"
                    if score_col and r.get(score_col, None) is not None and pd.notna(r.get(score_col)):
                        header += f" | 부정확률: `{float(r.get(score_col)):.2f}`"

                    st.markdown(header)
                    st.write(txt[:240] + ("…" if len(txt) > 240 else ""))
                    st.divider()


def main():

    data_dir = _resolve_data_dir()
    diag_base = _resolve_diag_base_dir(data_dir)

    months_asc = _list_months(diag_base)
    months_desc = list(reversed(months_asc))

    if not months_desc:
        st.error("월(YYYY-MM) 정보를 찾지 못했습니다. diag 폴더/CSV 파일명을 확인해주세요.")
        st.info(f"탐색 경로: {diag_base}")
        return

    # Sidebar
    st.sidebar.markdown("## 설정")
    view = st.sidebar.radio("보기", ["홈(요약)", "카테고리", "상품"], index=0)
    ym = st.sidebar.selectbox("기준 월(최신순)", options=months_desc, index=0)
    # 월 기준 정책은 그대로 두고, 화면(리뷰/키워드)만 특정 날짜로 좁힐 수 있는 옵션
    narrow_by_day = st.sidebar.checkbox("상세 날짜로 좁히기", value=False)
    selected_date: Optional[date] = None
    if narrow_by_day:
        try:
            per = pd.Period(str(ym), freq="M")
            month_start = per.start_time.date()
            month_end = per.end_time.date()
            selected_date = st.sidebar.date_input("상세 날짜", value=month_end, min_value=month_start, max_value=month_end)
        except Exception:
            # ym 파싱 실패 시: 제한 없이 날짜 입력
            selected_date = st.sidebar.date_input("상세 날짜", value=date.today())


    product_hint = ""
    category_filter = "전체"
    if view == "상품":
        st.sidebar.markdown("## 상품 화면 옵션")
        product_hint = st.sidebar.text_input("상품ID(숫자)", placeholder="예) 1203623707").strip()
        category_filter = st.sidebar.selectbox("카테고리(상품 화면에만 적용)", ["전체"] + CATEGORY_ORDER_KO, index=0)
        _sidebar_help()
    else:
        _sidebar_help()
    ym_prev = _prev_available_month(months_asc, ym)

    # Load history
    history = _load_history(diag_base)

    # Build policy queue (product-level) from history (canonical)
    policy_cur_raw = _build_policy_queue_from_history(history, ym)

    if policy_cur_raw.empty:
        st.warning("HISTORY에서 해당 월의 정책 큐를 만들 수 없어 QUEUE_TOP5/QUEUE_TOP20로 대체합니다.")
        # Load current month queues (fallback)
        top5_p = _find_queue_file(diag_base, "top5", ym)
        top20_p = _find_queue_file(diag_base, "top20", ym)

        if not top5_p or not top20_p:
            st.error("필수 CSV(HISTORY_ROW_PRED 또는 QUEUE_TOP5/QUEUE_TOP20)를 찾지 못했습니다.")
            st.info(f"탐색 경로: {diag_base}")
            return

        top5 = _prepare_queue(_read_csv(str(top5_p)), assume_level="위험")
        top20_raw = _prepare_queue(_read_csv(str(top20_p)), assume_level="주의")

        # Top20 파일이 state>=1 전체일 수 있으니 "주의"만 남김
        top20_yellow = top20_raw[top20_raw["등급"] == "주의"].copy() if "등급" in top20_raw.columns else top20_raw.copy()

        # 1차 보강: history
        top5 = _enrich_queue_with_history(top5, history, ym)
        top20_yellow = _enrich_queue_with_history(top20_yellow, history, ym)

        queue_all = _dedup_keep_highest(pd.concat([top5, top20_yellow], ignore_index=True))
    else:
        queue_all = _prepare_queue(policy_cur_raw, assume_level="주의")
        queue_all = _dedup_keep_highest(queue_all)
        queue_all = _enrich_queue_with_history(queue_all, history, ym)

    # ✅ review file (리뷰수/평점 None 해결)
    queue_all = _fill_reviews_and_rating_from_review_files(queue_all, ym, data_dir)

    # prev month queues for category Δ
    queue_prev_all = pd.DataFrame()
    if ym_prev:
        policy_prev_raw = _build_policy_queue_from_history(history, ym_prev)
        if not policy_prev_raw.empty:
            queue_prev_all = _prepare_queue(policy_prev_raw, assume_level="주의")
            queue_prev_all = _dedup_keep_highest(queue_prev_all)
            queue_prev_all = _enrich_queue_with_history(queue_prev_all, history, ym_prev)
            queue_prev_all = _fill_reviews_and_rating_from_review_files(queue_prev_all, ym_prev, data_dir)
        else:
            # fallback to queue files
            p5_prev = _find_queue_file(diag_base, "top5", ym_prev)
            p20_prev = _find_queue_file(diag_base, "top20", ym_prev)
            if p5_prev and p20_prev:
                top5_prev = _prepare_queue(_read_csv(str(p5_prev)), assume_level="위험")
                top20_prev_raw = _prepare_queue(_read_csv(str(p20_prev)), assume_level="주의")
                top20_prev = top20_prev_raw[top20_prev_raw["등급"] == "주의"].copy() if "등급" in top20_prev_raw.columns else top20_prev_raw
                queue_prev_all = _dedup_keep_highest(pd.concat([top5_prev, top20_prev], ignore_index=True))
                queue_prev_all = _fill_reviews_and_rating_from_review_files(queue_prev_all, ym_prev, data_dir)

    # Render views
    if view == "홈(요약)":
        _render_home(queue_all, data_dir, diag_base, ym, history)
    elif view == "카테고리":
        _render_category(queue_all, queue_prev_all, data_dir, diag_base, ym, months_asc)
    else:
        _render_product(queue_all, ym, product_hint, category_filter, data_dir, selected_date)

    # 화면 최상/최하단 잘림 방지용 여백
    st.markdown("<div style='height:56px;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()