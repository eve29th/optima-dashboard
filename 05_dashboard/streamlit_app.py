# -*- coding: utf-8 -*-
"""
Optima - 쿠팡 리뷰 불만 분석 대시보드
실행: streamlit run 05_dashboard/streamlit_app.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

ROOT = Path(__file__).parent.parent

# ── 폰트 설정 (로컬: Malgun Gothic / 배포: NanumGothic) ──
_local_font  = Path(r"C:\Windows\Fonts\malgunbd.ttf")
_deploy_font = ROOT / "fonts" / "NanumGothic.ttf"

if _local_font.exists():
    FONT_PATH = str(_local_font)
    plt.rcParams["font.family"] = "Malgun Gothic"
elif _deploy_font.exists():
    fm.fontManager.addfont(str(_deploy_font))
    FONT_PATH = str(_deploy_font)
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_deploy_font)).get_name()
else:
    FONT_PATH = None
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.color"]         = "white"
plt.rcParams["axes.labelcolor"]    = "white"
plt.rcParams["xtick.color"]        = "white"
plt.rcParams["ytick.color"]        = "white"
plt.rcParams["axes.edgecolor"]     = "white"

st.set_page_config(page_title="Optima 리뷰 분석", layout="wide")

# ── 전역 스타일 ──
st.markdown("""
<style>
/* 표 배경 제거 */
[data-testid="stDataFrame"] iframe { background: transparent !important; }
div[data-testid="stDataFrameResizable"] { background: transparent !important; }
.stDataFrame { background: transparent !important; }

/* metric 카드 스타일 */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 12px 16px;
}

/* 신호등 배지 */
.badge-danger  { background:#FF4B4B; color:white; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:700; }
.badge-warning { background:#FFA500; color:white; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:700; }
.badge-normal  { background:#21BA45; color:white; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 (CSV) ──
DATA_DIR = ROOT / "data" / "processed"

@st.cache_data
def load_data():
    reviews  = pd.read_csv(DATA_DIR / "reviews_slim.csv", encoding="utf-8-sig")
    daily    = pd.read_csv(DATA_DIR / "product_daily.csv",          encoding="utf-8-sig")
    kw_daily = pd.read_csv(DATA_DIR / "keyword_daily.csv",          encoding="utf-8-sig")
    rem      = pd.read_csv(DATA_DIR / "rem_metrics.csv",            encoding="utf-8-sig")

    # 날짜 파싱
    reviews["review_date"] = pd.to_datetime(
        reviews["date"].astype(str).str.replace(".", "-", regex=False), errors="coerce"
    )
    daily["review_date"]  = pd.to_datetime(daily["review_date"],  errors="coerce")
    rem["bucket_month"]   = pd.to_datetime(rem["bucket_month"],   errors="coerce")

    # complaint_type 컬럼명 통일
    if "complaint_type_3" in reviews.columns and "complaint_type" not in reviews.columns:
        reviews.rename(columns={"complaint_type_3": "complaint_type"}, inplace=True)

    return reviews, daily, kw_daily, rem

reviews, daily, kw_daily, rem = load_data()

CATEGORIES = sorted(reviews["category"].dropna().unique())
COMPLAINT_TYPES = ["defect", "transaction", "usability", "fit"]
COMPLAINT_LABELS = {
    "defect": "결함/불량",
    "transaction": "배송/CS",
    "usability": "사용성",
    "fit": "적합성",
}

def signal_badge(score):
    """리뷰 신호 지표 점수 → 신호등 배지"""
    if score < 33:
        return '<span class="badge-danger">⚠ 위험</span>'
    elif score < 66:
        return '<span class="badge-warning">△ 주의</span>'
    else:
        return '<span class="badge-normal">✓ 정상</span>'

# ══════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════
st.sidebar.title("필터")
sel_category = st.sidebar.multiselect("카테고리", CATEGORIES, default=CATEGORIES)
date_min = reviews["review_date"].min().date()
date_max = reviews["review_date"].max().date()
sel_date = st.sidebar.date_input(
    "기간", value=(date_min, date_max),
    min_value=date_min, max_value=date_max
)

if len(sel_date) == 2:
    d_from, d_to = pd.Timestamp(sel_date[0]), pd.Timestamp(sel_date[1])
else:
    d_from = d_to = pd.Timestamp(sel_date[0])

rv = reviews[reviews["category"].isin(sel_category) & reviews["review_date"].between(d_from, d_to)]
dy = daily[daily["category"].isin(sel_category) & daily["review_date"].between(d_from, d_to)]
rm = rem[rem["category"].isin(sel_category)]

# ══════════════════════════════════════════
# 탭 구성
# ══════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 전체 현황", "📈 일별 추이", "☁️ 키워드 분석", "🔍 상품별 리뷰", "💡 추천 액션"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 - 전체 현황
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.header("전체 현황")

    # ── KPI 지표 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 리뷰 수", f"{len(rv):,}")
    c2.metric("평균 별점", f"{rv['rating'].mean():.2f} ★")
    c3.metric("불만 리뷰 비율", f"{(rv['complaint_type'] != 'none').mean()*100:.1f}%")
    c4.metric("저평점 비율 (1–2★)", f"{(rv['rating'] <= 2).mean()*100:.1f}%")

    st.divider()

    # ── 리뷰 신호 지표 카테고리별 현황 ──
    st.subheader("📡 리뷰 신호 지표 카테고리별 현황")
    st.caption("리뷰 신호 지표(0–100): 높을수록 안정 / 위험 < 33 / 주의 33–66 / 정상 ≥ 66")

    latest_rem = rm.sort_values("bucket_month").groupby("category").last().reset_index()

    sig_cols = st.columns(len(latest_rem))
    for i, (_, row) in enumerate(latest_rem.iterrows()):
        score = row.get("rem_score", 0) or 0
        with sig_cols[i]:
            st.markdown(f"**{row['category']}**")
            st.markdown(signal_badge(score), unsafe_allow_html=True)
            st.metric("신호 점수", f"{score:.1f}")
            st.metric("리뷰 수", f"{int(row.get('n_reviews', 0)):,}")
            ns = row.get("NS", 0) or 0
            st.metric("저평점 비율", f"{ns*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("카테고리별 리뷰 수")
        cat_cnt = rv.groupby("category").size().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(cat_cnt.index, cat_cnt.values, color="#4C72B0", edgecolor="none")
        ax.set_xlabel("리뷰 수")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, transparent=True)
        plt.close()

    with col2:
        st.subheader("불만 유형 분포")
        comp = rv[rv["complaint_type"].isin(COMPLAINT_TYPES)]
        comp_cnt = comp["complaint_type"].value_counts()
        comp_cnt.index = [COMPLAINT_LABELS.get(i, i) for i in comp_cnt.index]
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.pie(comp_cnt.values, labels=comp_cnt.index,
               autopct="%1.1f%%", startangle=90,
               colors=["#E84545", "#4C72B0", "#55A868", "#C44E52"])
        st.pyplot(fig, transparent=True)
        plt.close()

    st.subheader("카테고리 × 불만 유형 히트맵")
    heat = rv[rv["complaint_type"].isin(COMPLAINT_TYPES)].groupby(
        ["category", "complaint_type"]
    ).size().unstack(fill_value=0)
    heat.columns = [COMPLAINT_LABELS.get(c, c) for c in heat.columns]
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    im = ax.imshow(heat.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.values[i,j]:,}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig, transparent=True)
    plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 - 일별 추이
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.header("일별 추이")

    st.subheader("일별 불만 비율")
    daily_comp = dy.groupby("review_date")["complaint_ratio"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.plot(daily_comp["review_date"], daily_comp["complaint_ratio"],
            linewidth=1, color="#E84545")
    ax.fill_between(daily_comp["review_date"], daily_comp["complaint_ratio"],
                    alpha=0.15, color="#E84545")
    ax.set_ylabel("불만 비율")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, transparent=True)
    plt.close()

    st.subheader("불만 유형별 일별 건수")
    type_cols   = ["defect_count", "transaction_count", "usability_count", "fit_count"]
    type_labels = ["결함/불량", "배송/CS", "사용성", "적합성"]
    colors      = ["#E84545", "#4C72B0", "#55A868", "#C44E52"]
    daily_type  = dy.groupby("review_date")[type_cols].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    for col, label, color in zip(type_cols, type_labels, colors):
        ax.plot(daily_type["review_date"], daily_type[col],
                label=label, linewidth=1.2, color=color)
    ax.legend(loc="upper left", framealpha=0)
    ax.set_ylabel("건수")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, transparent=True)
    plt.close()

    # ── 리뷰 신호 지표 추이 ──
    st.subheader("📡 리뷰 신호 지표 추이 (월별)")
    st.caption("신호 점수 기준선: 33(위험) / 66(주의)")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    for cat in sel_category:
        sub = rm[rm["category"] == cat].sort_values("bucket_month")
        if len(sub) == 0:
            continue
        ax.plot(sub["bucket_month"], sub["rem_score"],
                label=cat, linewidth=1.5, marker="o", markersize=3)
    ax.axhline(66, color="#21BA45", linestyle="--", linewidth=0.8, alpha=0.6, label="정상 기준(66)")
    ax.axhline(33, color="#FF4B4B", linestyle="--", linewidth=0.8, alpha=0.6, label="위험 기준(33)")
    ax.set_ylabel("신호 점수 (0–100)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, transparent=True)
    plt.close()

    # ── 리뷰 신호 지표 상세 테이블 ──
    st.subheader("리뷰 신호 지표 상세")
    latest = rm.sort_values("bucket_month").groupby("category").last().reset_index()
    show_cols = {
        "category": "카테고리",
        "rem_score": "신호 점수",
        "n_reviews": "리뷰 수",
        "NS": "저평점 비율",
        "dNS": "저평점 변화",
        "CT_shift": "불만 유형 변화",
        "V": "볼륨",
        "Q": "품질 점수",
    }
    disp = latest[[c for c in show_cols if c in latest.columns]].rename(columns=show_cols)

    # 신호등 컬럼 추가
    disp.insert(1, "신호", latest["rem_score"].apply(
        lambda s: "⚠ 위험" if s < 33 else ("△ 주의" if s < 66 else "✓ 정상")
    ))

    st.dataframe(
        disp.style
            .format({
                "신호 점수": "{:.1f}",
                "저평점 비율": "{:.1%}",
                "저평점 변화": "{:+.3f}",
                "불만 유형 변화": "{:.3f}",
                "볼륨": "{:.2f}",
                "품질 점수": "{:.2f}",
            })
            .set_properties(**{"background-color": "transparent"})
            .map(lambda v: "color: #FF4B4B; font-weight:700" if v == "⚠ 위험"
                 else ("color: #FFA500; font-weight:700" if v == "△ 주의"
                       else "color: #21BA45; font-weight:700"), subset=["신호"]),
        use_container_width=True, hide_index=True
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 - 키워드 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.header("키워드 분석")

    col1, col2 = st.columns(2)
    with col1:
        wc_category = st.selectbox("카테고리", ["전체"] + CATEGORIES, key="wc_cat")
    with col2:
        wc_type = st.selectbox(
            "불만 유형", ["전체"] + list(COMPLAINT_LABELS.keys()),
            format_func=lambda x: "전체" if x == "전체" else COMPLAINT_LABELS[x],
            key="wc_type"
        )

    wc_date = st.date_input(
        "기간 (키워드)", value=(date_min, date_max),
        min_value=date_min, max_value=date_max, key="wc_date"
    )
    wc_from = pd.Timestamp(wc_date[0]) if len(wc_date) >= 1 else pd.Timestamp(date_min)
    wc_to   = pd.Timestamp(wc_date[1]) if len(wc_date) == 2 else pd.Timestamp(date_max)

    kw_f = kw_daily.copy()
    kw_f["review_date"] = pd.to_datetime(kw_f["review_date"])
    kw_f = kw_f[kw_f["review_date"].between(wc_from, wc_to)]
    if wc_category != "전체":
        kw_f = kw_f[kw_f["category"] == wc_category]
    if wc_type != "전체":
        kw_f = kw_f[kw_f["complaint_type"] == wc_type]

    freq = kw_f.groupby("keyword")["count"].sum()
    freq = freq[freq > 0].to_dict()

    if freq:
        wc_img = WordCloud(
            font_path=FONT_PATH, width=900, height=400,
            background_color="white", colormap="RdYlBu_r", max_words=100
        ).generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc_img, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, transparent=True)
        plt.close()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("상위 키워드 Top 20")
            top_kw = (
                pd.DataFrame(list(freq.items()), columns=["키워드", "빈도"])
                .sort_values("빈도", ascending=False).head(20).reset_index(drop=True)
            )
            st.dataframe(
                top_kw.style.set_properties(**{"background-color": "transparent"}),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.subheader("키워드 빈도 막대")
            top15 = top_kw.head(15)
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            ax.barh(top15["키워드"][::-1], top15["빈도"][::-1], color="#4C72B0", edgecolor="none")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, transparent=True)
            plt.close()
    else:
        st.info("해당 조건에 키워드가 없습니다.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 - 상품별 리뷰
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.header("상품별 리뷰 분석")

    p_cat = st.selectbox("카테고리 선택", CATEGORIES, key="p_cat")
    cat_products = reviews[reviews["category"] == p_cat]["product_url"].unique()

    p_url = st.selectbox(
        "상품 선택", cat_products,
        format_func=lambda x: x.split("/")[-1] if x else x,
        key="p_url"
    )

    p_rv = reviews[reviews["product_url"] == p_url].sort_values("review_date", ascending=False)

    # ── 상품 신호 지표 ──
    c1, c2, c3, c4, c5 = st.columns(5)
    complaint_rate = (p_rv["complaint_type"] != "none").mean()
    low_rate       = (p_rv["rating"] <= 2).mean()
    signal_score   = max(0, 100 - complaint_rate * 100 - low_rate * 50)

    c1.metric("리뷰 수", f"{len(p_rv):,}")
    c2.metric("평균 별점", f"{p_rv['rating'].mean():.2f} ★")
    c3.metric("불만 비율", f"{complaint_rate*100:.1f}%")
    c4.metric("저평점 비율", f"{low_rate*100:.1f}%")
    with c5:
        st.metric("신호 점수", f"{signal_score:.0f}")
        st.markdown(signal_badge(signal_score), unsafe_allow_html=True)

    # ── 불만 유형 분포 ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("불만 유형 분포")
        p_comp = p_rv[p_rv["complaint_type"].isin(COMPLAINT_TYPES)]["complaint_type"].value_counts()
        if len(p_comp) > 0:
            p_comp.index = [COMPLAINT_LABELS.get(i, i) for i in p_comp.index]
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            ax.pie(p_comp.values, labels=p_comp.index, autopct="%1.1f%%", startangle=90,
                   colors=["#E84545", "#4C72B0", "#55A868", "#C44E52"])
            st.pyplot(fig, transparent=True)
            plt.close()

    with col2:
        st.subheader("키워드 워드클라우드")
        # keyword_daily에서 해당 상품 키워드 집계
        p_kw = kw_daily[kw_daily["product_url"] == p_url]
        p_freq = dict(zip(p_kw["keyword"], p_kw["count"])) if not p_kw.empty else {}

        if p_freq:
            wc_img = WordCloud(
                font_path=FONT_PATH, width=500, height=300,
                background_color="white", colormap="Reds", max_words=60
            ).generate_from_frequencies(p_freq)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc_img, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, transparent=True)
            plt.close()
        else:
            st.info("불만 키워드가 없습니다.")

    # ── 리뷰 목록 ──
    st.subheader("리뷰 목록")
    comp_filter = st.multiselect(
        "불만 유형 필터",
        ["none"] + COMPLAINT_TYPES,
        default=["none"] + COMPLAINT_TYPES,
        format_func=lambda x: "없음" if x == "none" else COMPLAINT_LABELS.get(x, x),
        key="comp_filter"
    )
    p_show = p_rv[p_rv["complaint_type"].isin(comp_filter)][
        ["review_date", "rating", "complaint_type", "content"]
    ].rename(columns={
        "review_date": "날짜", "rating": "별점",
        "complaint_type": "불만유형", "content": "리뷰내용"
    })

    def color_complaint(val):
        colors_map = {
            "defect": "color: #E84545",
            "transaction": "color: #4C72B0",
            "usability": "color: #55A868",
            "fit": "color: #C44E52",
        }
        return colors_map.get(val, "")

    st.dataframe(
        p_show.style
            .set_properties(**{"background-color": "transparent"})
            .map(color_complaint, subset=["불만유형"]),
        use_container_width=True, height=400, hide_index=True
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 - 추천 액션 (상품별)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.header("💡 추천 액션")
    st.caption("상품별 리뷰 패턴을 분석해 판매자가 취해야 할 구체적인 액션을 제안합니다.")

    a_cat = st.selectbox("카테고리", CATEGORIES, key="a_cat")
    cat_urls = reviews[reviews["category"] == a_cat]["product_url"].unique()

    # 상품별 요약 테이블
    prod_summary = []
    for url in cat_urls:
        p = reviews[reviews["product_url"] == url]
        total = len(p)
        if total < 5:
            continue
        complaint_r = (p["complaint_type"] != "none").mean()
        low_r       = (p["rating"] <= 2).mean()
        signal      = max(0, min(100, 100 - complaint_r * 100 - low_r * 50))
        prod_summary.append({
            "product_url": url,
            "product_id": url.split("/")[-1],
            "리뷰 수": total,
            "평균 별점": round(p["rating"].mean(), 2),
            "불만 비율": round(complaint_r * 100, 1),
            "저평점 비율": round(low_r * 100, 1),
            "신호 점수": round(signal, 1),
            "신호": "⚠ 위험" if signal < 33 else ("△ 주의" if signal < 66 else "✓ 정상"),
            "defect_r": (p["complaint_type"] == "defect").mean(),
            "trans_r":  (p["complaint_type"] == "transaction").mean(),
            "usability_r": (p["complaint_type"] == "usability").mean(),
            "fit_r":    (p["complaint_type"] == "fit").mean(),
        })

    if not prod_summary:
        st.info("리뷰 5개 이상 상품이 없습니다.")
        st.stop()

    summary_df = pd.DataFrame(prod_summary).sort_values("신호 점수")

    # 요약 테이블
    st.subheader(f"{a_cat} 상품별 신호 현황")
    disp_df = summary_df[["product_id","리뷰 수","평균 별점","불만 비율","저평점 비율","신호 점수","신호"]].rename(
        columns={"product_id": "상품 ID"}
    )
    st.dataframe(
        disp_df.style
            .set_properties(**{"background-color": "transparent"})
            .map(lambda v: "color:#FF4B4B;font-weight:700" if v == "⚠ 위험"
                 else ("color:#FFA500;font-weight:700" if v == "△ 주의"
                       else "color:#21BA45;font-weight:700"), subset=["신호"])
            .format({"불만 비율": "{:.1f}%", "저평점 비율": "{:.1f}%", "신호 점수": "{:.1f}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()

    # 상품 선택
    a_url = st.selectbox(
        "액션을 확인할 상품 선택",
        summary_df["product_url"].tolist(),
        format_func=lambda x: x.split("/")[-1],
        key="a_url"
    )

    row = summary_df[summary_df["product_url"] == a_url].iloc[0]
    p_rv_a = reviews[reviews["product_url"] == a_url]

    signal      = row["신호 점수"]
    grade       = row["신호"]
    grade_color = "#FF4B4B" if signal < 33 else ("#FFA500" if signal < 66 else "#21BA45")
    defect_r    = row["defect_r"]
    trans_r     = row["trans_r"]
    usability_r = row["usability_r"]
    fit_r       = row["fit_r"]
    complaint_r = (p_rv_a["complaint_type"] != "none").mean()
    low_r       = (p_rv_a["rating"] <= 2).mean()

    # 상품 헤더
    st.markdown(f"""
    <div style="border:1px solid {grade_color}; border-radius:12px; padding:16px 20px; margin-bottom:20px;">
    <h3 style="margin:0 0 4px 0;">상품 {a_url.split('/')[-1]}
    &nbsp;<span style="font-size:14px; background:{grade_color}; color:white;
    padding:3px 10px; border-radius:10px; font-weight:700;">{grade}</span>
    </h3>
    <span style="color:#aaa; font-size:13px;">
    리뷰 {int(row['리뷰 수']):,}개 &nbsp;|&nbsp; 평균 {row['평균 별점']:.2f}★ &nbsp;|&nbsp;
    불만 {complaint_r*100:.1f}% &nbsp;|&nbsp; 저평점 {low_r*100:.1f}%
    </span>
    </div>
    """, unsafe_allow_html=True)

    # 액션 생성
    actions = []

    if signal < 33:
        actions.append(("🚨 긴급 대응 필요",
            f"신호 점수 {signal:.1f}점으로 위험 수준입니다. "
            "이 상품의 최근 리뷰를 즉시 전수 검토하고, "
            "CS팀에 해당 상품 집중 모니터링을 요청하세요. "
            "필요시 일시 판매 중단 후 품질 점검을 권장합니다."))
    elif signal < 66:
        actions.append(("⚠ 주의 모니터링",
            f"신호 점수 {signal:.1f}점으로 주의 구간입니다. "
            "주 1–2회 신규 리뷰를 점검하고 불만 급증 여부를 확인하세요."))
    else:
        actions.append(("✅ 양호 — 긍정 마케팅 활용",
            f"신호 점수 {signal:.1f}점으로 정상입니다. "
            "만족도 높은 리뷰를 상품 페이지 대표 후기로 노출하고, "
            "이 상품을 중심으로 기획전·프로모션을 진행하세요."))

    if defect_r > 0.15:
        top_kw = (
            kw_daily[(kw_daily["category"] == a_cat) & (kw_daily["complaint_type"] == "defect")]
            .groupby("keyword")["count"].sum().sort_values(ascending=False).head(4).index.tolist()
        )
        kw_str = ", ".join(top_kw) if top_kw else "확인 필요"
        actions.append(("🔧 결함/불량 집중 대응",
            f"결함 불만이 {defect_r*100:.1f}%로 높습니다.\n\n"
            f"- 주요 불만 키워드: **{kw_str}**\n"
            "- 공급업체에 해당 키워드 중심의 품질 개선 요청서를 발송하세요\n"
            "- 최근 3개월 반품·교환 내역 중 결함 사유를 집계해 공유하세요\n"
            "- 동일 불량 반복 시 해당 SKU 입고 중단을 검토하세요"))

    if trans_r > 0.10:
        actions.append(("🚚 배송/CS 프로세스 점검",
            f"배송·CS 불만이 {trans_r*100:.1f}%입니다.\n\n"
            "- 최근 30일 배송 지연 건수 및 오배송 비율을 물류팀에 요청하세요\n"
            "- 반품·교환 안내 문구를 상품 페이지 상단에 명확히 표시하세요\n"
            "- CS 응대 평균 시간이 24시간을 초과하면 담당자 추가 배치를 요청하세요\n"
            "- 포장 불량 이슈라면 포장재 규격을 재점검하세요"))

    if usability_r > 0.10:
        top_kw = (
            kw_daily[(kw_daily["category"] == a_cat) & (kw_daily["complaint_type"] == "usability")]
            .groupby("keyword")["count"].sum().sort_values(ascending=False).head(4).index.tolist()
        )
        kw_str = ", ".join(top_kw) if top_kw else "확인 필요"
        actions.append(("📖 사용성 개선 안내",
            f"사용성 불만이 {usability_r*100:.1f}%입니다.\n\n"
            f"- 주요 불만 키워드: **{kw_str}**\n"
            "- 설치/연결 단계별 사진 가이드를 상품 이미지에 추가하세요\n"
            "- 자주 묻는 질문(FAQ)을 Q&A 게시판에 선제적으로 등록하세요\n"
            "- 호환 기기 목록, 필요 액세서리 정보를 상품 상세에 명시하세요\n"
            "- 유튜브 설치 영상 링크를 상품 설명에 포함하는 것도 효과적입니다"))

    if fit_r > 0.08:
        actions.append(("📐 적합성 정보 보강",
            f"적합성 불만이 {fit_r*100:.1f}%입니다.\n\n"
            "- 사이즈/규격 표를 상품 이미지에 시각적으로 추가하세요\n"
            "- 사용 대상(연령, 체형, 피부 타입, 반려동물 종/체중)을 명확히 기재하세요\n"
            "- '이 제품이 맞지 않는 경우'를 미리 안내해 반품률을 줄이세요\n"
            "- 구매 전 문의를 유도하는 메시지를 상품 페이지에 추가하세요"))

    if low_r > 0.20:
        actions.append(("⭐ 저평점 리뷰 관리",
            f"1–2★ 저평점 비율이 {low_r*100:.1f}%입니다.\n\n"
            "- 저평점 리뷰 전체에 48시간 이내 판매자 답변을 등록하세요\n"
            "- 반복되는 동일 불만은 상품 정보 수정으로 선제 대응하세요\n"
            "- 환불 처리 후에도 남은 저평점은 정중한 답변으로 신뢰를 회복하세요\n"
            "- 저평점이 3개월 이상 지속되면 상품 등록 재검토를 권장합니다"))

    # 액션 출력
    for title, desc in actions:
        with st.expander(title, expanded=True):
            st.markdown(desc)
