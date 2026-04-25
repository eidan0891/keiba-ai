from __future__ import annotations

import io
import re
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_TITLE = "にゃんこ競馬AI iPad現地版 v3"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# ----------------------------
# 読み込み・取得
# ----------------------------
def decode_bytes(raw: bytes) -> str:
    best = ""
    best_score = -10**9
    for enc in ["utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp"]:
        try:
            txt = raw.decode(enc, errors="replace")
        except Exception:
            continue
        score = (
            txt.count("馬名") * 50
            + txt.count("馬番") * 40
            + txt.count("騎手") * 20
            + txt.count("人気") * 20
            + txt.count("オッズ") * 20
            - txt.count("�") * 20
            - txt.count("郢") * 20
            - txt.count("ｽ") * 20
        )
        if score > best_score:
            best_score = score
            best = txt
    return best

def fetch_url_text(url: str, timeout: int = 8) -> str:
    try:
        if not url.strip():
            return ""
        res = requests.get(url.strip(), headers=HEADERS, timeout=timeout)
        res.raise_for_status()
        return decode_bytes(res.content)
    except Exception:
        return ""

def safe_read_html(html: str) -> list[pd.DataFrame]:
    if not html:
        return []
    try:
        return pd.read_html(io.StringIO(html))
    except Exception:
        return []

def safe_read_csv(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue()
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ----------------------------
# 整形
# ----------------------------
def flat_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x) for x in c if "Unnamed" not in str(x)]).strip("_")
            for c in out.columns
        ]
    out.columns = [str(c).strip().replace("　", "").replace(" ", "") for c in out.columns]
    return out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")].copy()

def table_score(df: pd.DataFrame) -> int:
    try:
        d = flat_cols(df)
        text = " ".join(map(str, d.columns)) + " " + " ".join(d.astype(str).head(8).fillna("").values.ravel().tolist())
        score = 0
        for k, p in {"馬番": 80, "馬名": 80, "騎手": 40, "人気": 35, "オッズ": 35, "単勝": 30, "枠": 20, "斤量": 15}.items():
            if k in text:
                score += p
        return score + min(len(d), 30)
    except Exception:
        return -999

def pick_best_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return flat_cols(max(tables, key=table_score))

def to_num_series(s, default=np.nan):
    try:
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce").fillna(default)
    except Exception:
        return pd.Series([default] * len(s))

def find_col(cols: list[str], keywords: list[str], deny: list[str] | None = None) -> str | None:
    deny = deny or []
    for c in cols:
        if any(k in c for k in keywords) and not any(d in c for d in deny):
            return c
    return None

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    src = flat_cols(df)
    out = pd.DataFrame(index=src.index)
    cols = list(src.columns)

    horse_no_col = find_col(cols, ["馬番"]) or find_col(cols, ["番号"])
    frame_col = find_col(cols, ["枠番"]) or (find_col(cols, ["枠"], deny=["枠順"]) if cols else None)
    name_col = find_col(cols, ["馬名"])
    jockey_col = find_col(cols, ["騎手"])
    pop_col = find_col(cols, ["人気"])
    odds_col = find_col(cols, ["オッズ"]) or find_col(cols, ["単勝"])
    weight_col = find_col(cols, ["斤量"])
    sex_col = find_col(cols, ["性齢"])
    last3f_col = find_col(cols, ["後3F"]) or find_col(cols, ["上がり"]) or find_col(cols, ["上り"])
    passing_col = find_col(cols, ["通過"])
    style_col = find_col(cols, ["脚質"])

    if horse_no_col:
        out["馬番"] = to_num_series(src[horse_no_col], 0).astype(int)
    else:
        out["馬番"] = np.arange(1, len(src) + 1)

    if frame_col:
        out["枠"] = to_num_series(src[frame_col], 0).astype(int)
    else:
        out["枠"] = ((out["馬番"] + 1) // 2).astype(int)

    if name_col:
        out["馬名"] = src[name_col].astype(str).replace("nan", "")
    else:
        # 日本語が一番多い列を名前候補にする
        best = None
        best_rate = -1
        for c in cols:
            rate = src[c].astype(str).str.contains(r"[一-龠ぁ-んァ-ヶー]", regex=True, na=False).mean()
            if rate > best_rate:
                best_rate = rate
                best = c
        out["馬名"] = src[best].astype(str) if best is not None and best_rate > 0.3 else [f"馬{x}" for x in out["馬番"]]

    out["騎手"] = src[jockey_col].astype(str) if jockey_col else ""
    out["人気"] = to_num_series(src[pop_col], np.nan) if pop_col else np.nan
    out["オッズ"] = to_num_series(src[odds_col], np.nan) if odds_col else np.nan
    out["斤量"] = to_num_series(src[weight_col], np.nan) if weight_col else np.nan
    out["性齢"] = src[sex_col].astype(str) if sex_col else ""
    out["後3F"] = to_num_series(src[last3f_col], np.nan) if last3f_col else np.nan
    out["通過順"] = src[passing_col].astype(str) if passing_col else ""
    out["脚質"] = src[style_col].astype(str) if style_col else ""

    # 空行・見出し行除外
    out = out[out["馬名"].astype(str).str.len() > 0].copy()
    out = out[~out["馬名"].astype(str).str.contains("馬名|出走取消|取消", na=False)].copy()
    out = out.drop_duplicates(subset=["馬番"], keep="first").reset_index(drop=True)

    # 馬番が全部0なら連番
    if len(out) and (out["馬番"] <= 0).all():
        out["馬番"] = np.arange(1, len(out) + 1)
    return out

def infer_style(row) -> str:
    s = str(row.get("脚質", ""))
    if s in ["逃げ", "先行", "差", "追込"]:
        return s
    nums = [int(x) for x in re.findall(r"\d+", str(row.get("通過順", "")))]
    if nums:
        if nums[0] <= 1: return "逃げ"
        if nums[0] <= 4: return "先行"
        if nums[0] <= 8: return "差"
        return "追込"
    return "不明"

# ----------------------------
# 予想ロジック
# ----------------------------
def predict(df: pd.DataFrame, mode: str, honmei_n: int, ana_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = standardize(df)
    if work.empty:
        return work, pd.DataFrame(), pd.DataFrame()

    work["脚質"] = work.apply(infer_style, axis=1)
    n = max(len(work), 1)

    pop = pd.to_numeric(work["人気"], errors="coerce")
    odds = pd.to_numeric(work["オッズ"], errors="coerce")
    last3f = pd.to_numeric(work["後3F"], errors="coerce")

    # 人気スコア。取れない場合でも全部同点にしない
    pop_score = (1 / pop.replace(0, np.nan)).fillna(0)
    if pop_score.sum() == 0:
        pop_score = pd.Series(np.linspace(1.0, 0.4, len(work)), index=work.index)

    # オッズスコア。低オッズ＝信頼、高オッズ＝妙味の2本を使う
    odds_trust = (1 / odds.replace(0, np.nan)).fillna(0)
    if odds_trust.sum() == 0:
        odds_trust = pd.Series(np.linspace(0.8, 0.4, len(work)), index=work.index)

    odds_value = odds.fillna(0).clip(0, 80) / 80.0

    # 上がりスコア
    if last3f.notna().sum() >= 2 and abs(last3f.max() - last3f.min()) > 1e-9:
        last3f_score = (last3f.max() - last3f) / (last3f.max() - last3f.min())
        last3f_score = last3f_score.fillna(0.5)
    else:
        last3f_score = pd.Series(0.5, index=work.index)

    style_bonus = work["脚質"].map({"逃げ":0.04, "先行":0.06, "差":0.04, "追込":0.02, "不明":0.0}).fillna(0)

    if mode == "的中率":
        score = pop_score*0.52 + odds_trust*0.28 + last3f_score*0.15 + style_bonus
    elif mode == "回収率":
        score = pop_score*0.26 + odds_trust*0.18 + odds_value*0.34 + last3f_score*0.16 + style_bonus
    elif mode == "三連複特化":
        score = pop_score*0.34 + odds_trust*0.22 + odds_value*0.18 + last3f_score*0.18 + style_bonus
    else:
        score = pop_score*0.38 + odds_trust*0.24 + odds_value*0.16 + last3f_score*0.16 + style_bonus

    # 同点回避。馬番順だけにならないよう、オッズ/人気/脚質で微差
    tie_break = (
        work["馬番"].astype(float).rank(ascending=False, pct=True) * 0.001
        + odds_value.fillna(0) * 0.003
        + last3f_score.fillna(0) * 0.002
    )
    work["AIスコア"] = pd.to_numeric(score + tie_break, errors="coerce").fillna(0)

    expv = np.exp(work["AIスコア"] - work["AIスコア"].max())
    work["勝率推定"] = expv / expv.sum() if expv.sum() else 1 / n

    ps_base = pop_score / pop_score.sum() if pop_score.sum() else pd.Series(1/n, index=work.index)
    work["複勝率推定"] = work["勝率推定"]*0.55 + ps_base*0.45

    if odds.notna().any() and odds_trust.sum() > 0:
        market = odds_trust / odds_trust.sum()
        work["妙味差"] = work["勝率推定"] - market
        work["期待値"] = work["勝率推定"] * odds.fillna(0)
    else:
        work["妙味差"] = work["AIスコア"] - work["AIスコア"].mean()
        work["期待値"] = work["AIスコア"]

    work = work.sort_values(["AIスコア", "勝率推定", "期待値"], ascending=False).reset_index(drop=True)
    marks = ["◎","○","▲","△","☆","注","穴"]
    work["印"] = ""
    for i, m in enumerate(marks):
        if i < len(work):
            work.loc[i, "印"] = m

    honmei = work.head(honmei_n).copy()
    ana = work.sort_values(["妙味差","期待値","AIスコア"], ascending=False).head(ana_n).copy()
    return work, honmei, ana

def build_wide(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    rows = []
    top = df.head(min(len(df), 10))
    for i, j in combinations(range(len(top)), 2):
        a, b = top.iloc[i], top.iloc[j]
        score = float(a["複勝率推定"]) + float(b["複勝率推定"]) + max(float(a["妙味差"]),0) + max(float(b["妙味差"]),0)
        rows.append({"券種":"ワイド","組み合わせ":f"{int(a['馬番'])}-{int(b['馬番'])}","馬1":a["馬名"],"馬2":b["馬名"],"score":round(score,4)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(limit) if rows else pd.DataFrame()

def build_trio(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    rows = []
    top = df.head(min(len(df), 10))
    for i, j, k in combinations(range(len(top)), 3):
        a,b,c = top.iloc[i], top.iloc[j], top.iloc[k]
        style_bonus = 0.05 if len({a["脚質"], b["脚質"], c["脚質"]}) >= 2 else 0
        score = float(a["複勝率推定"]) + float(b["複勝率推定"]) + float(c["複勝率推定"]) + style_bonus
        rows.append({"券種":"3連複","組み合わせ":f"{int(a['馬番'])}-{int(b['馬番'])}-{int(c['馬番'])}","馬1":a["馬名"],"馬2":b["馬名"],"馬3":c["馬名"],"score":round(score,4)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(limit) if rows else pd.DataFrame()

def build_trifecta(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    rows = []
    top = df.head(min(len(df), 7))
    for i,j,k in permutations(range(len(top)), 3):
        a,b,c = top.iloc[i], top.iloc[j], top.iloc[k]
        score = float(a["勝率推定"])*1.8 + float(b["複勝率推定"])*0.9 + float(c["複勝率推定"])*0.7 + max(float(c["妙味差"]),0)
        rows.append({"券種":"3連単","組み合わせ":f"{int(a['馬番'])}→{int(b['馬番'])}→{int(c['馬番'])}","1着":a["馬名"],"2着":b["馬名"],"3着":c["馬名"],"score":round(score,4)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(limit) if rows else pd.DataFrame()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🐾 " + APP_TITLE)
st.caption("URL取得→予想。取得した表を確認・編集してから予想できます。")

with st.sidebar:
    mode = st.selectbox("クイック設定", ["バランス","的中率","回収率","三連複特化"], index=0)
    honmei_n = st.slider("本命候補", 3, 10, 6)
    ana_n = st.slider("穴候補", 3, 15, 6)
    wide_n = st.slider("ワイド表示数", 3, 30, 10)
    trio_n = st.slider("3連複表示数", 3, 50, 15)
    trifecta_n = st.slider("3連単表示数", 3, 100, 20)

tab_url, tab_csv = st.tabs(["URL取得", "CSV"])

if "source_df" not in st.session_state:
    st.session_state["source_df"] = pd.DataFrame()

with tab_url:
    url = st.text_input("出馬表URL", placeholder="netkeiba / NAR の出馬表URL")
    if st.button("URL取得", type="primary", use_container_width=True):
        html = fetch_url_text(url)
        tables = safe_read_html(html)
        df = pick_best_table(tables)
        if df.empty:
            st.warning("取得できませんでした。CSVタブか、URLを確認してください。")
        else:
            st.session_state["source_df"] = df
            st.success(f"取得成功: {len(df)}行")

with tab_csv:
    up = st.file_uploader("CSVアップロード", type=["csv"])
    if st.button("CSV読み込み", use_container_width=True):
        df = safe_read_csv(up)
        if df.empty:
            st.warning("CSVを読み込めませんでした。")
        else:
            st.session_state["source_df"] = df
            st.success(f"CSV読込成功: {len(df)}行")

df0 = st.session_state["source_df"]

if not df0.empty:
    st.subheader("取得データ確認")
    base_df = standardize(df0)
    edited = st.data_editor(
        base_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "人気": st.column_config.NumberColumn("人気"),
            "オッズ": st.column_config.NumberColumn("オッズ"),
            "後3F": st.column_config.NumberColumn("後3F"),
            "脚質": st.column_config.SelectboxColumn("脚質", options=["不明","逃げ","先行","差","追込"]),
        },
    )

    if st.button("予想実行", type="primary", use_container_width=True):
        result, honmei, ana = predict(edited, mode, honmei_n, ana_n)

        st.subheader("AI予想ランキング")
        show_cols = ["印","馬番","枠","馬名","騎手","脚質","人気","オッズ","後3F","AIスコア","勝率推定","複勝率推定","妙味差","期待値"]
        st.dataframe(result[[c for c in show_cols if c in result.columns]], use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("本命候補")
            st.dataframe(honmei[["印","馬番","馬名","人気","オッズ","AIスコア","勝率推定"]], use_container_width=True, hide_index=True)
        with c2:
            st.subheader("穴候補")
            st.dataframe(ana[["印","馬番","馬名","人気","オッズ","妙味差","期待値"]], use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("ワイド")
            st.dataframe(build_wide(result, wide_n), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("3連複")
            st.dataframe(build_trio(result, trio_n), use_container_width=True, hide_index=True)
        with c3:
            st.subheader("3連単")
            st.dataframe(build_trifecta(result, trifecta_n), use_container_width=True, hide_index=True)

        st.download_button(
            "予想結果CSVダウンロード",
            result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="nyanko_prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
