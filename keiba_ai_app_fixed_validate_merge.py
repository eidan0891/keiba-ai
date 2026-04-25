from __future__ import annotations

import io
import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Iterable
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# =========================
# iPad / Cloud 安定設定
# =========================
SELENIUM_AVAILABLE = False
REQUEST_TIMEOUT = 8
APP_TITLE = "にゃんこ競馬AI（iPad版）"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# =========================
# Utility
# =========================
def decode_response_html(resp, url):
    raw = resp.content
    for enc in ["utf-8", "cp932", "shift_jis", "euc_jp"]:
        try:
            txt = raw.decode(enc, errors="replace")
            if "馬名" in txt:
                return txt
        except:
            pass
    return resp.text

def safe_read_html(html_text: str) -> List[pd.DataFrame]:
    try:
        if not html_text:
            return []
        return pd.read_html(io.StringIO(html_text))
    except:
        return []

def safe_num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def safe_text(df, col):
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str).fillna("")

# =========================
# Scraper
# =========================
class Scraper:
    def __init__(self):
        self.session = requests.Session()

    def get_html(self, url):
        try:
            r = self.session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return decode_response_html(r, url)
        except Exception as e:
            st.error(f"接続エラー: {e}")
            return ""

def pick_table(tables):
    best = None
    score_best = -999
    for t in tables:
        try:
            txt = "".join(t.columns.astype(str))
            score = 0
            if "馬名" in txt: score += 50
            if "馬番" in txt: score += 50
            if "人気" in txt: score += 30
            if "オッズ" in txt: score += 30
            score += len(t)
            if score > score_best:
                best = t
                score_best = score
        except:
            pass
    return best

# =========================
# AIロジック
# =========================
def predict(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "馬番" not in df.columns:
        df["馬番"] = np.arange(1, len(df) + 1)
    if "馬名" not in df.columns:
        df["馬名"] = ["馬" + str(i) for i in range(len(df))]

    df["人気"] = safe_num(df, "人気", 10)
    df["オッズ"] = safe_num(df, "オッズ", 50)

    # シンプルな計算式
    score_pop = 1 / df["人気"]
    score_odds = 1 / df["オッズ"]
    df["score"] = score_pop * 0.6 + score_odds * 0.4

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    df["印"] = ""
    for i in range(min(len(df), len(marks))):
        df.loc[i, "印"] = marks[i]
    
    return df

# =========================
# UI (Streamlit Main)
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# --- セッション状態の初期化 ---
if "master_df" not in st.session_state:
    st.session_state.master_df = None

scraper = Scraper()
tab1, tab2 = st.tabs(["URLから取得", "CSVをアップロード"])

# -----------------
# Tab 1: URL取得
# -----------------
with tab1:
    url_input = st.text_input("netkeibaなどの出馬表URLを入力")
    if st.button("データ取得実行"):
        if url_input:
            with st.spinner("スクレイピング中..."):
                html = scraper.get_html(url_input)
                if html:
                    tables = safe_read_html(html)
                    target_df = pick_table(tables)
                    if target_df is not None:
                        st.session_state.master_df = target_df
                        st.success("データの取得に成功しました！")
                    else:
                        st.warning("競馬の出馬表テーブルが見つかりませんでした。")
        else:
            st.info("URLを入力してください。")

# -----------------
# Tab 2: CSV
# -----------------
with tab2:
    uploaded_file = st.file_uploader("CSVファイルをドロップ", type=["csv"])
    if uploaded_file:
        try:
            st.session_state.master_df = pd.read_csv(uploaded_file)
            st.success("CSVを読み込みました。")
        except Exception as e:
            st.error(f"CSV読み込み失敗: {e}")

# -----------------
# 共通表示 & 予想実行
# -----------------
st.divider()

if st.session_state.master_df is not None:
    st.subheader("現在読み込んでいるデータ")
    st.dataframe(st.session_state.master_df, use_container_width=True)

    if st.button("AI予想を開始する", type="primary"):
        try:
            with st.spinner("解析中..."):
                result_df = predict(st.session_state.master_df)
                
                st.header("🏆 予想結果")
                # 印を左側に持ってくる
                cols = ["印", "馬番", "馬名", "人気", "オッズ"]
                existing_cols = [c for c in cols if c in result_df.columns]
                
                st.dataframe(
                    result_df[existing_cols].head(10).style.highlight_max(subset=["印"], color="#ffeb3b"),
                    use_container_width=True
                )
                st.balloons()
        except Exception as e:
            st.error(f"予想プロセスでエラーが発生しました: {e}")
else:
    st.info("上のタブからデータを読み込んでください。")
