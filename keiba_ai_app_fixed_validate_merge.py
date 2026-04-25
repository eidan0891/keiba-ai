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
# Scraper（Selenium完全排除）
# =========================
class Scraper:

    def __init__(self):
        self.session = requests.Session()

    def get_html(self, url):
        try:
            r = self.session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return decode_response_html(r, url)
        except:
            return ""


# =========================
# テーブル抽出
# =========================
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
# AIロジック（軽量版）
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
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

scraper = Scraper()

tab1, tab2 = st.tabs(["URL取得", "CSV"])

df = None


# -----------------
# URL取得
# -----------------
with tab1:

    url = st.text_input("出馬表URL")

    if st.button("取得"):

        html = scraper.get_html(url)

        if not html:
            st.warning("取得失敗")
        else:
            tables = safe_read_html(html)

            df = pick_table(tables)

            if df is None:
                st.warning("テーブル検出失敗")
            else:
                st.success("取得成功")
                st.dataframe(df)


# -----------------
# CSV
# -----------------
with tab2:

    file = st.file_uploader("CSVアップロード")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)


# -----------------
# 予想
# -----------------
if df is not None:

    if st.button("予想"):

        try:
            result = predict(df)

            st.subheader("予想結果")
            st.dataframe(result)

        except Exception as e:
            st.warning(f"予想失敗: {e}")
