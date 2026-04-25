from __future__ import annotations

import io
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 設定
# =========================
APP_TITLE = "にゃんこ競馬AI（完全版）"
REQUEST_TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}

# =========================
# Utility & Logic
# =========================
def safe_num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return pd.to_numeric(df[col].astype(str).str.replace('---', ''), errors="coerce").fillna(default)

def predict(df):
    df = df.copy()
    # カラム名の空白削除
    df.columns = [str(c).strip() for c in df.columns]

    # 必須カラムの補完
    if "馬番" not in df.columns:
        df["馬番"] = np.arange(1, len(df) + 1)
    if "馬名" not in df.columns:
        df["馬名"] = [f"馬{i+1}" for i in range(len(df))]

    # 数値変換
    pop = safe_num(df, "人気", 10)
    odds = safe_num(df, "オッズ", 50)

    # --- 予想ロジック ---
    # 人気とオッズの逆数でスコア化（ここをいじると予想が変わります）
    df["score"] = (1 / pop * 0.6) + (1 / odds * 0.4)

    # スコアが高い順に並び替え（ここで「12345...」の順序が崩れるはずです）
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # 印をつける（全頭分ループ）
    marks = ["◎", "○", "▲", "△", "☆"]
    df["印"] = ""
    for i in range(len(df)):
        if i < len(marks):
            df.loc[i, "印"] = marks[i]
        else:
            df.loc[i, "印"] = "-" # 6番手以降

    return df

# =========================
# UI (Streamlit)
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🏆 {APP_TITLE}")

if "master_df" not in st.session_state:
    st.session_state.master_df = None

tab1, tab2 = st.tabs(["URLから取得", "CSVから読み込み"])

with tab1:
    url = st.text_input("出馬表URLを入力")
    if st.button("データ取得"):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            # 文字コード自動判定
            r.encoding = r.apparent_encoding
            dfs = pd.read_html(io.StringIO(r.text))
            
            # 一番それっぽいテーブルを探す
            best_df = None
            max_score = -1
            for d in dfs:
                score = ("馬名" in d.columns) + ("馬番" in d.columns) + ("オッズ" in d.columns)
                if score > max_score:
                    max_score = score
                    best_df = d
            
            if best_df is not None:
                st.session_state.master_df = best_df
                st.success("取得完了！")
            else:
                st.error("競馬のテーブルが見つかりません。")
        except Exception as e:
            st.error(f"エラー: {e}")

with tab2:
    file = st.file_uploader("CSVファイルを選択", type="csv")
    if file:
        st.session_state.master_df = pd.read_csv(file)

# --- 共通メインエリア ---
if st.session_state.master_df is not None:
    st.divider()
    st.subheader("📋 読み込み中のデータ")
    st.dataframe(st.session_state.master_df, use_container_width=True)

    if st.button("🚀 AI予想を実行！", type="primary"):
        res = predict(st.session_state.master_df)
        
        st.divider()
        st.header("✨ 予想結果")
        
        # 表示する列を絞り込む
        display_cols = ["印", "馬番", "馬名", "人気", "オッズ"]
        actual_cols = [c for c in display_cols if c in res.columns]
        
        # 全頭表示（headを外しました）
        st.table(res[actual_cols]) 
        
        st.balloons()
