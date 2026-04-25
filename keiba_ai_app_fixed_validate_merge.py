from __future__ import annotations

import io
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 設定
# =========================
APP_TITLE = "にゃんこ競馬AI（オッズ修正版）"
REQUEST_TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}

# =========================
# 強力なオッズ抽出
# =========================
def extract_odds(val):
    """
    '1.5 (1)' や '1.5 ---' などの文字列から 1.5 だけを抜き出す
    """
    s = str(val)
    # 正規表現で浮動小数点数（1.5など）を検索
    match = re.search(r'(\d+\.\d+)', s)
    if match:
        return float(match.group(1))
    return np.nan

def safe_num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series([default] * len(df))
    
    # オッズ列の場合、特殊な抽出を行う
    if "オッズ" in col:
        return df[col].apply(extract_odds).fillna(default)
    
    return pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors="coerce").fillna(default)

# =========================
# AIロジック
# =========================
def predict(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "馬番" not in df.columns:
        df["馬番"] = np.arange(1, len(df) + 1)
    if "馬名" not in df.columns:
        df["馬名"] = [f"馬{i+1}" for i in range(len(df))]

    # 数値抽出の強化版を適用
    df["人気_num"] = safe_num(df, "人気", 10)
    df["オッズ_num"] = safe_num(df, "オッズ", 50)

    # 予想スコア計算（オッズが読み込めていればここで差が出る）
    df["score"] = (1 / df["人気_num"] * 0.6) + (1 / df["オッズ_num"] * 0.4)

    # スコア順にソート
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    marks = ["◎", "○", "▲", "△", "☆"]
    df["印"] = ""
    for i in range(len(df)):
        if i < len(marks):
            df.loc[i, "印"] = marks[i]
        else:
            df.loc[i, "印"] = "-"
    
    return df

# =========================
# UI (Streamlit)
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🐱 {APP_TITLE}")

if "master_df" not in st.session_state:
    st.session_state.master_df = None

tab1, tab2 = st.tabs(["URLから取得", "CSVから読み込み"])

with tab1:
    url = st.text_input("出馬表URLを入力 (netkeibaなど)")
    if st.button("データ取得"):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.encoding = r.apparent_encoding
            # HTML内の「---」などのゴミを置換してから読み込む
            clean_html = r.text.replace('---', '999')
            dfs = pd.read_html(io.StringIO(clean_html))
            
            best_df = None
            max_score = -1
            for d in dfs:
                # 文字列化した時に「馬名」が含まれるか
                combined_text = "".join(d.columns.astype(str)) + d.astype(str).to_string()
                score = ("馬名" in combined_text) + ("馬番" in combined_text) + ("オッズ" in combined_text)
                if score > max_score:
                    max_score = score
                    best_df = d
            
            if best_df is not None:
                # カラム名のクリーニング
                best_df.columns = [str(c).replace(' ', '') for c in best_df.columns]
                st.session_state.master_df = best_df
                st.success("取得完了！")
            else:
                st.error("出馬表テーブルが見つかりません。")
        except Exception as e:
            st.error(f"エラー: {e}")

with tab2:
    file = st.file_uploader("CSVファイルを選択", type="csv")
    if file:
        st.session_state.master_df = pd.read_csv(file)

if st.session_state.master_df is not None:
    st.divider()
    st.subheader("📋 取得データ（確認用）")
    st.dataframe(st.session_state.master_df, use_container_width=True)

    if st.button("🚀 AI予想を実行！", type="primary"):
        # 予想実行
        res = predict(st.session_state.master_df)
        
        st.divider()
        st.header("✨ 予想結果")
        
        # 重要な列だけを表示
        output_cols = ["印", "馬番", "馬名", "人気_num", "オッズ_num"]
        # カラムが存在するか確認してリネーム
        final_df = res[[c for c in output_cols if c in res.columns]].copy()
        final_df.columns = ["印", "馬番", "馬名", "人気", "オッズ"][:len(final_df.columns)]
        
        st.table(final_df)
        st.balloons()
