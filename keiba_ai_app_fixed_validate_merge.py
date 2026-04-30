# nyanko_keiba_ipad_cloud_20260428_safe_full.py
# ------------------------------------------------------------
# にゃんこ競馬AI iPad / Streamlit Cloud版 安全起動フル版
#
# 修正内容:
# - 起動直後にPKLを読まない。予想実行時だけ読む
# - Streamlit Cloudで Oh no になりにくいよう、main全体をtry/except
# - SimpleImputer _fill_dtype 補修
# - race_keyにsource_fileを混ぜて別レース混入を防止
# - 簡易CSVテンプレから実在馬名を削除
#
# 実行:
#   python -m streamlit run nyanko_keiba_ipad_cloud_20260428.py
# ------------------------------------------------------------

import io
import os
import re
from io import StringIO
from pathlib import Path

import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="にゃんこ競馬AI",
    page_icon="🐾",
    layout="wide"
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "nyanko_keiba_top3_model.pkl"
TARGET_CSV_PATH = APP_DIR / "yosou.csv"

COLS_52 = [
    "year", "month", "day", "kai", "place", "nichiji", "race_no", "race_name",
    "race_grade", "track_type", "course_kind", "distance", "going",
    "horse_name", "sex", "age", "jockey", "carried_weight",
    "field_size", "horse_no",
    "finish", "frame_no", "unknown_22",
    "odds", "popularity",
    "time_sec", "time_raw",
    "unknown_27",
    "pass1", "pass2", "pass3", "pass4",
    "last3f", "body_weight",
    "trainer", "belonging", "prize",
    "horse_id", "jockey_id", "trainer_id", "race_horse_id",
    "owner", "breeder",
    "sire", "dam", "broodmare_sire",
    "coat_color", "birthdate",
    "blank_48", "blank_49", "blank_50",
    "target_value"
]

NUMERIC_COLUMNS = [
    "year", "month", "day", "kai", "nichiji", "race_no", "race_grade",
    "course_kind", "distance", "age", "carried_weight", "field_size",
    "horse_no", "finish", "frame_no", "odds", "popularity", "time_sec",
    "pass1", "pass2", "pass3", "pass4", "last3f", "body_weight",
    "prize", "target_value"
]

JP_COLUMNS = {
    "mark": "印",
    "ml_rank": "AI順位",
    "horse_no": "馬番",
    "horse_name": "馬名",
    "sex": "性別",
    "age": "年齢",
    "jockey": "騎手",
    "carried_weight": "斤量",
    "odds": "オッズ",
    "popularity": "人気",
    "ml_top3_prob": "3着内確率",
    "expected_value": "期待値",
    "danger_popular": "危険人気馬",
    "value_horse": "穴候補",
    "jockey_top3_rate_prior": "騎手実績",
    "trainer_top3_rate_prior": "調教師実績",
    "sire_top3_rate_prior": "血統実績",
    "horse_distance_top3_rate_prior": "距離適性",
    "running_style": "脚質",
    "style_note": "脚質メモ",
    "value_score": "回収率スコア",
    "buy_flag": "判定",
    "buy_reason": "理由",
    "race_key": "レースID",
    "race_label": "レース"
}

DISPLAY_COLUMNS = [
    "mark", "ml_rank", "horse_no", "horse_name", "sex", "age", "jockey",
    "carried_weight", "odds", "popularity", "ml_top3_prob",
    "expected_value", "danger_popular", "value_horse", "running_style", "style_note",
    "jockey_top3_rate_prior", "trainer_top3_rate_prior",
    "sire_top3_rate_prior", "horse_distance_top3_rate_prior"
]

BASE_NUM_FEATURES = [
    "year_full", "month", "day", "race_no", "race_grade", "course_kind",
    "distance", "age", "carried_weight", "field_size", "horse_no", "frame_no",
    "odds", "popularity",
    "jockey_runs_prior", "jockey_win_rate_prior", "jockey_top3_rate_prior",
    "trainer_runs_prior", "trainer_win_rate_prior", "trainer_top3_rate_prior",
    "sire_runs_prior", "sire_win_rate_prior", "sire_top3_rate_prior",
    "horse_runs_prior", "horse_win_rate_prior", "horse_top3_rate_prior",
    "horse_distance_runs_prior", "horse_distance_top3_rate_prior",
    "horse_track_runs_prior", "horse_track_top3_rate_prior",
    "field_odds_rank", "field_pop_rank", "odds_gap_to_fav", "popularity_gap_to_fav"
]

CAT_FEATURES = [
    "place", "race_name", "track_type", "going", "sex", "jockey", "trainer",
    "belonging", "sire", "dam", "broodmare_sire"
]


def repair_simple_imputer(obj):
    seen = set()

    def walk(x):
        if x is None:
            return

        obj_id = id(x)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if x.__class__.__name__ == "SimpleImputer" and not hasattr(x, "_fill_dtype"):
            stat = getattr(x, "statistics_", None)
            try:
                x._fill_dtype = stat.dtype if stat is not None else np.dtype("float64")
            except Exception:
                x._fill_dtype = np.dtype("float64")

        for attr in ("steps", "transformers", "transformers_", "estimators", "estimators_"):
            if hasattr(x, attr):
                try:
                    for item in getattr(x, attr):
                        if isinstance(item, tuple):
                            for v in item:
                                if hasattr(v, "__dict__"):
                                    walk(v)
                        elif hasattr(item, "__dict__"):
                            walk(item)
                except Exception:
                    pass

        if hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                if hasattr(v, "__dict__"):
                    walk(v)
                elif isinstance(v, (list, tuple, set)):
                    for i in v:
                        if hasattr(i, "__dict__"):
                            walk(i)
                elif isinstance(v, dict):
                    for i in v.values():
                        if hasattr(i, "__dict__"):
                            walk(i)

    walk(obj)
    return obj



def parse_netkeiba_race_id(url: str) -> dict:
    m = re.search(r"race_id=(\d{12})", url or "")
    if not m:
        m = re.search(r"/race/(\d{12})", url or "")
    if not m:
        return {}

    race_id = m.group(1)
    place_map = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
    }
    return {
        "race_id": race_id,
        "year": int(race_id[0:4]),
        "place_code": race_id[4:6],
        "place": place_map.get(race_id[4:6], "不明"),
        "kai": int(race_id[6:8]),
        "nichiji": int(race_id[8:10]),
        "race_no": int(race_id[10:12]),
    }


def load_netkeiba_shutuba(url: str) -> pd.DataFrame:
    """
    netkeiba出馬表URLから発走前予想用CSV相当データを作る。
    発走前なので、着順・通過順・上がりは空。
    """
    info = parse_netkeiba_race_id(url)
    if not info:
        raise ValueError("netkeibaのrace_idをURLから取得できませんでした。")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://race.netkeiba.com/",
    }

    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
    except Exception as e:
        raise ValueError(f"netkeibaページ取得に失敗しました: {e}")

    html = res.text

    # Streamlit Cloudからnetkeibaへアクセスすると、実ページではなくブロック/別HTMLが返る場合がある。
    # ここで即エラーにせず、まずテーブル解析を試す。
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        snippet = html[:300].replace("\n", " ").replace("\r", " ")
        raise ValueError(
            "netkeibaの表を解析できませんでした。"
            "Streamlit CloudからのURL取得がブロックされている可能性があります。"
            "この場合は出馬表CSVアップロードを使ってください。"
            f" 詳細: {e} / HTML先頭: {snippet}"
        )

    if not tables:
        raise ValueError("出馬表テーブルを取得できませんでした。")

    src = None
    for t in tables:
        tmp_cols = t.columns
        if isinstance(tmp_cols, pd.MultiIndex):
            cols = ["_".join([str(x) for x in c if str(x) != "nan"]).strip("_") for c in tmp_cols]
        else:
            cols = [str(c) for c in tmp_cols]
        joined = " ".join(cols)
        if ("馬名" in joined or "馬番" in joined or "馬 番" in joined) and ("騎手" in joined or "斤量" in joined):
            src = t.copy()
            break

    if src is None:
        # テーブルはあるが出馬表ではない場合
        raise ValueError(
            "netkeibaから取得したHTML内に出馬表テーブルが見つかりません。"
            "Streamlit Cloudからのアクセス制限、またはURL違いの可能性があります。"
            "出馬表CSVアップロードなら予想できます。"
        )

    if isinstance(src.columns, pd.MultiIndex):
        src.columns = ["_".join([str(x) for x in c if str(x) != "nan"]).strip("_") for c in src.columns]
    else:
        src.columns = [str(c) for c in src.columns]

    rename = {}
    for c in src.columns:
        s = str(c)
        if "枠" == s or s.endswith("_枠") or "枠番" in s:
            rename[c] = "frame_no"
        elif "馬番" in s or "馬 番" in s:
            rename[c] = "horse_no"
        elif "馬名" in s:
            rename[c] = "horse_name"
        elif "性齢" in s or "性令" in s:
            rename[c] = "sex_age"
        elif "斤量" in s:
            rename[c] = "carried_weight"
        elif "騎手" in s:
            rename[c] = "jockey"
        elif "単勝" in s or "オッズ" in s:
            rename[c] = "odds"
        elif "人気" in s:
            rename[c] = "popularity"

    src = src.rename(columns=rename)

    missing = [c for c in ["horse_no", "horse_name"] if c not in src.columns]
    if missing:
        raise ValueError(f"出馬表から必要列を取得できませんでした: {missing} / columns={list(src.columns)}")

    src = src.dropna(subset=["horse_name"], how="all").copy()
    src["horse_name"] = src["horse_name"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    src = src[src["horse_name"].ne("")]
    src = src[~src["horse_name"].str.contains("馬名|出走取消|除外", na=False)]

    rows = []
    for i, r in src.iterrows():
        row = {c: "" for c in COLS_52}
        row["year"] = info["year"] - 2000
        row["month"] = 1
        row["day"] = 1
        row["kai"] = info.get("kai", 1)
        row["place"] = info.get("place", "不明")
        row["nichiji"] = info.get("nichiji", 1)
        row["race_no"] = info.get("race_no", 11)
        row["race_name"] = f"netkeiba_{info.get('race_id', '')}"
        row["race_grade"] = "3"
        row["track_type"] = ""
        row["course_kind"] = "0"
        row["distance"] = "0"
        row["going"] = ""
        row["horse_name"] = r.get("horse_name", "")
        sex_age = str(r.get("sex_age", "")).strip()
        if sex_age:
            row["sex"] = sex_age[0]
            age = re.sub(r"\D", "", sex_age[1:])
            row["age"] = age
        row["jockey"] = r.get("jockey", "")
        row["carried_weight"] = r.get("carried_weight", "")
        row["field_size"] = len(src)
        row["horse_no"] = r.get("horse_no", i + 1)
        row["frame_no"] = r.get("frame_no", "")
        row["odds"] = r.get("odds", "")
        row["popularity"] = r.get("popularity", "")
        rows.append([row[c] for c in COLS_52])

    df = pd.DataFrame(rows, columns=COLS_52)
    df["source_file"] = f"netkeiba_{info.get('race_id', '')}"
    return clean_types(df)


def load_uploaded_entry_csv(uploaded_csv, csv_mode: str) -> pd.DataFrame:
    """
    出馬表CSVを読む。
    - TARGET 52列CSV
    - 簡易CSV（馬番,馬名,性別,年齢,騎手,斤量,オッズ,人気）
    どちらでも自動判定する。
    """
    raw = uploaded_csv.read()

    # まずヘッダーありCSVとして読めるか確認
    header_df = None
    last_error = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            header_df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            break
        except Exception as e:
            last_error = e

    # 簡易CSVっぽい列がある場合は、画面選択に関係なく簡易CSVとして処理
    if header_df is not None:
        cols = set([str(c).strip() for c in header_df.columns])
        simple_markers = {
            "馬名", "horse_name",
            "騎手", "jockey",
            "オッズ", "odds",
            "人気", "popularity",
        }
        if len(cols & simple_markers) >= 3:
            return read_simple_csv_to_52(raw)

    # TARGET 52列形式を試す
    if csv_mode == "52列TARGET形式":
        try:
            df0 = read_csv_bytes(raw)
            return normalize_52cols(df0, uploaded_csv.name)
        except Exception as e:
            # 52列で失敗したら簡易CSVにフォールバック
            try:
                return read_simple_csv_to_52(raw)
            except Exception:
                raise e

    # 明示的に簡易CSV
    return read_simple_csv_to_52(raw)



def read_target_history_csv(path: Path) -> pd.DataFrame | None:
    """
    GitHub上のyosou.csv（TARGET過去CSV）を読み込む。
    日本語ヘッダー/英語ヘッダーの両方に寄せる。
    """
    if not path.exists():
        return None

    last_error = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            return normalize_target_history_columns(df)
        except Exception as e:
            last_error = e

    raise ValueError(f"TARGET過去CSVを読めませんでした: {last_error}")


def normalize_target_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TARGET CSVの列名をアプリ内部名へ変換する。
    例:
      確定着順/着順 -> finish
      単勝オッズ/オッズ -> odds
      馬名 -> horse_name
      騎手 -> jockey
      騎手コード -> jockey_id
      距離 -> distance
      場所/競馬場 -> place
      通過順1角 -> pass1
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "年": "year",
        "月": "month",
        "日": "day",
        "日付": "race_date",
        "回次": "kai",
        "場所": "place",
        "競馬場": "place",
        "日次": "nichiji",
        "レース番号": "race_no",
        "R": "race_no",
        "レース名": "race_name",
        "クラスコード": "race_grade",
        "芝・ダ": "track_type",
        "トラックコード": "track_type",
        "コース区分": "course_kind",
        "距離": "distance",
        "馬場状態": "going",
        "馬名": "horse_name",
        "性別": "sex",
        "性": "sex",
        "年齢": "age",
        "騎手": "jockey",
        "斤量": "carried_weight",
        "頭数": "field_size",
        "馬番": "horse_no",
        "枠番": "frame_no",
        "確定着順": "finish",
        "着順": "finish",
        "入線着順": "finish_raw",
        "単勝オッズ": "odds",
        "オッズ": "odds",
        "人気": "popularity",
        "走破タイム(秒)": "time_sec",
        "タイム": "time_raw",
        "通過順1角": "pass1",
        "通過順2角": "pass2",
        "通過順3角": "pass3",
        "通過順4角": "pass4",
        "上り3Fタイム": "last3f",
        "上がり3Fタイム": "last3f",
        "上り3F順位": "last3f_rank",
        "馬体重": "body_weight",
        "増減": "body_weight_diff",
        "調教師": "trainer",
        "所属": "belonging",
        "賞金": "prize",
        "騎手コード": "jockey_id",
        "調教師コード": "trainer_id",
        "血統登録番号": "horse_id",
        "父馬名": "sire",
        "母馬名": "dam",
        "母の父馬名": "broodmare_sire",
        "毛色": "coat_color",
        "生年月日": "birthdate",
    }

    # 完全一致
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # ゆらぎ対応
    dynamic_rename = {}
    for c in df.columns:
        s = str(c)
        if s in dynamic_rename:
            continue
        if "確定" in s and "着順" in s:
            dynamic_rename[c] = "finish"
        elif "単勝" in s and "オッズ" in s:
            dynamic_rename[c] = "odds"
        elif "通過" in s and "1" in s:
            dynamic_rename[c] = "pass1"
        elif "通過" in s and "2" in s:
            dynamic_rename[c] = "pass2"
        elif "通過" in s and "3" in s:
            dynamic_rename[c] = "pass3"
        elif "通過" in s and "4" in s:
            dynamic_rename[c] = "pass4"
        elif "上" in s and "3F" in s and "順位" not in s:
            dynamic_rename[c] = "last3f"
        elif "騎手" in s and "コード" in s:
            dynamic_rename[c] = "jockey_id"
        elif "調教師" in s and "コード" in s:
            dynamic_rename[c] = "trainer_id"

    if dynamic_rename:
        df = df.rename(columns=dynamic_rename)

    # 数値化
    for c in [
        "year", "month", "day", "race_no", "race_grade", "course_kind",
        "distance", "age", "carried_weight", "field_size", "horse_no",
        "frame_no", "finish", "odds", "popularity", "pass1", "pass2",
        "pass3", "pass4", "last3f", "body_weight", "prize"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["horse_name", "jockey", "trainer", "sire", "dam", "broodmare_sire", "place", "track_type", "going", "sex", "belonging"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})

    return df


def create_target_features(target_df: pd.DataFrame) -> dict:
    """
    TARGET過去CSVから、PKLが必要としている prior 特徴量を作る。
    """
    if target_df is None or target_df.empty:
        return {}

    df = target_df.copy()

    if "finish" not in df.columns:
        raise ValueError("TARGET過去CSVに 確定着順/着順 がありません。")

    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df[df["finish"].notna()].copy()

    if df.empty:
        return {}

    df["is_win"] = df["finish"].eq(1)
    df["is_top3"] = df["finish"].between(1, 3)

    features = {}

    # 騎手
    if "jockey" in df.columns:
        features["jockey_stats"] = (
            df.groupby("jockey", dropna=False)
            .agg(
                jockey_runs_prior=("finish", "count"),
                jockey_win_rate_prior=("is_win", "mean"),
                jockey_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 調教師
    if "trainer" in df.columns:
        features["trainer_stats"] = (
            df.groupby("trainer", dropna=False)
            .agg(
                trainer_runs_prior=("finish", "count"),
                trainer_win_rate_prior=("is_win", "mean"),
                trainer_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 種牡馬
    if "sire" in df.columns:
        features["sire_stats"] = (
            df.groupby("sire", dropna=False)
            .agg(
                sire_runs_prior=("finish", "count"),
                sire_win_rate_prior=("is_win", "mean"),
                sire_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 馬の総合成績
    if "horse_name" in df.columns:
        features["horse_stats"] = (
            df.groupby("horse_name", dropna=False)
            .agg(
                horse_runs_prior=("finish", "count"),
                horse_win_rate_prior=("is_win", "mean"),
                horse_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 馬×距離
    if "horse_name" in df.columns and "distance" in df.columns:
        features["horse_distance_stats"] = (
            df.groupby(["horse_name", "distance"], dropna=False)
            .agg(
                horse_distance_runs_prior=("finish", "count"),
                horse_distance_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 馬×競馬場
    if "horse_name" in df.columns and "place" in df.columns:
        features["horse_track_stats"] = (
            df.groupby(["horse_name", "place"], dropna=False)
            .agg(
                horse_track_runs_prior=("finish", "count"),
                horse_track_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    return features


@st.cache_data(show_spinner=False)
def load_target_features_cached():
    target_df = read_target_history_csv(TARGET_CSV_PATH)
    if target_df is None:
        return None, {}
    return target_df, create_target_features(target_df)


def merge_target_features(entry_df: pd.DataFrame) -> pd.DataFrame:
    """
    当日出馬表にTARGET過去CSV由来の特徴量を付与する。
    yosou.csvが無い場合は何もしない。
    """
    df = entry_df.copy()

    if not TARGET_CSV_PATH.exists():
        return df

    target_df, features = load_target_features_cached()
    if not features:
        return df

    if "jockey_stats" in features and "jockey" in df.columns:
        df = df.merge(features["jockey_stats"], on="jockey", how="left", suffixes=("", "_target"))

    if "trainer_stats" in features and "trainer" in df.columns:
        df = df.merge(features["trainer_stats"], on="trainer", how="left", suffixes=("", "_target"))

    if "sire_stats" in features and "sire" in df.columns:
        df = df.merge(features["sire_stats"], on="sire", how="left", suffixes=("", "_target"))

    if "horse_stats" in features and "horse_name" in df.columns:
        df = df.merge(features["horse_stats"], on="horse_name", how="left", suffixes=("", "_target"))

    if "horse_distance_stats" in features and {"horse_name", "distance"}.issubset(df.columns):
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df = df.merge(features["horse_distance_stats"], on=["horse_name", "distance"], how="left", suffixes=("", "_target"))

    if "horse_track_stats" in features and {"horse_name", "place"}.issubset(df.columns):
        df = df.merge(features["horse_track_stats"], on=["horse_name", "place"], how="left", suffixes=("", "_target"))

    # mergeで _target が付いた場合は空の元列を上書き
    for c in [
        "jockey_runs_prior", "jockey_win_rate_prior", "jockey_top3_rate_prior",
        "trainer_runs_prior", "trainer_win_rate_prior", "trainer_top3_rate_prior",
        "sire_runs_prior", "sire_win_rate_prior", "sire_top3_rate_prior",
        "horse_runs_prior", "horse_win_rate_prior", "horse_top3_rate_prior",
        "horse_distance_runs_prior", "horse_distance_top3_rate_prior",
        "horse_track_runs_prior", "horse_track_top3_rate_prior",
    ]:
        tc = f"{c}_target"
        if tc in df.columns:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[tc] = pd.to_numeric(df[tc], errors="coerce")
                df[c] = df[c].where(df[c].notna() & (df[c] != 0), df[tc])
            else:
                df[c] = df[tc]
            df = df.drop(columns=[tc])

    return df


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_error = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(io.BytesIO(raw), header=None, encoding=enc, dtype=str)
        except Exception as e:
            last_error = e
    raise ValueError(f"CSVを読めませんでした: {last_error}")


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": "", "None": "", "<NA>": ""})

    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["year"] = pd.to_numeric(df.get("year", 25), errors="coerce").fillna(25)
    df["month"] = pd.to_numeric(df.get("month", 4), errors="coerce").fillna(4)
    df["day"] = pd.to_numeric(df.get("day", 1), errors="coerce").fillna(1)
    df["race_no"] = pd.to_numeric(df.get("race_no", 11), errors="coerce").fillna(11)

    df["year_full"] = df["year"].apply(lambda x: int(x) + 2000 if pd.notna(x) and int(x) < 100 else int(x))
    df["date_int"] = (
        df["year_full"].fillna(0).astype(int) * 10000
        + df["month"].fillna(0).astype(int) * 100
        + df["day"].fillna(0).astype(int)
    )

    if "source_file" not in df.columns:
        df["source_file"] = ""

    # 別ファイル/別CSV由来の同一日・同一競馬場・同一R混入を防ぐ
    df["race_key"] = (
        df["date_int"].astype(str) + "_"
        + df.get("place", "").astype(str) + "_"
        + df["race_no"].fillna(0).astype(int).astype(str).str.zfill(2) + "_"
        + df["source_file"].astype(str)
    )

    df["race_label"] = (
        df["date_int"].astype(str) + " "
        + df.get("place", "").astype(str) + " "
        + df["race_no"].fillna(0).astype(int).astype(str) + "R "
        + df.get("race_name", "").astype(str)
    )

    return df


def normalize_52cols(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    need_cols = len(COLS_52)

    if len(df) > 0:
        first_row = df.iloc[0].astype(str).str.lower().tolist()
        if any(x in first_row for x in ["year", "horse_name", "source_file", "馬名"]):
            df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] > need_cols:
        first_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        second_col = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        if first_col.notna().mean() > 0.90 and second_col.dropna().between(0, 99).mean() > 0.50:
            df = df.iloc[:, 1:].copy()

    if df.shape[1] < need_cols:
        raise ValueError(f"列数不足です。52列必要ですが {df.shape[1]}列です。")

    source_series = None
    if df.shape[1] > need_cols:
        source_series = df.iloc[:, need_cols].astype(str).str.strip()

    df = df.iloc[:, :need_cols].copy()
    df.columns = COLS_52

    if source_series is not None and len(source_series) == len(df):
        df["source_file"] = source_series.values
    else:
        df["source_file"] = source_name

    return clean_types(df)


def read_simple_csv_to_52(raw: bytes, source_name: str = "simple_csv") -> pd.DataFrame:
    last_error = None
    src = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            src = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            break
        except Exception as e:
            last_error = e

    if src is None:
        raise ValueError(f"簡易CSVを読めませんでした: {last_error}")

    rename = {
        "馬名": "horse_name",
        "性別": "sex",
        "年齢": "age",
        "性齢": "sex_age",
        "騎手": "jockey",
        "斤量": "carried_weight",
        "オッズ": "odds",
        "単勝オッズ": "odds",
        "人気": "popularity",
        "年": "year",
        "月": "month",
        "日": "day",
        "競馬場": "place",
        "場所": "place",
        "レース番号": "race_no",
        "R": "race_no",
        "レース名": "race_name",
        "距離": "distance",
        "馬場": "going",
        "馬場状態": "going",
        "馬番": "horse_no",
        "枠番": "frame_no",
        "頭数": "field_size",
        "芝ダ": "track_type",
    }
    src = src.rename(columns=rename)

    # sex_age がある場合は 性別/年齢 に分解
    if "sex_age" in src.columns:
        if "sex" not in src.columns:
            src["sex"] = src["sex_age"].astype(str).str[0]
        if "age" not in src.columns:
            src["age"] = src["sex_age"].astype(str).str[1:].str.extract(r"(\d+)")[0]

    required = ["horse_name", "jockey", "carried_weight", "odds", "popularity"]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"簡易CSVの必須列が不足しています: {missing}")

    if "sex" not in src.columns:
        src["sex"] = ""
    if "age" not in src.columns:
        src["age"] = ""

    rows = []
    for i, r in src.iterrows():
        row = {c: "" for c in COLS_52}
        row["year"] = r.get("year", "25")
        row["month"] = r.get("month", "4")
        row["day"] = r.get("day", "1")
        row["kai"] = "1"
        row["place"] = r.get("place", "東京")
        row["nichiji"] = "1"
        row["race_no"] = r.get("race_no", "11")
        row["race_name"] = r.get("race_name", "未設定")
        row["race_grade"] = "3"
        row["track_type"] = r.get("track_type", "芝")
        row["course_kind"] = "0"
        row["distance"] = r.get("distance", "2000")
        row["going"] = r.get("going", "良")
        row["horse_name"] = r.get("horse_name", "")
        row["sex"] = r.get("sex", "")
        row["age"] = r.get("age", "")
        row["jockey"] = r.get("jockey", "")
        row["carried_weight"] = r.get("carried_weight", "")
        row["field_size"] = r.get("field_size", str(len(src)))
        row["horse_no"] = r.get("horse_no", str(i + 1))
        row["odds"] = r.get("odds", "")
        row["popularity"] = r.get("popularity", "")
        rows.append([row[c] for c in COLS_52])

    df = pd.DataFrame(rows, columns=COLS_52)
    df["source_file"] = source_name
    return clean_types(df)


def add_prior_stats_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in [
        "jockey_runs_prior", "jockey_win_rate_prior", "jockey_top3_rate_prior",
        "trainer_runs_prior", "trainer_win_rate_prior", "trainer_top3_rate_prior",
        "sire_runs_prior", "sire_win_rate_prior", "sire_top3_rate_prior",
        "horse_runs_prior", "horse_win_rate_prior", "horse_top3_rate_prior",
        "horse_distance_runs_prior", "horse_distance_top3_rate_prior",
        "horse_track_runs_prior", "horse_track_top3_rate_prior",
    ]:
        if c not in df.columns:
            df[c] = 0.0

    df["odds"] = pd.to_numeric(df.get("odds", 0), errors="coerce")
    df["popularity"] = pd.to_numeric(df.get("popularity", 99), errors="coerce")
    df["field_odds_rank"] = df.groupby("race_key")["odds"].rank(method="min", ascending=True)
    df["field_pop_rank"] = df.groupby("race_key")["popularity"].rank(method="min", ascending=True)

    fav_odds = df.groupby("race_key")["odds"].transform("min")
    fav_pop = df.groupby("race_key")["popularity"].transform("min")

    df["odds_gap_to_fav"] = df["odds"] - fav_odds
    df["popularity_gap_to_fav"] = df["popularity"] - fav_pop

    for c in BASE_NUM_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in CAT_FEATURES:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).fillna("")

    return df


def load_model_safely(uploaded_model):
    if uploaded_model is not None:
        model_obj = joblib.load(uploaded_model)
        return repair_simple_imputer(model_obj), "アップロードPKL"

    if MODEL_PATH.exists():
        model_obj = joblib.load(MODEL_PATH)
        return repair_simple_imputer(model_obj), "同梱PKL"

    return None, "未設定"


def get_pipeline_and_features(bundle):
    if isinstance(bundle, dict):
        feature_cols = bundle.get("feature_cols", BASE_NUM_FEATURES + CAT_FEATURES)
        pipe = bundle.get("pipeline") or bundle.get("model")
    else:
        feature_cols = BASE_NUM_FEATURES + CAT_FEATURES
        pipe = bundle

    if pipe is None:
        raise ValueError("PKL内に pipeline / model が見つかりません。")

    return pipe, feature_cols


def predict(bundle, df: pd.DataFrame) -> pd.DataFrame:
    df = add_prior_stats_for_prediction(df)
    df = add_running_style(df)
    pipe, feature_cols = get_pipeline_and_features(bundle)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"特徴量列が不足しています: {missing_features}")

    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(df[feature_cols])[:, 1]
    else:
        pred = pipe.predict(df[feature_cols])
        prob = np.asarray(pred, dtype=float)

    df["ml_top3_prob"] = prob
    df["ml_rank"] = df.groupby("race_key")["ml_top3_prob"].rank(ascending=False, method="first").astype(int)

    df["mark"] = df["ml_rank"].map({1: "◎", 2: "○", 3: "▲", 4: "△", 5: "☆"}).fillna("")
    df["expected_value"] = df["ml_top3_prob"] * df["odds"].fillna(0)
    df["danger_popular"] = ((df["popularity"].fillna(99) <= 3) & (df["ml_rank"] >= 5)).map({True: "危険", False: ""})
    df["value_horse"] = ((df["popularity"].fillna(0) >= 6) & (df["ml_rank"] <= 4)).map({True: "穴候補", False: ""})

    df = add_value_strategy(df)

    return df


def jp_view(df: pd.DataFrame, include_race_key=False) -> pd.DataFrame:
    cols = DISPLAY_COLUMNS.copy()
    if include_race_key:
        cols = ["race_label", "race_key"] + cols

    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    if "ml_top3_prob" in out.columns:
        out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    if "expected_value" in out.columns:
        out["expected_value"] = pd.to_numeric(out["expected_value"], errors="coerce").round(2)

    for c in ["jockey_top3_rate_prior", "trainer_top3_rate_prior", "sire_top3_rate_prior", "horse_distance_top3_rate_prior"]:
        if c in out.columns:
            out[c] = (pd.to_numeric(out[c], errors="coerce").fillna(0) * 100).round(1).astype(str) + "%"

    return out.rename(columns=JP_COLUMNS)



def add_running_style(df: pd.DataFrame) -> pd.DataFrame:
    """
    pass1〜pass4から脚質を推定する。
    目安:
      逃げ: 序盤で1〜2番手
      先行: 序盤で前目
      差し: 中団
      追込: 後方
    pass列が無い/空の場合は「不明」。
    """
    df = df.copy()

    for c in ["pass1", "pass2", "pass3", "pass4", "field_size", "finish"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def judge(row):
        passes = []
        for c in ["pass1", "pass2", "pass3", "pass4"]:
            v = row.get(c, np.nan)
            if pd.notna(v) and v > 0:
                passes.append(float(v))

        if not passes:
            return "不明", "通過順なし"

        field_size = row.get("field_size", np.nan)
        if pd.isna(field_size) or field_size <= 0:
            field_size = max(18, max(passes))

        early = passes[0]
        avg_pos = float(np.mean(passes))
        early_ratio = early / field_size
        avg_ratio = avg_pos / field_size

        # 逃げ: 最初の通過がかなり前
        if early <= 1.5 or early_ratio <= 0.12:
            return "逃げ", f"序盤{early:.0f}番手"

        # 先行: 前3〜4割
        if early_ratio <= 0.38 or avg_ratio <= 0.40:
            return "先行", f"前目 avg{avg_pos:.1f}"

        # 差し: 中団
        if avg_ratio <= 0.70:
            return "差し", f"中団 avg{avg_pos:.1f}"

        # 追込: 後方
        return "追込", f"後方 avg{avg_pos:.1f}"

    result = df.apply(judge, axis=1)
    df["running_style"] = [x[0] for x in result]
    df["style_note"] = [x[1] for x in result]
    return df


def make_style_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    finishがあるデータなら脚質別に勝率/3着内率を出す。
    予想用CSVでfinishが空なら件数だけ出る。
    """
    if "running_style" not in df.columns:
        df = add_running_style(df)

    tmp = df.copy()
    tmp["finish"] = pd.to_numeric(tmp.get("finish", np.nan), errors="coerce")
    tmp["is_win"] = tmp["finish"].eq(1)
    tmp["is_top3"] = tmp["finish"].between(1, 3)

    rows = []
    for style, g in tmp.groupby("running_style", dropna=False):
        runs = len(g)
        wins = int(g["is_win"].sum())
        top3 = int(g["is_top3"].sum())
        rows.append({
            "脚質": style,
            "件数": runs,
            "勝利数": wins,
            "3着内数": top3,
            "勝率": f"{(wins / runs * 100):.1f}%" if runs else "0.0%",
            "3着内率": f"{(top3 / runs * 100):.1f}%" if runs else "0.0%",
        })

    order = {"逃げ": 1, "先行": 2, "差し": 3, "追込": 4, "不明": 5}
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["脚質", "件数", "勝利数", "3着内数", "勝率", "3着内率"])
    out["_order"] = out["脚質"].map(order).fillna(99)
    return out.sort_values("_order").drop(columns=["_order"])


def show_style_tabs(pred_df: pd.DataFrame, race_df: pd.DataFrame):
    st.subheader("脚質分析")

    tab1, tab2, tab3 = st.tabs(["このレースの脚質", "脚質別成績", "脚質別AI順位"])

    with tab1:
        view_cols = ["mark", "ml_rank", "horse_no", "horse_name", "running_style", "style_note", "pass1", "pass2", "pass3", "pass4", "ml_top3_prob"]
        view_cols = [c for c in view_cols if c in race_df.columns]
        out = race_df.sort_values("ml_rank")[view_cols].copy()
        if "ml_top3_prob" in out.columns:
            out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
        st.dataframe(out.rename(columns=JP_COLUMNS), use_container_width=True, hide_index=True)

    with tab2:
        summary = make_style_summary(pred_df)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if "finish" not in pred_df.columns or pred_df["finish"].isna().all():
            st.caption("※予想CSVに着順finishが無い場合、勝率/3着内率は出ません。過去結果CSVを入れると集計できます。")

    with tab3:
        style_rank = (
            race_df.groupby("running_style", dropna=False)
            .agg(
                頭数=("horse_name", "count"),
                平均AI順位=("ml_rank", "mean"),
                平均3着内確率=("ml_top3_prob", "mean"),
            )
            .reset_index()
            .rename(columns={"running_style": "脚質"})
        )
        if not style_rank.empty:
            style_rank["平均AI順位"] = style_rank["平均AI順位"].round(2)
            style_rank["平均3着内確率"] = (style_rank["平均3着内確率"] * 100).round(1).astype(str) + "%"
        st.dataframe(style_rank, use_container_width=True, hide_index=True)



def add_value_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    的中率ではなく回収率寄りにするための補正。
    - value_score: 確率×オッズを中心にした期待値スコア
    - buy_flag: 買い/見送り
    - buy_reason: 理由
    """
    df = df.copy()

    df["odds"] = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df.get("popularity", 99), errors="coerce").fillna(99)
    df["ml_top3_prob"] = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)

    # 複勝・ワイド・三連複向けの基本期待値
    df["expected_value"] = (df["ml_top3_prob"] * df["odds"]).round(3)

    # 騎手補正: 3着内率が入っていれば加点。無ければ0扱い。
    jockey_rate = pd.to_numeric(df.get("jockey_top3_rate_prior", 0), errors="coerce").fillna(0)
    trainer_rate = pd.to_numeric(df.get("trainer_top3_rate_prior", 0), errors="coerce").fillna(0)
    sire_rate = pd.to_numeric(df.get("sire_top3_rate_prior", 0), errors="coerce").fillna(0)

    df["jockey_bonus"] = (jockey_rate - 0.25).clip(-0.10, 0.20)
    df["trainer_bonus"] = (trainer_rate - 0.25).clip(-0.05, 0.12)
    df["sire_bonus"] = (sire_rate - 0.25).clip(-0.05, 0.10)

    # 脚質補正: 極端に後ろすぎる追込は少しリスク、逃げ/先行は安定寄り
    style_bonus_map = {
        "逃げ": 0.04,
        "先行": 0.03,
        "差し": 0.00,
        "追込": -0.03,
        "不明": -0.01,
    }
    df["style_bonus"] = df.get("running_style", "不明").map(style_bonus_map).fillna(0)

    # 人気補正: 低人気でAI上位なら穴加点。人気馬でAI下位なら減点。
    df["ana_bonus"] = np.where((df["popularity"] >= 6) & (df["ml_rank"] <= 5), 0.12, 0.0)
    df["danger_penalty"] = np.where((df["popularity"] <= 3) & (df["ml_rank"] >= 5), -0.18, 0.0)

    # 最終スコア。1.0超えが一応買い候補。
    df["value_score"] = (
        df["expected_value"]
        * (1 + df["jockey_bonus"] + df["trainer_bonus"] + df["sire_bonus"] + df["style_bonus"] + df["ana_bonus"] + df["danger_penalty"])
    ).round(3)

    def judge(row):
        if row["ml_rank"] <= 3 and row["ml_top3_prob"] >= 0.22:
            return "買い", "AI上位・3着内確率高め"
        if row["value_score"] >= 1.10 and row["ml_rank"] <= 6:
            return "買い", "期待値高め"
        if row["value_score"] >= 0.95 and row["popularity"] >= 6 and row["ml_rank"] <= 5:
            return "買い", "穴期待"
        return "見送り", "期待値不足"

    judged = df.apply(judge, axis=1)
    df["buy_flag"] = [x[0] for x in judged]
    df["buy_reason"] = [x[1] for x in judged]

    return df


def make_value_summary(race_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mark", "ml_rank", "horse_no", "horse_name", "running_style",
        "odds", "popularity", "ml_top3_prob", "expected_value", "value_score",
        "buy_flag", "buy_reason", "danger_popular", "value_horse",
    ]
    cols = [c for c in cols if c in race_df.columns]
    out = race_df.sort_values(["buy_flag", "value_score", "ml_rank"], ascending=[True, False, True])[cols].copy()
    if "ml_top3_prob" in out.columns:
        out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    return out.rename(columns={
        **JP_COLUMNS,
        "value_score": "回収率スコア",
        "buy_flag": "判定",
        "buy_reason": "理由",
    })


def get_buy_candidates(race_df: pd.DataFrame, max_horses: int = 8) -> pd.DataFrame:
    """
    買い候補を返す。
    買い判定が少なすぎる場合は、value_score上位で補完。
    """
    r = race_df.sort_values(["value_score", "ml_top3_prob"], ascending=False).copy()
    buy = r[r["buy_flag"] == "買い"].copy()

    if len(buy) < 3:
        buy = r.head(max(3, min(max_horses, len(r)))).copy()

    # 重複除外して最大max_horses
    buy = buy.drop_duplicates(subset=["horse_no"]).head(max_horses)
    return buy


def generate_roi_bet_combinations(race_df: pd.DataFrame, max_count: int = 10) -> dict:
    """
    回収率寄りの買い目を最大10通り作る。
    AI順位だけではなく value_score と穴候補を使う。
    """
    r = race_df.sort_values(["value_score", "ml_top3_prob"], ascending=False).copy()
    buy = get_buy_candidates(race_df, max_horses=8)

    def no(row):
        try:
            return str(int(row["horse_no"]))
        except Exception:
            return str(row.get("horse_no", ""))

    def label(row):
        try:
            return f"{int(row['horse_no'])} {row['horse_name']}"
        except Exception:
            return str(row.get("horse_name", ""))

    nums = [no(row) for _, row in buy.iterrows() if no(row)]
    if not nums:
        return {}

    # 本命はAI順位1位、ただしvalue_scoreが低すぎる場合はvalue_score1位
    ai_top = race_df.sort_values("ml_rank").head(1)
    value_top = race_df.sort_values("value_score", ascending=False).head(1)
    if len(ai_top) and len(value_top):
        main = ai_top.iloc[0]
        if float(value_top.iloc[0]["value_score"]) > float(main.get("value_score", 0)) * 1.25:
            main = value_top.iloc[0]
    else:
        main = buy.iloc[0]

    main_no = no(main)

    # 穴候補
    ana = race_df[
        ((race_df["popularity"].fillna(0) >= 6) & (race_df["ml_rank"] <= 7))
        | (race_df["value_horse"] == "穴候補")
    ].sort_values("value_score", ascending=False)
    ana_nums = [no(row) for _, row in ana.head(5).iterrows() if no(row) and no(row) != main_no]

    combos = {}

    # 単勝: value_score上位。ただし単勝はAI1位も入れる
    tansho_rows = []
    for _, row in pd.concat([ai_top, r]).drop_duplicates(subset=["horse_no"]).head(max_count).iterrows():
        tansho_rows.append({"買い目": no(row), "馬名": label(row), "回収率スコア": row.get("value_score", 0), "理由": row.get("buy_reason", "")})
    combos["単勝"] = tansho_rows

    # 複勝: 3着内確率×期待値
    fukusho = []
    for _, row in r.head(max_count).iterrows():
        fukusho.append({"買い目": no(row), "馬名": label(row), "回収率スコア": row.get("value_score", 0), "理由": row.get("buy_reason", "")})
    combos["複勝"] = fukusho

    # 馬連: 本命軸 + value/穴
    umaren = []
    others = [n for n in nums if n != main_no]
    for n in others[:max_count]:
        umaren.append({"買い目": f"{main_no}-{n}", "狙い": "本命軸×期待値"})
    # 足りなければBOX
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            pair = f"{nums[i]}-{nums[j]}"
            if pair not in [x["買い目"] for x in umaren]:
                umaren.append({"買い目": pair, "狙い": "期待値BOX"})
            if len(umaren) >= max_count:
                break
        if len(umaren) >= max_count:
            break
    combos["馬連"] = umaren[:max_count]

    # 枠連
    wakuren = []
    if "frame_no" in race_df.columns and race_df["frame_no"].notna().any():
        frames = []
        for _, row in buy.iterrows():
            try:
                f = str(int(row["frame_no"]))
                if f:
                    frames.append(f)
            except Exception:
                pass
        for i in range(len(frames)):
            for j in range(i, len(frames)):
                pair = "-".join(sorted([frames[i], frames[j]]))
                if pair not in [x["買い目"] for x in wakuren]:
                    wakuren.append({"買い目": pair, "狙い": "枠連期待値"})
                if len(wakuren) >= max_count:
                    break
            if len(wakuren) >= max_count:
                break
    combos["枠連"] = wakuren or [{"買い目": "枠番データ不足", "狙い": "CSVに枠番が必要"}]

    # ワイド: 本命＋穴を厚め
    wide = []
    for n in (ana_nums + others):
        if n != main_no:
            wide.append({"買い目": f"{main_no}-{n}", "狙い": "本命×穴/期待値"})
        if len(wide) >= max_count:
            break
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            pair = f"{nums[i]}-{nums[j]}"
            if pair not in [x["買い目"] for x in wide]:
                wide.append({"買い目": pair, "狙い": "期待値ワイド"})
            if len(wide) >= max_count:
                break
        if len(wide) >= max_count:
            break
    combos["ワイド"] = wide[:max_count]

    # 馬単
    umatan = []
    for n in others[:max_count]:
        umatan.append({"買い目": f"{main_no}→{n}", "狙い": "本命頭固定"})
        if len(umatan) >= max_count:
            break
    # 穴頭も少し
    for a in ana_nums[:3]:
        if a != main_no:
            umatan.append({"買い目": f"{a}→{main_no}", "狙い": "穴頭リターン狙い"})
        if len(umatan) >= max_count:
            break
    combos["馬単"] = umatan[:max_count]

    # 三連複: 本命1頭軸 + 相手
    sanrenpuku = []
    partners = [n for n in nums if n != main_no]
    for i in range(len(partners)):
        for j in range(i + 1, len(partners)):
            sanrenpuku.append({"買い目": f"{main_no}-{partners[i]}-{partners[j]}", "狙い": "本命1頭軸"})
            if len(sanrenpuku) >= max_count:
                break
        if len(sanrenpuku) >= max_count:
            break
    combos["三連複"] = sanrenpuku[:max_count]

    # 三連単: 本命頭 + value相手 + 穴3着
    sanrentan = []
    seconds = partners[:5]
    thirds = list(dict.fromkeys(partners[:6] + ana_nums[:4]))
    for b in seconds:
        for c in thirds:
            if len({main_no, b, c}) == 3:
                sanrentan.append({"買い目": f"{main_no}→{b}→{c}", "狙い": "本命頭＋穴3着"})
            if len(sanrentan) >= max_count:
                break
        if len(sanrentan) >= max_count:
            break
    combos["三連単"] = sanrentan[:max_count]

    # 本命2頭＋穴
    honmei2_ana = []
    sorted_ai = race_df.sort_values("ml_rank")
    if len(sorted_ai) >= 2:
        h1 = no(sorted_ai.iloc[0])
        h2 = no(sorted_ai.iloc[1])
        use_ana = ana_nums or [n for n in nums if n not in [h1, h2]][:5]
        for a in use_ana:
            if a not in [h1, h2]:
                honmei2_ana.append({"買い目": f"{h1}-{h2}-{a}", "狙い": "本命2頭＋穴"})
            if len(honmei2_ana) >= max_count:
                break
    combos["本命2頭＋穴"] = honmei2_ana or [{"買い目": "穴候補なし", "狙い": "見送り推奨"}]

    # 本命1頭＋穴
    honmei1_ana = []
    use_ana = ana_nums or partners[:6]
    for a in use_ana:
        if a != main_no:
            honmei1_ana.append({"買い目": f"{main_no}-{a}", "狙い": "本命1頭＋穴"})
        if len(honmei1_ana) >= max_count:
            break
    combos["本命1頭＋穴"] = honmei1_ana or [{"買い目": "穴候補なし", "狙い": "見送り推奨"}]

    return combos


def show_roi_strategy(race_df: pd.DataFrame):
    st.subheader("回収率重視の買い/見送り判定")
    st.dataframe(make_value_summary(race_df), use_container_width=True, hide_index=True)

    buy_count = int((race_df["buy_flag"] == "買い").sum()) if "buy_flag" in race_df.columns else 0
    total = len(race_df)
    if buy_count == 0:
        st.warning("このレースは見送り寄りです。無理に買わない判定。")
    elif buy_count <= 3:
        st.info(f"買い候補は{buy_count}/{total}頭。絞れているので回収率重視向き。")
    else:
        st.info(f"買い候補は{buy_count}/{total}頭。BOXより軸流し推奨。")


def show_roi_ticket_tabs(race_df: pd.DataFrame):
    st.subheader("回収率重視TAB（各10通り）")
    combos = generate_roi_bet_combinations(race_df, max_count=10)
    order = ["単勝", "複勝", "馬連", "枠連", "ワイド", "馬単", "三連複", "三連単", "本命2頭＋穴", "本命1頭＋穴"]
    tabs = st.tabs(order)

    for tab, bet_type in zip(tabs, order):
        with tab:
            rows = combos.get(bet_type, [])
            if not rows:
                st.info("候補なし")
                continue
            df_show = pd.DataFrame(rows)
            df_show.insert(0, "No", range(1, len(df_show) + 1))
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            if bet_type in ["単勝", "複勝"]:
                st.caption("※回収率スコア上位。人気馬だけでなく妙味馬を含めます。")
            elif bet_type in ["ワイド", "三連複", "本命1頭＋穴", "本命2頭＋穴"]:
                st.caption("※本命＋穴を優先。複勝圏狙い。")
            elif bet_type in ["馬単", "三連単"]:
                st.caption("※順序あり。リターン重視なので点数を絞って使う想定。")

def make_tickets(race_df: pd.DataFrame) -> dict:
    """画面上部の簡易サマリー用"""
    r = race_df.sort_values("ml_rank").copy()

    def horse_label(row):
        try:
            return f"{int(row['horse_no'])} {row['horse_name']}"
        except Exception:
            return str(row.get("horse_name", ""))

    top = r.head(6)
    danger = r[r["danger_popular"] == "危険"]
    value = r[r["value_horse"] == "穴候補"].copy()

    if value.empty:
        value = r[(r["popularity"].fillna(0) >= 6) & (r["ml_rank"] <= 8)].copy()

    return {
        "本命": horse_label(top.iloc[0]) if len(top) else "",
        "単勝": horse_label(top.iloc[0]) if len(top) else "",
        "複勝": " / ".join([horse_label(row) for _, row in top.head(3).iterrows()]),
        "危険人気馬": " / ".join([horse_label(row) for _, row in danger.iterrows()]) or "なし",
        "穴候補": " / ".join([horse_label(row) for _, row in value.head(5).iterrows()]) or "なし",
    }


def _horse_no(row) -> str:
    try:
        return str(int(row["horse_no"]))
    except Exception:
        return str(row.get("horse_no", ""))


def _horse_label(row) -> str:
    try:
        return f"{int(row['horse_no'])} {row['horse_name']}"
    except Exception:
        return str(row.get("horse_name", ""))


def _frame_no(row) -> str:
    try:
        if pd.notna(row.get("frame_no", np.nan)):
            return str(int(row["frame_no"]))
    except Exception:
        pass
    return ""


def generate_bet_combinations(race_df: pd.DataFrame, max_count: int = 10) -> dict:
    """
    馬券種別ごとにおすすめ買い目を最大10通り作る。
    AI順位を基本に、穴候補も一部混ぜる。
    """
    r = race_df.sort_values(["ml_rank", "popularity"]).copy()
    r = r[pd.notna(r["horse_no"])].copy()

    if r.empty:
        return {}

    top = r.head(8).copy()
    top_nums = [_horse_no(row) for _, row in top.iterrows() if _horse_no(row)]

    # 穴候補: value_horse優先。無ければ人気6番人気以下かつAI順位8位以内
    ana = r[r["value_horse"] == "穴候補"].copy()
    if ana.empty:
        ana = r[(r["popularity"].fillna(0) >= 6) & (r["ml_rank"] <= 8)].copy()
    ana_nums = [_horse_no(row) for _, row in ana.head(5).iterrows() if _horse_no(row)]

    # 上位だけだと堅すぎるので、穴を末尾に混ぜた候補リスト
    candidate_nums = []
    for n in top_nums + ana_nums:
        if n and n not in candidate_nums:
            candidate_nums.append(n)

    candidate_nums = candidate_nums[:8]

    def name_of(num):
        hit = r[r["horse_no"].astype("Int64", errors="ignore").astype(str) == str(num)]
        if len(hit):
            return _horse_label(hit.iloc[0])
        return str(num)

    combos = {}

    # 単勝: AI順位上位10
    combos["単勝"] = [
        {"買い目": _horse_no(row), "馬名": _horse_label(row), "狙い": "AI上位"}
        for _, row in r.head(max_count).iterrows()
    ]

    # 複勝: AI順位上位10
    combos["複勝"] = [
        {"買い目": _horse_no(row), "馬名": _horse_label(row), "狙い": "3着内狙い"}
        for _, row in r.head(max_count).iterrows()
    ]

    # 馬連: 上位5頭BOX中心 + 穴絡み
    umaren = []
    base = candidate_nums[:6]
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            umaren.append({"買い目": f"{base[i]}-{base[j]}", "狙い": "上位BOX/穴絡み"})
            if len(umaren) >= max_count:
                break
        if len(umaren) >= max_count:
            break
    combos["馬連"] = umaren

    # 枠連: 枠番がある場合のみ
    wakuren = []
    frame_pairs = []
    if "frame_no" in r.columns and r["frame_no"].notna().any():
        tmp = top.head(8).copy()
        frames = []
        for _, row in tmp.iterrows():
            f = _frame_no(row)
            if f:
                frames.append(f)
        for i in range(len(frames)):
            for j in range(i, len(frames)):
                pair = "-".join(sorted([frames[i], frames[j]]))
                if pair not in frame_pairs:
                    frame_pairs.append(pair)
                    wakuren.append({"買い目": pair, "狙い": "枠番上位"})
                if len(wakuren) >= max_count:
                    break
            if len(wakuren) >= max_count:
                break
    if not wakuren:
        wakuren = [{"買い目": "枠番データ不足", "狙い": "CSVにframe_no/枠番が必要"}]
    combos["枠連"] = wakuren

    # ワイド: 馬連より広め。上位+穴
    wide = []
    wide_base = candidate_nums[:7]
    for i in range(len(wide_base)):
        for j in range(i + 1, len(wide_base)):
            wide.append({"買い目": f"{wide_base[i]}-{wide_base[j]}", "狙い": "3着内2頭狙い"})
            if len(wide) >= max_count:
                break
        if len(wide) >= max_count:
            break
    combos["ワイド"] = wide

    # 馬単: 1〜3位を頭にして流す
    umatan = []
    heads = candidate_nums[:3]
    tails = candidate_nums[:7]
    for h in heads:
        for t in tails:
            if h != t:
                umatan.append({"買い目": f"{h}→{t}", "狙い": "AI上位頭固定"})
            if len(umatan) >= max_count:
                break
        if len(umatan) >= max_count:
            break
    combos["馬単"] = umatan

    # 三連複: 上位+穴BOX
    sanrenpuku = []
    tri_base = candidate_nums[:7]
    for i in range(len(tri_base)):
        for j in range(i + 1, len(tri_base)):
            for k in range(j + 1, len(tri_base)):
                sanrenpuku.append({"買い目": f"{tri_base[i]}-{tri_base[j]}-{tri_base[k]}", "狙い": "上位+穴BOX"})
                if len(sanrenpuku) >= max_count:
                    break
            if len(sanrenpuku) >= max_count:
                break
        if len(sanrenpuku) >= max_count:
            break
    combos["三連複"] = sanrenpuku

    # 三連単: 1〜3位を中心に順序付き
    sanrentan = []
    firsts = candidate_nums[:3]
    seconds = candidate_nums[:5]
    thirds = candidate_nums[:6]
    for a in firsts:
        for b in seconds:
            for c in thirds:
                if len({a, b, c}) == 3:
                    sanrentan.append({"買い目": f"{a}→{b}→{c}", "狙い": "AI上位順序"})
                if len(sanrentan) >= max_count:
                    break
            if len(sanrentan) >= max_count:
                break
        if len(sanrentan) >= max_count:
            break
    combos["三連単"] = sanrentan

    # 本命2頭＋穴
    honmei2_ana = []
    if len(candidate_nums) >= 2:
        h1, h2 = candidate_nums[0], candidate_nums[1]
        ana_use = ana_nums if ana_nums else candidate_nums[2:7]
        for a in ana_use:
            if a not in [h1, h2]:
                honmei2_ana.append({"買い目": f"{h1}-{h2}-{a}", "狙い": "本命2頭＋穴"})
            if len(honmei2_ana) >= max_count:
                break
    combos["本命2頭＋穴"] = honmei2_ana or [{"買い目": "穴候補なし", "狙い": "人気/AI順位から穴が拾えません"}]

    # 本命1頭＋穴
    honmei1_ana = []
    if candidate_nums:
        h1 = candidate_nums[0]
        ana_use = ana_nums if ana_nums else candidate_nums[1:8]
        for a in ana_use:
            if a != h1:
                honmei1_ana.append({"買い目": f"{h1}-{a}", "狙い": "本命1頭＋穴"})
            if len(honmei1_ana) >= max_count:
                break
    combos["本命1頭＋穴"] = honmei1_ana or [{"買い目": "穴候補なし", "狙い": "人気/AI順位から穴が拾えません"}]

    return combos


def show_ticket_tabs(race_df: pd.DataFrame):
    st.subheader("馬券おすすめ（TAB別・各10通り）")

    combos = generate_bet_combinations(race_df, max_count=10)
    order = ["単勝", "複勝", "馬連", "枠連", "ワイド", "馬単", "三連複", "三連単", "本命2頭＋穴", "本命1頭＋穴"]

    tabs = st.tabs(order)

    for tab, bet_type in zip(tabs, order):
        with tab:
            rows = combos.get(bet_type, [])
            if not rows:
                st.info("候補なし")
                continue

            df_show = pd.DataFrame(rows)

            # 見やすくする
            if "買い目" not in df_show.columns:
                df_show["買い目"] = ""
            if "狙い" not in df_show.columns:
                df_show["狙い"] = ""

            df_show.insert(0, "No", range(1, len(df_show) + 1))
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            if bet_type in ["三連単", "馬単"]:
                st.caption("※順序あり。左から着順指定。")
            elif bet_type in ["馬連", "ワイド", "三連複", "枠連"]:
                st.caption("※順序なし。BOX/流し候補。")

def app_main():
    st.title("🐾 にゃんこ競馬AI")
    st.caption("iPad / Streamlit Cloud対応版。netkeiba URLまたは出馬表CSVから発走前予想できます。")

    with st.sidebar:
        st.header("設定")
        uploaded_model = st.file_uploader("学習済みモデルPKL", type=["pkl"])
        csv_mode = st.radio("予想CSV形式", ["52列TARGET形式", "簡易CSV形式"], index=0)
        st.info("GitHubの models/nyanko_keiba_top3_model.pkl にPKLがあれば、iPadではPKLアップロード不要です。")
        if MODEL_PATH.exists():
            st.success(f"同梱PKLあり: {MODEL_PATH.name}")
        else:
            st.warning("同梱PKLなし。画面からPKLをアップロードしてください。")

        if TARGET_CSV_PATH.exists():
            st.success(f"TARGET過去CSVあり: {TARGET_CSV_PATH.name}")
        else:
            st.info("TARGET過去CSVなし: yosou.csv をリポジトリ直下に置くと補正します。")

    st.subheader("入力方法")

    input_method = st.radio(
        "入力方法を選択",
        ["出馬表CSV", "netkeiba URL"],
        horizontal=True,
        index=0
    )

    race_url = ""
    uploaded_csv = None

    if input_method == "netkeiba URL":
        race_url = st.text_input(
            "netkeiba 出馬表URL",
            placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=202605020111"
        )
        st.caption("URL取得はnetkeiba側にブロックされる場合があります。その場合は出馬表CSVを使ってください。")
    else:
        uploaded_csv = st.file_uploader("予想CSVをアップロード", type=["csv"])
        st.caption("TARGET 52列CSV、または簡易CSVを使えます。CSV選択時はURL欄が残っていても無視します。")

    if input_method == "netkeiba URL" and not (race_url and race_url.strip()):
        st.info("netkeiba 出馬表URLを入力してください。")
        return

    if input_method == "出馬表CSV" and uploaded_csv is None:
        st.info("出馬表CSVをアップロードしてください。")
        return

    if st.button("予想する", type="primary"):
        try:
            bundle, model_status = load_model_safely(uploaded_model)
            if bundle is None:
                st.error("学習済みモデルPKLがありません。modelsフォルダに置くか、サイドバーからアップロードしてください。")
                return

            st.success(f"モデル読込: {model_status}")

            if input_method == "netkeiba URL":
                pred_src = load_netkeiba_shutuba(race_url.strip())
                st.success("netkeiba出馬表URLから取得しました。")
            else:
                pred_src = load_uploaded_entry_csv(uploaded_csv, csv_mode)
                st.success("出馬表CSVから取得しました。")

            # TARGET過去CSV（yosou.csv）があれば、騎手・調教師・血統・馬の適性を結合
            pred_src = merge_target_features(pred_src)

            if TARGET_CSV_PATH.exists():
                st.success("TARGET過去CSV（yosou.csv）を結合しました。")
            else:
                st.info("TARGET過去CSV（yosou.csv）は未配置です。URL/CSV単体で予想します。")

            pred_df = predict(bundle, pred_src)

            st.subheader("予想結果")
            race_options = (
                pred_df[["race_key", "race_label"]]
                .drop_duplicates()
                .sort_values("race_label")
            )
            label_map = dict(zip(race_options["race_label"], race_options["race_key"]))
            selected_label = st.selectbox("レース選択", list(label_map.keys()))
            selected_race = label_map[selected_label]

            race_df = pred_df[pred_df["race_key"] == selected_race].sort_values("ml_rank")
            st.dataframe(jp_view(race_df), use_container_width=True, hide_index=True)

            tickets = make_tickets(race_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("本命", tickets["本命"])
            c2.metric("単勝", tickets["単勝"])
            c3.metric("複勝", tickets["複勝"])

            show_ticket_tabs(race_df)

            show_roi_strategy(race_df)
            show_roi_ticket_tabs(race_df)

            show_style_tabs(pred_df, race_df)

            c4, c5 = st.columns(2)
            c4.info(f"危険人気馬: {tickets.get('危険人気馬', 'なし')}")
            c5.success(f"穴候補: {tickets.get('穴候補', 'なし')}")

            st.subheader("全レース")
            all_jp = jp_view(pred_df.sort_values(["race_key", "ml_rank"]), include_race_key=True)
            st.dataframe(all_jp, use_container_width=True, hide_index=True)

            csv_bytes = all_jp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "日本語CSVダウンロード",
                data=csv_bytes,
                file_name="nyanko_keiba_prediction_jp.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"予想できませんでした: {e}")
            st.exception(e)

    st.divider()
    with st.expander("簡易CSVテンプレ"):
        st.caption("※これは入力例です。実在馬名は入れていません。実際の出走馬CSVをアップロードしてください。")
        st.code("""馬番,馬名,性別,年齢,騎手,斤量,オッズ,人気,競馬場,レース番号,レース名,距離,馬場,頭数,芝ダ
1,サンプルホースA,牡,5,サンプル騎手A,58.0,2.8,1,東京,11,サンプルレース,2000,良,18,芝
2,サンプルホースB,牝,4,サンプル騎手B,56.0,8.5,5,東京,11,サンプルレース,2000,良,18,芝
""", language="csv")


try:
    app_main()
except Exception as e:
    st.error("アプリ起動時エラーです。下の詳細を確認してください。")
    st.exception(e)
