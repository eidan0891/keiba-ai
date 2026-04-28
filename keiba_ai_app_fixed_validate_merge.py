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
#   python -m streamlit run nyanko_keiba_ipad_cloud_20260428_target_pkl_fixed.py
# ------------------------------------------------------------

import io
from pathlib import Path

import joblib
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
        "騎手": "jockey",
        "斤量": "carried_weight",
        "オッズ": "odds",
        "人気": "popularity",
        "年": "year",
        "月": "month",
        "日": "day",
        "競馬場": "place",
        "レース番号": "race_no",
        "R": "race_no",
        "レース名": "race_name",
        "距離": "distance",
        "馬場": "going",
        "馬番": "horse_no",
        "頭数": "field_size",
        "芝ダ": "track_type",
    }
    src = src.rename(columns=rename)

    required = ["horse_name", "sex", "age", "jockey", "carried_weight", "odds", "popularity"]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"簡易CSVの必須列が不足しています: {missing}")

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



# ------------------------------------------------------------
# PKL内TARGETデータから事前成績を復元する補修
# ------------------------------------------------------------
def normalize_text_key(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\u3000", " ").strip().replace(" ", "")


def flatten_bundle_dataframes(obj, prefix="pkl", seen=None):
    if seen is None:
        seen = set()
    out = []
    if obj is None:
        return out
    oid = id(obj)
    if oid in seen:
        return out
    seen.add(oid)
    if isinstance(obj, pd.DataFrame):
        return [(prefix, obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten_bundle_dataframes(v, f"{prefix}.{k}", seen))
        return out
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(flatten_bundle_dataframes(v, f"{prefix}[{i}]", seen))
        return out
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if isinstance(v, pd.DataFrame):
                out.append((f"{prefix}.{k}", v))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, pd.DataFrame):
                        out.append((f"{prefix}.{k}.{kk}", vv))
    return out


def normalize_history_df_for_priors(src: pd.DataFrame) -> pd.DataFrame:
    df = src.copy()
    if df.shape[1] >= 52 and not any(c in df.columns for c in ["horse_name", "馬名"]):
        df = df.iloc[:, :52].copy()
        df.columns = COLS_52
    rename = {
        "馬名": "horse_name", "騎手": "jockey", "調教師": "trainer", "種牡馬": "sire",
        "母": "dam", "母父": "broodmare_sire", "距離": "distance", "芝ダ": "track_type",
        "馬場": "going", "競馬場": "place", "着順": "finish", "人気": "popularity",
        "オッズ": "odds", "年": "year", "月": "month", "日": "day", "R": "race_no",
        "レース番号": "race_no", "馬番": "horse_no", "枠番": "frame_no", "頭数": "field_size",
        "通過1": "pass1", "通過2": "pass2", "通過3": "pass3", "通過4": "pass4",
        "上がり3F": "last3f", "horse": "horse_name", "jockey_name": "jockey",
        "trainer_name": "trainer", "sire_name": "sire",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "finish" not in df.columns:
        return pd.DataFrame()
    if not any(c in df.columns for c in ["horse_name", "jockey", "trainer", "sire"]):
        return pd.DataFrame()
    for c in ["year", "month", "day", "race_no", "distance", "finish", "horse_no", "frame_no", "field_size", "odds", "popularity", "pass1", "pass2", "pass3", "pass4", "last3f"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date_int" not in df.columns:
        if all(c in df.columns for c in ["year", "month", "day"]):
            y = pd.to_numeric(df["year"], errors="coerce").fillna(0)
            y = y.apply(lambda x: int(x) + 2000 if int(x) < 100 and int(x) > 0 else int(x))
            df["date_int"] = y * 10000 + pd.to_numeric(df["month"], errors="coerce").fillna(0).astype(int) * 100 + pd.to_numeric(df["day"], errors="coerce").fillna(0).astype(int)
        else:
            df["date_int"] = 0
    for c in ["horse_name", "jockey", "trainer", "sire", "track_type", "place"]:
        if c not in df.columns:
            df[c] = ""
        df[c + "_key"] = df[c].map(normalize_text_key)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df[df["finish"].notna()].copy()
    df = df[df["finish"] > 0].copy()
    df["is_win"] = df["finish"].eq(1)
    df["is_top3"] = df["finish"].between(1, 3)
    return df


def build_rate_table(hist: pd.DataFrame, key_cols, prefix: str) -> pd.DataFrame:
    if hist.empty or any(c not in hist.columns for c in key_cols):
        return pd.DataFrame()
    g = hist.groupby(key_cols, dropna=False).agg(runs=("finish", "count"), wins=("is_win", "sum"), top3=("is_top3", "sum")).reset_index()
    g[f"{prefix}_runs_prior"] = g["runs"].astype(float)
    g[f"{prefix}_win_rate_prior"] = (g["wins"] / g["runs"]).fillna(0.0)
    g[f"{prefix}_top3_rate_prior"] = (g["top3"] / g["runs"]).fillna(0.0)
    return g.drop(columns=["runs", "wins", "top3"])


def add_target_priors_from_pkl(bundle, df: pd.DataFrame):
    base = df.copy()
    debug = []
    for c in ["horse_name", "jockey", "trainer", "sire", "track_type", "place"]:
        if c not in base.columns:
            base[c] = ""
        base[c + "_key"] = base[c].map(normalize_text_key)
    if "distance" not in base.columns:
        base["distance"] = 0
    base["distance"] = pd.to_numeric(base["distance"], errors="coerce").fillna(0)

    frames = flatten_bundle_dataframes(bundle)
    hist_list = []
    for name, f in frames:
        h = normalize_history_df_for_priors(f)
        if not h.empty and len(h) >= 10:
            hist_list.append(h)
            debug.append(f"{name}: {len(h)}行")

    if not hist_list:
        base.attrs["pkl_prior_debug"] = "PKL内にTARGET過去成績DataFrameを検出できず。0%表示のまま。"
        return add_prior_stats_for_prediction(base)

    hist = pd.concat(hist_list, ignore_index=True, sort=False).drop_duplicates()
    if "date_int" in base.columns and base["date_int"].notna().any():
        min_pred_date = pd.to_numeric(base["date_int"], errors="coerce").min()
        if pd.notna(min_pred_date) and min_pred_date > 0:
            old = hist[pd.to_numeric(hist["date_int"], errors="coerce").fillna(0) < min_pred_date].copy()
            if not old.empty:
                hist = old

    for keys, prefix in [
        (["jockey_key"], "jockey"),
        (["trainer_key"], "trainer"),
        (["sire_key"], "sire"),
        (["horse_name_key"], "horse"),
        (["horse_name_key", "distance"], "horse_distance"),
        (["horse_name_key", "track_type_key"], "horse_track"),
    ]:
        tbl = build_rate_table(hist, keys, prefix)
        if not tbl.empty:
            base = base.merge(tbl, on=keys, how="left")

    base = add_prior_stats_for_prediction(base)
    nonzero = {}
    for c in ["jockey_top3_rate_prior", "trainer_top3_rate_prior", "sire_top3_rate_prior", "horse_distance_top3_rate_prior"]:
        if c in base.columns:
            nonzero[c] = int((pd.to_numeric(base[c], errors="coerce").fillna(0) > 0).sum())
    base.attrs["pkl_prior_debug"] = " / ".join(debug[:5]) + f" / 非ゼロ件数: {nonzero}"
    return base


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
    df = add_target_priors_from_pkl(bundle, df)
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
    st.caption("iPad / Streamlit Cloud対応版。単勝・複勝・馬連・枠連・ワイド・馬単・三連複・三連単・本命＋穴まで出します。")

    with st.sidebar:
        st.header("設定")
        uploaded_model = st.file_uploader("学習済みモデルPKL", type=["pkl"])
        csv_mode = st.radio("予想CSV形式", ["52列TARGET形式", "簡易CSV形式"], index=0)
        st.info("GitHubの models/nyanko_keiba_top3_model.pkl にPKLがあれば、iPadではPKLアップロード不要です。")
        if MODEL_PATH.exists():
            st.success(f"同梱PKLあり: {MODEL_PATH.name}")
        else:
            st.warning("同梱PKLなし。画面からPKLをアップロードしてください。")

    uploaded_csv = st.file_uploader("予想CSVをアップロード", type=["csv"])

    if uploaded_csv is None:
        st.info("予想CSVをアップロードしてください。")
        return

    if st.button("予想する", type="primary"):
        try:
            bundle, model_status = load_model_safely(uploaded_model)
            if bundle is None:
                st.error("学習済みモデルPKLがありません。modelsフォルダに置くか、サイドバーからアップロードしてください。")
                return

            st.success(f"モデル読込: {model_status}")

            raw = uploaded_csv.read()
            if csv_mode == "52列TARGET形式":
                df0 = read_csv_bytes(raw)
                pred_src = normalize_52cols(df0, uploaded_csv.name)
            else:
                pred_src = read_simple_csv_to_52(raw, uploaded_csv.name)

            pred_df = predict(bundle, pred_src)
            prior_debug = pred_df.attrs.get("pkl_prior_debug", "")
            if prior_debug:
                st.info(f"PKL/TARGETデータ連携: {prior_debug}")

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
