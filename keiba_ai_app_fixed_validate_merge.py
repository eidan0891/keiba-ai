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
    "race_key": "レースID",
    "race_label": "レース"
}

DISPLAY_COLUMNS = [
    "mark", "ml_rank", "horse_no", "horse_name", "sex", "age", "jockey",
    "carried_weight", "odds", "popularity", "ml_top3_prob",
    "expected_value", "danger_popular", "value_horse",
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
