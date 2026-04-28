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
    r = race_df.sort_values("ml_rank")

    nums = []
    for _, row in r.head(5).iterrows():
        if pd.notna(row["horse_no"]):
            nums.append(str(int(row["horse_no"])))

    def n(row):
        return f"{int(row['horse_no'])} {row['horse_name']}" if pd.notna(row["horse_no"]) else str(row["horse_name"])

    top = r.head(5)
    danger = r[r["danger_popular"] == "危険"]
    value = r[r["value_horse"] == "穴候補"]

    return {
        "本命": n(top.iloc[0]) if len(top) else "",
        "馬連BOX": " - ".join(nums[:4]),
        "三連複BOX": " - ".join(nums[:5]),
        "危険人気馬": " / ".join([n(row) for _, row in danger.iterrows()]) or "なし",
        "穴候補": " / ".join([n(row) for _, row in value.iterrows()]) or "なし",
    }


def app_main():
    st.title("🐾 にゃんこ競馬AI")
    st.caption("iPad / Streamlit Cloud対応版。URLを開いて予想CSVを入れるだけ。")

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
            c2.metric("馬連BOX", tickets["馬連BOX"])
            c3.metric("三連複BOX", tickets["三連複BOX"])

            st.write(f"**危険人気馬:** {tickets['危険人気馬']}")
            st.write(f"**穴候補:** {tickets['穴候補']}")

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
