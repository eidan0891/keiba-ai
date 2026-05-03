# nyanko_keiba_ipad_cloud_20260428_safe_full_rankfix.py
# ------------------------------------------------------------
# にゃんこ競馬AI iPad / Streamlit Cloud版 安全起動フル版
#
# 修正内容:
# - 起動直後にPKLを読まない。予想実行時だけ読む
# - Streamlit Cloudで Oh no になりにくいよう、main全体をtry/except
# - SimpleImputer _fill_dtype 補修
# - race_keyにsource_fileを混ぜて別レース混入を防止
# - 簡易CSVテンプレから実在馬名を削除
# - AI順位を印より前に表示、☆以降も×/無印で順位が分かるように修正
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
DATA_DIR = APP_DIR / "data"

def find_target_csv_path() -> Path | None:
    """
    TARGET過去CSVの場所を自動検出する。
    空ファイルは無視する。
    """
    candidates = [
        APP_DIR / "yosou.csv",
        APP_DIR / "yosou_clean.csv",
        Path.cwd() / "yosou.csv",
        Path.cwd() / "yosou_clean.csv",
        Path.cwd() / "254" / "yosou.csv",
        Path.cwd() / "254" / "yosou_clean.csv",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return p
        except Exception:
            pass
    return None

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
    "ml_rank", "mark", "horse_no", "horse_name", "sex", "age", "jockey",
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




PLACE_MAP = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}
PLACE_CODE_MAP = {v: k for k, v in PLACE_MAP.items()}


# ------------------------------------------------------------
# v4: TARGET過去CSVと出馬表CSVの表記ゆれ吸収
# - 馬名/騎手/競馬場の空白・全角空白を除去
# - 騎手の短縮表記（北村友 → 北村友一 など）を寄せる
# - 距離/通過順/着順/上がり3Fを数値化
# これを入れないと、CSV内にクロワデュノール等が存在しても
# 予想側と完全一致せず「未取得」になりやすい。
# ------------------------------------------------------------
JOCKEY_ALIAS_MAP = {
    "岩田望": "岩田望来",
    "北村友": "北村友一",
    "横山武": "横山武史",
    "横山和": "横山和生",
    "横山典": "横山典弘",
    "鮫島駿": "鮫島克駿",
    "鮫島克": "鮫島克駿",
    "佐々木": "佐々木大輔",
    "佐々木大": "佐々木大輔",
    "松山": "松山弘平",
    "坂井": "坂井瑠星",
    "武豊": "武豊",
    "ルメール": "ルメール",
    "Ｃ．ルメール": "ルメール",
    "C.ルメール": "ルメール",
    "Ｍ．デム": "Ｍ．デムーロ",
    "M.デム": "Ｍ．デムーロ",
    "Ｍデムーロ": "Ｍ．デムーロ",
    "戸崎": "戸崎圭太",
    "川田": "川田将雅",
    "丹内": "丹内祐次",
    "池添": "池添謙一",
    "浜中": "浜中俊",
    "藤岡佑": "藤岡佑介",
    "田口": "田口貫太",
    "高杉": "高杉吏麒",
    "吉村": "吉村誠之助",
    "吉村誠": "吉村誠之助",
    "小沢": "小沢大仁",
    "斎藤": "斎藤新",
    "富田": "富田暁",
    "古川奈": "古川奈穂",
    "小林勝": "小林勝太",
    "小林凌": "小林凌大",
    "角田河": "角田大河",
    "角田和": "角田大和",
    "団野": "団野大成",
    "西村淳": "西村淳也",
    "菅原明": "菅原明良",
    "津村": "津村明秀",
    "三浦": "三浦皇成",
    "内田博": "内田博幸",
    "菱田": "菱田裕二",
    "幸": "幸英明",
    "和田竜": "和田竜二",
}


def _norm_text_value(x) -> str:
    s = str(x).strip()
    if s in ["nan", "None", "<NA>"]:
        return ""
    return (
        s.replace("　", "")
         .replace(" ", "")
         .replace("・", "")
         .replace("．", ".")
         .strip()
    )


def _norm_jockey_value(x) -> str:
    s = _norm_text_value(x)
    if not s:
        return ""
    # 外国人騎手の表記を少し寄せる
    s = s.replace("Ｃ.", "C.").replace("Ｍ.", "M.")
    if s in JOCKEY_ALIAS_MAP:
        return JOCKEY_ALIAS_MAP[s]
    # 既にフル表記のものも、短縮表記側に寄せたい場合があるため逆引きしやすい形にする
    return s


def normalize_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    """TARGET過去CSVと出馬表CSVの結合キーを正規化する。"""
    df = df.copy()

    if "horse_name" in df.columns:
        df["horse_name"] = df["horse_name"].apply(_norm_text_value)

    if "jockey" in df.columns:
        df["jockey"] = df["jockey"].apply(_norm_jockey_value)

    if "place" in df.columns:
        df["place"] = (
            df["place"].apply(_norm_text_value)
            .str.replace("競馬場", "", regex=False)
        )

    for c in ["distance", "finish", "pass1", "pass2", "pass3", "pass4", "last3f", "odds", "popularity", "horse_no", "frame_no", "field_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def extract_race_id(text: str) -> str:
    text = str(text).strip()
    m = re.search(r"race_id=(\d{12})", text)
    if m:
        return m.group(1)
    m = re.search(r"/race/(\d{12})", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d{12})", text)
    if m:
        return m.group(1)
    return ""


def race_id_to_info(race_id: str) -> dict:
    race_id = str(race_id)
    return {
        "race_id": race_id,
        "year": int(race_id[0:4]),
        "place_code": race_id[4:6],
        "place": PLACE_MAP.get(race_id[4:6], "不明"),
        "kai": int(race_id[6:8]),
        "nichiji": int(race_id[8:10]),
        "race_no": int(race_id[10:12]),
    }


def build_race_ids(year: int, place_name: str, kai: int, nichiji_list: list[int], race_start: int, race_end: int) -> list[str]:
    place_code = PLACE_CODE_MAP[place_name]
    ids = []
    for nichiji in nichiji_list:
        for r in range(race_start, race_end + 1):
            ids.append(f"{year}{place_code}{kai:02d}{nichiji:02d}{r:02d}")
    return ids


def make_netkeiba_url(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


def flatten_html_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in c if str(x) != "nan"]).strip("_")
            for c in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def pick_shutuba_table_from_html(tables):
    for t in tables:
        tmp = flatten_html_columns(t)
        joined = " ".join([str(c) for c in tmp.columns])
        if ("馬名" in joined or "馬番" in joined or "馬 番" in joined) and ("騎手" in joined or "斤量" in joined):
            return tmp
    return None


def fetch_netkeiba_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://race.netkeiba.com/",
    }
    res = requests.get(url, headers=headers, timeout=20)
    res.raise_for_status()
    return res.text


def netkeiba_table_to_52cols(src: pd.DataFrame, race_id: str) -> pd.DataFrame:
    info = race_id_to_info(race_id)
    src = flatten_html_columns(src)

    rename = {}
    for c in src.columns:
        s = str(c)
        if s == "枠" or "枠番" in s or s.endswith("_枠"):
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
        elif "厩舎" in s or "調教師" in s:
            rename[c] = "trainer"
        elif "馬体重" in s:
            rename[c] = "body_weight"

    src = src.rename(columns=rename)

    if "horse_name" not in src.columns:
        raise ValueError(f"馬名列が見つかりません: columns={list(src.columns)}")

    src = src.dropna(subset=["horse_name"], how="all").copy()
    src["horse_name"] = (
        src["horse_name"].astype(str)
        .str.replace("\\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )
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
        row["race_name"] = f"netkeiba_{race_id}"
        row["race_grade"] = "3"
        row["track_type"] = ""
        row["course_kind"] = "0"
        row["distance"] = "0"
        row["going"] = ""
        row["horse_name"] = r.get("horse_name", "")

        sex_age = str(r.get("sex_age", "")).strip()
        if sex_age:
            row["sex"] = sex_age[0]
            m = re.search(r"(\\d+)", sex_age[1:])
            row["age"] = m.group(1) if m else ""

        row["jockey"] = r.get("jockey", "")
        row["carried_weight"] = r.get("carried_weight", "")
        row["field_size"] = len(src)
        row["horse_no"] = r.get("horse_no", i + 1)
        row["frame_no"] = r.get("frame_no", "")
        row["odds"] = r.get("odds", "")
        row["popularity"] = r.get("popularity", "")
        row["trainer"] = r.get("trainer", "")
        row["body_weight"] = r.get("body_weight", "")

        rows.append([row[c] for c in COLS_52])

    out = pd.DataFrame(rows, columns=COLS_52)
    out["source_file"] = f"netkeiba_{race_id}"
    return clean_types(out)


def fetch_netkeiba_race_to_52cols(race_id_or_url: str) -> pd.DataFrame:
    race_id = extract_race_id(race_id_or_url)
    if not race_id:
        raise ValueError("race_idを取得できませんでした。URLまたは12桁race_idを確認してください。")

    url = make_netkeiba_url(race_id)
    html = fetch_netkeiba_html(url)

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        snippet = html[:300].replace("\\n", " ").replace("\\r", " ")
        raise ValueError(f"netkeiba HTML解析失敗: {e} / HTML先頭: {snippet}")

    table = pick_shutuba_table_from_html(tables)
    if table is None:
        raise ValueError("出馬表テーブルが見つかりません。netkeiba側ブロック、またはURL違いの可能性があります。")

    return netkeiba_table_to_52cols(table, race_id)


def fetch_many_netkeiba_to_52cols(race_ids_or_urls: list[str], sleep_sec: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    import time

    frames = []
    errors = []

    for idx, item in enumerate(race_ids_or_urls, start=1):
        rid = extract_race_id(item)
        if not rid:
            errors.append({"入力": item, "エラー": "race_id取得不可"})
            continue

        try:
            df = fetch_netkeiba_race_to_52cols(rid)
            frames.append(df)
        except Exception as e:
            errors.append({"race_id": rid, "エラー": str(e)})

        time.sleep(float(sleep_sec))

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = pd.DataFrame()

    return all_df, pd.DataFrame(errors)


def convert_52_to_simple_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = {
        "source_file": "source_file",
        "place": "競馬場",
        "race_no": "レース番号",
        "frame_no": "枠番",
        "horse_no": "馬番",
        "horse_name": "馬名",
        "sex": "性別",
        "age": "年齢",
        "jockey": "騎手",
        "carried_weight": "斤量",
        "odds": "オッズ",
        "popularity": "人気",
    }
    use = [c for c in cols if c in out.columns]
    return out[use].rename(columns=cols)


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
        row["pass1"] = r.get("pass1", "")
        row["pass2"] = r.get("pass2", "")
        row["pass3"] = r.get("pass3", "")
        row["pass4"] = r.get("pass4", "")
        row["trainer"] = r.get("trainer", "")
        row["sire"] = r.get("sire", "")
        row["dam"] = r.get("dam", "")
        row["broodmare_sire"] = r.get("broodmare_sire", "")
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
    v7:
      - 空ファイルは None 扱い
      - UTF-8-SIG / UTF-8 / cp932 / shift_jis を順に試す
      - ヘッダーあり/なし両対応
      - 10列CSV（馬名,騎手,着順,距離,競馬場,通過1,通過2,通過3,通過4,上がり3F）を内部列名へ変換
    """
    if path is None or not Path(path).exists():
        return None

    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return None
    except Exception:
        return None

    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error = None

    for enc in encodings:
        # まずヘッダーありとして読む
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            if df is not None and not df.empty:
                return normalize_target_history_columns(df)
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            last_error = e

        # 次にヘッダーなしとして読む
        try:
            df = pd.read_csv(path, encoding=enc, header=None, dtype=str)
            if df is None or df.empty:
                continue

            if df.shape[1] >= 10:
                df = df.iloc[:, :10].copy()
                df.columns = [
                    "horse_name", "jockey", "finish", "distance", "place",
                    "pass1", "pass2", "pass3", "pass4", "last3f"
                ]
                return normalize_target_history_columns(df)
        except pd.errors.EmptyDataError:
            return None
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

    # v4: 出馬表側と結合できるように馬名・騎手・競馬場・距離を正規化
    df = normalize_match_keys(df)

    return df



def _judge_running_style_from_pass_values(pass_values, field_size=18):
    vals = []
    for v in pass_values:
        try:
            f = float(v)
            if f > 0:
                vals.append(f)
        except Exception:
            pass
    if not vals:
        return "", ""
    try:
        fs = float(field_size)
        if fs <= 0:
            fs = max(18.0, max(vals))
    except Exception:
        fs = max(18.0, max(vals))
    early = vals[0]
    avg_pos = float(np.mean(vals))
    early_ratio = early / fs
    avg_ratio = avg_pos / fs
    if early <= 1.5 or early_ratio <= 0.12:
        return "逃げ", f"過去通過:序盤{early:.0f}番手"
    if early_ratio <= 0.38 or avg_ratio <= 0.40:
        return "先行", f"過去通過:前目 avg{avg_pos:.1f}"
    if avg_ratio <= 0.70:
        return "差し", f"過去通過:中団 avg{avg_pos:.1f}"
    return "追込", f"過去通過:後方 avg{avg_pos:.1f}"


def _nyanko_norm_text(s):
    return (
        pd.Series(s).astype(str)
        .str.replace("\u3000", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )

def _nyanko_prepare_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    v8:
    予想側に距離・競馬場が無い/0でも補完できるように、
    馬名キー・騎手キーを作る。
    """
    df = df.copy()
    if "horse_name" in df.columns:
        df["horse_name_key"] = _nyanko_norm_text(df["horse_name"])
    else:
        df["horse_name_key"] = ""
    if "jockey" in df.columns:
        df["jockey_key"] = _nyanko_norm_text(df["jockey"])
    else:
        df["jockey_key"] = ""
    if "place" in df.columns:
        df["place_key"] = _nyanko_norm_text(df["place"])
    else:
        df["place_key"] = ""
    if "distance" in df.columns:
        df["distance_key"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int)
    else:
        df["distance_key"] = 0
    return df


def create_target_features(target_df: pd.DataFrame) -> dict:
    """
    TARGET過去CSVから prior 特徴量を作る。
    v8:
      - 馬名/騎手/競馬場の表記ゆれ吸収
      - 距離・競馬場が予想側に無くても、馬名だけで脚質・馬成績を補完
      - 馬名×距離が無理なら馬名だけにフォールバック
    """
    if target_df is None or target_df.empty:
        return {}

    df = target_df.copy()
    df = _nyanko_prepare_match_keys(df)

    if "finish" not in df.columns:
        return {}

    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df[df["finish"].notna() & (df["finish"] > 0)].copy()

    if df.empty:
        return {}

    # 数値化
    for c in ["distance", "distance_key", "pass1", "pass2", "pass3", "pass4", "last3f"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_win"] = df["finish"].eq(1)
    df["is_top3"] = df["finish"].between(1, 3)

    features = {}

    # 騎手
    if "jockey_key" in df.columns:
        features["jockey_stats"] = (
            df.groupby("jockey_key", dropna=False)
            .agg(
                jockey_runs_prior=("finish", "count"),
                jockey_win_rate_prior=("is_win", "mean"),
                jockey_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 調教師（CSVにあれば）
    if "trainer" in df.columns:
        df["trainer_key"] = _nyanko_norm_text(df["trainer"])
        features["trainer_stats"] = (
            df.groupby("trainer_key", dropna=False)
            .agg(
                trainer_runs_prior=("finish", "count"),
                trainer_win_rate_prior=("is_win", "mean"),
                trainer_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 種牡馬（CSVにあれば）
    if "sire" in df.columns:
        df["sire_key"] = _nyanko_norm_text(df["sire"])
        features["sire_stats"] = (
            df.groupby("sire_key", dropna=False)
            .agg(
                sire_runs_prior=("finish", "count"),
                sire_win_rate_prior=("is_win", "mean"),
                sire_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 馬の総合成績（距離/競馬場が無くても使える）
    if "horse_name_key" in df.columns:
        features["horse_stats"] = (
            df.groupby("horse_name_key", dropna=False)
            .agg(
                horse_runs_prior=("finish", "count"),
                horse_win_rate_prior=("is_win", "mean"),
                horse_top3_rate_prior=("is_top3", "mean"),
                horse_last3f_mean=("last3f", "mean"),
                pass1_mean=("pass1", "mean"),
                pass2_mean=("pass2", "mean"),
                pass3_mean=("pass3", "mean"),
                pass4_mean=("pass4", "mean"),
            )
            .reset_index()
        )

    # 馬×距離
    if "horse_name_key" in df.columns and "distance_key" in df.columns:
        features["horse_distance_stats"] = (
            df.groupby(["horse_name_key", "distance_key"], dropna=False)
            .agg(
                horse_distance_runs_prior=("finish", "count"),
                horse_distance_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 馬×競馬場
    if "horse_name_key" in df.columns and "place_key" in df.columns:
        features["horse_track_stats"] = (
            df.groupby(["horse_name_key", "place_key"], dropna=False)
            .agg(
                horse_track_runs_prior=("finish", "count"),
                horse_track_top3_rate_prior=("is_top3", "mean"),
            )
            .reset_index()
        )

    # 馬ごとの調教師・血統プロファイル
    # 予想側CSVに trainer/sire が無い場合でも、過去CSVの馬名から補完する
    if "horse_name_key" in df.columns:
        profile_cols = ["horse_name_key"]
        if "trainer" in df.columns:
            df["trainer_key"] = _nyanko_norm_text(df["trainer"])
            profile_cols.append("trainer_key")
        if "sire" in df.columns:
            df["sire_key"] = _nyanko_norm_text(df["sire"])
            profile_cols.append("sire_key")

        if len(profile_cols) > 1:
            profiles = []
            for horse_key, g in df.groupby("horse_name_key", dropna=False):
                row = {"horse_name_key": horse_key}
                if "trainer_key" in g.columns:
                    vc = g["trainer_key"].dropna()
                    vc = vc[vc.astype(str).str.len() > 0]
                    row["trainer_key"] = vc.mode().iloc[0] if not vc.empty else ""
                if "sire_key" in g.columns:
                    vc = g["sire_key"].dropna()
                    vc = vc[vc.astype(str).str.len() > 0]
                    row["sire_key"] = vc.mode().iloc[0] if not vc.empty else ""
                profiles.append(row)
            features["horse_profile"] = pd.DataFrame(profiles)

    # v10: 馬名だけで調教師実績・血統実績を確実に補完するための直引きテーブル
    # 予想側CSVに trainer/sire が無くても、過去CSVの同一馬から代表trainer/sireを拾い、
    # そのtrainer/sire全体の3着内率を表示用に入れる。
    if "horse_name_key" in df.columns:
        horse_direct_rows = []

        trainer_rate_map = {}
        if "trainer_key" in df.columns:
            trainer_tmp = (
                df.groupby("trainer_key", dropna=False)
                .agg(trainer_top3_rate_direct=("is_top3", "mean"))
                .reset_index()
            )
            trainer_rate_map = dict(zip(trainer_tmp["trainer_key"], trainer_tmp["trainer_top3_rate_direct"]))

        sire_rate_map = {}
        if "sire_key" in df.columns:
            sire_tmp = (
                df.groupby("sire_key", dropna=False)
                .agg(sire_top3_rate_direct=("is_top3", "mean"))
                .reset_index()
            )
            sire_rate_map = dict(zip(sire_tmp["sire_key"], sire_tmp["sire_top3_rate_direct"]))

        for horse_key, g in df.groupby("horse_name_key", dropna=False):
            row = {"horse_name_key": horse_key}

            if "trainer_key" in g.columns:
                vc = g["trainer_key"].dropna()
                vc = vc[vc.astype(str).str.len() > 0]
                trainer_key = vc.mode().iloc[0] if not vc.empty else ""
                row["trainer_key_direct"] = trainer_key
                row["trainer_top3_rate_prior_direct"] = trainer_rate_map.get(trainer_key, np.nan)

            if "sire_key" in g.columns:
                vc = g["sire_key"].dropna()
                vc = vc[vc.astype(str).str.len() > 0]
                sire_key = vc.mode().iloc[0] if not vc.empty else ""
                row["sire_key_direct"] = sire_key
                row["sire_top3_rate_prior_direct"] = sire_rate_map.get(sire_key, np.nan)

            horse_direct_rows.append(row)

        if horse_direct_rows:
            features["horse_direct_profile_stats"] = pd.DataFrame(horse_direct_rows)

    # 脚質補完用：馬ごとの平均通過順
    if "horse_name_key" in df.columns:
        style_cols = [c for c in ["pass1", "pass2", "pass3", "pass4"] if c in df.columns]
        if style_cols:
            features["horse_style_stats"] = (
                df.groupby("horse_name_key", dropna=False)[style_cols]
                .mean()
                .reset_index()
            )

    return features


def load_target_features_cached():
    """
    CSVを入れ替えても古いキャッシュを使わない。
    """
    path = find_target_csv_path()
    if path is None:
        return None, {}
    target_df = read_target_history_csv(path)
    if target_df is None or target_df.empty:
        return None, {}
    return target_df, create_target_features(target_df)


def merge_target_features(entry_df: pd.DataFrame) -> pd.DataFrame:
    """
    当日出馬表にTARGET過去CSV由来の特徴量を付与する。
    v8:
      - 予想CSV側にdistance/placeが無くても horse_name_key で補完
      - 騎手実績は jockey_key で補完
      - 距離適性がマッチしない場合は horse_top3_rate_prior を代用
      - 通過順が無ければ過去平均通過順を入れて脚質判定可能にする
    """
    df = entry_df.copy()
    df = _nyanko_prepare_match_keys(df)

    if find_target_csv_path() is None:
        return df

    target_df, features = load_target_features_cached()
    if not features:
        return df

    # 予想側に trainer/sire が無い場合、馬名から過去CSVの代表 trainer/sire を補完
    if "horse_profile" in features and "horse_name_key" in df.columns:
        df = df.merge(features["horse_profile"], on="horse_name_key", how="left", suffixes=("", "_profile"))

        if "trainer_key_profile" in df.columns:
            if "trainer_key" not in df.columns:
                df["trainer_key"] = df["trainer_key_profile"]
            else:
                df["trainer_key"] = df["trainer_key"].where(
                    df["trainer_key"].astype(str).str.len() > 0,
                    df["trainer_key_profile"]
                )
            df = df.drop(columns=["trainer_key_profile"])

        if "sire_key_profile" in df.columns:
            if "sire_key" not in df.columns:
                df["sire_key"] = df["sire_key_profile"]
            else:
                df["sire_key"] = df["sire_key"].where(
                    df["sire_key"].astype(str).str.len() > 0,
                    df["sire_key_profile"]
                )
            df = df.drop(columns=["sire_key_profile"])

    # 騎手
    if "jockey_stats" in features and "jockey_key" in df.columns:
        df = df.merge(features["jockey_stats"], on="jockey_key", how="left", suffixes=("", "_target"))

    # 調教師
    if "trainer_stats" in features:
        if "trainer_key" not in df.columns:
            if "trainer" in df.columns:
                df["trainer_key"] = _nyanko_norm_text(df["trainer"])
            else:
                df["trainer_key"] = ""
        df = df.merge(features["trainer_stats"], on="trainer_key", how="left", suffixes=("", "_target"))

    # 血統
    if "sire_stats" in features:
        if "sire_key" not in df.columns:
            if "sire" in df.columns:
                df["sire_key"] = _nyanko_norm_text(df["sire"])
            else:
                df["sire_key"] = ""
        df = df.merge(features["sire_stats"], on="sire_key", how="left", suffixes=("", "_target"))

    # 馬の総合成績
    if "horse_stats" in features and "horse_name_key" in df.columns:
        df = df.merge(features["horse_stats"], on="horse_name_key", how="left", suffixes=("", "_target"))

    # 馬×距離
    if "horse_distance_stats" in features and {"horse_name_key", "distance_key"}.issubset(df.columns):
        df = df.merge(features["horse_distance_stats"], on=["horse_name_key", "distance_key"], how="left", suffixes=("", "_target"))

    # 馬×競馬場
    if "horse_track_stats" in features and {"horse_name_key", "place_key"}.issubset(df.columns):
        df = df.merge(features["horse_track_stats"], on=["horse_name_key", "place_key"], how="left", suffixes=("", "_target"))

    # 脚質補完: 予想側に通過順が無い場合、過去平均通過順を入れる
    if "horse_style_stats" in features and "horse_name_key" in df.columns:
        style_stats = features["horse_style_stats"].rename(columns={
            "pass1": "pass1_hist", "pass2": "pass2_hist", "pass3": "pass3_hist", "pass4": "pass4_hist"
        })
        df = df.merge(style_stats, on="horse_name_key", how="left", suffixes=("", "_style"))
        for c in ["pass1", "pass2", "pass3", "pass4"]:
            hc = f"{c}_hist"
            if hc in df.columns:
                if c not in df.columns:
                    df[c] = df[hc]
                else:
                    cur = pd.to_numeric(df[c], errors="coerce")
                    hist = pd.to_numeric(df[hc], errors="coerce")
                    df[c] = cur.where(cur.notna() & (cur > 0), hist)
                df = df.drop(columns=[hc])

    # _target列を元列へ反映
    for c in [
        "jockey_runs_prior", "jockey_win_rate_prior", "jockey_top3_rate_prior",
        "trainer_runs_prior", "trainer_win_rate_prior", "trainer_top3_rate_prior",
        "sire_runs_prior", "sire_win_rate_prior", "sire_top3_rate_prior",
        "horse_runs_prior", "horse_win_rate_prior", "horse_top3_rate_prior",
        "horse_distance_runs_prior", "horse_distance_top3_rate_prior",
        "horse_track_runs_prior", "horse_track_top3_rate_prior",
        "horse_last3f_mean",
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

    # 距離適性が取れない時は馬の総合3着内率を代用
    if "horse_distance_top3_rate_prior" not in df.columns:
        df["horse_distance_top3_rate_prior"] = np.nan
    if "horse_top3_rate_prior" in df.columns:
        df["horse_distance_top3_rate_prior"] = pd.to_numeric(df["horse_distance_top3_rate_prior"], errors="coerce")
        df["horse_top3_rate_prior"] = pd.to_numeric(df["horse_top3_rate_prior"], errors="coerce")
        df["horse_distance_top3_rate_prior"] = df["horse_distance_top3_rate_prior"].where(
            df["horse_distance_top3_rate_prior"].notna() & (df["horse_distance_top3_rate_prior"] > 0),
            df["horse_top3_rate_prior"]
        )


    # v10: 最終フォールバック
    # ここまでで調教師/血統が未取得なら、馬名直引きテーブルから直接入れる。
    if "horse_direct_profile_stats" in features and "horse_name_key" in df.columns:
        direct = features["horse_direct_profile_stats"]
        df = df.merge(direct, on="horse_name_key", how="left", suffixes=("", "_direct2"))

        if "trainer_top3_rate_prior_direct" in df.columns:
            if "trainer_top3_rate_prior" not in df.columns:
                df["trainer_top3_rate_prior"] = np.nan
            cur = pd.to_numeric(df["trainer_top3_rate_prior"], errors="coerce")
            val = pd.to_numeric(df["trainer_top3_rate_prior_direct"], errors="coerce")
            df["trainer_top3_rate_prior"] = cur.where(cur.notna() & (cur > 0), val)

        if "sire_top3_rate_prior_direct" in df.columns:
            if "sire_top3_rate_prior" not in df.columns:
                df["sire_top3_rate_prior"] = np.nan
            cur = pd.to_numeric(df["sire_top3_rate_prior"], errors="coerce")
            val = pd.to_numeric(df["sire_top3_rate_prior_direct"], errors="coerce")
            df["sire_top3_rate_prior"] = cur.where(cur.notna() & (cur > 0), val)

        # 表示/デバッグ用にキーも補完
        if "trainer_key_direct" in df.columns:
            if "trainer_key" not in df.columns:
                df["trainer_key"] = df["trainer_key_direct"]
            else:
                df["trainer_key"] = df["trainer_key"].where(
                    df["trainer_key"].astype(str).str.len() > 0,
                    df["trainer_key_direct"]
                )

        if "sire_key_direct" in df.columns:
            if "sire_key" not in df.columns:
                df["sire_key"] = df["sire_key_direct"]
            else:
                df["sire_key"] = df["sire_key"].where(
                    df["sire_key"].astype(str).str.len() > 0,
                    df["sire_key_direct"]
                )

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

    # v4: 出馬表側も結合キーを正規化
    df = normalize_match_keys(df)

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
        "脚質": "running_style",
        "脚質メモ": "style_note",
        "通過順1角": "pass1",
        "通過順2角": "pass2",
        "通過順3角": "pass3",
        "通過順4角": "pass4",
        "1角": "pass1",
        "2角": "pass2",
        "3角": "pass3",
        "4角": "pass4",
        "調教師": "trainer",
        "厩舎": "trainer",
        "父馬名": "sire",
        "父": "sire",
        "母馬名": "dam",
        "母": "dam",
        "母の父馬名": "broodmare_sire",
        "母父": "broodmare_sire",
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
    # 簡易CSVに脚質/脚質メモがある場合は、52列外の補助列として保持する
    if "running_style" in src.columns:
        df["running_style"] = src["running_style"].astype(str).values
    if "style_note" in src.columns:
        df["style_note"] = src["style_note"].astype(str).values
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

    # 5位までしか印が無いと、☆以降が馬番昇順のように見えるため、
    # 6〜8位にも補助印を付けてAI順位の連続性を見える化する。
    df["mark"] = df["ml_rank"].map({
        1: "◎", 2: "○", 3: "▲", 4: "△", 5: "☆",
        6: "×", 7: "×", 8: "×"
    }).fillna("")
    df["expected_value"] = df["ml_top3_prob"] * df["odds"].fillna(0)
    df["danger_popular"] = ((df["popularity"].fillna(99) <= 3) & (df["ml_rank"] >= 5)).map({True: "危険", False: ""})
    df["value_horse"] = ((df["popularity"].fillna(0) >= 6) & (df["ml_rank"] <= 4)).map({True: "穴候補", False: ""})

    df = add_value_strategy(df)

    return df


def jp_view(df: pd.DataFrame, include_race_key=False) -> pd.DataFrame:
    """
    予想結果の日本語表示用。
    v3修正:
    - 列は非表示にしない
    - 脚質/実績が入っていればそのまま表示
    - 取得できないものは 0.00% のままではなく「未取得」と表示して原因を明確化
    """
    cols = DISPLAY_COLUMNS.copy()
    if include_race_key:
        cols = ["race_label", "race_key"] + cols

    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    if "running_style" in out.columns:
        out["running_style"] = (
            out["running_style"].astype(str)
            .replace({"nan": "", "None": "", "不明": "未取得", "": "未取得"})
        )
    if "style_note" in out.columns:
        out["style_note"] = (
            out["style_note"].astype(str)
            .replace({"nan": "", "None": "", "通過順なし": "通過順データなし", "": "データなし"})
        )

    if "ml_top3_prob" in out.columns:
        out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    if "expected_value" in out.columns:
        out["expected_value"] = pd.to_numeric(out["expected_value"], errors="coerce").round(2)

    for c in ["jockey_top3_rate_prior", "trainer_top3_rate_prior", "sire_top3_rate_prior", "horse_distance_top3_rate_prior"]:
        if c in out.columns:
            vals = pd.to_numeric(out[c], errors="coerce")
            out[c] = np.where(
                vals.notna() & (vals > 0),
                (vals * 100).round(1).astype(str) + "%",
                "未取得"
            )

    return out.rename(columns=JP_COLUMNS)

def add_running_style(df: pd.DataFrame) -> pd.DataFrame:
    """
    pass1〜pass4から脚質を推定する。
    目安:
      逃げ: 序盤で1〜2番手
      先行: 序盤で前目
      差し: 中団
      追込: 後方
    pass列が無い/空の場合は「未取得」。
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
            existing_style = str(row.get("running_style", "")).strip()
            existing_note = str(row.get("style_note", "")).strip()
            if existing_style and existing_style not in ["nan", "None", "不明", "未取得"]:
                return existing_style, existing_note if existing_note and existing_note not in ["nan", "None"] else "CSV/TARGET補完"
            return "未取得", "通過順データなし"

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

    order = {"逃げ": 1, "先行": 2, "差し": 3, "追込": 4, "未取得": 5, "不明": 6}
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
        out = race_df.sort_values(["ml_rank", "value_score", "horse_no"], ascending=[True, False, True])[view_cols].copy()
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
        "未取得": 0.00,
        "不明": 0.00,
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
    tmp = race_df.copy()
    tmp["_buy_order"] = tmp.get("buy_flag", "").map({"買い": 0, "見送り": 1}).fillna(9)
    out = tmp.sort_values(["_buy_order", "value_score", "ml_top3_prob", "ml_rank"], ascending=[True, False, False, True])[cols].copy()
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



def _ensure_10_rows(rows: list, race_df: pd.DataFrame, bet_type: str, max_count: int = 10) -> list:
    """
    表示名が「各10通り」なのに件数不足になる問題を防ぐ。
    候補が足りない券種は、AI順位・回収率スコア上位から補完して必ず10行にする。
    ただし三連系など物理的に組み合わせが足りない場合も、最後は見送り行で10行に揃える。
    """
    rows = list(rows or [])

    def key_of(x):
        return str(x.get("買い目", ""))

    seen = set()
    clean = []
    for r0 in rows:
        if not isinstance(r0, dict):
            continue
        k = key_of(r0)
        if k and k not in seen:
            clean.append(r0)
            seen.add(k)
    rows = clean

    r = race_df.copy()
    if "value_score" not in r.columns:
        r["value_score"] = 0
    if "ml_top3_prob" not in r.columns:
        r["ml_top3_prob"] = 0
    if "ml_rank" not in r.columns:
        r["ml_rank"] = range(1, len(r) + 1)

    r = r[pd.notna(r.get("horse_no", np.nan))].copy()
    r = r.sort_values(["value_score", "ml_top3_prob", "ml_rank"], ascending=[False, False, True])

    nums = []
    labels = {}
    frames = {}

    for _, row in r.iterrows():
        n = _horse_no(row)
        if not n or n in nums:
            continue
        nums.append(n)
        labels[n] = _horse_label(row)
        try:
            frames[n] = str(int(row.get("frame_no"))) if pd.notna(row.get("frame_no")) else ""
        except Exception:
            frames[n] = ""

    def add(item):
        k = str(item.get("買い目", ""))
        if k and k not in seen and len(rows) < max_count:
            rows.append(item)
            seen.add(k)

    # 単勝・複勝は馬単体なので上位馬で補完
    if bet_type in ["単勝", "複勝"]:
        for n in nums:
            add({"買い目": n, "馬名": labels.get(n, n), "狙い": "AI/回収率上位で補完"})

    # 枠連は枠番がある場合だけ組み合わせ補完
    elif bet_type == "枠連":
        frame_list = []
        for n in nums:
            f = frames.get(n, "")
            if f and f not in frame_list:
                frame_list.append(f)
        for i in range(len(frame_list)):
            for j in range(i, len(frame_list)):
                add({"買い目": f"{frame_list[i]}-{frame_list[j]}", "狙い": "枠連補完"})
                if len(rows) >= max_count:
                    break
            if len(rows) >= max_count:
                break

    # 馬連・ワイド・本命1頭＋穴は2頭組み合わせで補完
    elif bet_type in ["馬連", "ワイド", "本命1頭＋穴"]:
        if nums:
            main = nums[0]
            for n in nums[1:]:
                add({"買い目": f"{main}-{n}", "狙い": "本命軸補完"})
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    add({"買い目": f"{nums[i]}-{nums[j]}", "狙い": "BOX補完"})
                    if len(rows) >= max_count:
                        break
                if len(rows) >= max_count:
                    break

    # 馬単は順序付き2頭
    elif bet_type == "馬単":
        if nums:
            main = nums[0]
            for n in nums[1:]:
                add({"買い目": f"{main}→{n}", "狙い": "本命頭補完"})
            for n in nums[1:]:
                add({"買い目": f"{n}→{main}", "狙い": "相手頭補完"})
            for a in nums:
                for b in nums:
                    if a != b:
                        add({"買い目": f"{a}→{b}", "狙い": "順序補完"})
                    if len(rows) >= max_count:
                        break
                if len(rows) >= max_count:
                    break

    # 三連複・本命2頭＋穴は3頭組み合わせ
    elif bet_type in ["三連複", "本命2頭＋穴"]:
        if len(nums) >= 3:
            h1 = nums[0]
            h2 = nums[1] if len(nums) > 1 else None
            if h2:
                for n in nums[2:]:
                    add({"買い目": f"{h1}-{h2}-{n}", "狙い": "本命2頭軸補完"})
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    for k in range(j + 1, len(nums)):
                        add({"買い目": f"{nums[i]}-{nums[j]}-{nums[k]}", "狙い": "三連複補完"})
                        if len(rows) >= max_count:
                            break
                    if len(rows) >= max_count:
                        break
                if len(rows) >= max_count:
                    break

    # 三連単は順序付き3頭
    elif bet_type == "三連単":
        if len(nums) >= 3:
            firsts = nums[:4]
            seconds = nums[:6]
            thirds = nums[:8]
            for a in firsts:
                for b in seconds:
                    for c in thirds:
                        if len({a, b, c}) == 3:
                            add({"買い目": f"{a}→{b}→{c}", "狙い": "三連単補完"})
                        if len(rows) >= max_count:
                            break
                    if len(rows) >= max_count:
                        break
                if len(rows) >= max_count:
                    break

    # 最後の保険。どうしても足りない場合も10行表示に揃える
    while len(rows) < max_count:
        rows.append({
            "買い目": f"候補不足{len(rows)+1}",
            "狙い": "候補不足。実買いは見送り推奨"
        })

    return rows[:max_count]


def _ensure_combo_dict_10(combos: dict, race_df: pd.DataFrame, max_count: int = 10) -> dict:
    order = ["単勝", "複勝", "馬連", "枠連", "ワイド", "馬単", "三連複", "三連単", "本命2頭＋穴", "本命1頭＋穴"]
    out = dict(combos or {})
    for bet_type in order:
        out[bet_type] = _ensure_10_rows(out.get(bet_type, []), race_df, bet_type, max_count=max_count)
    return out

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
    ai_top = race_df.sort_values(["ml_rank", "value_score", "horse_no"], ascending=[True, False, True]).head(1)
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
    sorted_ai = race_df.sort_values(["ml_rank", "value_score", "horse_no"], ascending=[True, False, True])
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

    return _ensure_combo_dict_10(combos, race_df, max_count=max_count)


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
    st.subheader("回収率重視TAB（必ず各10通り）")
    combos = _ensure_combo_dict_10(generate_roi_bet_combinations(race_df, max_count=10), race_df, max_count=10)
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
    r = race_df.sort_values(["ml_rank", "value_score", "horse_no"], ascending=[True, False, True]).copy()

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

    return _ensure_combo_dict_10(combos, race_df, max_count=max_count)


def show_ticket_tabs(race_df: pd.DataFrame):
    st.subheader("馬券おすすめ（TAB別・必ず各10通り）")

    combos = _ensure_combo_dict_10(generate_bet_combinations(race_df, max_count=10), race_df, max_count=10)
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


def list_preloaded_csv_files() -> list[Path]:
    """
    GitHubリポジトリ内 data/ 配下のCSVを一覧する。
    例:
      data/tokyo_1R.csv
      data/tokyo_2R.csv
      data/tokyo_12R.csv
    """
    if not DATA_DIR.exists():
        return []

    files = sorted(DATA_DIR.glob("*.csv"))
    return [p for p in files if p.is_file()]


def make_preloaded_file_label(path: Path) -> str:
    """
    iPadで見やすい表示名にする。
    """
    name = path.stem

    # tokyo_1R / 東京1R などを見やすくする
    m = re.search(r"(\d{1,2})\s*[RrＲｒ]", name)
    if m:
        return f"{m.group(1)}R：{path.name}"

    return path.name


def load_preloaded_entry_csv(path: Path, csv_mode: str) -> pd.DataFrame:
    """
    data/ 配下に事前配置したCSVを読む。
    TARGET 52列でも簡易CSVでも自動判定する。
    """
    if not path.exists():
        raise ValueError(f"事前CSVが見つかりません: {path}")

    raw = path.read_bytes()

    # まずヘッダーありCSVとして読み、簡易CSVっぽければ簡易CSVへ
    header_df = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            header_df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            break
        except Exception:
            pass

    if header_df is not None:
        cols = set([str(c).strip() for c in header_df.columns])
        simple_markers = {"馬名", "horse_name", "騎手", "jockey", "オッズ", "odds", "人気", "popularity"}
        if len(cols & simple_markers) >= 3:
            return read_simple_csv_to_52(raw, source_name=path.name)

    # TARGET 52列を試す。失敗したら簡易CSVへフォールバック。
    try:
        df0 = read_csv_bytes(raw)
        return normalize_52cols(df0, path.name)
    except Exception as e52:
        try:
            return read_simple_csv_to_52(raw, source_name=path.name)
        except Exception:
            raise e52


def load_many_preloaded_entry_csv(paths: list[Path], csv_mode: str) -> pd.DataFrame:
    """
    data/ 配下の複数CSVをまとめて読む。
    東京1R〜12Rをまとめて予想する用途。
    """
    frames = []
    errors = []

    for p in paths:
        try:
            frames.append(load_preloaded_entry_csv(p, csv_mode))
        except Exception as e:
            errors.append({"ファイル": p.name, "エラー": str(e)})

    if not frames:
        if errors:
            raise ValueError("事前CSVを1件も読めませんでした: " + str(errors[:3]))
        raise ValueError("事前CSVがありません。dataフォルダにCSVを置いてください。")

    df = pd.concat(frames, ignore_index=True)
    if errors:
        st.warning(f"読めなかった事前CSVがあります: {len(errors)}件")
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

    return df



def nyanko_safe_show_prediction_table(pred_df: pd.DataFrame):
    """
    v12安全版:
    重い買い目生成をしない。
    予想結果表だけを確実に表示する。
    """
    if pred_df is None or pred_df.empty:
        st.warning("予想結果が空です。")
        return

    st.markdown("---")
    st.subheader("予想結果")

    show_df = pred_df.copy()

    # 並び順
    sort_cols = []
    if "race_key" in show_df.columns:
        sort_cols.append("race_key")
    if "ml_rank" in show_df.columns:
        sort_cols.append("ml_rank")

    if sort_cols:
        show_df = show_df.sort_values(sort_cols)

    try:
        view = jp_view(show_df, include_race_key=False)
    except Exception:
        view = show_df

    st.dataframe(view, use_container_width=True, hide_index=True)

    try:
        csv_bytes = view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "予想結果CSVをダウンロード",
            data=csv_bytes,
            file_name="nyanko_prediction_result.csv",
            mime="text/csv",
            key="download_prediction_result_v12"
        )
    except Exception as e:
        st.caption(f"CSVダウンロード生成をスキップ: {e}")


def app_main():
    st.title("🐾 にゃんこ競馬AI")

        # v5: TARGET過去CSVの読込状況を画面に出す
    st.caption("iPad / Streamlit Cloud対応版。事前CSV・netkeiba URL・出馬表CSVから発走前予想できます。")

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
        ["事前CSVから選択", "netkeiba一括取得→そのまま予想", "出馬表CSV", "netkeiba URL単発"],
        horizontal=True,
        index=0
    )

    preloaded_paths = []
    selected_preloaded_paths = []
    pred_src_preloaded = None
    uploaded_csv = None
    race_url = ""

    if input_method == "事前CSVから選択":
        st.caption("GitHubの data/ フォルダに置いたCSVを選ぶだけで予想できます。WINS/iPad向け。")
        preloaded_paths = list_preloaded_csv_files()

        if not preloaded_paths:
            st.warning("dataフォルダにCSVがありません。GitHubで data/tokyo_1R.csv 〜 data/tokyo_12R.csv のように置いてください。")
        else:
            labels = [make_preloaded_file_label(p) for p in preloaded_paths]
            mode = st.radio("読み込み方法", ["1レースだけ選ぶ", "全部まとめて読む"], horizontal=True, index=0)

            if mode == "1レースだけ選ぶ":
                selected_label = st.selectbox("事前CSVを選択", labels)
                selected_preloaded_paths = [preloaded_paths[labels.index(selected_label)]]
            else:
                selected_preloaded_paths = preloaded_paths
                st.info(f"dataフォルダ内のCSVを全部読みます: {len(selected_preloaded_paths)}件")

            with st.expander("検出した事前CSV"):
                st.write([p.name for p in preloaded_paths])

    elif input_method == "netkeiba一括取得→そのまま予想":
        st.caption("race_id/URL一覧、または開催情報から一括取得して、そのまま予想できます。取得CSVのダウンロードも可能です。")
        make_mode = st.radio(
            "一括取得方法",
            ["race_id / URL一覧", "開催情報から自動生成"],
            horizontal=True,
            index=0
        )

        race_items = []
        if make_mode == "race_id / URL一覧":
            sample = "202605020111\n202605020112\n202605020113"
            text = st.text_area("race_id または URLを1行ずつ入力", value=sample, height=120)
            race_items = [x.strip() for x in text.splitlines() if x.strip()]
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                year = st.number_input("年", min_value=2020, max_value=2035, value=2026, step=1)
            with c2:
                place_name = st.selectbox("競馬場", list(PLACE_CODE_MAP.keys()), index=list(PLACE_CODE_MAP.keys()).index("東京"))
            with c3:
                kai = st.number_input("開催回", min_value=1, max_value=10, value=2, step=1)
            with c4:
                nichiji_text = st.text_input("日次（カンマ区切り）", value="1,2")

            c5, c6 = st.columns(2)
            with c5:
                race_start = st.number_input("開始R", min_value=1, max_value=12, value=1, step=1)
            with c6:
                race_end = st.number_input("終了R", min_value=1, max_value=12, value=12, step=1)

            nichiji_list = []
            for x in nichiji_text.split(","):
                x = x.strip()
                if x.isdigit():
                    nichiji_list.append(int(x))

            race_items = build_race_ids(int(year), place_name, int(kai), nichiji_list, int(race_start), int(race_end))

        st.write("取得予定レース数:", len(race_items))
        with st.expander("取得予定race_id"):
            st.write([extract_race_id(x) for x in race_items if extract_race_id(x)])

        sleep_sec = st.slider("アクセス間隔（秒）", min_value=0.2, max_value=3.0, value=0.8, step=0.1)

    elif input_method == "出馬表CSV":
        uploaded_csv = st.file_uploader("予想CSVをアップロード", type=["csv"])
        st.caption("TARGET 52列CSV、または簡易CSVを使えます。")

    else:
        race_url = st.text_input(
            "netkeiba 出馬表URL",
            placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=202605020111"
        )
        st.caption("単発URL取得。ブロックされる場合はCSVか一括取得後ダウンロードを使ってください。")


    if input_method == "事前CSVから選択" and not selected_preloaded_paths:
        st.info("dataフォルダにCSVを置くか、事前CSVを選択してください。")
        return

    if input_method == "netkeiba一括取得→そのまま予想" and not race_items:
        st.info("race_id/URLを入力するか、開催情報を指定してください。")
        return

    if input_method == "出馬表CSV" and uploaded_csv is None:
        st.info("出馬表CSVをアップロードしてください。")
        return

    if input_method == "netkeiba URL単発" and not (race_url and race_url.strip()):
        st.info("netkeiba 出馬表URLを入力してください。")
        return

    if st.button("予想する", type="primary"):
        try:
            bundle, model_status = load_model_safely(uploaded_model)
            if bundle is None:
                st.error("学習済みモデルPKLがありません。modelsフォルダに置くか、サイドバーからアップロードしてください。")
                return

            st.success(f"モデル読込: {model_status}")

            if input_method == "事前CSVから選択":
                with st.spinner("事前CSVを読み込み中..."):
                    pred_src = load_many_preloaded_entry_csv(selected_preloaded_paths, csv_mode)
                st.success(f"事前CSVから取得しました: {pred_src['race_key'].nunique()}レース / {len(pred_src)}頭")

                export_simple = convert_52_to_simple_export(pred_src)
                st.download_button(
                    "読み込んだ出馬表CSVをダウンロード",
                    data=export_simple.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="preloaded_entry_races.csv",
                    mime="text/csv",
                )

            elif input_method == "netkeiba一括取得→そのまま予想":
                with st.spinner("netkeibaから出馬表を一括取得中..."):
                    pred_src, fetch_errors = fetch_many_netkeiba_to_52cols(race_items, sleep_sec=sleep_sec)

                if pred_src.empty:
                    st.error("1レースも取得できませんでした。netkeiba側のアクセス制限、またはrace_id違いの可能性があります。")
                    if not fetch_errors.empty:
                        st.dataframe(fetch_errors, use_container_width=True, hide_index=True)
                    return

                st.success(f"netkeibaから取得しました: {pred_src['race_key'].nunique()}レース / {len(pred_src)}頭")

                export_simple = convert_52_to_simple_export(pred_src)
                st.download_button(
                    "取得した出馬表CSVをダウンロード",
                    data=export_simple.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="entry_all_races.csv",
                    mime="text/csv",
                )

                if not fetch_errors.empty:
                    st.warning(f"取得失敗: {len(fetch_errors)}件")
                    st.dataframe(fetch_errors, use_container_width=True, hide_index=True)

            elif input_method == "netkeiba URL単発":
                pred_src = fetch_netkeiba_race_to_52cols(race_url.strip())
                st.success("netkeiba出馬表URLから取得しました。")

                export_simple = convert_52_to_simple_export(pred_src)
                st.download_button(
                    "取得した出馬表CSVをダウンロード",
                    data=export_simple.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="entry_race.csv",
                    mime="text/csv",
                )

            else:
                pred_src = load_uploaded_entry_csv(uploaded_csv, csv_mode)
                st.success("出馬表CSVから取得しました。")

            # TARGET過去CSV（yosou.csv）があれば、騎手・調教師・血統・馬の適性を結合
            pred_src = merge_target_features(pred_src)

            if TARGET_CSV_PATH.exists():
                try:
                    _target_df_check, _features_check = load_target_features_cached()
                    if _features_check:
                        st.success("TARGET過去CSV（yosou.csv）を結合しました。")
                    else:
                        st.info("yosou.csv はありますが、着順が無いため過去補正なしで予想します。")
                except Exception:
                    st.info("yosou.csv はありますが、過去補正に使えないため出馬表単体で予想します。")
            else:
                st.info("TARGET過去CSV（yosou.csv）は未配置です。URL/CSV単体で予想します。")

            pred_df = predict(bundle, pred_src)
            st.success(f"予想完了: {len(pred_df)}頭")
            nyanko_safe_show_prediction_table(pred_df)

            st.subheader("予想結果")
            race_options = (
                pred_df[["race_key", "race_label"]]
                .drop_duplicates()
                .sort_values("race_label")
            )
            label_map = dict(zip(race_options["race_label"], race_options["race_key"]))
            selected_label = st.selectbox("レース選択", list(label_map.keys()))
            selected_race = label_map[selected_label]

            race_df = pred_df[pred_df["race_key"] == selected_race].sort_values(["ml_rank", "value_score", "horse_no"], ascending=[True, False, True])
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
