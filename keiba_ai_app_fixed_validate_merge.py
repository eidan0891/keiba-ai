# nyanko_keiba_ipad_cloud_v23.py
# ------------------------------------------------------------
# にゃんこ競馬AI v23 (三連複改善 + 日付バグ修正版)
#
# 【v23 修正内容】
#
# 【🔴 BUG FIX 1】簡易CSV日付バグ修正
#   - read_simple_csv_to_52() のデフォルト値が year=25, month=4, day=1 に
#     ハードコードされていたため、簡易CSV使用時に date_int=20250401 固定になっていた
#   - 修正: 実行時の現在年月日を動的に取得してデフォルト値に使用
#   - さらに CSV側に年月日列があれば正しくそちらを優先するよう整理
#
# 【🟢 IMPROVEMENT 1】三連複5頭BOX 選出ロジック改善
#   - 旧: EV乖離に過度に依存 → 人気馬が除外されて3着圏を外しやすかった
#   - 新(回収率): ml_top3_prob 重みを増やし、EV乖離は補助に抑制
#         score = prob*0.45 + ev_score*0.25 + value_score*0.15 - rank*0.01 + ev正ボーナス0.05
#   - 新(的中率): prob 重みをさらに強化、人気・AI順位を重視
#         score = prob*0.55 + 人気正規化*0.25 + value_score*0.15 - rank*0.02
#
# 【🟢 IMPROVEMENT 2】三連複ゾーン・相手B選出の安定化
#   - 旧(回収率): EV乖離>0の馬のみ → 候補が0〜1頭になることが多かった
#   - 新(回収率): EV乖離>0を優先しつつ、不足時はml_top3_prob上位で補完
#     → 相手B は必ず3頭以上確保するよう保証
#   - 的中率モードの相手B: 人気3〜6位に絞り込みを微調整（7位を除外）
#
# 【🟢 IMPROVEMENT 3】三連複1頭軸フォーメーション 組み合わせ品質向上
#   - 軸馬を ml_rank<=2 かつ ml_top3_prob が最も高い馬を優先選出に変更
#   - 相手A×相手B の組み合わせ時にスプレッドスコア下限(0.2)を設けて
#     馬番が固まりすぎる組み合わせを排除
#   - 最低6点は生成されるようフォールバック強化
#
# 【🟢 IMPROVEMENT 4】三連複2頭軸フォーメーション 組み合わせ品質向上
#   - 軸2頭が ml_top3_prob 合計で上位2頭になるよう優先選出
#   - 相手候補が不足した場合に safe_df全体からフォールバック補完
#
# 【v21/v22 BUG FIX 引継ぎ】
#   - EV計算式修正 / EVフロア適用 / 穴強制インクルード廃止
# ------------------------------------------------------------

import io
import os
import re
import itertools
from datetime import date
from io import StringIO
from pathlib import Path

import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="にゃんこ競馬AI v23",
    page_icon="🐾",
    layout="wide"
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "nyanko_keiba_top3_model.pkl"
TARGET_CSV_PATH = APP_DIR / "yosou.csv"
DATA_DIR = APP_DIR / "data"

VERSION = "v23 (三連複改善 + 日付バグ修正版)"

# ============================================================
# 定数
# ============================================================

TANSHO_DEDUCTION   = 0.20
FUKUSHO_DEDUCTION  = 0.20
UMAREN_DEDUCTION   = 0.225
WIDE_DEDUCTION     = 0.225
WAKUREN_DEDUCTION  = 0.225
UMATAN_DEDUCTION   = 0.225
SANRENPUKU_DEDUCTION = 0.25
SANRENTAN_DEDUCTION  = 0.25

SANRENPUKU_EV_FLOOR = 0.80
ODDS_TO_TOP3_RATIO = 3.0

# 予想モード定数
STRATEGY_MODE_ROI    = "回収率重視"
STRATEGY_MODE_HITRATE = "的中率重視"
STRATEGY_MODE_OPTIONS = [STRATEGY_MODE_ROI, STRATEGY_MODE_HITRATE]


def find_target_csv_path() -> Path | None:
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
    "ev_score": "EV乖離スコア",
    "implied_top3": "市場暗示3着内確率",
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
    "expected_value", "ev_score", "implied_top3",
    "danger_popular", "value_horse",
    "running_style", "style_note",
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

PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
PLACE_CODE_MAP = {v: k for k, v in PLACE_MAP.items()}

JOCKEY_ALIAS_MAP = {
    "岩田望": "岩田望来", "北村友": "北村友一", "横山武": "横山武史",
    "横山和": "横山和生", "横山典": "横山典弘", "鮫島駿": "鮫島克駿",
    "鮫島克": "鮫島克駿", "佐々木": "佐々木大輔", "佐々木大": "佐々木大輔",
    "松山": "松山弘平", "坂井": "坂井瑠星", "武豊": "武豊",
    "ルメール": "ルメール", "Ｃ．ルメール": "ルメール", "C.ルメール": "ルメール",
    "Ｍ．デム": "Ｍ．デムーロ", "M.デム": "Ｍ．デムーロ", "Ｍデムーロ": "Ｍ．デムーロ",
    "戸崎": "戸崎圭太", "川田": "川田将雅", "丹内": "丹内祐次",
    "池添": "池添謙一", "浜中": "浜中俊", "藤岡佑": "藤岡佑介",
    "田口": "田口貫太", "高杉": "高杉吏麒", "吉村": "吉村誠之助",
    "吉村誠": "吉村誠之助", "小沢": "小沢大仁", "斎藤": "斎藤新",
    "富田": "富田暁", "古川奈": "古川奈穂", "小林勝": "小林勝太",
    "小林凌": "小林凌大", "角田河": "角田大河", "角田和": "角田大和",
    "団野": "団野大成", "西村淳": "西村淳也", "菅原明": "菅原明良",
    "津村": "津村明秀", "三浦": "三浦皇成", "内田博": "内田博幸",
    "菱田": "菱田裕二", "幸": "幸英明", "和田竜": "和田竜二",
}


# ============================================================
# ユーティリティ
# ============================================================

def _norm_text_value(x) -> str:
    s = str(x).strip()
    if s in ["nan", "None", "<NA>"]:
        return ""
    return s.replace("　", "").replace(" ", "").replace("・", "").replace("．", ".").strip()


def _norm_jockey_value(x) -> str:
    s = _norm_text_value(x)
    if not s:
        return ""
    s = s.replace("Ｃ.", "C.").replace("Ｍ.", "M.")
    return JOCKEY_ALIAS_MAP.get(s, s)


def normalize_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "horse_name" in df.columns:
        df["horse_name"] = df["horse_name"].apply(_norm_text_value)
    if "jockey" in df.columns:
        df["jockey"] = df["jockey"].apply(_norm_jockey_value)
    if "place" in df.columns:
        df["place"] = df["place"].apply(_norm_text_value).str.replace("競馬場", "", regex=False)
    for c in ["distance", "finish", "pass1", "pass2", "pass3", "pass4", "last3f",
              "odds", "popularity", "horse_no", "frame_no", "field_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def extract_race_id(text: str) -> str:
    text = str(text).strip()
    for pattern in [r"race_id=(\d{12})", r"/race/(\d{12})", r"(\d{12})"]:
        m = re.search(pattern, text)
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


def make_netkeiba_url(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


def build_race_ids(year: int, place_name: str, kai: int, nichiji_list: list[int],
                   race_start: int, race_end: int) -> list[str]:
    place_code = PLACE_CODE_MAP[place_name]
    return [
        f"{year}{place_code}{kai:02d}{nichiji:02d}{r:02d}"
        for nichiji in nichiji_list
        for r in range(race_start, race_end + 1)
    ]


# ============================================================
# repair_simple_imputer
# ============================================================

def repair_simple_imputer(obj, _seen=None, _depth=0, _max_depth=20):
    if _seen is None:
        _seen = set()
    if _depth > _max_depth:
        return obj
    obj_id = id(obj)
    if obj_id in _seen:
        return obj
    _seen.add(obj_id)
    if obj.__class__.__name__ == "SimpleImputer" and not hasattr(obj, "_fill_dtype"):
        stat = getattr(obj, "statistics_", None)
        try:
            obj._fill_dtype = stat.dtype if stat is not None else np.dtype("float64")
        except Exception:
            obj._fill_dtype = np.dtype("float64")
    for attr in ("steps", "transformers", "transformers_", "estimators", "estimators_"):
        if hasattr(obj, attr):
            try:
                for item in getattr(obj, attr):
                    children = item if not isinstance(item, tuple) else item
                    if isinstance(children, tuple):
                        for v in children:
                            if hasattr(v, "__dict__"):
                                repair_simple_imputer(v, _seen, _depth + 1, _max_depth)
                    elif hasattr(children, "__dict__"):
                        repair_simple_imputer(children, _seen, _depth + 1, _max_depth)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            if hasattr(v, "__dict__"):
                repair_simple_imputer(v, _seen, _depth + 1, _max_depth)
            elif isinstance(v, (list, tuple, set)):
                for i in v:
                    if hasattr(i, "__dict__"):
                        repair_simple_imputer(i, _seen, _depth + 1, _max_depth)
            elif isinstance(v, dict):
                for i in v.values():
                    if hasattr(i, "__dict__"):
                        repair_simple_imputer(i, _seen, _depth + 1, _max_depth)
    return obj


# ============================================================
# netkeiba取得
# ============================================================

def _fetch_netkeiba_html(url: str) -> str:
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


def _flatten_html_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in c if str(x) != "nan"]).strip("_")
            for c in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _pick_shutuba_table(tables) -> pd.DataFrame | None:
    for t in tables:
        tmp = _flatten_html_columns(t)
        joined = " ".join(str(c) for c in tmp.columns)
        if ("馬名" in joined or "馬番" in joined or "馬 番" in joined) and \
           ("騎手" in joined or "斤量" in joined):
            return tmp
    return None


def _rename_shutuba_columns(src: pd.DataFrame) -> pd.DataFrame:
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
    return src.rename(columns=rename)


def _shutuba_table_to_52cols(src: pd.DataFrame, race_id: str) -> pd.DataFrame:
    info = race_id_to_info(race_id)
    src = _flatten_html_columns(src)
    src = _rename_shutuba_columns(src)
    if "horse_name" not in src.columns:
        raise ValueError(f"馬名列が見つかりません: columns={list(src.columns)}")
    src = src.dropna(subset=["horse_name"], how="all").copy()
    src["horse_name"] = (
        src["horse_name"].astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )
    src = src[src["horse_name"].ne("")]
    src = src[~src["horse_name"].str.contains("馬名|出走取消|除外", na=False)]
    rows = []
    for i, r in src.iterrows():
        row = {c: "" for c in COLS_52}
        row.update({
            "year": info["year"] - 2000,
            "month": 1, "day": 1,
            "kai": info.get("kai", 1),
            "place": info.get("place", "不明"),
            "nichiji": info.get("nichiji", 1),
            "race_no": info.get("race_no", 11),
            "race_name": f"netkeiba_{race_id}",
            "race_grade": "3",
            "track_type": "", "course_kind": "0", "distance": "0", "going": "",
            "horse_name": r.get("horse_name", ""),
        })
        sex_age = str(r.get("sex_age", "")).strip()
        if sex_age:
            row["sex"] = sex_age[0]
            m = re.search(r"(\d+)", sex_age[1:])
            row["age"] = m.group(1) if m else ""
        row.update({
            "jockey": r.get("jockey", ""),
            "carried_weight": r.get("carried_weight", ""),
            "field_size": len(src),
            "horse_no": r.get("horse_no", i + 1),
            "frame_no": r.get("frame_no", ""),
            "odds": r.get("odds", ""),
            "popularity": r.get("popularity", ""),
            "trainer": r.get("trainer", ""),
            "body_weight": r.get("body_weight", ""),
            "pass1": r.get("pass1", ""), "pass2": r.get("pass2", ""),
            "pass3": r.get("pass3", ""), "pass4": r.get("pass4", ""),
            "sire": r.get("sire", ""), "dam": r.get("dam", ""),
            "broodmare_sire": r.get("broodmare_sire", ""),
        })
        rows.append([row[c] for c in COLS_52])
    out = pd.DataFrame(rows, columns=COLS_52)
    out["source_file"] = f"netkeiba_{race_id}"
    return clean_types(out)


def fetch_netkeiba_race_to_52cols(race_id_or_url: str) -> pd.DataFrame:
    race_id = extract_race_id(race_id_or_url)
    if not race_id:
        raise ValueError("race_idを取得できませんでした。")
    url = make_netkeiba_url(race_id)
    html = _fetch_netkeiba_html(url)
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        raise ValueError(f"netkeibaの表を解析できませんでした。{e}")
    table = _pick_shutuba_table(tables)
    if table is None:
        raise ValueError("出馬表テーブルが見つかりません。")
    return _shutuba_table_to_52cols(table, race_id)


def fetch_many_netkeiba_to_52cols(race_ids_or_urls: list[str],
                                   sleep_sec: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    import time
    frames, errors = [], []
    for item in race_ids_or_urls:
        rid = extract_race_id(item)
        if not rid:
            errors.append({"入力": item, "エラー": "race_id取得不可"})
            continue
        try:
            frames.append(fetch_netkeiba_race_to_52cols(rid))
        except Exception as e:
            errors.append({"race_id": rid, "エラー": str(e)})
        time.sleep(float(sleep_sec))
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, pd.DataFrame(errors)


def convert_52_to_simple_export(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "source_file": "source_file", "place": "競馬場", "race_no": "レース番号",
        "frame_no": "枠番", "horse_no": "馬番", "horse_name": "馬名",
        "sex": "性別", "age": "年齢", "jockey": "騎手",
        "carried_weight": "斤量", "odds": "オッズ", "popularity": "人気",
    }
    use = [c for c in cols if c in df.columns]
    return df[use].rename(columns=cols)


# ============================================================
# CSV読込
# ============================================================

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
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = pd.to_numeric(df.get("year", 25), errors="coerce").fillna(25)
    df["month"] = pd.to_numeric(df.get("month", 4), errors="coerce").fillna(4)
    df["day"] = pd.to_numeric(df.get("day", 1), errors="coerce").fillna(1)
    df["race_no"] = pd.to_numeric(df.get("race_no", 11), errors="coerce").fillna(11)
    df["year_full"] = df["year"].apply(
        lambda x: int(x) + 2000 if pd.notna(x) and int(x) < 100 else int(x)
    )
    df["date_int"] = (
        df["year_full"].fillna(0).astype(int) * 10000
        + df["month"].fillna(0).astype(int) * 100
        + df["day"].fillna(0).astype(int)
    )
    if "source_file" not in df.columns:
        df["source_file"] = ""
    df = normalize_match_keys(df)
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


# ============================================================
# 【v23 BUG FIX 1】簡易CSV読込 - 日付デフォルト値を動的取得に修正
# ============================================================

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
        "馬名": "horse_name", "性別": "sex", "年齢": "age", "性齢": "sex_age",
        "騎手": "jockey", "斤量": "carried_weight", "オッズ": "odds",
        "単勝オッズ": "odds", "人気": "popularity", "年": "year",
        "月": "month", "日": "day", "競馬場": "place", "場所": "place",
        "レース番号": "race_no", "R": "race_no", "レース名": "race_name",
        "距離": "distance", "馬場": "going", "馬場状態": "going",
        "馬番": "horse_no", "枠番": "frame_no", "頭数": "field_size",
        "芝ダ": "track_type", "脚質": "running_style", "脚質メモ": "style_note",
        "通過順1角": "pass1", "通過順2角": "pass2", "通過順3角": "pass3",
        "通過順4角": "pass4", "1角": "pass1", "2角": "pass2",
        "3角": "pass3", "4角": "pass4",
        "調教師": "trainer", "厩舎": "trainer",
        "父馬名": "sire", "父": "sire", "母馬名": "dam", "母": "dam",
        "母の父馬名": "broodmare_sire", "母父": "broodmare_sire",
        # v23追加: 日付まとめ列対応
        "日付": "race_date_str", "開催日": "race_date_str",
    }
    src = src.rename(columns=rename)

    # v23 BUG FIX: 日付を動的に取得
    today = date.today()
    default_year = str(today.year - 2000)   # 例: 2025 → "25"
    default_month = str(today.month)         # 例: 5
    default_day = str(today.day)             # 例: 10

    # "日付" 列が "20250510" や "2025/05/10" 形式で存在する場合はそこから分解
    if "race_date_str" in src.columns:
        def parse_date_col(s):
            s = str(s).strip().replace("/", "").replace("-", "").replace(".", "")
            if len(s) == 8 and s.isdigit():
                return s[2:4], s[4:6], s[6:8]  # YY, MM, DD
            return None, None, None
        parsed = src["race_date_str"].apply(parse_date_col)
        if "year" not in src.columns:
            src["year"] = [p[0] if p[0] else default_year for p in parsed]
        if "month" not in src.columns:
            src["month"] = [p[1] if p[1] else default_month for p in parsed]
        if "day" not in src.columns:
            src["day"] = [p[2] if p[2] else default_day for p in parsed]

    if "sex_age" in src.columns:
        if "sex" not in src.columns:
            src["sex"] = src["sex_age"].astype(str).str[0]
        if "age" not in src.columns:
            src["age"] = src["sex_age"].astype(str).str[1:].str.extract(r"(\d+)")[0]

    required = ["horse_name", "jockey", "carried_weight", "odds", "popularity"]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"簡易CSVの必須列が不足しています: {missing}")

    for c in ["sex", "age"]:
        if c not in src.columns:
            src[c] = ""

    rows = []
    for i, r in src.iterrows():
        row = {c: "" for c in COLS_52}
        row.update({
            # v23 BUG FIX: ハードコード値をデフォルト変数で置き換え
            "year":     r.get("year",     default_year),
            "month":    r.get("month",    default_month),
            "day":      r.get("day",      default_day),
            "kai": "1",
            "place":    r.get("place",    "東京"),
            "nichiji": "1",
            "race_no":  r.get("race_no",  "11"),
            "race_name": r.get("race_name", "未設定"),
            "race_grade": "3",
            "track_type": r.get("track_type", "芝"),
            "course_kind": "0",
            "distance": r.get("distance", "2000"),
            "going":    r.get("going",    "良"),
            "horse_name": r.get("horse_name", ""),
            "sex":       r.get("sex", ""),
            "age":       r.get("age", ""),
            "jockey":    r.get("jockey", ""),
            "carried_weight": r.get("carried_weight", ""),
            "field_size": r.get("field_size", str(len(src))),
            "horse_no":  r.get("horse_no", str(i + 1)),
            "odds":      r.get("odds", ""),
            "popularity": r.get("popularity", ""),
        })
        rows.append([row[c] for c in COLS_52])

    df = pd.DataFrame(rows, columns=COLS_52)
    df["source_file"] = source_name
    if "running_style" in src.columns:
        df["running_style"] = src["running_style"].astype(str).values
    if "style_note" in src.columns:
        df["style_note"] = src["style_note"].astype(str).values
    return clean_types(df)


def load_uploaded_entry_csv(uploaded_csv, csv_mode: str) -> pd.DataFrame:
    raw = uploaded_csv.read()
    header_df = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            header_df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            break
        except Exception:
            pass
    if header_df is not None:
        cols = set(str(c).strip() for c in header_df.columns)
        simple_markers = {"馬名", "horse_name", "騎手", "jockey", "オッズ", "odds", "人気", "popularity"}
        if len(cols & simple_markers) >= 3:
            return read_simple_csv_to_52(raw)
    if csv_mode == "52列TARGET形式":
        try:
            df0 = read_csv_bytes(raw)
            return normalize_52cols(df0, uploaded_csv.name)
        except Exception as e:
            try:
                return read_simple_csv_to_52(raw)
            except Exception:
                raise e
    return read_simple_csv_to_52(raw)


# ============================================================
# TARGET過去CSV
# ============================================================

def normalize_target_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "年": "year", "月": "month", "日": "day", "日付": "race_date",
        "回次": "kai", "場所": "place", "競馬場": "place", "日次": "nichiji",
        "レース番号": "race_no", "R": "race_no", "レース名": "race_name",
        "クラスコード": "race_grade", "芝・ダ": "track_type",
        "トラックコード": "track_type", "コース区分": "course_kind",
        "距離": "distance", "馬場状態": "going", "馬名": "horse_name",
        "性別": "sex", "性": "sex", "年齢": "age", "騎手": "jockey",
        "斤量": "carried_weight", "頭数": "field_size", "馬番": "horse_no",
        "枠番": "frame_no", "確定着順": "finish", "着順": "finish",
        "入線着順": "finish_raw", "単勝オッズ": "odds", "オッズ": "odds",
        "人気": "popularity", "走破タイム(秒)": "time_sec", "タイム": "time_raw",
        "通過順1角": "pass1", "通過順2角": "pass2",
        "通過順3角": "pass3", "通過順4角": "pass4",
        "上り3Fタイム": "last3f", "上がり3Fタイム": "last3f",
        "上り3F順位": "last3f_rank", "馬体重": "body_weight",
        "増減": "body_weight_diff", "調教師": "trainer", "所属": "belonging",
        "賞金": "prize", "騎手コード": "jockey_id", "調教師コード": "trainer_id",
        "血統登録番号": "horse_id", "父馬名": "sire", "母馬名": "dam",
        "母の父馬名": "broodmare_sire", "毛色": "coat_color", "生年月日": "birthdate",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    for c in ["year", "month", "day", "race_no", "race_grade", "course_kind",
              "distance", "age", "carried_weight", "field_size", "horse_no",
              "frame_no", "finish", "odds", "popularity", "pass1", "pass2",
              "pass3", "pass4", "last3f", "body_weight", "prize"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["horse_name", "jockey", "trainer", "sire", "dam", "broodmare_sire",
              "place", "track_type", "going", "sex", "belonging"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    df = normalize_match_keys(df)
    return df


def read_target_history_csv(path: Path) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return None
    except Exception:
        return None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            if df is not None and not df.empty:
                return normalize_target_history_columns(df)
        except pd.errors.EmptyDataError:
            return None
        except Exception:
            pass
    raise ValueError(f"TARGET過去CSVを読めませんでした: {path}")


# ============================================================
# TARGET特徴量
# ============================================================

def _nyanko_norm_text(s):
    return (
        pd.Series(s).astype(str)
        .str.replace("\u3000", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def _nyanko_prepare_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["horse_name_key"] = _nyanko_norm_text(df["horse_name"]) if "horse_name" in df.columns else ""
    df["jockey_key"] = _nyanko_norm_text(df["jockey"]) if "jockey" in df.columns else ""
    df["place_key"] = _nyanko_norm_text(df["place"]) if "place" in df.columns else ""
    df["distance_key"] = (
        pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int)
        if "distance" in df.columns else 0
    )
    return df


def create_target_features(target_df: pd.DataFrame) -> dict:
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
    for c in ["distance", "distance_key", "pass1", "pass2", "pass3", "pass4", "last3f"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_win"] = df["finish"].eq(1)
    df["is_top3"] = df["finish"].between(1, 3)
    features = {}
    if "jockey_key" in df.columns:
        features["jockey_stats"] = (
            df.groupby("jockey_key", dropna=False)
            .agg(jockey_runs_prior=("finish", "count"),
                 jockey_win_rate_prior=("is_win", "mean"),
                 jockey_top3_rate_prior=("is_top3", "mean"))
            .reset_index()
        )
    if "trainer" in df.columns:
        df["trainer_key"] = _nyanko_norm_text(df["trainer"])
        features["trainer_stats"] = (
            df.groupby("trainer_key", dropna=False)
            .agg(trainer_runs_prior=("finish", "count"),
                 trainer_win_rate_prior=("is_win", "mean"),
                 trainer_top3_rate_prior=("is_top3", "mean"))
            .reset_index()
        )
    if "sire" in df.columns:
        df["sire_key"] = _nyanko_norm_text(df["sire"])
        features["sire_stats"] = (
            df.groupby("sire_key", dropna=False)
            .agg(sire_runs_prior=("finish", "count"),
                 sire_win_rate_prior=("is_win", "mean"),
                 sire_top3_rate_prior=("is_top3", "mean"))
            .reset_index()
        )
    if "horse_name_key" in df.columns:
        features["horse_stats"] = (
            df.groupby("horse_name_key", dropna=False)
            .agg(horse_runs_prior=("finish", "count"),
                 horse_win_rate_prior=("is_win", "mean"),
                 horse_top3_rate_prior=("is_top3", "mean"),
                 horse_last3f_mean=("last3f", "mean"),
                 pass1_mean=("pass1", "mean"), pass2_mean=("pass2", "mean"),
                 pass3_mean=("pass3", "mean"), pass4_mean=("pass4", "mean"))
            .reset_index()
        )
    if "horse_name_key" in df.columns and "distance_key" in df.columns:
        features["horse_distance_stats"] = (
            df.groupby(["horse_name_key", "distance_key"], dropna=False)
            .agg(horse_distance_runs_prior=("finish", "count"),
                 horse_distance_top3_rate_prior=("is_top3", "mean"))
            .reset_index()
        )
    if "horse_name_key" in df.columns and "place_key" in df.columns:
        features["horse_track_stats"] = (
            df.groupby(["horse_name_key", "place_key"], dropna=False)
            .agg(horse_track_runs_prior=("finish", "count"),
                 horse_track_top3_rate_prior=("is_top3", "mean"))
            .reset_index()
        )
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
                for key_col in ["trainer_key", "sire_key"]:
                    if key_col in g.columns:
                        vc = g[key_col].dropna()
                        vc = vc[vc.astype(str).str.len() > 0]
                        row[key_col] = vc.mode().iloc[0] if not vc.empty else ""
                profiles.append(row)
            features["horse_profile"] = pd.DataFrame(profiles)
    if "horse_name_key" in df.columns:
        trainer_rate_map, sire_rate_map = {}, {}
        if "trainer_key" in df.columns:
            tmp = df.groupby("trainer_key", dropna=False).agg(
                trainer_top3_rate_direct=("is_top3", "mean")).reset_index()
            trainer_rate_map = dict(zip(tmp["trainer_key"], tmp["trainer_top3_rate_direct"]))
        if "sire_key" in df.columns:
            tmp = df.groupby("sire_key", dropna=False).agg(
                sire_top3_rate_direct=("is_top3", "mean")).reset_index()
            sire_rate_map = dict(zip(tmp["sire_key"], tmp["sire_top3_rate_direct"]))
        horse_direct_rows = []
        for horse_key, g in df.groupby("horse_name_key", dropna=False):
            row = {"horse_name_key": horse_key}
            if "trainer_key" in g.columns:
                vc = g["trainer_key"].dropna()
                vc = vc[vc.astype(str).str.len() > 0]
                tk = vc.mode().iloc[0] if not vc.empty else ""
                row["trainer_key_direct"] = tk
                row["trainer_top3_rate_prior_direct"] = trainer_rate_map.get(tk, np.nan)
            if "sire_key" in g.columns:
                vc = g["sire_key"].dropna()
                vc = vc[vc.astype(str).str.len() > 0]
                sk = vc.mode().iloc[0] if not vc.empty else ""
                row["sire_key_direct"] = sk
                row["sire_top3_rate_prior_direct"] = sire_rate_map.get(sk, np.nan)
            horse_direct_rows.append(row)
        if horse_direct_rows:
            features["horse_direct_profile_stats"] = pd.DataFrame(horse_direct_rows)
    if "horse_name_key" in df.columns:
        style_cols = [c for c in ["pass1", "pass2", "pass3", "pass4"] if c in df.columns]
        if style_cols:
            features["horse_style_stats"] = (
                df.groupby("horse_name_key", dropna=False)[style_cols].mean().reset_index()
            )
    return features


@st.cache_data
def load_target_features_cached():
    path = find_target_csv_path()
    if path is None:
        return None, {}
    target_df = read_target_history_csv(path)
    if target_df is None or target_df.empty:
        return None, {}
    return target_df, create_target_features(target_df)


def merge_target_features(entry_df: pd.DataFrame) -> pd.DataFrame:
    df = entry_df.copy()
    df = _nyanko_prepare_match_keys(df)
    if find_target_csv_path() is None:
        return df
    target_df, features = load_target_features_cached()
    if not features:
        return df
    if "horse_profile" in features and "horse_name_key" in df.columns:
        df = df.merge(features["horse_profile"], on="horse_name_key", how="left", suffixes=("", "_profile"))
        for key_col in ["trainer_key", "sire_key"]:
            prof_col = f"{key_col}_profile"
            if prof_col in df.columns:
                if key_col not in df.columns:
                    df[key_col] = df[prof_col]
                else:
                    df[key_col] = df[key_col].where(df[key_col].astype(str).str.len() > 0, df[prof_col])
                df = df.drop(columns=[prof_col])
    if "jockey_stats" in features and "jockey_key" in df.columns:
        df = df.merge(features["jockey_stats"], on="jockey_key", how="left", suffixes=("", "_target"))
    if "trainer_stats" in features:
        if "trainer_key" not in df.columns:
            df["trainer_key"] = _nyanko_norm_text(df["trainer"]) if "trainer" in df.columns else ""
        df = df.merge(features["trainer_stats"], on="trainer_key", how="left", suffixes=("", "_target"))
    if "sire_stats" in features:
        if "sire_key" not in df.columns:
            df["sire_key"] = _nyanko_norm_text(df["sire"]) if "sire" in df.columns else ""
        df = df.merge(features["sire_stats"], on="sire_key", how="left", suffixes=("", "_target"))
    if "horse_stats" in features and "horse_name_key" in df.columns:
        df = df.merge(features["horse_stats"], on="horse_name_key", how="left", suffixes=("", "_target"))
    if "horse_distance_stats" in features and {"horse_name_key", "distance_key"}.issubset(df.columns):
        df = df.merge(features["horse_distance_stats"], on=["horse_name_key", "distance_key"],
                      how="left", suffixes=("", "_target"))
    if "horse_track_stats" in features and {"horse_name_key", "place_key"}.issubset(df.columns):
        df = df.merge(features["horse_track_stats"], on=["horse_name_key", "place_key"],
                      how="left", suffixes=("", "_target"))
    if "horse_style_stats" in features and "horse_name_key" in df.columns:
        style_stats = features["horse_style_stats"].rename(columns={
            "pass1": "pass1_hist", "pass2": "pass2_hist",
            "pass3": "pass3_hist", "pass4": "pass4_hist"
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
    prior_cols = [
        "jockey_runs_prior", "jockey_win_rate_prior", "jockey_top3_rate_prior",
        "trainer_runs_prior", "trainer_win_rate_prior", "trainer_top3_rate_prior",
        "sire_runs_prior", "sire_win_rate_prior", "sire_top3_rate_prior",
        "horse_runs_prior", "horse_win_rate_prior", "horse_top3_rate_prior",
        "horse_distance_runs_prior", "horse_distance_top3_rate_prior",
        "horse_track_runs_prior", "horse_track_top3_rate_prior",
        "horse_last3f_mean",
    ]
    for c in prior_cols:
        tc = f"{c}_target"
        if tc in df.columns:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[tc] = pd.to_numeric(df[tc], errors="coerce")
                df[c] = df[c].where(df[c].notna() & (df[c] != 0), df[tc])
            else:
                df[c] = df[tc]
            df = df.drop(columns=[tc])
    if "horse_distance_top3_rate_prior" not in df.columns:
        df["horse_distance_top3_rate_prior"] = np.nan
    if "horse_top3_rate_prior" in df.columns:
        df["horse_distance_top3_rate_prior"] = pd.to_numeric(
            df["horse_distance_top3_rate_prior"], errors="coerce")
        df["horse_top3_rate_prior"] = pd.to_numeric(df["horse_top3_rate_prior"], errors="coerce")
        df["horse_distance_top3_rate_prior"] = df["horse_distance_top3_rate_prior"].where(
            df["horse_distance_top3_rate_prior"].notna() & (df["horse_distance_top3_rate_prior"] > 0),
            df["horse_top3_rate_prior"]
        )
    if "horse_direct_profile_stats" in features and "horse_name_key" in df.columns:
        direct = features["horse_direct_profile_stats"]
        df = df.merge(direct, on="horse_name_key", how="left", suffixes=("", "_direct2"))
        for rate_col, direct_col, key_col, direct_key_col in [
            ("trainer_top3_rate_prior", "trainer_top3_rate_prior_direct",
             "trainer_key", "trainer_key_direct"),
            ("sire_top3_rate_prior", "sire_top3_rate_prior_direct",
             "sire_key", "sire_key_direct"),
        ]:
            if direct_col in df.columns:
                if rate_col not in df.columns:
                    df[rate_col] = np.nan
                cur = pd.to_numeric(df[rate_col], errors="coerce")
                val = pd.to_numeric(df[direct_col], errors="coerce")
                df[rate_col] = cur.where(cur.notna() & (cur > 0), val)
            if direct_key_col in df.columns:
                if key_col not in df.columns:
                    df[key_col] = df[direct_key_col]
                else:
                    df[key_col] = df[key_col].where(
                        df[key_col].astype(str).str.len() > 0, df[direct_key_col])
    return df


# ============================================================
# 脚質
# ============================================================

def add_running_style(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["pass1", "pass2", "pass3", "pass4", "field_size", "finish"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def judge(row):
        passes = [float(row.get(c, np.nan)) for c in ["pass1", "pass2", "pass3", "pass4"]
                  if pd.notna(row.get(c, np.nan)) and row.get(c, 0) > 0]
        if not passes:
            existing_style = str(row.get("running_style", "")).strip()
            existing_note = str(row.get("style_note", "")).strip()
            if existing_style and existing_style not in ["nan", "None", "不明", "未取得"]:
                note = existing_note if existing_note and existing_note not in ["nan", "None"] else "CSV/TARGET補完"
                return existing_style, note
            return "未取得", "通過順データなし"
        field_size = row.get("field_size", np.nan)
        if pd.isna(field_size) or field_size <= 0:
            field_size = max(18, max(passes))
        early = passes[0]
        avg_pos = float(np.mean(passes))
        early_ratio = early / field_size
        avg_ratio = avg_pos / field_size
        if early <= 1.5 or early_ratio <= 0.12:
            return "逃げ", f"序盤{early:.0f}番手"
        if early_ratio <= 0.38 or avg_ratio <= 0.40:
            return "先行", f"前目 avg{avg_pos:.1f}"
        if avg_ratio <= 0.70:
            return "差し", f"中団 avg{avg_pos:.1f}"
        return "追込", f"後方 avg{avg_pos:.1f}"

    result = df.apply(judge, axis=1)
    df["running_style"] = [x[0] for x in result]
    df["style_note"] = [x[1] for x in result]
    return df


def make_style_summary(df: pd.DataFrame) -> pd.DataFrame:
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
            "脚質": style, "件数": runs, "勝利数": wins, "3着内数": top3,
            "勝率": f"{wins / runs * 100:.1f}%" if runs else "0.0%",
            "3着内率": f"{top3 / runs * 100:.1f}%" if runs else "0.0%",
        })
    order = {"逃げ": 1, "先行": 2, "差し": 3, "追込": 4, "未取得": 5, "不明": 6}
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["脚質", "件数", "勝利数", "3着内数", "勝率", "3着内率"])
    out["_order"] = out["脚質"].map(order).fillna(99)
    return out.sort_values("_order").drop(columns=["_order"])


# ============================================================
# 予想コア
# ============================================================

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


def predict(bundle, df: pd.DataFrame, strategy_mode: str = STRATEGY_MODE_ROI) -> pd.DataFrame:
    df = add_prior_stats_for_prediction(df)
    df = add_running_style(df)
    pipe, feature_cols = get_pipeline_and_features(bundle)
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"特徴量列が不足しています: {missing_features}")
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(df[feature_cols])[:, 1]
    else:
        prob = np.asarray(pipe.predict(df[feature_cols]), dtype=float)
    df["ml_top3_prob"] = prob
    df["ml_rank"] = df.groupby("race_key")["ml_top3_prob"].rank(
        ascending=False, method="first").astype(int)
    df["mark"] = df["ml_rank"].map({
        1: "◎", 2: "○", 3: "▲", 4: "△", 5: "☆", 6: "×", 7: "×", 8: "×"
    }).fillna("")
    df["expected_value"] = df["ml_top3_prob"] * df["odds"].fillna(0)
    df["danger_popular"] = ((df["popularity"].fillna(99) <= 3) & (df["ml_rank"] >= 5)).map(
        {True: "危険", False: ""})
    df["value_horse"] = ((df["popularity"].fillna(0) >= 6) & (df["ml_rank"] <= 4)).map(
        {True: "穴候補", False: ""})
    df = add_ev_score(df)
    df = add_value_strategy(df, strategy_mode=strategy_mode)
    return df


# ============================================================
# EV乖離スコア
# ============================================================

def add_ev_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    odds = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)
    prob = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)
    valid_odds = odds.where(odds > 1.0, np.nan)
    implied_top3_raw = (ODDS_TO_TOP3_RATIO / valid_odds).clip(upper=0.95)
    implied_top3 = implied_top3_raw * (1.0 - FUKUSHO_DEDUCTION)
    df["implied_top3"] = implied_top3.fillna(0).round(4)
    df["ev_score"] = (prob - implied_top3).fillna(0).round(4)
    return df


# ============================================================
# 回収率戦略 - モード別buy_flag判定
# ============================================================

def add_value_strategy(df: pd.DataFrame, strategy_mode: str = STRATEGY_MODE_ROI) -> pd.DataFrame:
    df = df.copy()
    df["odds"] = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df.get("popularity", 99), errors="coerce").fillna(99)
    df["ml_top3_prob"] = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)
    df["expected_value"] = (df["ml_top3_prob"] * df["odds"]).round(3)

    ev_score = pd.to_numeric(df.get("ev_score", 0), errors="coerce").fillna(0)
    jockey_rate = pd.to_numeric(df.get("jockey_top3_rate_prior", 0), errors="coerce").fillna(0)
    trainer_rate = pd.to_numeric(df.get("trainer_top3_rate_prior", 0), errors="coerce").fillna(0)
    sire_rate = pd.to_numeric(df.get("sire_top3_rate_prior", 0), errors="coerce").fillna(0)

    df["jockey_bonus"] = (jockey_rate - 0.25).clip(-0.10, 0.20)
    df["trainer_bonus"] = (trainer_rate - 0.25).clip(-0.04, 0.10)
    df["sire_bonus"] = (sire_rate - 0.25).clip(-0.04, 0.08)

    if strategy_mode == STRATEGY_MODE_ROI:
        df["ev_bonus"] = (ev_score * 1.5).clip(-0.20, 0.30)
        df["ana_bonus"] = np.where((df["popularity"] >= 5) & (df["ml_rank"] <= 5), 0.12, 0.0)
    else:
        df["ev_bonus"] = (ev_score * 0.5).clip(-0.10, 0.10)
        df["ana_bonus"] = 0.0

    style_bonus_map = {"逃げ": 0.03, "先行": 0.02, "差し": 0.00, "追込": -0.02,
                       "未取得": 0.00, "不明": 0.00}
    df["style_bonus"] = df.get("running_style", "不明").map(style_bonus_map).fillna(0)
    df["danger_penalty"] = np.where((df["popularity"] <= 3) & (df["ml_rank"] >= 5), -0.40, 0.0)
    df["fav_weak_penalty"] = np.where(
        (df["popularity"] == 1) & (df["ml_rank"] >= 4), -0.15, 0.0)

    df["value_score"] = (
        df["expected_value"]
        * (1.0
           + df["jockey_bonus"]
           + df["trainer_bonus"]
           + df["sire_bonus"]
           + df["ev_bonus"]
           + df["style_bonus"]
           + df["ana_bonus"]
           + df["danger_penalty"]
           + df["fav_weak_penalty"])
    ).round(3)

    df["_sanrenpuku_ev_approx"] = (
        df["ml_top3_prob"] * df["odds"] * (1.0 - SANRENPUKU_DEDUCTION)
    ).round(3)

    if strategy_mode == STRATEGY_MODE_ROI:
        def judge_roi(row):
            ev = row.get("ev_score", 0)
            pop = row.get("popularity", 99)
            ml_rank = row.get("ml_rank", 99)
            prob = row.get("ml_top3_prob", 0)
            vs = row.get("value_score", 0)
            san_ev = row.get("_sanrenpuku_ev_approx", 0)
            if row.get("danger_popular", "") == "危険":
                return "見送り", "危険人気馬(除外)"
            if san_ev < SANRENPUKU_EV_FLOOR and ml_rank > 2:
                return "見送り", f"三連複EV不足({san_ev:.2f}<{SANRENPUKU_EV_FLOOR})"
            if ml_rank <= 3 and prob >= 0.22:
                return "買い", "AI上位・3着内確率高め"
            if ev >= 0.06 and ml_rank <= 5:
                return "買い", f"市場過小評価(EV+{ev:.3f})"
            if vs >= 1.15 and ml_rank <= 5:
                return "買い", "期待値高め"
            if vs >= 1.00 and pop >= 5 and ml_rank <= 5:
                return "買い", "穴期待"
            return "見送り", "期待値不足"
        judged = df.apply(judge_roi, axis=1)
    else:
        def judge_hitrate(row):
            ml_rank = row.get("ml_rank", 99)
            prob = row.get("ml_top3_prob", 0)
            pop = row.get("popularity", 99)
            vs = row.get("value_score", 0)
            if row.get("danger_popular", "") == "危険":
                return "見送り", "危険人気馬(除外)"
            if ml_rank <= 3:
                return "買い", f"AI上位{ml_rank}位(的中率重視)"
            if ml_rank <= 5 and pop <= 5 and prob >= 0.18:
                return "買い", f"人気{pop}・AI{ml_rank}位(安定圏)"
            if ml_rank <= 6 and vs >= 0.90:
                return "買い", "実績スコア高め(保険)"
            return "見送り", "AI順位・実績不足"
        judged = df.apply(judge_hitrate, axis=1)

    df["buy_flag"] = [x[0] for x in judged]
    df["buy_reason"] = [x[1] for x in judged]
    return df


def get_buy_candidates(race_df: pd.DataFrame, max_horses: int = 8,
                       strategy_mode: str = STRATEGY_MODE_ROI) -> pd.DataFrame:
    r = race_df.sort_values(["value_score", "ml_top3_prob"], ascending=False).copy()
    safe = r[r.get("danger_popular", "") != "危険"] if "danger_popular" in r.columns else r
    buy = safe[safe["buy_flag"] == "買い"].copy() if "buy_flag" in safe.columns else safe.copy()
    if strategy_mode == STRATEGY_MODE_HITRATE:
        buy = buy.sort_values(["ml_rank", "ml_top3_prob"], ascending=[True, False])
        if len(buy) < 3:
            buy = safe.sort_values("ml_rank").head(max(3, min(max_horses, len(safe)))).copy()
    else:
        if len(buy) < 3:
            buy = safe.head(max(3, min(max_horses, len(safe)))).copy()
    return buy.drop_duplicates(subset=["horse_no"]).head(max_horses)


def make_value_summary(race_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["mark", "ml_rank", "horse_no", "horse_name", "running_style",
            "odds", "popularity", "ml_top3_prob", "expected_value",
            "ev_score", "implied_top3", "value_score", "buy_flag", "buy_reason",
            "danger_popular", "value_horse"]
    cols = [c for c in cols if c in race_df.columns]
    tmp = race_df.copy()
    tmp["_buy_order"] = tmp.get("buy_flag", "").map({"買い": 0, "見送り": 1}).fillna(9)
    out = tmp.sort_values(
        ["_buy_order", "value_score", "ml_top3_prob", "ml_rank"],
        ascending=[True, False, False, True]
    )[cols].copy()
    if "ml_top3_prob" in out.columns:
        out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    if "implied_top3" in out.columns:
        out["implied_top3"] = (out["implied_top3"] * 100).round(1).astype(str) + "%"
    return out.rename(columns={**JP_COLUMNS, "value_score": "回収率スコア",
                                "buy_flag": "判定", "buy_reason": "理由"})


def make_tickets(race_df: pd.DataFrame) -> dict:
    r = race_df.sort_values(
        ["ml_rank", "value_score", "horse_no"], ascending=[True, False, True]).copy()

    def horse_label(row):
        try:
            return f"{int(row['horse_no'])} {row['horse_name']}"
        except Exception:
            return str(row.get("horse_name", ""))

    top = r.head(6)
    danger = r[r["danger_popular"] == "危険"] if "danger_popular" in r.columns else r.iloc[0:0]
    value = r[r["value_horse"] == "穴候補"].copy() if "value_horse" in r.columns else r.iloc[0:0]
    if value.empty:
        value = r[(r["popularity"].fillna(0) >= 5) & (r["ml_rank"] <= 8)].copy()

    return {
        "本命": horse_label(top.iloc[0]) if len(top) else "",
        "単勝": horse_label(top.iloc[0]) if len(top) else "",
        "複勝": " / ".join(horse_label(row) for _, row in top.head(3).iterrows()),
        "危険人気馬": " / ".join(horse_label(row) for _, row in danger.iterrows()) or "なし",
        "穴候補": " / ".join(horse_label(row) for _, row in value.head(5).iterrows()) or "なし",
    }


# ============================================================
# 分散スコア計算
# ============================================================

def _calc_spread_score(combo_nums: list[str], all_nums: list[str]) -> float:
    if len(combo_nums) < 2 or not all_nums:
        return 0.5
    try:
        ns = [int(n) for n in combo_nums if str(n).isdigit()]
        an = [int(n) for n in all_nums if str(n).isdigit()]
        if not ns or not an:
            return 0.5
        horse_range = max(an) - min(an)
        if horse_range <= 0:
            return 0.5
        spread = np.std(ns) / (horse_range / 2)
        return float(np.clip(spread, 0, 1))
    except Exception:
        return 0.5


# ============================================================
# 【v23 IMPROVEMENT 2】三連複ゾーンデータ - 相手B安定化
# ============================================================

def build_sanrenpuku_zone_data(race_df: pd.DataFrame,
                                strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    r = race_df.copy()
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity", "ev_score", "horse_no"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    safe = r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy() \
        if "danger_popular" in r.columns else r.copy()
    if safe.empty:
        safe = r.copy()

    all_nums = [str(int(n)) for n in r["horse_no"].dropna() if n > 0]

    def _no(row):
        return str(int(row["horse_no"])) if row["horse_no"] > 0 else ""

    # v23: 軸選出を ml_top3_prob 優先に変更
    if strategy_mode == STRATEGY_MODE_ROI:
        sorted_ai = safe.sort_values(
            ["ml_rank", "ml_top3_prob", "value_score"], ascending=[True, False, False])
    else:
        sorted_ai = safe.sort_values(
            ["ml_rank", "ml_top3_prob", "popularity"], ascending=[True, False, True])

    pivot_row = sorted_ai.iloc[0] if len(sorted_ai) >= 1 else None
    pivot2_row = sorted_ai.iloc[1] if len(sorted_ai) >= 2 else None

    pivot_no = _no(pivot_row) if pivot_row is not None else ""
    pivot2_no = _no(pivot2_row) if pivot2_row is not None else ""

    aite_a_df = sorted_ai[sorted_ai["ml_rank"].between(2, 6)].copy()
    aite_a_nums = [_no(row) for _, row in aite_a_df.iterrows()
                   if _no(row) not in [pivot_no, pivot2_no]][:5]

    if strategy_mode == STRATEGY_MODE_ROI:
        # v23 IMPROVEMENT: EV乖離正の馬を優先しつつ、不足時はml_top3_prob上位で補完
        aite_b_df = safe.sort_values(
            ["ev_score", "ml_top3_prob"], ascending=[False, False]).copy()
        # まずEV正の馬
        aite_b_nums = [_no(row) for _, row in aite_b_df.iterrows()
                       if _no(row) not in [pivot_no, pivot2_no] + aite_a_nums
                       and float(row.get("ev_score", 0)) > 0][:5]
        # 3頭未満の場合は確率上位で補完（EV問わず）
        if len(aite_b_nums) < 3:
            for _, row in safe.sort_values("ml_top3_prob", ascending=False).iterrows():
                n = _no(row)
                if n and n not in [pivot_no, pivot2_no] + aite_a_nums + aite_b_nums:
                    aite_b_nums.append(n)
                if len(aite_b_nums) >= 5:
                    break
    else:
        # v23 IMPROVEMENT: 的中率モードは人気3〜6位に絞り（7位を除外）
        aite_b_df = safe[safe["popularity"].between(3, 6)].sort_values(
            ["ml_top3_prob", "popularity"], ascending=[False, True]).copy()
        aite_b_nums = [_no(row) for _, row in aite_b_df.iterrows()
                       if _no(row) not in [pivot_no, pivot2_no] + aite_a_nums][:5]
        # 不足時はml_top3_prob上位で補完
        if len(aite_b_nums) < 3:
            for _, row in safe.sort_values("ml_top3_prob", ascending=False).iterrows():
                n = _no(row)
                if n and n not in [pivot_no, pivot2_no] + aite_a_nums + aite_b_nums:
                    aite_b_nums.append(n)
                if len(aite_b_nums) >= 5:
                    break

    return {
        "safe_df": safe,
        "all_nums": all_nums,
        "pivot_row": pivot_row,
        "pivot2_row": pivot2_row,
        "pivot_no": pivot_no,
        "pivot2_no": pivot2_no,
        "aite_a_nums": aite_a_nums,
        "aite_b_nums": aite_b_nums,
        "aite_b_df": aite_b_df,
        "sorted_ai": sorted_ai,
        "strategy_mode": strategy_mode,
    }


# ============================================================
# 【v23 IMPROVEMENT 3】三連複1頭軸フォーメーション
# ============================================================

def generate_sanrenpuku_1jiku(zone: dict, max_count: int = 10) -> list[dict]:
    pivot_no = zone["pivot_no"]
    aite_a = zone["aite_a_nums"]
    aite_b = zone["aite_b_nums"]
    all_nums = zone["all_nums"]
    safe_df = zone.get("safe_df", pd.DataFrame())
    if not pivot_no:
        return []

    combos = []
    seen = set()
    SPREAD_MIN = 0.15  # v23: 馬番が固まりすぎる組み合わせを排除

    # 相手A×相手B の組み合わせ
    for ha in aite_a[:6]:
        for hb in aite_b[:7]:
            if ha == hb:
                continue
            tri = tuple(sorted([pivot_no, ha, hb]))
            if len(set(tri)) == 3 and tri not in seen:
                spread = _calc_spread_score(list(tri), all_nums)
                if spread >= SPREAD_MIN or len(combos) < 4:  # 最低4点は品質問わず生成
                    combos.append({
                        "買い目": f"{pivot_no}-{ha}-{hb}",
                        "軸": pivot_no, "相手A": ha, "相手B": hb,
                        "狙い": "1頭軸×中間×EV/安定",
                        "分散スコア": round(spread, 3),
                    })
                    seen.add(tri)
            if len(combos) >= max_count:
                break
        if len(combos) >= max_count:
            break

    # 相手A同士の組み合わせ
    for i in range(len(aite_a)):
        for j in range(i + 1, len(aite_a)):
            tri = tuple(sorted([pivot_no, aite_a[i], aite_a[j]]))
            if len(set(tri)) == 3 and tri not in seen:
                spread = _calc_spread_score(list(tri), all_nums)
                combos.append({
                    "買い目": f"{pivot_no}-{aite_a[i]}-{aite_a[j]}",
                    "軸": pivot_no, "相手A": aite_a[i], "相手B": aite_a[j],
                    "狙い": "1頭軸×相手A×相手A",
                    "分散スコア": round(spread, 3),
                })
                seen.add(tri)
            if len(combos) >= max_count:
                break
        if len(combos) >= max_count:
            break

    # v23 フォールバック: 6点未満なら safe_df全体から補完
    if len(combos) < 6 and not safe_df.empty:
        fallback_nums = [
            str(int(row["horse_no"])) for _, row in
            safe_df.sort_values("ml_top3_prob", ascending=False).iterrows()
            if row["horse_no"] > 0
        ]
        for ha in fallback_nums:
            for hb in fallback_nums:
                if ha >= hb or ha == pivot_no or hb == pivot_no:
                    continue
                tri = tuple(sorted([pivot_no, ha, hb]))
                if len(set(tri)) == 3 and tri not in seen:
                    spread = _calc_spread_score(list(tri), all_nums)
                    combos.append({
                        "買い目": f"{pivot_no}-{ha}-{hb}",
                        "軸": pivot_no, "相手A": ha, "相手B": hb,
                        "狙い": "1頭軸(補完)",
                        "分散スコア": round(spread, 3),
                    })
                    seen.add(tri)
                if len(combos) >= max_count:
                    break
            if len(combos) >= max_count:
                break

    combos.sort(key=lambda x: -x["分散スコア"])
    return combos[:max_count]


# ============================================================
# 【v23 IMPROVEMENT 4】三連複2頭軸フォーメーション
# ============================================================

def generate_sanrenpuku_2jiku(zone: dict, max_count: int = 10) -> list[dict]:
    pivot_no = zone["pivot_no"]
    pivot2_no = zone["pivot2_no"]
    aite_a = zone["aite_a_nums"]
    aite_b = zone["aite_b_nums"]
    all_nums = zone["all_nums"]
    safe_df = zone.get("safe_df", pd.DataFrame())
    if not pivot_no or not pivot2_no:
        return []

    all_aite = list(dict.fromkeys(aite_b + aite_a))  # v23: 相手Bを先に試す

    combos = []
    seen = set()

    for hb in all_aite[:8]:
        if hb in [pivot_no, pivot2_no]:
            continue
        tri = tuple(sorted([pivot_no, pivot2_no, hb]))
        if len(set(tri)) == 3 and tri not in seen:
            spread = _calc_spread_score(list(tri), all_nums)
            tag = "2頭軸×EV高め/安定" if hb in aite_b else "2頭軸×中間"
            combos.append({
                "買い目": f"{pivot_no}-{pivot2_no}-{hb}",
                "軸1": pivot_no, "軸2": pivot2_no, "相手": hb,
                "狙い": tag,
                "分散スコア": round(spread, 3),
            })
            seen.add(tri)
        if len(combos) >= max_count:
            break

    # v23 フォールバック: 候補不足時に safe_df から補完
    if len(combos) < 6 and not safe_df.empty:
        fallback_nums = [
            str(int(row["horse_no"])) for _, row in
            safe_df.sort_values("ml_top3_prob", ascending=False).iterrows()
            if row["horse_no"] > 0
        ]
        for hb in fallback_nums:
            if hb in [pivot_no, pivot2_no]:
                continue
            tri = tuple(sorted([pivot_no, pivot2_no, hb]))
            if len(set(tri)) == 3 and tri not in seen:
                spread = _calc_spread_score(list(tri), all_nums)
                combos.append({
                    "買い目": f"{pivot_no}-{pivot2_no}-{hb}",
                    "軸1": pivot_no, "軸2": pivot2_no, "相手": hb,
                    "狙い": "2頭軸(補完)",
                    "分散スコア": round(spread, 3),
                })
                seen.add(tri)
            if len(combos) >= max_count:
                break

    combos.sort(key=lambda x: -x["分散スコア"])
    return combos[:max_count]


# ============================================================
# 【v23 IMPROVEMENT 1】三連複5頭BOX - 選出スコア改善
# ============================================================

def generate_sanrenpuku_5box(race_df: pd.DataFrame,
                              strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    r = race_df.copy()
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity", "ev_score", "horse_no"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    safe = r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy() \
        if "danger_popular" in r.columns else r.copy()
    if safe.empty:
        safe = r.copy()

    if strategy_mode == STRATEGY_MODE_ROI:
        # v23 IMPROVEMENT: ml_top3_prob の重みを増やし、EV乖離は補助に抑制
        # 旧: ev_score*0.50 + prob*0.35 + value_score*0.10 - rank*0.01 + ev正ボーナス0.05
        # 新: prob*0.45 + ev_score*0.25 + value_score*0.15 - rank*0.01 + ev正ボーナス0.05
        safe["_box_score"] = (
            safe["ml_top3_prob"] * 0.45
            + safe["ev_score"] * 0.25
            + safe["value_score"] * 0.15
            - (safe["ml_rank"] - 1) * 0.01
            + np.where(safe["ev_score"] > 0, 0.05, 0.0)
        )
    else:
        # 的中率: AI確率・人気を重視
        pop_norm = (1.0 / safe["popularity"].clip(lower=1))
        safe["_box_score"] = (
            safe["ml_top3_prob"] * 0.55
            + pop_norm * 0.25
            + safe["value_score"] * 0.15
            - (safe["ml_rank"] - 1) * 0.02
        )

    top5 = safe.sort_values("_box_score", ascending=False).head(5)

    def _label(row):
        try:
            return f"{int(row['horse_no'])} {row['horse_name']}"
        except Exception:
            return str(row.get("horse_name", ""))

    def _no(row):
        try:
            return str(int(row["horse_no"]))
        except Exception:
            return str(row.get("horse_no", ""))

    horses = []
    nums = []
    for _, row in top5.iterrows():
        n = _no(row)
        nums.append(n)
        horses.append({
            "馬番": n,
            "馬名": row.get("horse_name", ""),
            "AI順位": int(row.get("ml_rank", 0)),
            "人気": int(row.get("popularity", 0)),
            "オッズ": round(float(row.get("odds", 0)), 1),
            "3着内確率": f"{float(row.get('ml_top3_prob', 0)) * 100:.1f}%",
            "EV乖離": round(float(row.get("ev_score", 0)), 4),
            "市場暗示3着内": f"{float(row.get('implied_top3', 0)) * 100:.1f}%",
            "BOX選出スコア": round(float(row.get("_box_score", 0)), 3),
            "穴候補": "✓" if row.get("value_horse", "") == "穴候補" else "",
        })

    combos = []
    for tri in itertools.combinations(range(len(nums)), 3):
        tri_nums = [nums[i] for i in tri]
        combos.append({
            "No": len(combos) + 1,
            "買い目": "-".join(sorted(tri_nums, key=lambda x: int(x))),
            "馬番①": tri_nums[0], "馬番②": tri_nums[1], "馬番③": tri_nums[2],
        })

    alt_horse = None
    remaining = safe.sort_values("_box_score", ascending=False).iloc[5:6]
    if not remaining.empty:
        row = remaining.iloc[0]
        alt_horse = {
            "馬番": _no(row), "馬名": row.get("horse_name", ""),
            "AI順位": int(row.get("ml_rank", 0)),
            "人気": int(row.get("popularity", 0)),
            "EV乖離": round(float(row.get("ev_score", 0)), 4),
            "BOX選出スコア": round(float(row.get("_box_score", 0)), 3),
        }

    ana_included = any(h["穴候補"] == "✓" for h in horses)
    ev_positive_count = sum(1 for h in horses if h["EV乖離"] > 0)
    avg_prob = np.mean([float(h["3着内確率"].replace("%", "")) for h in horses])

    if strategy_mode == STRATEGY_MODE_ROI:
        if ana_included:
            selection_note = "✅ [回収率] 穴候補を含むBOX構成です。"
        elif ev_positive_count >= 3:
            selection_note = f"✅ [回収率] EV乖離プラス馬が{ev_positive_count}頭含まれています。"
        else:
            selection_note = (
                f"⚠️ [回収率] EV乖離プラス馬が少ないです。"
                f"平均3着内確率{avg_prob:.1f}%の構成です。"
            )
    else:
        avg_pop = np.mean([h["人気"] for h in horses])
        selection_note = (
            f"✅ [的中率] AI確率・人気重視で{len(horses)}頭選出。"
            f"平均人気{avg_pop:.1f}番人気 / 平均3着内確率{avg_prob:.1f}%の安定構成です。"
        )

    return {
        "horses": horses, "combos": combos, "alt_horse": alt_horse,
        "selection_note": selection_note, "nums": nums,
    }


# ============================================================
# 三連単フォーメーション
# ============================================================

def _generate_sanrentan_formation(race_df: pd.DataFrame, max_count: int = 10,
                                   strategy_mode: str = STRATEGY_MODE_ROI) -> list[dict]:
    r = race_df.copy()
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity", "ev_score", "horse_no"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    all_nums = [str(int(n)) for n in r["horse_no"].dropna() if n > 0]
    safe = r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy() \
        if "danger_popular" in r.columns else r.copy()
    if safe.empty:
        safe = r.copy()

    def get_nums(zone_df, max_n=6, exclude=None):
        nums = []
        for _, row in zone_df.iterrows():
            try:
                n = str(int(row["horse_no"])) if row["horse_no"] > 0 else ""
            except Exception:
                n = ""
            if n and n not in nums and (exclude is None or n not in exclude):
                nums.append(n)
            if len(nums) >= max_n:
                break
        return nums

    sorted_ai = safe.sort_values(["ml_rank", "ev_score"], ascending=[True, False])

    if strategy_mode == STRATEGY_MODE_ROI:
        sorted_ev = safe.sort_values("ev_score", ascending=False)
        first_candidates = get_nums(sorted_ai, max_n=2)
        ev_top = get_nums(sorted_ev, max_n=1)
        if ev_top and ev_top[0] not in first_candidates:
            first_candidates.insert(0, ev_top[0])
        first_candidates = list(dict.fromkeys(first_candidates))[:2]
        ana_zone = safe[(safe["popularity"] >= 5) | (safe.get("value_horse", pd.Series([""] * len(safe))) == "穴候補")]
        second_candidates = get_nums(sorted_ai, max_n=5, exclude=set(first_candidates))
        third_candidates = get_nums(
            ana_zone.sort_values("ev_score", ascending=False),
            max_n=6, exclude=set(first_candidates + second_candidates[:2])
        )
        if len(third_candidates) < 3:
            third_candidates += get_nums(
                sorted_ev, max_n=6, exclude=set(first_candidates + second_candidates[:2]))
        third_candidates = list(dict.fromkeys(third_candidates))
    else:
        first_candidates = get_nums(sorted_ai, max_n=2)
        second_candidates = get_nums(
            safe.sort_values(["popularity", "ml_rank"], ascending=[True, True]),
            max_n=5, exclude=set(first_candidates))
        third_candidates = get_nums(
            safe.sort_values("ml_top3_prob", ascending=False),
            max_n=6, exclude=set(first_candidates + second_candidates[:2]))
        if len(third_candidates) < 3:
            third_candidates += get_nums(
                sorted_ai, max_n=6, exclude=set(first_candidates + second_candidates[:2]))
        third_candidates = list(dict.fromkeys(third_candidates))

    combos = []
    seen = set()
    for h1 in first_candidates:
        for h2 in second_candidates:
            for h3 in third_candidates:
                if len({h1, h2, h3}) == 3:
                    key = f"{h1}→{h2}→{h3}"
                    if key not in seen:
                        spread = _calc_spread_score([h1, h2, h3], all_nums)
                        note = "1着EV×穴3着" if strategy_mode == STRATEGY_MODE_ROI else "AI上位固定"
                        combos.append({"買い目": key, "spread": spread, "狙い": note})
                        seen.add(key)

    combos.sort(key=lambda x: -x["spread"])
    return [{"買い目": c["買い目"], "狙い": c["狙い"],
              "分散スコア": round(c["spread"], 3)} for c in combos[:max_count]]


# ============================================================
# 買い目生成 (その他券種)
# ============================================================

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
        v = row.get("frame_no", np.nan)
        return str(int(v)) if pd.notna(v) else ""
    except Exception:
        return ""


def _ensure_10_rows(rows: list, race_df: pd.DataFrame, bet_type: str,
                    max_count: int = 10) -> list:
    rows = list(rows or [])
    seen = set()
    clean = []
    for r0 in rows:
        if not isinstance(r0, dict):
            continue
        k = str(r0.get("買い目", ""))
        if k and k not in seen:
            clean.append(r0)
            seen.add(k)
    rows = clean
    r = race_df.copy()
    for col, default in [("value_score", 0), ("ml_top3_prob", 0)]:
        if col not in r.columns:
            r[col] = default
    if "ml_rank" not in r.columns:
        r["ml_rank"] = range(1, len(r) + 1)
    r = r[pd.notna(r.get("horse_no", np.nan))].copy()
    r = r.sort_values(["value_score", "ml_top3_prob", "ml_rank"], ascending=[False, False, True])
    nums, labels, frames = [], {}, {}
    for _, row in r.iterrows():
        n = _horse_no(row)
        if not n or n in nums:
            continue
        nums.append(n)
        labels[n] = _horse_label(row)
        frames[n] = _frame_no(row)

    def add(item):
        k = str(item.get("買い目", ""))
        if k and k not in seen and len(rows) < max_count:
            rows.append(item)
            seen.add(k)

    if bet_type in ["単勝", "複勝"]:
        for n in nums:
            add({"買い目": n, "馬名": labels.get(n, n), "狙い": "AI/回収率上位で補完"})
    elif bet_type == "枠連":
        frame_list = list(dict.fromkeys(f for n in nums for f in [frames.get(n, "")] if f))
        for i in range(len(frame_list)):
            for j in range(i, len(frame_list)):
                add({"買い目": f"{frame_list[i]}-{frame_list[j]}", "狙い": "枠連補完"})
                if len(rows) >= max_count:
                    break
            if len(rows) >= max_count:
                break
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
    elif bet_type == "馬単":
        if nums:
            main = nums[0]
            for n in nums[1:]:
                add({"買い目": f"{main}→{n}", "狙い": "本命頭補完"})
                add({"買い目": f"{n}→{main}", "狙い": "相手頭補完"})
    elif bet_type in ["三連複", "本命2頭＋穴"]:
        if len(nums) >= 3:
            h1, h2 = nums[0], nums[1]
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
    elif bet_type == "三連単":
        if len(nums) >= 3:
            for a in nums[:4]:
                for b in nums[:6]:
                    for c in nums[:8]:
                        if len({a, b, c}) == 3:
                            add({"買い目": f"{a}→{b}→{c}", "狙い": "三連単補完"})
                        if len(rows) >= max_count:
                            break
                    if len(rows) >= max_count:
                        break
                if len(rows) >= max_count:
                    break
    while len(rows) < max_count:
        rows.append({"買い目": f"候補不足{len(rows) + 1}", "狙い": "候補不足。実買いは見送り推奨"})
    return rows[:max_count]


def _ensure_combo_dict_10(combos: dict, race_df: pd.DataFrame, max_count: int = 10) -> dict:
    order = ["単勝", "複勝", "馬連", "枠連", "ワイド", "馬単", "三連複", "三連単", "本命2頭＋穴", "本命1頭＋穴"]
    out = dict(combos or {})
    for bet_type in order:
        out[bet_type] = _ensure_10_rows(out.get(bet_type, []), race_df, bet_type, max_count=max_count)
    return out


def generate_roi_bet_combinations(race_df: pd.DataFrame, max_count: int = 10,
                                   strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    r = race_df.sort_values(["value_score", "ml_top3_prob"], ascending=False).copy()
    buy = get_buy_candidates(race_df, max_horses=8, strategy_mode=strategy_mode)

    nums = [_horse_no(row) for _, row in buy.iterrows() if _horse_no(row)]
    if not nums:
        return {}

    ai_top = race_df.sort_values(
        ["ml_rank", "value_score", "horse_no"], ascending=[True, False, True]).head(1)
    value_top = race_df.sort_values("value_score", ascending=False).head(1)
    main = ai_top.iloc[0]

    if strategy_mode == STRATEGY_MODE_ROI:
        if len(value_top) and float(value_top.iloc[0]["value_score"]) > float(main.get("value_score", 0)) * 1.25:
            main = value_top.iloc[0]

    main_no = _horse_no(main)

    if strategy_mode == STRATEGY_MODE_ROI:
        ana = race_df[
            ((race_df["popularity"].fillna(0) >= 5) & (race_df["ml_rank"] <= 7))
            | (race_df.get("value_horse", pd.Series([""] * len(race_df))) == "穴候補")
        ].sort_values("ev_score" if "ev_score" in race_df.columns else "value_score", ascending=False)
        ana_nums = [_horse_no(row) for _, row in ana.head(6).iterrows()
                    if _horse_no(row) != main_no]
    else:
        ana = race_df.sort_values(["popularity", "ml_rank"], ascending=[True, True])
        ana_nums = [_horse_no(row) for _, row in ana.head(8).iterrows()
                    if _horse_no(row) != main_no]

    combos = {}

    combos["単勝"] = [
        {"買い目": _horse_no(row), "馬名": _horse_label(row),
         "回収率スコア": row.get("value_score", 0), "EV乖離": row.get("ev_score", 0),
         "理由": row.get("buy_reason", "")}
        for _, row in pd.concat([ai_top, r]).drop_duplicates(subset=["horse_no"]).head(max_count).iterrows()
    ]

    combos["複勝"] = [
        {"買い目": _horse_no(row), "馬名": _horse_label(row),
         "回収率スコア": row.get("value_score", 0), "EV乖離": row.get("ev_score", 0),
         "理由": row.get("buy_reason", "")}
        for _, row in r.head(max_count).iterrows()
    ]

    others = [n for n in nums if n != main_no]
    umaren, seen_u = [], set()
    for n in others[:max_count]:
        k = f"{main_no}-{n}"
        if k not in seen_u:
            umaren.append({"買い目": k, "狙い": "本命軸×期待値"})
            seen_u.add(k)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            k = f"{nums[i]}-{nums[j]}"
            if k not in seen_u:
                umaren.append({"買い目": k, "狙い": "期待値BOX"})
                seen_u.add(k)
            if len(umaren) >= max_count:
                break
        if len(umaren) >= max_count:
            break
    combos["馬連"] = umaren[:max_count]

    wakuren = []
    if "frame_no" in race_df.columns and race_df["frame_no"].notna().any():
        flist = list(dict.fromkeys(_frame_no(row) for _, row in buy.iterrows() if _frame_no(row)))
        seen_w = set()
        for i in range(len(flist)):
            for j in range(i, len(flist)):
                k = "-".join(sorted([flist[i], flist[j]]))
                if k not in seen_w:
                    wakuren.append({"買い目": k, "狙い": "枠連期待値"})
                    seen_w.add(k)
                if len(wakuren) >= max_count:
                    break
            if len(wakuren) >= max_count:
                break
    combos["枠連"] = wakuren or [{"買い目": "枠番データ不足", "狙い": "CSVに枠番が必要"}]

    wide, seen_w2 = [], set()
    for n in ana_nums + others:
        if n != main_no:
            k = f"{main_no}-{n}"
            if k not in seen_w2:
                wide.append({"買い目": k, "狙い": "本命×穴/期待値"})
                seen_w2.add(k)
        if len(wide) >= max_count:
            break
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            k = f"{nums[i]}-{nums[j]}"
            if k not in seen_w2:
                wide.append({"買い目": k, "狙い": "期待値ワイド"})
                seen_w2.add(k)
            if len(wide) >= max_count:
                break
        if len(wide) >= max_count:
            break
    combos["ワイド"] = wide[:max_count]

    umatan, seen_ut = [], set()
    for n in others[:max_count]:
        k = f"{main_no}→{n}"
        if k not in seen_ut:
            umatan.append({"買い目": k, "狙い": "本命頭固定"})
            seen_ut.add(k)
    if strategy_mode == STRATEGY_MODE_ROI:
        for a in ana_nums[:4]:
            if a != main_no:
                k = f"{a}→{main_no}"
                if k not in seen_ut:
                    umatan.append({"買い目": k, "狙い": "穴頭リターン狙い"})
                    seen_ut.add(k)
            if len(umatan) >= max_count:
                break
    combos["馬単"] = umatan[:max_count]

    zone = build_sanrenpuku_zone_data(race_df, strategy_mode=strategy_mode)
    san1j = generate_sanrenpuku_1jiku(zone, max_count=max_count)
    combos["三連複"] = [{"買い目": c["買い目"], "狙い": c["狙い"],
                          "分散スコア": c["分散スコア"]} for c in san1j] or \
                       [{"買い目": "候補不足", "狙い": "見送り推奨"}]

    combos["三連単"] = _generate_sanrentan_formation(race_df, max_count=max_count,
                                                     strategy_mode=strategy_mode)

    sorted_ai = race_df.sort_values(["ml_rank", "value_score", "horse_no"], ascending=[True, False, True])
    honmei2_ana = []
    if len(sorted_ai) >= 2:
        h1, h2 = _horse_no(sorted_ai.iloc[0]), _horse_no(sorted_ai.iloc[1])
        use_ana = ana_nums or [n for n in nums if n not in [h1, h2]][:5]
        seen_h2 = set()
        for a in use_ana:
            if a not in [h1, h2]:
                k = f"{h1}-{h2}-{a}"
                if k not in seen_h2:
                    honmei2_ana.append({"買い目": k, "狙い": "本命2頭＋穴/安定"})
                    seen_h2.add(k)
            if len(honmei2_ana) >= max_count:
                break
    combos["本命2頭＋穴"] = honmei2_ana or [{"買い目": "穴候補なし", "狙い": "見送り推奨"}]

    honmei1_ana, seen_h1 = [], set()
    use_ana = ana_nums or others[:6]
    for a in use_ana:
        if a != main_no:
            k = f"{main_no}-{a}"
            if k not in seen_h1:
                honmei1_ana.append({"買い目": k, "狙い": "本命1頭＋穴/安定"})
                seen_h1.add(k)
        if len(honmei1_ana) >= max_count:
            break
    combos["本命1頭＋穴"] = honmei1_ana or [{"買い目": "穴候補なし", "狙い": "見送り推奨"}]

    return _ensure_combo_dict_10(combos, race_df, max_count=max_count)


# ============================================================
# 三連複専用タブ表示
# ============================================================

def show_sanrenpuku_tabs(race_df: pd.DataFrame, strategy_mode: str = STRATEGY_MODE_ROI):
    mode_label = "🎯" if strategy_mode == STRATEGY_MODE_ROI else "🏆"
    st.subheader(f"{mode_label} 三連複 詳細フォーメーション＆BOX [{strategy_mode}]")

    zone = build_sanrenpuku_zone_data(race_df, strategy_mode=strategy_mode)

    with st.expander("📋 ゾーン構成（軸・相手・選出根拠）"):
        pivot_row = zone.get("pivot_row")
        pivot2_row = zone.get("pivot2_row")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**軸候補 (最有力)**")
            if pivot_row is not None:
                st.markdown(f"◎ 馬番{zone['pivot_no']} {pivot_row.get('horse_name', '')}")
                st.caption(
                    f"AI{int(pivot_row.get('ml_rank', 0))}位 / "
                    f"3着内確率{float(pivot_row.get('ml_top3_prob', 0)) * 100:.1f}% / "
                    f"EV乖離{float(pivot_row.get('ev_score', 0)):.3f}"
                )
            if pivot2_row is not None:
                st.markdown(f"○ 馬番{zone['pivot2_no']} {pivot2_row.get('horse_name', '')}")
                st.caption(
                    f"AI{int(pivot2_row.get('ml_rank', 0))}位 / "
                    f"3着内確率{float(pivot2_row.get('ml_top3_prob', 0)) * 100:.1f}% / "
                    f"EV乖離{float(pivot2_row.get('ev_score', 0)):.3f}"
                )
        with col2:
            st.markdown("**相手A (中間人気帯)**")
            for n in zone["aite_a_nums"]:
                try:
                    row = race_df[race_df["horse_no"] == int(n)]
                    if not row.empty:
                        r = row.iloc[0]
                        st.markdown(f"馬番{n} {r.get('horse_name', '')}")
                        st.caption(f"AI{int(r.get('ml_rank', 0))}位 / 人気{int(r.get('popularity', 0))} / EV{float(r.get('ev_score', 0)):.3f}")
                except Exception:
                    pass
        with col3:
            b_label = "相手B (EV高め/補完)" if strategy_mode == STRATEGY_MODE_ROI else "相手B (人気3〜6位)"
            st.markdown(f"**{b_label}**")
            for n in zone["aite_b_nums"]:
                try:
                    row = race_df[race_df["horse_no"] == int(n)]
                    if not row.empty:
                        r = row.iloc[0]
                        ev = float(r.get('ev_score', 0))
                        icon = "💎" if ev > 0 else "📊"
                        st.markdown(f"馬番{n} {r.get('horse_name', '')} {icon}")
                        st.caption(f"AI{int(r.get('ml_rank', 0))}位 / 人気{int(r.get('popularity', 0))} / EV{ev:.3f}")
                except Exception:
                    pass

    tab1, tab2, tab3 = st.tabs([
        "① 1頭軸フォーメーション",
        "② 2頭軸フォーメーション",
        "③ 5頭BOX（10点）"
    ])

    with tab1:
        pivot_no = zone["pivot_no"]
        pivot_name = pivot_row.get("horse_name", "") if pivot_row is not None else ""
        aite_a = zone["aite_a_nums"]
        aite_b = zone["aite_b_nums"]
        st.markdown(f"**軸: 馬番{pivot_no} {pivot_name}**")
        st.markdown(f"**相手A**: 馬番 {', '.join(aite_a[:5])}  **相手B**: 馬番 {', '.join(aite_b[:5])}")
        combos_1j = generate_sanrenpuku_1jiku(zone, max_count=10)
        if combos_1j:
            df_1j = pd.DataFrame(combos_1j)
            df_1j.insert(0, "No", range(1, len(df_1j) + 1))
            st.dataframe(df_1j, use_container_width=True, hide_index=True)
            csv_1j = df_1j.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("1頭軸CSV", data=csv_1j, file_name="sanrenpuku_1jiku.csv",
                               mime="text/csv", key="dl_1jiku")
        else:
            st.info("1頭軸フォーメーションの候補が生成できませんでした。")

    with tab2:
        pivot_no = zone["pivot_no"]
        pivot2_no = zone["pivot2_no"]
        pivot_name = pivot_row.get("horse_name", "") if pivot_row is not None else ""
        pivot2_name = pivot2_row.get("horse_name", "") if pivot2_row is not None else ""
        all_aite = list(dict.fromkeys(zone["aite_a_nums"] + zone["aite_b_nums"]))
        st.markdown(f"**軸1: 馬番{pivot_no} {pivot_name}**　**軸2: 馬番{pivot2_no} {pivot2_name}**")
        st.markdown(f"**相手**: 馬番 {', '.join(all_aite[:7])}")
        combos_2j = generate_sanrenpuku_2jiku(zone, max_count=10)
        if combos_2j:
            df_2j = pd.DataFrame(combos_2j)
            df_2j.insert(0, "No", range(1, len(df_2j) + 1))
            st.dataframe(df_2j, use_container_width=True, hide_index=True)
            csv_2j = df_2j.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("2頭軸CSV", data=csv_2j, file_name="sanrenpuku_2jiku.csv",
                               mime="text/csv", key="dl_2jiku")
        else:
            st.info("2頭軸フォーメーションの候補が生成できませんでした。")

    with tab3:
        box_data = generate_sanrenpuku_5box(race_df, strategy_mode=strategy_mode)
        st.markdown("#### 選出5頭")
        st.info(box_data["selection_note"])
        horses_df = pd.DataFrame(box_data["horses"])
        st.dataframe(horses_df, use_container_width=True, hide_index=True)
        if box_data["alt_horse"]:
            alt = box_data["alt_horse"]
            st.markdown(
                f"**置き換え候補（6位）**: 馬番{alt['馬番']} {alt['馬名']} "
                f"（AI{alt['AI順位']}位 / 人気{alt['人気']} / EV乖離{alt['EV乖離']:.4f}）"
            )
        st.markdown("#### 買い目一覧（10点固定）")
        combos_df = pd.DataFrame(box_data["combos"])
        st.dataframe(combos_df, use_container_width=True, hide_index=True)
        csv_box = combos_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("5頭BOX CSV", data=csv_box, file_name="sanrenpuku_5box.csv",
                           mime="text/csv", key="dl_5box")


# ============================================================
# EV乖離ランキング
# ============================================================

def show_ev_ranking(race_df: pd.DataFrame):
    st.subheader("📊 EV乖離スコアランキング（市場過小評価馬）")
    r = race_df.copy()
    for c in ["ev_score", "ml_top3_prob", "odds", "popularity", "value_score", "implied_top3"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)
    r = r.sort_values("ev_score", ascending=False)
    display_cols = [c for c in ["mark", "horse_no", "horse_name", "popularity", "odds",
                                  "ml_top3_prob", "implied_top3", "ev_score", "value_score",
                                  "buy_flag", "buy_reason"] if c in r.columns]
    out = r[display_cols].copy()
    if "ml_top3_prob" in out.columns:
        out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    if "implied_top3" in out.columns:
        out["implied_top3"] = (out["implied_top3"] * 100).round(1).astype(str) + "%"

    def row_color(row):
        try:
            ev = float(str(row.get("EV乖離スコア", row.get("ev_score", 0))).replace("%", ""))
        except Exception:
            ev = 0
        if ev >= 0.06:
            return ["background-color: #d4edda"] * len(row)
        if ev >= 0.02:
            return ["background-color: #fff3cd"] * len(row)
        if ev <= -0.06:
            return ["background-color: #f8d7da"] * len(row)
        return [""] * len(row)

    try:
        styled = out.rename(columns=JP_COLUMNS).style.apply(row_color, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(out.rename(columns=JP_COLUMNS), use_container_width=True, hide_index=True)

    buy_count = int((race_df.get("buy_flag", pd.Series()) == "買い").sum()) \
        if "buy_flag" in race_df.columns else 0
    high_ev = int((r["ev_score"] >= 0.06).sum())
    avg_odds = float(r["odds"].mean())
    if avg_odds <= 10:
        rec_san = f"{max(3, min(buy_count * 2, 6))}点"
    elif avg_odds <= 30:
        rec_san = f"{max(4, min(buy_count * 2, 8))}点"
    else:
        rec_san = f"{max(5, min(buy_count * 3, 10))}点"

    st.caption(
        f"💡 買い判定: {buy_count}頭 / EV高め(0.06以上): {high_ev}頭 / 三連複推奨: {rec_san}\n"
        "📌 EV乖離 = AI3着内確率 - 市場暗示3着内確率(3.0/単勝オッズ×0.80)"
    )


# ============================================================
# 事前CSV
# ============================================================

def list_preloaded_csv_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.glob("*.csv") if p.is_file()])


def make_preloaded_file_label(path: Path) -> str:
    name = path.stem
    m = re.search(r"(\d{1,2})\s*[RrＲｒ]", name)
    return f"{m.group(1)}R：{path.name}" if m else path.name


def load_preloaded_entry_csv(path: Path, csv_mode: str) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"事前CSVが見つかりません: {path}")
    raw = path.read_bytes()
    header_df = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            header_df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            break
        except Exception:
            pass
    if header_df is not None:
        cols = set(str(c).strip() for c in header_df.columns)
        simple_markers = {"馬名", "horse_name", "騎手", "jockey", "オッズ", "odds", "人気", "popularity"}
        if len(cols & simple_markers) >= 3:
            return read_simple_csv_to_52(raw, source_name=path.name)
    try:
        return normalize_52cols(read_csv_bytes(raw), path.name)
    except Exception as e52:
        try:
            return read_simple_csv_to_52(raw, source_name=path.name)
        except Exception:
            raise e52


def load_many_preloaded_entry_csv(paths: list[Path], csv_mode: str) -> pd.DataFrame:
    frames, errors = [], []
    for p in paths:
        try:
            frames.append(load_preloaded_entry_csv(p, csv_mode))
        except Exception as e:
            errors.append({"ファイル": p.name, "エラー": str(e)})
    if not frames:
        msg = "事前CSVを1件も読めませんでした: " + str(errors[:3]) if errors \
            else "事前CSVがありません。dataフォルダにCSVを置いてください。"
        raise ValueError(msg)
    df = pd.concat(frames, ignore_index=True)
    if errors:
        st.warning(f"読めなかった事前CSVがあります: {len(errors)}件")
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    return df


# ============================================================
# 表示ヘルパー
# ============================================================

def jp_view(df: pd.DataFrame, include_race_key=False) -> pd.DataFrame:
    cols = DISPLAY_COLUMNS.copy()
    if include_race_key:
        cols = ["race_label", "race_key"] + cols
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    if "running_style" in out.columns:
        out["running_style"] = out["running_style"].astype(str).replace(
            {"nan": "", "None": "", "不明": "未取得", "": "未取得"})
    if "style_note" in out.columns:
        out["style_note"] = out["style_note"].astype(str).replace(
            {"nan": "", "None": "", "通過順なし": "通過順データなし", "": "データなし"})
    if "ml_top3_prob" in out.columns:
        out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    if "implied_top3" in out.columns:
        out["implied_top3"] = (out["implied_top3"] * 100).round(1).astype(str) + "%"
    if "expected_value" in out.columns:
        out["expected_value"] = pd.to_numeric(out["expected_value"], errors="coerce").round(2)
    if "ev_score" in out.columns:
        out["ev_score"] = pd.to_numeric(out["ev_score"], errors="coerce").round(4)
    for c in ["jockey_top3_rate_prior", "trainer_top3_rate_prior",
              "sire_top3_rate_prior", "horse_distance_top3_rate_prior"]:
        if c in out.columns:
            vals = pd.to_numeric(out[c], errors="coerce")
            out[c] = np.where(
                vals.notna() & (vals > 0),
                (vals * 100).round(1).astype(str) + "%",
                "未取得"
            )
    return out.rename(columns=JP_COLUMNS)


def show_style_tabs(pred_df: pd.DataFrame, race_df: pd.DataFrame):
    st.subheader("脚質分析")
    tab1, tab2, tab3 = st.tabs(["このレースの脚質", "脚質別成績", "脚質別AI順位"])
    with tab1:
        view_cols = [c for c in ["mark", "ml_rank", "horse_no", "horse_name", "running_style",
                                  "style_note", "pass1", "pass2", "pass3", "pass4", "ml_top3_prob"]
                     if c in race_df.columns]
        out = race_df.sort_values(
            ["ml_rank", "value_score", "horse_no"], ascending=[True, False, True])[view_cols].copy()
        if "ml_top3_prob" in out.columns:
            out["ml_top3_prob"] = (out["ml_top3_prob"] * 100).round(1).astype(str) + "%"
        st.dataframe(out.rename(columns=JP_COLUMNS), use_container_width=True, hide_index=True)
    with tab2:
        summary = make_style_summary(pred_df)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with tab3:
        style_rank = (
            race_df.groupby("running_style", dropna=False)
            .agg(頭数=("horse_name", "count"),
                 平均AI順位=("ml_rank", "mean"),
                 平均3着内確率=("ml_top3_prob", "mean"))
            .reset_index().rename(columns={"running_style": "脚質"})
        )
        if not style_rank.empty:
            style_rank["平均AI順位"] = style_rank["平均AI順位"].round(2)
            style_rank["平均3着内確率"] = (style_rank["平均3着内確率"] * 100).round(1).astype(str) + "%"
        st.dataframe(style_rank, use_container_width=True, hide_index=True)


def show_roi_strategy(race_df: pd.DataFrame, strategy_mode: str = STRATEGY_MODE_ROI):
    mode_icon = "💰" if strategy_mode == STRATEGY_MODE_ROI else "🏆"
    st.subheader(f"{mode_icon} 買い/見送り判定 [{strategy_mode}]")
    st.dataframe(make_value_summary(race_df), use_container_width=True, hide_index=True)
    buy_count = int((race_df["buy_flag"] == "買い").sum()) if "buy_flag" in race_df.columns else 0
    total = len(race_df)
    if buy_count == 0:
        st.warning("このレースは見送り寄りです。")
    elif buy_count <= 3:
        st.info(f"買い候補: {buy_count}/{total}頭。絞れているので回収率/的中率ともに向き。")
    else:
        st.info(f"買い候補: {buy_count}/{total}頭。BOXより軸流し推奨。")


def show_bets(pred_df: pd.DataFrame, key_prefix: str = "bets",
              strategy_mode: str = STRATEGY_MODE_ROI):
    if pred_df is None or pred_df.empty:
        st.warning("買い目候補: 予想結果が空です。")
        return
    st.markdown("---")
    mode_icon = "💰" if strategy_mode == STRATEGY_MODE_ROI else "🏆"
    st.subheader(f"{mode_icon} 買い目候補 [{strategy_mode}]")
    try:
        race_keys = (
            list(pred_df["race_key"].dropna().unique())
            if "race_key" in pred_df.columns else [None]
        )
        for idx, rk in enumerate(race_keys, start=1):
            race_df = pred_df if rk is None else pred_df[pred_df["race_key"] == rk].copy()
            if race_df.empty:
                continue
            if "ml_rank" in race_df.columns:
                race_df = race_df.sort_values("ml_rank")
            label = str(race_df["race_label"].iloc[0]) if "race_label" in race_df.columns \
                else f"レース{idx}"
            if len(race_keys) > 1:
                st.markdown(f"### {label}")
            combos = _ensure_combo_dict_10(
                generate_roi_bet_combinations(race_df, max_count=10, strategy_mode=strategy_mode),
                race_df, max_count=10)
            if not combos:
                st.info("買い目候補がありません。")
                continue
            tabs = st.tabs(list(combos.keys()))
            for tab, (bet_type, rows) in zip(tabs, combos.items()):
                with tab:
                    df_rows = pd.DataFrame(rows)
                    df_rows.insert(0, "No", range(1, len(df_rows) + 1))
                    st.dataframe(df_rows, use_container_width=True, hide_index=True)
                    try:
                        bet_csv = df_rows.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button(
                            f"{bet_type} CSV",
                            data=bet_csv,
                            file_name=f"nyanko_bets_{bet_type}.csv",
                            mime="text/csv",
                            key=f"{key_prefix}_{idx}_{bet_type}"
                        )
                    except Exception:
                        pass
    except Exception as e:
        st.error(f"買い目生成エラー: {e}")


def show_ticket_tabs(race_df: pd.DataFrame, strategy_mode: str = STRATEGY_MODE_ROI):
    mode_icon = "💰" if strategy_mode == STRATEGY_MODE_ROI else "🏆"
    st.subheader(f"{mode_icon} 馬券おすすめ [{strategy_mode}]")
    combos = _ensure_combo_dict_10(
        generate_roi_bet_combinations(race_df, max_count=10, strategy_mode=strategy_mode),
        race_df, max_count=10)
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


# ============================================================
# メイン
# ============================================================

def app_main():
    st.title("🐾 にゃんこ競馬AI")
    st.success(f"起動版: {VERSION}")
    st.caption(
        "v23: ①三連複選出ロジック改善（確率重視・相手B安定化・フォールバック強化）"
        "　②簡易CSV日付バグ修正（20250401固定 → 実行日付を動的取得）"
    )

    with st.sidebar:
        st.header("設定")

        st.markdown("### 🎯 予想モード")
        strategy_mode = st.radio(
            "モードを選択",
            STRATEGY_MODE_OPTIONS,
            index=0,
            help=(
                "**回収率重視**: EV乖離・穴馬評価を強調。高配当狙い。\n\n"
                "**的中率重視**: AI上位馬・人気馬を優先。複勝圏の安定的中狙い。"
            )
        )
        if strategy_mode == STRATEGY_MODE_ROI:
            st.info("💰 回収率重視: 穴馬・EV高め馬を積極評価。高配当を狙います。")
        else:
            st.success("🏆 的中率重視: AI上位・人気馬を中心に絞り込み。安定的中を狙います。")

        st.markdown("---")
        uploaded_model = st.file_uploader("学習済みモデルPKL", type=["pkl"])
        csv_mode = st.radio("予想CSV形式", ["52列TARGET形式", "簡易CSV形式"], index=0)

        if MODEL_PATH.exists():
            st.success(f"同梱PKLあり: {MODEL_PATH.name}")
        else:
            st.warning("同梱PKLなし。画面からPKLをアップロードしてください。")
        if TARGET_CSV_PATH.exists():
            st.success(f"TARGET過去CSVあり: {TARGET_CSV_PATH.name}")
        else:
            st.info("TARGET過去CSVなし: yosou.csv をリポジトリ直下に置くと補正します。")

        st.markdown("---")
        st.caption(
            "**v23 修正内容**\n\n"
            "**🔴 BUG FIX: 簡易CSV日付**\n"
            "- year=25/month=4/day=1固定 → 実行日付を動的取得\n"
            "- 「日付」列(20250510形式)があれば自動分解\n\n"
            "**🟢 三連複5頭BOX改善**\n"
            "- 選出スコア: EV依存を抑制、確率重視に\n"
            "  旧: ev*0.50+prob*0.35\n"
            "  新: prob*0.45+ev*0.25\n\n"
            "**🟢 三連複相手B安定化**\n"
            "- EV正の馬のみ→不足時は確率上位で自動補完\n"
            "- 必ず3頭以上を確保\n\n"
            "**🟢 1頭軸/2頭軸フォーメーション**\n"
            "- 6点未満時のフォールバック補完追加\n"
            "- 2頭軸: 相手BをAより先に試す順序に変更\n"
            "- 分散スコア下限(0.15)で馬番密集を排除\n\n"
            "**v22モード切替は全て引継ぎ**"
        )

    st.subheader("入力方法")
    input_method = st.radio(
        "入力方法を選択",
        ["事前CSVから選択", "netkeiba一括取得→そのまま予想", "出馬表CSV", "netkeiba URL単発"],
        horizontal=True, index=0
    )

    selected_preloaded_paths = []
    uploaded_csv = None
    race_url = ""
    race_items = []

    if input_method == "事前CSVから選択":
        st.caption("GitHubの data/ フォルダに置いたCSVを選ぶだけで予想できます。")
        preloaded_paths = list_preloaded_csv_files()
        if not preloaded_paths:
            st.warning("dataフォルダにCSVがありません。")
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
        st.caption("race_id/URL一覧、または開催情報から一括取得して予想できます。")
        make_mode = st.radio("一括取得方法", ["race_id / URL一覧", "開催情報から自動生成"],
                              horizontal=True, index=0)
        if make_mode == "race_id / URL一覧":
            text = st.text_area("race_id または URLを1行ずつ入力",
                                value="202605020111\n202605020112\n202605020113", height=120)
            race_items = [x.strip() for x in text.splitlines() if x.strip()]
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                year = st.number_input("年", min_value=2020, max_value=2035, value=2026, step=1)
            with c2:
                place_name = st.selectbox("競馬場", list(PLACE_CODE_MAP.keys()),
                                          index=list(PLACE_CODE_MAP.keys()).index("東京"))
            with c3:
                kai = st.number_input("開催回", min_value=1, max_value=10, value=2, step=1)
            with c4:
                nichiji_text = st.text_input("日次（カンマ区切り）", value="1,2")
            c5, c6 = st.columns(2)
            with c5:
                race_start = st.number_input("開始R", min_value=1, max_value=12, value=1, step=1)
            with c6:
                race_end = st.number_input("終了R", min_value=1, max_value=12, value=12, step=1)
            nichiji_list = [int(x.strip()) for x in nichiji_text.split(",") if x.strip().isdigit()]
            race_items = build_race_ids(int(year), place_name, int(kai), nichiji_list,
                                        int(race_start), int(race_end))
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
                st.error("学習済みモデルPKLがありません。")
                return
            st.success(f"モデル読込: {model_status} / モード: {strategy_mode}")

            if input_method == "事前CSVから選択":
                with st.spinner("事前CSVを読み込み中..."):
                    pred_src = load_many_preloaded_entry_csv(selected_preloaded_paths, csv_mode)
                st.success(f"取得: {pred_src['race_key'].nunique()}レース / {len(pred_src)}頭")

            elif input_method == "netkeiba一括取得→そのまま予想":
                with st.spinner("netkeibaから一括取得中..."):
                    pred_src, fetch_errors = fetch_many_netkeiba_to_52cols(race_items, sleep_sec=sleep_sec)
                if pred_src.empty:
                    st.error("1レースも取得できませんでした。")
                    if not fetch_errors.empty:
                        st.dataframe(fetch_errors, use_container_width=True, hide_index=True)
                    return
                st.success(f"取得: {pred_src['race_key'].nunique()}レース / {len(pred_src)}頭")
                if not fetch_errors.empty:
                    st.warning(f"取得失敗: {len(fetch_errors)}件")
                    st.dataframe(fetch_errors, use_container_width=True, hide_index=True)

            elif input_method == "netkeiba URL単発":
                pred_src = fetch_netkeiba_race_to_52cols(race_url.strip())
                st.success("netkeiba URLから取得しました。")

            else:
                pred_src = load_uploaded_entry_csv(uploaded_csv, csv_mode)
                st.success("CSVから取得しました。")

            export_simple = convert_52_to_simple_export(pred_src)
            st.download_button(
                "読み込んだ出馬表CSV",
                data=export_simple.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="entry_races.csv", mime="text/csv",
            )

            pred_src = merge_target_features(pred_src)
            if TARGET_CSV_PATH.exists():
                try:
                    _, _features_check = load_target_features_cached()
                    if _features_check:
                        st.success("TARGET過去CSV（yosou.csv）を結合しました。")
                    else:
                        st.info("yosou.csv はありますが着順なし→補正なしで予想します。")
                except Exception:
                    st.info("yosou.csv は利用できないため出馬表単体で予想します。")
            else:
                st.info("TARGET過去CSV（yosou.csv）は未配置です。出馬表単体で予想します。")

            pred_df = predict(bundle, pred_src, strategy_mode=strategy_mode)
            st.success(f"予想完了: {len(pred_df)}頭 [{strategy_mode}]")

            st.markdown("---")
            st.subheader("予想結果")
            show_df = pred_df.sort_values(
                ["race_key", "ml_rank"] if "race_key" in pred_df.columns else ["ml_rank"])
            try:
                view = jp_view(show_df, include_race_key=False)
            except Exception:
                view = show_df
            st.dataframe(view, use_container_width=True, hide_index=True)
            try:
                csv_bytes = view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("予想結果CSV", data=csv_bytes,
                                   file_name="nyanko_prediction_result.csv", mime="text/csv",
                                   key="download_prediction_result")
            except Exception as e:
                st.caption(f"CSVダウンロード生成をスキップ: {e}")

            show_bets(pred_df, key_prefix="main_bets", strategy_mode=strategy_mode)

            st.markdown("---")
            st.subheader("レース詳細")
            race_options = (
                pred_df[["race_key", "race_label"]].drop_duplicates().sort_values("race_label")
            )
            label_map = dict(zip(race_options["race_label"], race_options["race_key"]))
            selected_label = st.selectbox("レース選択", list(label_map.keys()))
            selected_race = label_map[selected_label]

            race_df = pred_df[pred_df["race_key"] == selected_race].sort_values(
                ["ml_rank", "value_score", "horse_no"], ascending=[True, False, True])
            st.dataframe(jp_view(race_df), use_container_width=True, hide_index=True)

            tickets = make_tickets(race_df)
            c1, c2, c3 = st.columns(3)
            c1.metric("本命", tickets["本命"])
            c2.metric("単勝", tickets["単勝"])
            c3.metric("複勝", tickets["複勝"])

            show_ev_ranking(race_df)

            st.markdown("---")
            show_sanrenpuku_tabs(race_df, strategy_mode=strategy_mode)

            st.markdown("---")
            show_ticket_tabs(race_df, strategy_mode=strategy_mode)
            show_roi_strategy(race_df, strategy_mode=strategy_mode)

            # 2モード比較表示
            st.markdown("---")
            st.subheader("📊 2モード比較（同一レース）")
            st.caption("同じAIスコアから、モードによって買い判定がどう変わるか比較できます。")
            col_roi, col_hit = st.columns(2)
            with col_roi:
                st.markdown("**💰 回収率重視**")
                df_roi_compare = add_value_strategy(race_df.copy(), strategy_mode=STRATEGY_MODE_ROI)
                buy_roi = df_roi_compare[df_roi_compare["buy_flag"] == "買い"][["horse_name", "buy_flag", "buy_reason"]]
                st.dataframe(buy_roi.rename(columns=JP_COLUMNS), use_container_width=True, hide_index=True)
            with col_hit:
                st.markdown("**🏆 的中率重視**")
                df_hit_compare = add_value_strategy(race_df.copy(), strategy_mode=STRATEGY_MODE_HITRATE)
                buy_hit = df_hit_compare[df_hit_compare["buy_flag"] == "買い"][["horse_name", "buy_flag", "buy_reason"]]
                st.dataframe(buy_hit.rename(columns=JP_COLUMNS), use_container_width=True, hide_index=True)

            show_style_tabs(pred_df, race_df)

            c4, c5 = st.columns(2)
            c4.info(f"危険人気馬: {tickets.get('危険人気馬', 'なし')}")
            c5.success(f"穴候補: {tickets.get('穴候補', 'なし')}")

            st.subheader("全レース")
            all_jp = jp_view(pred_df.sort_values(["race_key", "ml_rank"]), include_race_key=True)
            st.dataframe(all_jp, use_container_width=True, hide_index=True)
            csv_bytes = all_jp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("日本語CSVダウンロード", data=csv_bytes,
                               file_name="nyanko_keiba_prediction_jp.csv", mime="text/csv")

        except Exception as e:
            st.error(f"予想できませんでした: {e}")
            st.exception(e)

    st.divider()
    with st.expander("簡易CSVテンプレ（v23: 日付列対応）"):
        st.caption("日付列を入れると正しい日付でレースが識別されます。")
        st.code(
            "日付,馬番,馬名,性別,年齢,騎手,斤量,オッズ,人気,競馬場,レース番号,レース名,距離,馬場,頭数,芝ダ\n"
            "20260510,1,サンプルホースA,牡,5,サンプル騎手A,58.0,2.8,1,東京,11,サンプルレース,2000,良,18,芝\n"
            "20260510,2,サンプルホースB,牝,4,サンプル騎手B,56.0,8.5,5,東京,11,サンプルレース,2000,良,18,芝\n",
            language="csv"
        )
        st.caption("日付列なしでも動作します（実行日付が自動で入ります）。")


try:
    app_main()
except Exception as e:
    st.error("アプリ起動時エラーです。下の詳細を確認してください。")
    st.exception(e)
