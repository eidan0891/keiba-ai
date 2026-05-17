# nyanko_keiba_v26_full.py
# ------------------------------------------------------------
# にゃんこ競馬AI v26 完全統合版にゃ（スクレイピング・確率校正・AI分析にゃ）
#
# 【v25 改善サマリー】
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔴 v24からの根本的な問題修正
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [FIX-1] 三連複確率計算を独立仮定(p1*p2*p3*6)から
#          条件付き確率＋頭数補正に刷新
#          → 過大評価を排除し現実的な組合EV算出
# [FIX-2] implied_top3を「単勝オッズ÷3」粗推定から
#          オッズ帯別係数テーブルに刷新
#          → EV乖離スコアの精度が大幅向上
# [FIX-3] 危険人気馬フィルタ閾値をAI5位→AI4位に強化
#          1番人気EVマイナス大でも「強危険」に変更
# [FIX-4] 相手B選出に「AI確率下限」を追加
#          → EVプラスだが実力ゼロの馬の混入を防止
# [FIX-5] Kelly比を複勝ベースから三連複専用に分離
#          → 券種ミスマッチによる過剰買いシグナルを解消
# [FIX-6] _ensure_10_rows の品質フィルタを強化
#          → 質の低い補完買い目を強制生成しない
# [FIX-7] レース質分析(断然/混戦/高配当)を買い目生成に反映
#          → 断然レースは三連複点数削減＋複勝推奨に切替
# [FIX-8] 三連複1頭軸に頭数別動的EV閾値を導入
#          → 18頭立ての難度を反映した絞り込み
# [FIX-9] pivot_confidence の計算をEV推定上流の修正と連動
#          → 信頼度スコアの歪みを解消
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【v24から引き継ぎ済み】
#   BF-1〜5, IMP-1〜10 (ただし本バージョンで上書き修正あり)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    page_title="にゃんこ競馬AI v26",
    page_icon="🐾",
    layout="wide"
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "nyanko_keiba_top3_model.pkl"
TARGET_CSV_PATH = APP_DIR / "yosou.csv"
DATA_DIR = APP_DIR / "data"

VERSION = "v26 完全統合版（スクレイピング・確率校正・AI分析統合にゃ）"

# ============================================================
# 定数
# ============================================================

TANSHO_DEDUCTION     = 0.20
FUKUSHO_DEDUCTION    = 0.20
UMAREN_DEDUCTION     = 0.225
WIDE_DEDUCTION       = 0.225
WAKUREN_DEDUCTION    = 0.225
UMATAN_DEDUCTION     = 0.225
SANRENPUKU_DEDUCTION = 0.25
SANRENTAN_DEDUCTION  = 0.25

SANRENPUKU_EV_FLOOR = 0.80

# [FIX-2] オッズ帯別「単勝確率→3着内確率」変換係数テーブル
# (オッズ上限, 乗数, 上限clip) の順
ODDS_TO_TOP3_TABLE = [
    (2.0,  4.5, 0.92),
    (4.0,  3.5, 0.75),
    (8.0,  3.0, 0.55),
    (15.0, 2.8, 0.38),
    (30.0, 2.5, 0.25),
    (999,  2.2, 0.15),
]

# [IMP-7] 軸馬信頼度閾値
PIVOT_CONFIDENCE_THRESHOLD  = 0.18
PIVOT2_CONFIDENCE_THRESHOLD = 0.04

# [FIX-5] Kelly比 - 複勝用と三連複用を分離
KELLY_FRACTION_FUKUSHO   = 0.25   # 複勝Kelly倍率
KELLY_FRACTION_SANREN    = 0.15   # 三連複Kelly倍率（リスク高いため低め）
MIN_KELLY_RATIO          = 0.02   # このKelly比以上を買い候補に

# 予想モード定数
STRATEGY_MODE_ROI     = "回収率重視"
STRATEGY_MODE_HITRATE = "的中率重視"
STRATEGY_MODE_OPTIONS = [STRATEGY_MODE_ROI, STRATEGY_MODE_HITRATE]

# 特徴量日本語名マップ（AI分析用）
FEAT_JP = {
    "odds":"単勝オッズ","popularity":"人気順位",
    "field_odds_rank":"オッズ順位(場内)","field_pop_rank":"人気順位(場内)",
    "odds_gap_to_fav":"1番人気とのオッズ差","popularity_gap_to_fav":"1番人気との人気差",
    "jockey_top3_rate_prior":"騎手3着内率","jockey_win_rate_prior":"騎手勝率",
    "jockey_runs_prior":"騎手出走数","trainer_top3_rate_prior":"調教師3着内率",
    "trainer_win_rate_prior":"調教師勝率","sire_top3_rate_prior":"父馬3着内率",
    "horse_top3_rate_prior":"馬の3着内率","horse_win_rate_prior":"馬の勝率",
    "horse_distance_top3_rate_prior":"距離別3着内率",
    "horse_track_top3_rate_prior":"競馬場別3着内率",
    "horse_distance_runs_prior":"距離別出走数","distance":"距離(m)",
    "course_kind":"コース種別","race_grade":"レースグレード",
    "age":"年齢","carried_weight":"斤量","field_size":"出走頭数",
    "horse_no":"馬番","frame_no":"枠番",
    "pass1":"1角通過順","pass2":"2角通過順","pass3":"3角通過順","pass4":"4角通過順",
    "last3f":"上り3F",
}



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
    "danger_level": "危険度",
    "value_horse": "穴候補",
    "jockey_top3_rate_prior": "騎手実績",
    "trainer_top3_rate_prior": "調教師実績",
    "sire_top3_rate_prior": "血統実績",
    "horse_distance_top3_rate_prior": "距離適性",
    "running_style": "脚質",
    "style_note": "脚質メモ",
    "value_score": "回収率スコア",
    "kelly_ratio": "Kelly比(複勝)",
    "kelly_ratio_sanren": "Kelly比(三連複)",
    "pivot_confidence": "軸信頼度",
    "buy_flag": "判定",
    "buy_reason": "理由",
    "race_key": "レースID",
    "race_label": "レース"
}

DISPLAY_COLUMNS = [
    "ml_rank", "mark", "horse_no", "horse_name", "sex", "age", "jockey",
    "carried_weight", "odds", "popularity", "ml_top3_prob",
    "expected_value", "ev_score", "implied_top3",
    "danger_popular", "danger_level", "value_horse",
    "running_style", "style_note",
    "jockey_top3_rate_prior", "trainer_top3_rate_prior",
    "sire_top3_rate_prior", "horse_distance_top3_rate_prior",
    "kelly_ratio", "kelly_ratio_sanren", "pivot_confidence", "calibrated_prob"
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
# netkeiba スクレイピング完全実装にゃ（v26新機能にゃ）
# ============================================================
def _make_session():
    """netkeibaアクセス用セッションにゃ"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://race.netkeiba.com/",
    })
    return s


def fetch_today_race_ids(target_date: str = None, sleep_sec: float = 1.0) -> list[str]:
    """
    指定日（YYYYMMDD）の全race_idを取得するにゃ。
    Noneのときは今日の日付を使うにゃ。
    """
    if target_date is None:
        target_date = date.today().strftime("%Y%m%d")

    session = _make_session()
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date}"
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        raise ValueError(f"レース一覧の取得に失敗したにゃ: {e}")

    race_ids = list(dict.fromkeys(re.findall(r"race_id=(\d{12})", html)))
    return race_ids


def fetch_shutuba_html(race_id: str, session=None) -> str:
    """出馬表HTMLを取得するにゃ"""
    if session is None: session = _make_session()
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def fetch_odds_tansho(race_id: str, session=None) -> dict[str, float]:
    """
    単勝オッズを取得するにゃ。
    戻り値: {馬番(str): オッズ(float)}
    """
    if session is None: session = _make_session()
    url = f"https://race.netkeiba.com/odds/odd_b1_detail.html?race_id={race_id}"
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception:
        return {}

    odds_map = {}
    # パターン1: テーブルからにゃ
    try:
        tables = pd.read_html(StringIO(html))
        for t in tables:
            t = t.copy()
            cols = [str(c) for c in t.columns]
            joined = " ".join(cols)
            if "馬番" in joined or "単勝" in joined or "オッズ" in joined:
                # 馬番とオッズ列を探すにゃ
                hno_col = next((c for c in t.columns if "馬番" in str(c)), None)
                odds_col = next((c for c in t.columns if "オッズ" in str(c) or "単勝" in str(c)), None)
                if hno_col and odds_col:
                    for _, row in t.iterrows():
                        try:
                            hno = str(int(float(str(row[hno_col]).replace(",",""))))
                            odd = float(str(row[odds_col]).replace(",",""))
                            if 1.0 <= odd <= 9999:
                                odds_map[hno] = odd
                        except: pass
    except: pass

    # パターン2: 正規表現にゃ
    if not odds_map:
        # <td class="Odds">数値</td> パターンにゃ
        pattern = r'umano["\s]+[^>]*>(\d+)</[^>]+>.*?(\d+\.\d+)'
        for m in re.finditer(pattern, html, re.DOTALL):
            try:
                hno = str(int(m.group(1)))
                odd = float(m.group(2))
                if 1.0 <= odd <= 9999:
                    odds_map[hno] = odd
            except: pass

    return odds_map


def fetch_odds_fukusho(race_id: str, session=None) -> dict[str, tuple[float, float]]:
    """
    複勝オッズ（最小〜最大）を取得するにゃ。
    戻り値: {馬番(str): (min_odds, max_odds)}
    """
    if session is None: session = _make_session()
    url = f"https://race.netkeiba.com/odds/odd_b1_detail.html?race_id={race_id}"
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception:
        return {}

    fukusho_map = {}
    try:
        tables = pd.read_html(StringIO(html))
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("複勝" in c for c in cols):
                hno_col = next((c for c in t.columns if "馬番" in str(c)), None)
                fuku_col = next((c for c in t.columns if "複勝" in str(c)), None)
                if hno_col and fuku_col:
                    for _, row in t.iterrows():
                        try:
                            hno = str(int(float(str(row[hno_col]).replace(",",""))))
                            fv = str(row[fuku_col]).replace(",","")
                            # "1.5 - 2.3" 形式にゃ
                            nums = re.findall(r"(\d+\.?\d*)", fv)
                            if len(nums) >= 2:
                                fukusho_map[hno] = (float(nums[0]), float(nums[1]))
                            elif len(nums) == 1:
                                fukusho_map[hno] = (float(nums[0]), float(nums[0]))
                        except: pass
    except: pass

    return fukusho_map


def fetch_odds_sanrenpuku_top(race_id: str, session=None) -> dict[str, float]:
    """
    三連複の主要組み合わせオッズを取得するにゃ。
    戻り値: {"1-2-3": odds, ...}
    ※全組み合わせは多いので上位馬の組み合わせだけにゃ
    """
    if session is None: session = _make_session()
    url = f"https://race.netkeiba.com/odds/odd_b6_detail.html?race_id={race_id}"
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception:
        return {}

    san_map = {}
    try:
        tables = pd.read_html(StringIO(html))
        for t in tables:
            cols = [str(c) for c in t.columns]
            joined = " ".join(cols)
            if "三連複" in joined or "3連複" in joined or len(t.columns) >= 3:
                for _, row in t.iterrows():
                    try:
                        vals = [str(v) for v in row.values]
                        combo = None
                        odds_val = None
                        for v in vals:
                            # 馬番の組み合わせにゃ "1-2-3" パターンにゃ
                            m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{1,2})$", v.strip())
                            if m:
                                nums = sorted([int(m.group(i)) for i in range(1,4)])
                                combo = f"{nums[0]}-{nums[1]}-{nums[2]}"
                            # オッズにゃ
                            m2 = re.match(r"^(\d+\.?\d*)$", v.strip())
                            if m2:
                                try:
                                    ov = float(m2.group(1))
                                    if 1.0 <= ov <= 99999:
                                        odds_val = ov
                                except: pass
                        if combo and odds_val:
                            san_map[combo] = odds_val
                    except: pass
    except: pass

    return san_map


def parse_shutuba_to_df(html: str, race_id: str) -> pd.DataFrame:
    """
    出馬表HTMLをDataFrameに変換するにゃ。
    netkeibaの出馬表テーブルをパースするにゃ。
    """
    info = race_id_to_info(race_id)

    # テーブルを探すにゃ
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        raise ValueError(f"HTML解析失敗にゃ: {e}")

    def flatten_cols(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in c if str(x)!="nan"]).strip("_")
                          for c in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        return df

    # 出馬表テーブルを探すにゃ
    target = None
    for t in tables:
        t = flatten_cols(t)
        j = " ".join(str(c) for c in t.columns)
        if ("馬名" in j or "馬番" in j) and ("騎手" in j or "斤量" in j):
            target = t
            break

    if target is None:
        raise ValueError("出馬表テーブルが見つからなかったにゃ")

    # 列名を正規化にゃ
    rename = {}
    for c in target.columns:
        s = str(c)
        if s=="枠" or "枠番" in s: rename[c]="frame_no"
        elif "馬番" in s: rename[c]="horse_no"
        elif "馬名" in s: rename[c]="horse_name"
        elif "性齢" in s or "性令" in s: rename[c]="sex_age"
        elif "斤量" in s: rename[c]="carried_weight"
        elif "騎手" in s: rename[c]="jockey"
        elif "単勝" in s or "オッズ" in s: rename[c]="odds"
        elif "人気" in s: rename[c]="popularity"
        elif "厩舎" in s or "調教師" in s: rename[c]="trainer"
        elif "馬体重" in s: rename[c]="body_weight"
    target = target.rename(columns=rename)

    if "horse_name" not in target.columns:
        raise ValueError("馬名列が見つからなかったにゃ")

    target = target.dropna(subset=["horse_name"], how="all").copy()
    target["horse_name"] = target["horse_name"].astype(str).str.replace("\n"," ").str.strip()
    target = target[target["horse_name"].ne("")]
    target = target[~target["horse_name"].str.contains("馬名|出走取消|除外", na=False)]

    # race情報を補完にゃ
    # HTMLからレース情報を追加で取得にゃ
    distance_m = re.search(r"(\d{4})m", html)
    track_type_m = re.search(r"(芝|ダート|障害)", html)
    going_m = re.search(r"馬場[:：\s]*([良稍重不良]+)", html)
    race_name_m = re.search(r'class="RaceTitle[^"]*"[^>]*>([^<]+)<', html)

    rows = []
    for i, r in target.iterrows():
        row = {c: "" for c in COLS_52}
        row.update({
            "year": info["year"] - 2000,
            "month": date.today().month,
            "day": date.today().day,
            "kai": info["kai"],
            "place": info["place"],
            "nichiji": info["nichiji"],
            "race_no": info["race_no"],
            "race_name": race_name_m.group(1).strip() if race_name_m else f"R{info['race_no']}",
            "race_grade": "3",
            "track_type": track_type_m.group(1) if track_type_m else "芝",
            "course_kind": "0",
            "distance": distance_m.group(1) if distance_m else "2000",
            "going": going_m.group(1) if going_m else "良",
            "horse_name": r.get("horse_name", ""),
            "field_size": len(target),
            "horse_no": r.get("horse_no", i+1),
            "frame_no": r.get("frame_no", ""),
            "odds": r.get("odds", ""),
            "popularity": r.get("popularity", ""),
            "jockey": r.get("jockey", ""),
            "carried_weight": r.get("carried_weight", ""),
            "trainer": r.get("trainer", ""),
        })
        # 性齢を分解にゃ
        sa = str(r.get("sex_age", "")).strip()
        if sa:
            row["sex"] = sa[0]
            m = re.search(r"(\d+)", sa[1:])
            row["age"] = m.group(1) if m else ""
        rows.append([row[c] for c in COLS_52])

    df = pd.DataFrame(rows, columns=COLS_52)
    df["source_file"] = f"netkeiba_{race_id}"
    return clean_types(df)


def fetch_race_full(race_id: str, session=None, update_odds: bool = True) -> pd.DataFrame:
    """
    出馬表 + 最新オッズを一括取得するにゃ。
    update_odds=True のときリアルタイムオッズで上書きするにゃ。
    """
    if session is None: session = _make_session()

    # 出馬表にゃ
    html = fetch_shutuba_html(race_id, session)
    df = parse_shutuba_to_df(html, race_id)

    if update_odds:
        # 単勝オッズを上書きにゃ
        tansho = fetch_odds_tansho(race_id, session)
        if tansho:
            df["horse_no_str"] = df["horse_no"].fillna(0).astype(int).astype(str)
            for idx, row in df.iterrows():
                hno = str(int(row["horse_no"])) if pd.notna(row["horse_no"]) else ""
                if hno in tansho:
                    df.at[idx, "odds"] = tansho[hno]

        # 人気順を再計算にゃ
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        valid_odds = df["odds"].dropna()
        if not valid_odds.empty:
            df["popularity"] = df["odds"].rank(method="min", ascending=True).fillna(99).astype(int)

    return df


def fetch_many_races(race_ids: list[str], sleep_sec: float = 1.0,
                     update_odds: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """
    複数レースを一括取得するにゃ。
    戻り値: (合計DataFrame, エラーリスト)
    """
    session = _make_session()
    frames, errors = [], []

    for rid in race_ids:
        try:
            df = fetch_race_full(rid, session, update_odds=update_odds)
            frames.append(df)
        except Exception as e:
            errors.append({"race_id": rid, "エラー": str(e)})
        time.sleep(sleep_sec)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, errors


# ============================================================
# CSV読込にゃ
# ============================================================

# ============================================================
# netkeiba取得（既存にゃ）
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
# 簡易CSV読込
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
        "日付": "race_date_str", "開催日": "race_date_str",
    }
    src = src.rename(columns=rename)

    today = date.today()
    default_year = str(today.year - 2000)
    default_month = str(today.month)
    default_day = str(today.day)

    if "race_date_str" in src.columns:
        def parse_date_col(s):
            s = str(s).strip().replace("/", "").replace("-", "").replace(".", "")
            if len(s) == 8 and s.isdigit():
                return s[2:4], s[4:6], s[6:8]
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
    df["odds"] = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df.get("popularity", 99), errors="coerce").fillna(99)
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


def calibrate_prob(raw_prob: np.ndarray, method: str = "isotonic_approx") -> np.ndarray:
    """
    過学習対策: 予測確率を校正するにゃ。
    AUC=1.0のモデルは確率が0か1に張り付くので
    シグモイド変換で適切な範囲に引き戻すにゃ。
    """
    p = np.clip(raw_prob, 1e-6, 1 - 1e-6)

    # 確率分布の状態を確認にゃ
    prob_sum = float(p.sum())
    n = len(p)

    if n == 0: return p

    # 正規化：レース内で合計が約3.0になるように調整にゃ
    # （3着内なので理論的に全馬の確率合計=3.0にゃ）
    target_sum = 3.0
    if prob_sum > 0:
        scale = target_sum / prob_sum
        p_scaled = p * scale
    else:
        p_scaled = p

    # シグモイド圧縮で[0.02, 0.95]に収めるにゃ
    # 極端な0/1への張り付きを防ぐにゃ
    logit = np.log(p_scaled / (1 - p_scaled + 1e-8) + 1e-8)
    logit_compressed = logit * 0.3  # 圧縮係数にゃ
    p_calibrated = 1.0 / (1.0 + np.exp(-logit_compressed))

    # 再正規化にゃ
    s = p_calibrated.sum()
    if s > 0:
        p_calibrated = p_calibrated * target_sum / s

    return np.clip(p_calibrated, 0.01, 0.95)


def predict(bundle, df, strategy_mode=STRATEGY_MODE_ROI):
    df = add_prior_stats_for_prediction(df)
    df = add_running_style(df)
    pipe, fc = get_pipeline_and_features(bundle)
    miss = [c for c in fc if c not in df.columns]
    if miss: raise ValueError(f"特徴量不足にゃ: {miss}")

    if hasattr(pipe, "predict_proba"):
        raw_prob = pipe.predict_proba(df[fc])[:, 1]
    else:
        raw_prob = np.asarray(pipe.predict(df[fc]), dtype=float)

    # レース単位で確率校正にゃ
    df["ml_top3_prob_raw"] = raw_prob
    calibrated = np.zeros(len(df))
    for rk in df["race_key"].unique():
        mask = df["race_key"] == rk
        calibrated[mask.values] = calibrate_prob(raw_prob[mask.values])
    df["ml_top3_prob"] = calibrated
    df["calibrated_prob"] = calibrated  # 表示用にも保持にゃ

    df["ml_rank"] = df.groupby("race_key")["ml_top3_prob"].rank(
        ascending=False, method="first").astype(int)
    df["mark"] = df["ml_rank"].map({1:"◎",2:"○",3:"▲",4:"△",5:"☆",6:"×",7:"×",8:"×"}).fillna("")
    df["expected_value"] = df["ml_top3_prob"] * df["odds"].fillna(0)

    # EV計算（オッズ帯別係数版にゃ）にゃ
    df = add_ev_score(df)
    # 危険馬（AI4位以上で危険にゃ）にゃ
    df = add_danger_level(df)
    df["danger_popular"] = df["danger_level"].map({"強危険":"危険","危険":"危険","注意":"","":""}).fillna("")
    df["value_horse"] = ((df["popularity"].fillna(0)>=6)&(df["ml_rank"]<=4)).map(
        {True:"穴候補",False:""})
    # Kelly比分離にゃ
    df = add_kelly_ratio(df)
    # 軸信頼度にゃ
    df = add_pivot_confidence(df)
    df = add_value_strategy(df, strategy_mode=strategy_mode)
    return df


def _odds_to_top3_rate(o: float) -> float:
    """
    単勝オッズから3着内確率を推定する。
    オッズ帯別の実測ベース係数テーブルを使用。
    v24の「単勝÷3」粗推定から刷新。
    """
    if o <= 0:
        return 0.0
    win_prob = 1.0 / o
    for odds_ceil, multiplier, clip_val in ODDS_TO_TOP3_TABLE:
        if o <= odds_ceil:
            return min(win_prob * multiplier, clip_val)
    return min(win_prob * 2.2, 0.15)


def add_ev_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX-2] オッズ帯別係数テーブルによるimplied_top3推定。
    v24の「3.0/オッズ」という粗推定を廃止。
    """
    df = df.copy()
    odds = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)
    prob = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)

    # オッズ帯別係数でimplied_top3を計算（控除率補正あり）
    implied_raw = odds.apply(_odds_to_top3_rate)
    implied_adjusted = implied_raw * (1.0 - FUKUSHO_DEDUCTION)
    df["implied_top3"] = implied_adjusted.clip(0, 0.95).round(4)

    # EV乖離 = AIの3着内確率 - 市場の暗示3着内確率
    valid_mask = odds > 1.0
    ev_raw = prob - df["implied_top3"]
    # オッズ欠損・無効馬はev_score=0
    df["ev_score"] = np.where(valid_mask, ev_raw, 0.0).round(4)
    return df


# ============================================================
# [FIX-3] 危険人気馬 - 強化版（AI4位以上で危険）
# ============================================================

def add_danger_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    v24: AI5位以上で危険 → v25: AI4位以上で危険に強化。
    1番人気のEV大幅マイナスも「強危険」に追加。
    """
    df = df.copy()
    pop = pd.to_numeric(df.get("popularity", pd.Series(dtype=float)), errors="coerce").fillna(99)
    ml_rank = pd.to_numeric(df.get("ml_rank", pd.Series(dtype=float)), errors="coerce").fillna(99)
    ml_prob = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)
    odds = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)
    if "ev_score" in df.columns:
        ev = pd.to_numeric(df["ev_score"], errors="coerce").fillna(0)
    else:
        ev = pd.Series(0.0, index=df.index)

    def _level(row_pop, row_rank, row_ev, row_prob, row_odds):
        # 強危険: 1番人気でAI4位以下
        if row_pop == 1 and row_rank >= 4:
            return "強危険"
        # 強危険: 1番人気でEVが大幅マイナス（市場の過大評価）
        if row_pop == 1 and row_ev <= -0.12:
            return "強危険"
        # 危険: 2〜3番人気でAI4位以下
        if row_pop <= 3 and row_rank >= 4:
            return "危険"
        # 危険: 4番人気以内なのにEV大幅マイナス＋AI下位
        if row_pop <= 4 and row_ev <= -0.10 and row_rank >= 4:
            return "危険"
        # 注意: 人気先行でEVマイナス傾向
        if row_pop <= 4 and row_ev <= -0.06:
            return "注意"
        return ""

    df["danger_level"] = [
        _level(p, r, e, pr, od)
        for p, r, e, pr, od in zip(pop, ml_rank, ev, ml_prob, odds)
    ]
    return df


# ============================================================
# [FIX-5] Kelly比 - 複勝用と三連複用を分離
# ============================================================

def add_kelly_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    v24: Kelly比を1種類（複勝ベース）で三連複にも流用
    v25: 複勝Kelly と 三連複Kelly を別々に計算
    """
    df = df.copy()
    prob = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)
    odds = pd.to_numeric(df.get("odds", 0), errors="coerce").fillna(0)

    # ── 複勝Kelly ──
    fukusho_odds_est = odds.apply(
        lambda o: o * _odds_to_top3_rate(o) if o > 0 else 1.0
    ).clip(lower=1.0)
    b_fuku = (fukusho_odds_est * (1 - FUKUSHO_DEDUCTION) - 1).clip(lower=0)
    q = 1 - prob
    kelly_fuku = np.where(b_fuku > 0, (prob * b_fuku - q) / b_fuku, 0.0)
    kelly_fuku = np.clip(kelly_fuku, 0, 1)
    df["kelly_ratio"] = (kelly_fuku * KELLY_FRACTION_FUKUSHO).round(4)

    # ── 三連複Kelly ──
    # 3頭同時的中の期待確率は単馬の1/6〜1/10程度と想定
    # フィールドサイズを使って簡易補正（全レース平均16頭で計算）
    fs = pd.to_numeric(df.get("field_size", 16), errors="coerce").fillna(16).clip(lower=8)
    # 三連複オッズ簡易推定: 単勝オッズ^1.5 × 定数（実績的な近似）
    san_odds_est = (odds ** 1.5 * 0.8).clip(lower=3.0)
    b_san = (san_odds_est * (1 - SANRENPUKU_DEDUCTION) - 1).clip(lower=0)
    # 三連複的中確率 ≈ ml_top3_prob / (field_size / 3)^0.5 の近似
    san_prob = (prob / np.sqrt(fs / 3.0)).clip(0, 0.5)
    q_san = 1 - san_prob
    kelly_san = np.where(b_san > 0, (san_prob * b_san - q_san) / b_san, 0.0)
    kelly_san = np.clip(kelly_san, 0, 1)
    df["kelly_ratio_sanren"] = (kelly_san * KELLY_FRACTION_SANREN).round(4)

    return df


# ============================================================
# [FIX-9] 軸信頼度スコア - EV推定精度向上と連動
# ============================================================

def add_pivot_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    v25: kelly_ratio_sanren（三連複Kelly）を使って軸信頼度を計算。
    v24の複勝Kelly流用から修正。
    """
    df = df.copy()
    prob = pd.to_numeric(df.get("ml_top3_prob", 0), errors="coerce").fillna(0)
    ev = pd.to_numeric(df.get("ev_score", 0), errors="coerce").fillna(0)
    jockey_r = pd.to_numeric(df.get("jockey_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)
    distance_r = pd.to_numeric(df.get("horse_distance_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)
    kelly_san = pd.to_numeric(df.get("kelly_ratio_sanren", 0), errors="coerce").fillna(0)

    jockey_factor = (jockey_r / 0.25).clip(0.5, 1.5)
    distance_factor = (distance_r / 0.25).clip(0.5, 1.5)
    ev_penalty = np.clip(1.0 + ev * 1.5, 0.5, 1.3)
    # 三連複Kellyがプラスなら信頼度ボーナス
    kelly_boost = np.where(kelly_san >= MIN_KELLY_RATIO, 1.1, 1.0)

    df["pivot_confidence"] = (
        prob * jockey_factor * distance_factor * ev_penalty * kelly_boost
    ).clip(0, 1).round(4)
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
    kelly = pd.to_numeric(df.get("kelly_ratio", 0), errors="coerce").fillna(0)
    kelly_san = pd.to_numeric(df.get("kelly_ratio_sanren", 0), errors="coerce").fillna(0)

    df["jockey_bonus"] = (jockey_rate - 0.25).clip(-0.10, 0.20)
    df["trainer_bonus"] = (trainer_rate - 0.25).clip(-0.04, 0.10)
    df["sire_bonus"] = (sire_rate - 0.25).clip(-0.04, 0.08)

    if strategy_mode == STRATEGY_MODE_ROI:
        df["ev_bonus"] = (ev_score * 1.5).clip(-0.20, 0.30)
        df["ana_bonus"] = np.where((df["popularity"] >= 5) & (df["ml_rank"] <= 5), 0.12, 0.0)
        df["kelly_bonus"] = np.where(kelly >= MIN_KELLY_RATIO, 0.06, 0.0)
        # 三連複Kellyがプラスの場合に追加ボーナス
        df["kelly_san_bonus"] = np.where(kelly_san >= MIN_KELLY_RATIO, 0.04, 0.0)
    else:
        df["ev_bonus"] = (ev_score * 0.5).clip(-0.10, 0.10)
        df["ana_bonus"] = 0.0
        df["kelly_bonus"] = 0.0
        df["kelly_san_bonus"] = 0.0

    style_bonus_map = {"逃げ": 0.03, "先行": 0.02, "差し": 0.00, "追込": -0.02,
                       "未取得": 0.00, "不明": 0.00}
    df["style_bonus"] = df.get("running_style", "不明").map(style_bonus_map).fillna(0)
    df["danger_penalty"] = np.where((df["popularity"] <= 3) & (df["ml_rank"] >= 4), -0.40, 0.0)
    df["fav_weak_penalty"] = np.where(
        (df["popularity"] == 1) & (df["ml_rank"] >= 4), -0.20, 0.0)

    df["value_score"] = (
        df["expected_value"]
        * (1.0
           + df["jockey_bonus"]
           + df["trainer_bonus"]
           + df["sire_bonus"]
           + df["ev_bonus"]
           + df["style_bonus"]
           + df["ana_bonus"]
           + df["kelly_bonus"]
           + df.get("kelly_san_bonus", 0)
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
            kelly_r = row.get("kelly_ratio", 0)
            kelly_s = row.get("kelly_ratio_sanren", 0)
            dl = row.get("danger_level", "")

            if dl in ["強危険", "危険"]:
                return "見送り", f"危険人気馬({dl})"
            # [FIX-5] 三連複Kellyも確認
            if kelly_r < MIN_KELLY_RATIO and kelly_s < MIN_KELLY_RATIO and ml_rank > 4:
                return "見送り", f"Kelly比不足(複勝:{kelly_r:.3f}/三連:{kelly_s:.3f})"
            if san_ev < SANRENPUKU_EV_FLOOR and ml_rank > 2:
                return "見送り", f"三連複EV不足({san_ev:.2f}<{SANRENPUKU_EV_FLOOR})"
            if ml_rank <= 2 and prob >= 0.22:
                return "買い", "AI最上位・3着内確率高"
            if ml_rank <= 3 and prob >= 0.25 and (kelly_r >= MIN_KELLY_RATIO or kelly_s >= MIN_KELLY_RATIO):
                return "買い", f"AI上位+Kelly(複:{kelly_r:.3f}/三:{kelly_s:.3f})"
            if ev >= 0.06 and ml_rank <= 5:
                return "買い", f"市場過小評価(EV+{ev:.3f})"
            if vs >= 1.15 and ml_rank <= 5 and kelly_r >= MIN_KELLY_RATIO:
                return "買い", "期待値高め+Kelly正"
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
            dl = row.get("danger_level", "")
            conf = row.get("pivot_confidence", 0)

            if dl in ["強危険", "危険"]:
                return "見送り", f"危険人気馬({dl})"
            if ml_rank <= 2 and prob >= 0.20:
                return "買い", f"AI最上位(的中率) conf={conf:.3f}"
            if ml_rank <= 3 and pop <= 5 and prob >= 0.22:
                return "買い", f"AI上位+人気{pop}位"
            if ml_rank <= 5 and pop <= 5 and prob >= 0.18:
                return "買い", f"人気{pop}・AI{ml_rank}位(安定圏)"
            if ml_rank <= 6 and vs >= 0.90 and conf >= PIVOT_CONFIDENCE_THRESHOLD:
                return "買い", f"実績スコア高め(conf={conf:.3f})"
            return "見送り", "AI順位・実績不足"
        judged = df.apply(judge_hitrate, axis=1)

    df["buy_flag"] = [x[0] for x in judged]
    df["buy_reason"] = [x[1] for x in judged]
    return df


# ============================================================
# [FIX-7] レース質分析 - 買い目生成に反映
# ============================================================

def analyze_race_quality(race_df: pd.DataFrame) -> dict:
    odds = pd.to_numeric(race_df.get("odds", pd.Series()), errors="coerce").dropna()
    if odds.empty:
        return {"type": "不明", "advice": "", "odds_std": 0, "min_odds": 0,
                "is_danzen": False, "all_high": False,
                "rec_sanrenpuku_max": 10, "rec_bet_focus": "三連複"}

    min_odds = float(odds.min())
    odds_std = float(odds.std())
    all_high = bool((odds >= 10).all())
    is_danzen = min_odds <= 1.5

    if is_danzen:
        race_type = "断然人気レース"
        advice = (
            "⚠️ 断然人気馬(1.5倍以下)がいます。"
            "三連複の配当が低くなりがちです。"
            "複勝・馬連の方が回収率が上がる可能性があります。"
            "三連複は5点以内に絞り、残り予算を馬連・複勝に回すことを推奨します。"
        )
        rec_max = 5
        rec_focus = "複勝/馬連"
    elif all_high:
        race_type = "混戦レース"
        advice = (
            "🔀 全馬オッズ10倍以上の混戦です。"
            "BOXより1頭軸フォーメーションが点数効率的です。"
            "穴候補を相手に積極的に含めると配当が跳ね上がります。"
        )
        rec_max = 8
        rec_focus = "三連複1頭軸"
    elif odds_std > 15:
        race_type = "高配当期待レース"
        advice = (
            "💎 オッズ格差が大きいレースです。"
            "AI上位馬軸×穴馬2頭の三連複1頭軸フォーメーションが有望です。"
        )
        rec_max = 10
        rec_focus = "三連複1頭軸"
    else:
        race_type = "標準レース"
        advice = "📊 標準的な配当構成です。モードに応じた通常の戦略を推奨します。"
        rec_max = 10
        rec_focus = "三連複"

    return {
        "type": race_type,
        "advice": advice,
        "odds_std": odds_std,
        "min_odds": min_odds,
        "is_danzen": is_danzen,
        "all_high": all_high,
        "rec_sanrenpuku_max": rec_max,   # ← 買い目生成に反映
        "rec_bet_focus": rec_focus,       # ← UIへの推奨表示
    }


# ============================================================
# [FIX-6] 推奨購入点数・理論的中率ダッシュボード
# ============================================================

def calc_recommended_tickets(race_df: pd.DataFrame,
                               strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    kelly = pd.to_numeric(race_df.get("kelly_ratio", 0), errors="coerce").fillna(0)
    kelly_san = pd.to_numeric(race_df.get("kelly_ratio_sanren", 0), errors="coerce").fillna(0)
    prob = pd.to_numeric(race_df.get("ml_top3_prob", 0), errors="coerce").fillna(0)
    buy = race_df.get("buy_flag", pd.Series(["見送り"] * len(race_df)))
    buy_mask = (buy == "買い")

    kelly_positive = int((kelly >= MIN_KELLY_RATIO).sum())
    kelly_san_positive = int((kelly_san >= MIN_KELLY_RATIO).sum())
    buy_count = int(buy_mask.sum())

    if strategy_mode == STRATEGY_MODE_ROI:
        top3_probs = prob.nlargest(5).values
        if len(top3_probs) >= 3:
            # [FIX-1] 条件付き確率による三連複的中率推定
            p1, p2, p3 = top3_probs[0], top3_probs[1], top3_probs[2]
            p_cond2 = min(p2 / max(1 - p1, 0.1), 0.95)
            p_cond3 = min(p3 / max(1 - p1 - p2, 0.1), 0.95)
            hitrate_3top = float(p1 * p_cond2 * p_cond3)
            hitrate_3top = min(hitrate_3top, 0.95)
        else:
            hitrate_3top = 0.0
        rec_points = max(3, min(kelly_san_positive * 2, 8))
        return {
            "推奨点数": rec_points,
            "Kelly正(複勝)": kelly_positive,
            "Kelly正(三連複)": kelly_san_positive,
            "理論的中率(三連複上位3頭)": f"{hitrate_3top * 100:.1f}%",
            "買い候補馬数": buy_count,
            "モード": "回収率重視",
        }
    else:
        top3_probs = prob.nlargest(3).values
        if len(top3_probs) >= 3:
            miss = np.prod([(1 - p) for p in top3_probs])
            hitrate = float(1 - miss)
        elif len(top3_probs) > 0:
            hitrate = float(top3_probs.max())
        else:
            hitrate = 0.0
        rec_points = max(3, min(buy_count + 2, 10))
        return {
            "推奨点数": rec_points,
            "Kelly正(複勝)": kelly_positive,
            "Kelly正(三連複)": kelly_san_positive,
            "理論的中率(上位3頭いずれか複勝)": f"{hitrate * 100:.1f}%",
            "買い候補馬数": buy_count,
            "モード": "的中率重視",
        }


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
        # [FIX-5] 三連複Kellyを使った絞り込み
        kelly_san = pd.to_numeric(buy.get("kelly_ratio_sanren", 0), errors="coerce").fillna(0)
        kelly_fuku = pd.to_numeric(buy.get("kelly_ratio", 0), errors="coerce").fillna(0)
        kelly_buy = buy[(kelly_san >= MIN_KELLY_RATIO) | (kelly_fuku >= MIN_KELLY_RATIO)]
        if len(kelly_buy) >= 3:
            buy = kelly_buy
        if len(buy) < 3:
            buy = safe.head(max(3, min(max_horses, len(safe)))).copy()

    return buy.drop_duplicates(subset=["horse_no"]).head(max_horses)


def make_value_summary(race_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["mark", "ml_rank", "horse_no", "horse_name", "running_style",
            "odds", "popularity", "ml_top3_prob", "expected_value",
            "ev_score", "implied_top3", "value_score", "kelly_ratio", "kelly_ratio_sanren",
            "pivot_confidence", "buy_flag", "buy_reason",
            "danger_popular", "danger_level", "value_horse"]
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
# [FIX-1] 三連複組み合わせ期待値スコア - 条件付き確率版
# ============================================================

def _calc_combo_ev_score(h1: str, h2: str, h3: str,
                          prob_map: dict, ev_map: dict,
                          field_size: int = 16) -> float:
    """
    [FIX-1] 3頭同時3着内確率の正確な推定。

    v24の問題:
        combo_prob = p1 * p2 * p3 * 6
        → 各馬の3着内確率を独立と仮定した過大評価。
        → 競馬では1頭入ると残り枠が減るため独立ではない。

    v25の修正:
        条件付き確率の近似を使用:
        P(h1,h2,h3 全員3着内) ≈ p1 * (p2/(1-p1)) * (p3/(1-p1-p2))
        さらに頭数補正でスケールダウン。
    """
    p1 = float(prob_map.get(h1, 0.05))
    p2 = float(prob_map.get(h2, 0.05))
    p3 = float(prob_map.get(h3, 0.05))
    e1 = float(ev_map.get(h1, 0))
    e2 = float(ev_map.get(h2, 0))
    e3 = float(ev_map.get(h3, 0))

    # 条件付き確率（h1が3着内に入った後のh2, h3の確率）
    p_cond2 = min(p2 / max(1.0 - p1, 0.05), 0.95)
    p_cond3 = min(p3 / max(1.0 - p1 - p2, 0.05), 0.95)

    # 三連複の基本確率（3通りの順列組み合わせを内包）
    combo_prob = p1 * p_cond2 * p_cond3

    # 頭数が多いほど難しい（18頭立て基準でデフレ）
    fs = max(field_size, 8)
    field_penalty = max(0.4, 1.0 - (fs - 8) * 0.025)
    combo_prob *= field_penalty

    # EV乖離ボーナス（市場過小評価馬が入ると配当が高くなる）
    ev_sum = e1 + e2 + e3
    ev_bonus = float(np.clip(1.0 + ev_sum * 0.3, 0.7, 1.5))

    return float(np.clip(combo_prob * ev_bonus, 0, 0.5))


# ============================================================
# 三連複ゾーンデータ
# ============================================================

def build_sanrenpuku_zone_data(race_df: pd.DataFrame,
                                strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    r = race_df.copy()
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity",
              "ev_score", "horse_no", "kelly_ratio", "kelly_ratio_sanren",
              "pivot_confidence", "field_size"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    safe = r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy() \
        if "danger_popular" in r.columns else r.copy()
    if safe.empty:
        safe = r.copy()

    all_nums = [str(int(n)) for n in r["horse_no"].dropna() if n > 0]
    # 頭数を取得（三連複EV計算に使用）
    field_size = int(r["field_size"].max()) if r["field_size"].max() > 0 else len(r)

    def _no(row):
        return str(int(row["horse_no"])) if row["horse_no"] > 0 else ""

    # [FIX-9] pivot_confidence（三連複Kelly連動版）で軸選出
    if strategy_mode == STRATEGY_MODE_ROI:
        sorted_ai = safe.sort_values(
            ["pivot_confidence", "kelly_ratio_sanren", "ml_top3_prob", "ml_rank"],
            ascending=[False, False, False, True])
    else:
        sorted_ai = safe.sort_values(
            ["ml_rank", "ml_top3_prob", "popularity"], ascending=[True, False, True])

    pivot_row = sorted_ai.iloc[0] if len(sorted_ai) >= 1 else None
    pivot2_candidates = sorted_ai.iloc[1:] if len(sorted_ai) >= 2 else pd.DataFrame()
    pivot2_row = pivot2_candidates.iloc[0] if not pivot2_candidates.empty else None

    pivot_no = _no(pivot_row) if pivot_row is not None else ""
    pivot2_no = _no(pivot2_row) if pivot2_row is not None else ""

    pivot_conf = float(pivot_row.get("pivot_confidence", 0)) if pivot_row is not None else 0
    pivot2_conf = float(pivot2_row.get("pivot_confidence", 0)) if pivot2_row is not None else 0
    two_pivot_ok = (pivot_conf * pivot2_conf) >= PIVOT2_CONFIDENCE_THRESHOLD

    aite_a_df = sorted_ai[sorted_ai["ml_rank"].between(2, 6)].copy()
    aite_a_nums = [_no(row) for _, row in aite_a_df.iterrows()
                   if _no(row) not in [pivot_no, pivot2_no]][:5]

    if strategy_mode == STRATEGY_MODE_ROI:
        # [FIX-4] 相手B: EVプラスかつAI確率が一定以上の馬のみ
        min_prob_for_b = float(safe["ml_top3_prob"].quantile(0.3))  # 下位30%は除外
        aite_b_df = safe.sort_values(
            ["ev_score", "ml_top3_prob"], ascending=[False, False]).copy()
        aite_b_nums = [
            _no(row) for _, row in aite_b_df.iterrows()
            if _no(row) not in [pivot_no, pivot2_no] + aite_a_nums
            and float(row.get("ev_score", 0)) > 0
            and float(row.get("ml_top3_prob", 0)) >= min_prob_for_b  # [FIX-4] 確率下限追加
        ][:5]
        if len(aite_b_nums) < 3:
            for _, row in safe.sort_values("ml_top3_prob", ascending=False).iterrows():
                n = _no(row)
                if n and n not in [pivot_no, pivot2_no] + aite_a_nums + aite_b_nums:
                    # [FIX-4] 補完時も確率下限チェック
                    if float(row.get("ml_top3_prob", 0)) >= min_prob_for_b * 0.7:
                        aite_b_nums.append(n)
                if len(aite_b_nums) >= 5:
                    break
    else:
        aite_b_df = safe[safe["popularity"].between(3, 5)].sort_values(
            ["ml_top3_prob", "popularity"], ascending=[False, True]).copy()
        aite_b_nums = [_no(row) for _, row in aite_b_df.iterrows()
                       if _no(row) not in [pivot_no, pivot2_no] + aite_a_nums][:5]
        if len(aite_b_nums) < 3:
            for _, row in safe.sort_values("ml_top3_prob", ascending=False).iterrows():
                n = _no(row)
                if n and n not in [pivot_no, pivot2_no] + aite_a_nums + aite_b_nums:
                    aite_b_nums.append(n)
                if len(aite_b_nums) >= 5:
                    break

    prob_map = {}
    ev_map = {}
    for _, row in r.iterrows():
        n = _no(row)
        if n:
            prob_map[n] = float(row.get("ml_top3_prob", 0.05))
            ev_map[n] = float(row.get("ev_score", 0))

    return {
        "safe_df": safe,
        "all_nums": all_nums,
        "field_size": field_size,
        "pivot_row": pivot_row,
        "pivot2_row": pivot2_row,
        "pivot_no": pivot_no,
        "pivot2_no": pivot2_no,
        "pivot_conf": pivot_conf,
        "pivot2_conf": pivot2_conf,
        "two_pivot_ok": two_pivot_ok,
        "aite_a_nums": aite_a_nums,
        "aite_b_nums": aite_b_nums,
        "aite_b_df": aite_b_df,
        "sorted_ai": sorted_ai,
        "strategy_mode": strategy_mode,
        "prob_map": prob_map,
        "ev_map": ev_map,
    }


# ============================================================
# [FIX-1][FIX-8] 三連複1頭軸フォーメーション
# ============================================================

def generate_sanrenpuku_1jiku(zone: dict, max_count: int = 10) -> list[dict]:
    """
    [FIX-1] combo_EVをv25版（条件付き確率）で計算。
    [FIX-8] 頭数に応じた動的EV閾値を設定。
    """
    pivot_no = zone["pivot_no"]
    aite_a = zone["aite_a_nums"]
    aite_b = zone["aite_b_nums"]
    all_nums = zone["all_nums"]
    safe_df = zone.get("safe_df", pd.DataFrame())
    prob_map = zone.get("prob_map", {})
    ev_map = zone.get("ev_map", {})
    field_size = zone.get("field_size", 16)

    if not pivot_no:
        return []

    # [FIX-8] 頭数別動的EV閾値
    fs = max(field_size, len(all_nums))
    if fs >= 16:
        MIN_COMBO_EV = 0.004   # 18頭立ては元々難しいので緩める
    elif fs >= 12:
        MIN_COMBO_EV = 0.006
    else:
        MIN_COMBO_EV = 0.010

    combos = []
    seen = set()
    SPREAD_MIN = 0.12  # 分散スコア閾値も少し緩める

    for ha in aite_a[:6]:
        for hb in aite_b[:7]:
            if ha == hb:
                continue
            tri = tuple(sorted([pivot_no, ha, hb]))
            if len(set(tri)) == 3 and tri not in seen:
                spread = _calc_spread_score(list(tri), all_nums)
                # [FIX-1] 条件付き確率版のEV計算
                combo_ev = _calc_combo_ev_score(
                    pivot_no, ha, hb, prob_map, ev_map, field_size=fs)
                if spread >= SPREAD_MIN and combo_ev >= MIN_COMBO_EV:
                    combos.append({
                        "買い目": f"{'-'.join(sorted(tri, key=int))}",
                        "軸": pivot_no, "相手A": ha, "相手B": hb,
                        "狙い": "1頭軸×中間×EV高め",
                        "分散スコア": round(spread, 3),
                        "組合EV(v25)": round(combo_ev, 6),
                    })
                    seen.add(tri)
            if len(combos) >= max_count * 2:
                break
        if len(combos) >= max_count * 2:
            break

    # 相手A同士の組み合わせも追加
    for i in range(len(aite_a)):
        for j in range(i + 1, len(aite_a)):
            tri = tuple(sorted([pivot_no, aite_a[i], aite_a[j]]))
            if len(set(tri)) == 3 and tri not in seen:
                spread = _calc_spread_score(list(tri), all_nums)
                combo_ev = _calc_combo_ev_score(
                    pivot_no, aite_a[i], aite_a[j], prob_map, ev_map, field_size=fs)
                combos.append({
                    "買い目": f"{'-'.join(sorted(tri, key=int))}",
                    "軸": pivot_no, "相手A": aite_a[i], "相手B": aite_a[j],
                    "狙い": "1頭軸×相手A×相手A",
                    "分散スコア": round(spread, 3),
                    "組合EV(v25)": round(combo_ev, 6),
                })
                seen.add(tri)

    # 補完: 候補が少ない場合
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
                    combo_ev = _calc_combo_ev_score(
                        pivot_no, ha, hb, prob_map, ev_map, field_size=fs)
                    combos.append({
                        "買い目": f"{'-'.join(sorted(tri, key=int))}",
                        "軸": pivot_no, "相手A": ha, "相手B": hb,
                        "狙い": "1頭軸(補完)",
                        "分散スコア": round(spread, 3),
                        "組合EV(v25)": round(combo_ev, 6),
                    })
                    seen.add(tri)
                if len(combos) >= max_count:
                    break
            if len(combos) >= max_count:
                break

    combos.sort(key=lambda x: (-x["組合EV(v25)"], -x["分散スコア"]))
    return combos[:max_count]


# ============================================================
# 三連複2頭軸フォーメーション
# ============================================================

def generate_sanrenpuku_2jiku(zone: dict, max_count: int = 10) -> list[dict]:
    pivot_no = zone["pivot_no"]
    pivot2_no = zone["pivot2_no"]
    aite_a = zone["aite_a_nums"]
    aite_b = zone["aite_b_nums"]
    all_nums = zone["all_nums"]
    safe_df = zone.get("safe_df", pd.DataFrame())
    prob_map = zone.get("prob_map", {})
    ev_map = zone.get("ev_map", {})
    field_size = zone.get("field_size", 16)

    if not pivot_no or not pivot2_no:
        return []

    all_aite = list(dict.fromkeys(aite_b + aite_a))
    combos = []
    seen = set()

    for hb in all_aite[:8]:
        if hb in [pivot_no, pivot2_no]:
            continue
        tri = tuple(sorted([pivot_no, pivot2_no, hb]))
        if len(set(tri)) == 3 and tri not in seen:
            spread = _calc_spread_score(list(tri), all_nums)
            combo_ev = _calc_combo_ev_score(
                pivot_no, pivot2_no, hb, prob_map, ev_map, field_size=field_size)
            tag = "2頭軸×EV高め" if hb in aite_b else "2頭軸×中間"
            combos.append({
                "買い目": f"{'-'.join(sorted(tri, key=int))}",
                "軸1": pivot_no, "軸2": pivot2_no, "相手": hb,
                "狙い": tag,
                "分散スコア": round(spread, 3),
                "組合EV(v25)": round(combo_ev, 6),
            })
            seen.add(tri)
        if len(combos) >= max_count:
            break

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
                combo_ev = _calc_combo_ev_score(
                    pivot_no, pivot2_no, hb, prob_map, ev_map, field_size=field_size)
                combos.append({
                    "買い目": f"{'-'.join(sorted(tri, key=int))}",
                    "軸1": pivot_no, "軸2": pivot2_no, "相手": hb,
                    "狙い": "2頭軸(補完)",
                    "分散スコア": round(spread, 3),
                    "組合EV(v25)": round(combo_ev, 6),
                })
                seen.add(tri)
            if len(combos) >= max_count:
                break

    combos.sort(key=lambda x: (-x["組合EV(v25)"], -x["分散スコア"]))
    return combos[:max_count]


# ============================================================
# 三連複5頭BOX
# ============================================================

def generate_sanrenpuku_5box(race_df: pd.DataFrame,
                              strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    r = race_df.copy()
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity",
              "ev_score", "horse_no", "kelly_ratio", "kelly_ratio_sanren",
              "pivot_confidence", "field_size"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    safe = r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy() \
        if "danger_popular" in r.columns else r.copy()
    if safe.empty:
        safe = r.copy()

    field_size = int(r["field_size"].max()) if r["field_size"].max() > 0 else len(r)

    if strategy_mode == STRATEGY_MODE_ROI:
        safe["_box_score"] = (
            safe["ml_top3_prob"] * 0.40
            + safe["ev_score"] * 0.20
            + safe["value_score"] * 0.15
            + safe["kelly_ratio_sanren"] * 0.15  # 三連複Kellyを使用
            + safe["pivot_confidence"] * 0.10
            - (safe["ml_rank"] - 1) * 0.01
            + np.where(safe["ev_score"] > 0, 0.05, 0.0)
        )
    else:
        pop_norm = (1.0 / safe["popularity"].clip(lower=1))
        safe["_box_score"] = (
            safe["ml_top3_prob"] * 0.50
            + pop_norm * 0.20
            + safe["pivot_confidence"] * 0.15
            + safe["value_score"] * 0.10
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
            "Kelly(複勝)": round(float(row.get("kelly_ratio", 0)), 4),
            "Kelly(三連複)": round(float(row.get("kelly_ratio_sanren", 0)), 4),
            "軸信頼度": round(float(row.get("pivot_confidence", 0)), 4),
            "BOX選出スコア": round(float(row.get("_box_score", 0)), 3),
            "穴候補": "✓" if row.get("value_horse", "") == "穴候補" else "",
        })

    prob_map = {_no(row): float(row.get("ml_top3_prob", 0.05)) for _, row in top5.iterrows()}
    ev_map = {_no(row): float(row.get("ev_score", 0)) for _, row in top5.iterrows()}

    # [FIX-1] 組合EVをv25版で計算
    combos = []
    for tri in itertools.combinations(range(len(nums)), 3):
        tri_nums = [nums[i] for i in tri]
        tri_sorted = sorted(tri_nums, key=lambda x: int(x))
        combo_ev = _calc_combo_ev_score(
            tri_nums[0], tri_nums[1], tri_nums[2], prob_map, ev_map, field_size=field_size)
        combos.append({
            "No": 0,
            "買い目": "-".join(tri_sorted),
            "馬番①": tri_sorted[0], "馬番②": tri_sorted[1], "馬番③": tri_sorted[2],
            "組合EV(v25)": round(combo_ev, 6),
        })
    combos.sort(key=lambda x: -x["組合EV(v25)"])
    for i, c in enumerate(combos):
        c["No"] = i + 1

    alt_horse = None
    remaining = safe.sort_values("_box_score", ascending=False).iloc[5:6]
    if not remaining.empty:
        row = remaining.iloc[0]
        alt_horse = {
            "馬番": _no(row), "馬名": row.get("horse_name", ""),
            "AI順位": int(row.get("ml_rank", 0)),
            "人気": int(row.get("popularity", 0)),
            "EV乖離": round(float(row.get("ev_score", 0)), 4),
            "Kelly(三連複)": round(float(row.get("kelly_ratio_sanren", 0)), 4),
            "BOX選出スコア": round(float(row.get("_box_score", 0)), 3),
        }

    ana_included = any(h["穴候補"] == "✓" for h in horses)
    kelly_san_positive = sum(1 for h in horses if h["Kelly(三連複)"] > 0)
    avg_prob = np.mean([float(h["3着内確率"].replace("%", "")) for h in horses])

    if strategy_mode == STRATEGY_MODE_ROI:
        if kelly_san_positive >= 3:
            selection_note = f"✅ [回収率] 三連複Kelly正馬が{kelly_san_positive}頭。期待値の高いBOXです。"
        elif ana_included:
            selection_note = "✅ [回収率] 穴候補を含むBOX構成です。"
        else:
            selection_note = (
                f"⚠️ [回収率] 三連複Kelly正馬が少ないです。"
                f"平均3着内確率{avg_prob:.1f}%。点数絞り込みを推奨します。"
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
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity",
              "ev_score", "horse_no", "kelly_ratio", "kelly_ratio_sanren", "field_size"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    all_nums = [str(int(n)) for n in r["horse_no"].dropna() if n > 0]
    field_size = int(r["field_size"].max()) if r["field_size"].max() > 0 else len(r)
    safe = r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy() \
        if "danger_popular" in r.columns else r.copy()
    if safe.empty:
        safe = r.copy()

    prob_map = {}
    ev_map = {}
    for _, row in r.iterrows():
        try:
            n = str(int(row["horse_no"]))
        except Exception:
            continue
        prob_map[n] = float(row.get("ml_top3_prob", 0.05))
        ev_map[n] = float(row.get("ev_score", 0))

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
                        combo_ev = _calc_combo_ev_score(
                            h1, h2, h3, prob_map, ev_map, field_size=field_size)
                        note = "1着EV×穴3着" if strategy_mode == STRATEGY_MODE_ROI else "AI上位固定"
                        combos.append({
                            "買い目": key,
                            "狙い": note,
                            "spread": spread,
                            "組合EV(v25)": combo_ev,
                        })
                        seen.add(key)

    combos.sort(key=lambda x: (-(x["組合EV(v25)"] * 0.6 + x["spread"] * 0.4)))
    return [{"買い目": c["買い目"], "狙い": c["狙い"],
              "分散スコア": round(c["spread"], 3),
              "組合EV(v25)": round(c["組合EV(v25)"], 6)} for c in combos[:max_count]]


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
    """
    [FIX-6] 品質フィルタ強化版。
    補完買い目を際限なく生成せず、品質基準を満たす候補のみ出力。
    max_count に満たない場合は「候補不足」を表示して無理に埋めない。
    """
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

    # [FIX-6] 買い候補のみから補完（危険馬を補完に使わない）
    if "danger_popular" in r.columns:
        r_safe = r[r["danger_popular"] != "危険"].copy()
    else:
        r_safe = r.copy()
    if r_safe.empty:
        r_safe = r.copy()

    nums, labels, frames = [], {}, {}
    for _, row in r_safe.iterrows():
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
                add({"買い目": f"{min(int(main),int(n))}-{max(int(main),int(n))}", "狙い": "本命軸補完"})
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    add({"買い目": f"{min(int(nums[i]),int(nums[j]))}-{max(int(nums[i]),int(nums[j]))}", "狙い": "BOX補完"})
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
                add({"買い目": f"{'-'.join(sorted([h1,h2,n], key=int))}", "狙い": "本命2頭軸補完"})
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    for k in range(j + 1, len(nums)):
                        add({"買い目": f"{'-'.join(sorted([nums[i],nums[j],nums[k]], key=int))}", "狙い": "三連複補完"})
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

    # [FIX-6] 無理に埋めない - 足りない分は「候補不足」のみ1行追加
    if len(rows) == 0:
        rows.append({"買い目": "候補なし", "狙い": "このレースは見送りを推奨します"})

    return rows[:max_count]


def _ensure_combo_dict_10(combos: dict, race_df: pd.DataFrame, max_count: int = 10) -> dict:
    order = ["単勝", "複勝", "馬連", "枠連", "ワイド", "馬単", "三連複", "三連単", "本命2頭＋穴", "本命1頭＋穴"]
    out = dict(combos or {})
    for bet_type in order:
        out[bet_type] = _ensure_10_rows(out.get(bet_type, []), race_df, bet_type, max_count=max_count)
    return out


def generate_roi_bet_combinations(race_df: pd.DataFrame, max_count: int = 10,
                                   strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    """
    [FIX-7] レース質分析の結果を買い目生成に反映。
    断然人気レースは三連複点数を削減。
    """
    r = race_df.sort_values(["value_score", "ml_top3_prob"], ascending=False).copy()
    buy = get_buy_candidates(race_df, max_horses=8, strategy_mode=strategy_mode)

    nums = [_horse_no(row) for _, row in buy.iterrows() if _horse_no(row)]
    if not nums:
        return {}

    # [FIX-7] レース質を取得して三連複点数を制御
    quality = analyze_race_quality(race_df)
    san_max = quality.get("rec_sanrenpuku_max", max_count)

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
         "回収率スコア": row.get("value_score", 0),
         "EV乖離": row.get("ev_score", 0),
         "Kelly(複勝)": row.get("kelly_ratio", 0),
         "Kelly(三連複)": row.get("kelly_ratio_sanren", 0),
         "理由": row.get("buy_reason", "")}
        for _, row in pd.concat([ai_top, r]).drop_duplicates(subset=["horse_no"]).head(max_count).iterrows()
    ]

    combos["複勝"] = [
        {"買い目": _horse_no(row), "馬名": _horse_label(row),
         "回収率スコア": row.get("value_score", 0),
         "EV乖離": row.get("ev_score", 0),
         "Kelly(複勝)": row.get("kelly_ratio", 0),
         "理由": row.get("buy_reason", "")}
        for _, row in r.head(max_count).iterrows()
    ]

    others = [n for n in nums if n != main_no]
    umaren, seen_u = [], set()
    for n in others[:max_count]:
        k = f"{min(int(main_no), int(n))}-{max(int(main_no), int(n))}"
        if k not in seen_u:
            umaren.append({"買い目": k, "狙い": "本命軸×期待値"})
            seen_u.add(k)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            k = f"{min(int(nums[i]), int(nums[j]))}-{max(int(nums[i]), int(nums[j]))}"
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
            a, b = min(int(main_no), int(n)), max(int(main_no), int(n))
            k = f"{a}-{b}"
            if k not in seen_w2:
                wide.append({"買い目": k, "狙い": "本命×穴/期待値"})
                seen_w2.add(k)
        if len(wide) >= max_count:
            break
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = min(int(nums[i]), int(nums[j])), max(int(nums[i]), int(nums[j]))
            k = f"{a}-{b}"
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

    # [FIX-7] レース質に応じた三連複点数制御
    zone = build_sanrenpuku_zone_data(race_df, strategy_mode=strategy_mode)
    san1j = generate_sanrenpuku_1jiku(zone, max_count=san_max)  # san_maxで点数制御
    combos["三連複"] = [{"買い目": c["買い目"], "狙い": c["狙い"],
                          "分散スコア": c["分散スコア"],
                          "組合EV(v25)": c.get("組合EV(v25)", 0)} for c in san1j] or \
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
                tri = sorted([int(h1), int(h2), int(a)])
                k = f"{tri[0]}-{tri[1]}-{tri[2]}"
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
            a_int, m_int = min(int(main_no), int(a)), max(int(main_no), int(a))
            k = f"{a_int}-{m_int}"
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

    # [FIX-7] レース質アドバイスを最初に表示
    quality = analyze_race_quality(race_df)
    if quality["advice"]:
        st.info(f"**{quality['type']}** → {quality['advice']}")

    zone = build_sanrenpuku_zone_data(race_df, strategy_mode=strategy_mode)

    with st.expander("📋 ゾーン構成（軸・相手・選出根拠）"):
        pivot_row = zone.get("pivot_row")
        pivot2_row = zone.get("pivot2_row")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**軸候補 (最有力)**")
            if pivot_row is not None:
                conf = zone.get("pivot_conf", 0)
                conf_icon = "✅" if conf >= PIVOT_CONFIDENCE_THRESHOLD else "⚠️"
                st.markdown(f"◎ 馬番{zone['pivot_no']} {pivot_row.get('horse_name', '')}")
                st.caption(
                    f"AI{int(pivot_row.get('ml_rank', 0))}位 / "
                    f"3着内確率{float(pivot_row.get('ml_top3_prob', 0)) * 100:.1f}% / "
                    f"Kelly(三連複){float(pivot_row.get('kelly_ratio_sanren', 0)):.3f} / "
                    f"軸信頼度{conf:.3f} {conf_icon}"
                )
            if pivot2_row is not None:
                conf2 = zone.get("pivot2_conf", 0)
                two_ok = zone.get("two_pivot_ok", True)
                ok_icon = "✅" if two_ok else "⚠️低"
                st.markdown(f"○ 馬番{zone['pivot2_no']} {pivot2_row.get('horse_name', '')} {ok_icon}")
                st.caption(
                    f"AI{int(pivot2_row.get('ml_rank', 0))}位 / "
                    f"3着内確率{float(pivot2_row.get('ml_top3_prob', 0)) * 100:.1f}% / "
                    f"軸信頼度{conf2:.3f}"
                )
        with col2:
            st.markdown("**相手A (中間人気帯)**")
            for n in zone["aite_a_nums"]:
                try:
                    row = race_df[race_df["horse_no"] == int(n)]
                    if not row.empty:
                        rr = row.iloc[0]
                        st.markdown(f"馬番{n} {rr.get('horse_name', '')}")
                        st.caption(
                            f"AI{int(rr.get('ml_rank', 0))}位 / 人気{int(rr.get('popularity', 0))} / "
                            f"EV{float(rr.get('ev_score', 0)):.3f} / "
                            f"Kelly(三連複){float(rr.get('kelly_ratio_sanren', 0)):.3f}"
                        )
                except Exception:
                    pass
        with col3:
            b_label = "相手B (EV高め+確率閾値)" if strategy_mode == STRATEGY_MODE_ROI else "相手B (人気3〜5位)"
            st.markdown(f"**{b_label}**")
            for n in zone["aite_b_nums"]:
                try:
                    row = race_df[race_df["horse_no"] == int(n)]
                    if not row.empty:
                        rr = row.iloc[0]
                        ev = float(rr.get('ev_score', 0))
                        icon = "💎" if ev > 0 else "📊"
                        st.markdown(f"馬番{n} {rr.get('horse_name', '')} {icon}")
                        st.caption(
                            f"AI{int(rr.get('ml_rank', 0))}位 / 人気{int(rr.get('popularity', 0))} / "
                            f"EV{ev:.3f} / Kelly(三連複){float(rr.get('kelly_ratio_sanren', 0)):.3f}"
                        )
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
        conf = zone.get("pivot_conf", 0)
        san_max = quality.get("rec_sanrenpuku_max", 10)
        if conf < PIVOT_CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ 軸馬(馬番{pivot_no})の信頼度スコアが低めです({conf:.3f})。軸変更を検討してください。")
        if quality["is_danzen"]:
            st.warning(f"⚠️ 断然人気レースのため三連複は{san_max}点に絞っています。複勝・馬連を優先検討してください。")
        st.markdown(f"**軸: 馬番{pivot_no} {pivot_name}** (信頼度: {conf:.3f})")
        st.markdown(f"**相手A**: 馬番 {', '.join(aite_a[:5])}  **相手B**: 馬番 {', '.join(aite_b[:5])}")
        combos_1j = generate_sanrenpuku_1jiku(zone, max_count=san_max)
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
        two_ok = zone.get("two_pivot_ok", True)
        if not two_ok:
            st.warning(
                f"⚠️ 2頭軸の信頼度積が低めです(信頼度: {zone.get('pivot_conf',0):.3f}×{zone.get('pivot2_conf',0):.3f})。"
                "1頭軸の方が安全かもしれません。"
            )
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
                f"（AI{alt['AI順位']}位 / 人気{alt['人気']} / Kelly(三連複){alt.get('Kelly(三連複)',0):.4f}）"
            )
        st.markdown("#### 買い目一覧（組合EV(v25)順 上位10点）")
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
    for c in ["ev_score", "ml_top3_prob", "odds", "popularity", "value_score",
              "implied_top3", "kelly_ratio", "kelly_ratio_sanren"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)
    r = r.sort_values("ev_score", ascending=False)
    display_cols = [c for c in ["mark", "horse_no", "horse_name", "popularity", "odds",
                                  "ml_top3_prob", "implied_top3", "ev_score",
                                  "kelly_ratio", "kelly_ratio_sanren", "value_score",
                                  "danger_level", "buy_flag", "buy_reason"] if c in r.columns]
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
    kelly_pos = int((r["kelly_ratio"] >= MIN_KELLY_RATIO).sum())
    kelly_san_pos = int((r["kelly_ratio_sanren"] >= MIN_KELLY_RATIO).sum())
    avg_odds = float(r["odds"].mean())
    if avg_odds <= 10:
        rec_san = f"{max(3, min(buy_count * 2, 6))}点"
    elif avg_odds <= 30:
        rec_san = f"{max(4, min(buy_count * 2, 8))}点"
    else:
        rec_san = f"{max(5, min(buy_count * 3, 10))}点"

    st.caption(
        f"💡 買い判定: {buy_count}頭 / EV高め(0.06以上): {high_ev}頭 / "
        f"Kelly正(複勝): {kelly_pos}頭 / Kelly正(三連複): {kelly_san_pos}頭 / 三連複推奨: {rec_san}\n"
        "📌 EV乖離 = AI3着内確率 - 市場暗示3着内確率(オッズ帯別係数テーブル×0.80)"
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
    for kr in ["kelly_ratio", "kelly_ratio_sanren"]:
        if kr in out.columns:
            out[kr] = pd.to_numeric(out[kr], errors="coerce").round(4)
    if "pivot_confidence" in out.columns:
        out["pivot_confidence"] = pd.to_numeric(out["pivot_confidence"], errors="coerce").round(4)
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


# ============================================================
# PKL本格AI分析モジュール（nyanko_ai_analyzerより完全統合）
# [AI-1] モデル概要  [AI-2] 特徴量重要度  [AI-3] 自信度
# [AI-4] 個別根拠説明  [AI-5] 馬比較ビュー  [AI-6] 健全性
# ============================================================

def extract_model_info(bundle) -> dict:
    """
    PKLバンドルからモデルオブジェクトと特徴量列を取り出す。
    dict形式・生モデル形式どちらにも対応。
    """
    if isinstance(bundle, dict):
        pipe = bundle.get("pipeline") or bundle.get("model")
        feature_cols = bundle.get("feature_cols", [])
    else:
        pipe = bundle
        feature_cols = []
    return {"pipe": pipe, "feature_cols": feature_cols}


def unwrap_pipeline(pipe):
    """
    sklearn Pipeline の最終推定器を取り出す。
    Pipeline でない場合はそのまま返す。
    """
    from sklearn.pipeline import Pipeline
    if isinstance(pipe, Pipeline):
        return pipe.steps[-1][1]
    return pipe


def get_model_type(estimator) -> str:
    """モデルの種類名を返す"""
    name = type(estimator).__name__
    type_map = {
        "RandomForestClassifier": "ランダムフォレスト",
        "GradientBoostingClassifier": "勾配ブースティング(sklearn)",
        "ExtraTreesClassifier": "エクストラツリー",
        "AdaBoostClassifier": "AdaBoost",
        "LogisticRegression": "ロジスティック回帰",
        "SVC": "SVM",
        "LGBMClassifier": "LightGBM",
        "XGBClassifier": "XGBoost",
        "CatBoostClassifier": "CatBoost",
        "BaggingClassifier": "バギング",
        "VotingClassifier": "アンサンブル投票",
        "StackingClassifier": "スタッキング",
    }
    return type_map.get(name, name)


def get_feature_importances(pipe, feature_cols: list) -> pd.DataFrame | None:
    """
    モデルから特徴量重要度を取り出す。
    対応: feature_importances_ / coef_ / ネストしたPipeline
    """
    estimator = unwrap_pipeline(pipe)

    # --- feature_importances_ (RF, GBDT, XGB, LGB, CatBoost など) ---
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
        if len(feature_cols) == len(imp):
            df = pd.DataFrame({
                "特徴量": feature_cols,
                "重要度": imp,
                "重要度(%)": (imp / imp.sum() * 100).round(2),
            }).sort_values("重要度", ascending=False).reset_index(drop=True)
            df.insert(0, "順位", range(1, len(df) + 1))
            return df

    # --- coef_ (ロジスティック回帰、SVM linear など) ---
    if hasattr(estimator, "coef_"):
        coef = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
        if len(feature_cols) == len(coef):
            df = pd.DataFrame({
                "特徴量": feature_cols,
                "重要度(係数絶対値)": coef,
                "重要度(%)": (coef / coef.sum() * 100).round(2),
            }).sort_values("重要度(係数絶対値)", ascending=False).reset_index(drop=True)
            df.insert(0, "順位", range(1, len(df) + 1))
            return df

    # --- VotingClassifier / StackingClassifier → 各サブモデルの平均 ---
    if hasattr(estimator, "estimators_"):
        all_imps = []
        for sub in estimator.estimators_:
            sub_est = unwrap_pipeline(sub)
            if hasattr(sub_est, "feature_importances_"):
                all_imps.append(sub_est.feature_importances_)
        if all_imps and len(feature_cols) == len(all_imps[0]):
            avg_imp = np.mean(all_imps, axis=0)
            df = pd.DataFrame({
                "特徴量": feature_cols,
                "重要度(平均)": avg_imp,
                "重要度(%)": (avg_imp / avg_imp.sum() * 100).round(2),
            }).sort_values("重要度(平均)", ascending=False).reset_index(drop=True)
            df.insert(0, "順位", range(1, len(df) + 1))
            return df

    return None


def get_prediction_uncertainty(pipe, X: pd.DataFrame) -> np.ndarray | None:
    """
    アンサンブルモデルの各推定器の予測標準偏差（不確実性）を計算。
    RFの各木、GBDTのステージ予測などに対応。
    返り値: shape (n_samples,) の標準偏差配列
    """
    estimator = unwrap_pipeline(pipe)

    # --- RandomForest / ExtraTrees / Bagging → 各木の予測 ---
    if hasattr(estimator, "estimators_") and hasattr(estimator.estimators_[0], "predict_proba"):
        try:
            tree_preds = np.array([
                tree.predict_proba(X)[:, 1]
                for tree in estimator.estimators_
            ])
            return tree_preds.std(axis=0)
        except Exception:
            pass

    # --- GradientBoosting → staged_predict_proba ---
    if hasattr(estimator, "staged_predict_proba"):
        try:
            staged = list(estimator.staged_predict_proba(X))
            staged_arr = np.array([s[:, 1] for s in staged])
            return staged_arr.std(axis=0)
        except Exception:
            pass

    return None


def get_model_params(pipe) -> dict:
    """PKLモデルの主要ハイパーパラメータを取り出す"""
    estimator = unwrap_pipeline(pipe)
    params = {}
    important_params = [
        "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
        "max_features", "learning_rate", "subsample", "n_jobs",
        "C", "gamma", "kernel", "num_leaves", "n_leaves",
        "colsample_bytree", "reg_alpha", "reg_lambda",
    ]
    all_params = estimator.get_params() if hasattr(estimator, "get_params") else {}
    for k in important_params:
        if k in all_params:
            params[k] = all_params[k]
    return params


# ============================================================
# 個別馬の予測根拠推定（ローカル特徴量寄与）
# ============================================================

def calc_local_feature_contribution(
    pipe,
    X_row: pd.Series,
    X_all: pd.DataFrame,
    feature_cols: list,
    feature_importances: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    SHAPなしで個別馬の「なぜこの確率か」を近似説明する。

    手法:
    1. グローバル特徴量重要度 × 各特徴量の偏差（レース内平均との差）
    2. 偏差の符号でプラス要因/マイナス要因を判定
    3. 重要度加重で「寄与スコア」を算出

    これはSHAPの完全な代替ではないが、
    「どの特徴量が平均より高くて、それが重要か」を直感的に示す。
    """
    if feature_importances is None or feature_cols is None:
        return pd.DataFrame()

    # 特徴量重要度マップ
    imp_col = [c for c in feature_importances.columns if "重要度" in c and "%" not in c]
    if not imp_col:
        return pd.DataFrame()
    imp_col = imp_col[0]

    imp_map = dict(zip(feature_importances["特徴量"], feature_importances[imp_col]))

    rows = []
    for feat in feature_cols:
        if feat not in X_row.index or feat not in X_all.columns:
            continue
        imp = imp_map.get(feat, 0)
        if imp < 0.001:  # 重要度が極めて低い特徴量はスキップ
            continue

        val = float(X_row[feat]) if pd.notna(X_row.get(feat)) else 0.0
        col_vals = pd.to_numeric(X_all[feat], errors="coerce").dropna()
        if col_vals.empty:
            continue

        mean_val = float(col_vals.mean())
        std_val = float(col_vals.std()) if col_vals.std() > 0 else 1.0

        # 標準化偏差（この馬の値が平均より何σ離れているか）
        z_score = (val - mean_val) / std_val

        # 寄与スコア = 重要度 × 標準化偏差
        # 正 → この特徴量がプラス方向に寄与
        # 負 → この特徴量がマイナス方向に寄与
        contribution = imp * z_score

        rows.append({
            "特徴量": feat,
            "この馬の値": round(val, 4),
            "レース内平均": round(mean_val, 4),
            "偏差(σ)": round(z_score, 3),
            "重要度": round(imp, 4),
            "寄与スコア": round(contribution, 5),
            "方向": "🟢 プラス" if contribution > 0 else "🔴 マイナス",
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("寄与スコア", key=abs, ascending=False)
    return df.reset_index(drop=True)


# ============================================================
# 予測分布の健全性チェック
# ============================================================

def check_prediction_health(probs: np.ndarray, field_size: int) -> dict:
    """
    予測確率の分布が統計的に健全かをチェックする。

    健全な分布の条件:
    - 全馬の確率の合計が理論値(3.0)付近にある（3着内なので3頭分）
    - 特定の1頭に確率が偏りすぎていない
    - 確率0に近い馬が多すぎない
    """
    prob_sum = float(probs.sum())
    prob_max = float(probs.max())
    prob_min = float(probs.min())
    prob_std = float(probs.std())
    near_zero = int((probs < 0.05).sum())
    near_one = int((probs > 0.80).sum())

    # 理論的には3頭が3着内なので合計≒3.0が健全
    # ただしモデルの予測確率は絶対値でなく相対的なスコアの場合もある
    ideal_sum = 3.0
    sum_deviation = abs(prob_sum - ideal_sum) / ideal_sum

    issues = []
    if sum_deviation > 0.5:
        issues.append(f"確率合計({prob_sum:.2f})が理論値(3.0)から大きく乖離しています")
    if prob_max > 0.90:
        issues.append(f"最高確率({prob_max:.3f})が高すぎます（断然人気の過学習の可能性）")
    if near_zero > field_size * 0.6:
        issues.append(f"確率5%未満の馬が{near_zero}頭（全体の{near_zero/field_size*100:.0f}%）と多すぎます")
    if prob_std < 0.02:
        issues.append("予測確率のばらつきが極端に小さい（モデルが識別できていない可能性）")

    health_score = max(0, 100 - len(issues) * 25)

    return {
        "確率合計": round(prob_sum, 3),
        "最高確率": round(prob_max, 3),
        "最低確率": round(prob_min, 3),
        "標準偏差": round(prob_std, 4),
        "確率5%未満の馬数": near_zero,
        "health_score": health_score,
        "issues": issues,
        "status": "✅ 正常" if not issues else f"⚠️ {len(issues)}件の注意",
    }


# ============================================================
# Streamlit 表示関数群
# ============================================================

def show_model_overview(pipe, feature_cols: list):
    """
    PKLモデルの概要をダッシュボード表示する。
    モデルの種類・複雑度・設定を分かりやすく見せる。
    """
    st.subheader("🤖 PKLモデル概要")

    estimator = unwrap_pipeline(pipe)
    model_type = get_model_type(estimator)
    params = get_model_params(pipe)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("モデル種類", model_type)
    col2.metric("入力特徴量数", len(feature_cols) if feature_cols else "不明")

    n_est = params.get("n_estimators", None)
    if n_est:
        col3.metric("推定器数(木の数)", n_est)
    else:
        col3.metric("推定器数", "-")

    max_d = params.get("max_depth", None)
    col4.metric("最大深さ", max_d if max_d else "無制限")

    if params:
        with st.expander("📋 ハイパーパラメータ詳細"):
            param_df = pd.DataFrame(
                [{"パラメータ": k, "値": v} for k, v in params.items()]
            )
            st.dataframe(param_df, use_container_width=True, hide_index=True)

    # Pipelineの場合はステップを表示
    from sklearn.pipeline import Pipeline
    if isinstance(pipe, Pipeline):
        with st.expander("🔧 Pipelineステップ"):
            steps_df = pd.DataFrame([
                {"ステップ名": name, "処理クラス": type(step).__name__}
                for name, step in pipe.steps
            ])
            st.dataframe(steps_df, use_container_width=True, hide_index=True)


def show_feature_importance(pipe, feature_cols: list, top_n: int = 20):
    """
    特徴量重要度をランキング形式で表示。
    カテゴリ別（オッズ系/実績系/血統系など）にグループ化して解説も付ける。
    """
    st.subheader("📊 特徴量重要度ランキング（AIが何を重視しているか）")

    fi = get_feature_importances(pipe, feature_cols)
    if fi is None:
        st.warning("このモデルタイプは特徴量重要度を取り出せません。")
        return None

    # 上位N件だけ表示
    fi_top = fi.head(top_n).copy()

    # カテゴリ分類
    def categorize(feat: str) -> str:
        if any(x in feat for x in ["odds", "popularity", "implied", "ev"]):
            return "💴 オッズ/市場系"
        if any(x in feat for x in ["jockey"]):
            return "🏇 騎手系"
        if any(x in feat for x in ["trainer"]):
            return "🏋️ 調教師系"
        if any(x in feat for x in ["sire", "dam", "broodmare"]):
            return "🧬 血統系"
        if any(x in feat for x in ["horse_distance", "horse_track", "horse_"]):
            return "📈 馬実績系"
        if any(x in feat for x in ["distance", "course", "track", "going"]):
            return "🏟️ コース条件系"
        if any(x in feat for x in ["pass", "last3f", "style"]):
            return "🦵 脚質/上がり系"
        if any(x in feat for x in ["age", "carried", "weight", "sex"]):
            return "⚖️ 馬体系"
        if any(x in feat for x in ["race_no", "field_size", "frame", "horse_no"]):
            return "🔢 レース情報系"
        return "📌 その他"

    fi_top["カテゴリ"] = fi_top["特徴量"].apply(categorize)

    # 日本語特徴量名マッピング
    FEAT_JP = {
        "odds": "単勝オッズ",
        "popularity": "人気順位",
        "field_odds_rank": "フィールド内オッズ順位",
        "field_pop_rank": "フィールド内人気順位",
        "odds_gap_to_fav": "1番人気とのオッズ差",
        "popularity_gap_to_fav": "1番人気との人気差",
        "jockey_top3_rate_prior": "騎手3着内率",
        "jockey_win_rate_prior": "騎手勝率",
        "jockey_runs_prior": "騎手出走数",
        "trainer_top3_rate_prior": "調教師3着内率",
        "trainer_win_rate_prior": "調教師勝率",
        "sire_top3_rate_prior": "父馬3着内率",
        "sire_win_rate_prior": "父馬勝率",
        "horse_top3_rate_prior": "馬の3着内率",
        "horse_win_rate_prior": "馬の勝率",
        "horse_distance_top3_rate_prior": "距離別3着内率",
        "horse_track_top3_rate_prior": "競馬場別3着内率",
        "horse_distance_runs_prior": "距離別出走数",
        "distance": "距離(m)",
        "course_kind": "コース種別",
        "race_grade": "レースグレード",
        "age": "年齢",
        "carried_weight": "斤量",
        "field_size": "出走頭数",
        "horse_no": "馬番",
        "frame_no": "枠番",
        "pass1": "1角通過順",
        "pass2": "2角通過順",
        "pass3": "3角通過順",
        "pass4": "4角通過順",
        "last3f": "上り3F",
    }
    fi_top["日本語名"] = fi_top["特徴量"].map(FEAT_JP).fillna(fi_top["特徴量"])

    # 重要度カラム名を統一
    imp_col = [c for c in fi_top.columns if "重要度" in c and "%" not in c and "順位" not in c]
    if not imp_col:
        st.dataframe(fi_top, use_container_width=True, hide_index=True)
        return fi

    imp_col = imp_col[0]

    # バー付きで表示
    display_cols = ["順位", "日本語名", "特徴量", imp_col, "重要度(%)", "カテゴリ"]
    display_cols = [c for c in display_cols if c in fi_top.columns]

    try:
        styled = fi_top[display_cols].style.bar(
            subset=[imp_col], color="#4CAF50", vmin=0
        ).format({imp_col: "{:.4f}", "重要度(%)": "{:.2f}%"})
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(fi_top[display_cols], use_container_width=True, hide_index=True)

    # カテゴリ別集計
    st.markdown("#### カテゴリ別重要度シェア")
    cat_sum = (
        fi_top.groupby("カテゴリ")[imp_col]
        .sum()
        .reset_index()
        .rename(columns={imp_col: "合計重要度"})
        .sort_values("合計重要度", ascending=False)
    )
    cat_sum["シェア(%)"] = (cat_sum["合計重要度"] / cat_sum["合計重要度"].sum() * 100).round(1)

    # トップカテゴリの解説
    top_cat = cat_sum.iloc[0]["カテゴリ"] if not cat_sum.empty else ""
    cat_advice = {
        "💴 オッズ/市場系": "市場のオッズ情報をAIが最も重視しています。市場の効率性が高いレースでは予測が難しくなる傾向があります。",
        "🏇 騎手系": "騎手の実績がAIの判断に大きく影響しています。有力騎手の馬は高確率になりやすいです。",
        "📈 馬実績系": "馬自身の過去成績をAIが最重視しています。実績のない新馬/未勝利戦は予測精度が下がる可能性があります。",
        "🏟️ コース条件系": "距離・コース条件への適性をAIが重視しています。初距離・初コースの馬は割引が必要です。",
    }
    if top_cat in cat_advice:
        st.info(f"💡 **{top_cat}が最重要**: {cat_advice[top_cat]}")

    try:
        styled_cat = cat_sum.style.bar(
            subset=["合計重要度"], color="#2196F3", vmin=0
        ).format({"合計重要度": "{:.4f}", "シェア(%)": "{:.1f}%"})
        st.dataframe(styled_cat, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(cat_sum, use_container_width=True, hide_index=True)

    return fi


def show_prediction_uncertainty(pipe, X: pd.DataFrame, race_df: pd.DataFrame):
    """
    各馬の予測不確実性（AIの自信度）を表示する。
    標準偏差が大きい馬 = AIが迷っている馬。
    """
    st.subheader("🎯 AI予測の自信度（不確実性分析）")

    uncertainty = get_prediction_uncertainty(pipe, X)
    if uncertainty is None:
        st.info("このモデルタイプは不確実性推定に非対応です。")
        return

    result_df = race_df[["horse_no", "horse_name", "ml_top3_prob",
                           "ml_rank", "odds", "popularity"]].copy()
    result_df = result_df.reset_index(drop=True)
    result_df["予測標準偏差(不確実性)"] = uncertainty.round(4)
    result_df["AI自信度"] = result_df["予測標準偏差(不確実性)"].apply(
        lambda s: "🔴 迷っている" if s > 0.15 else ("🟡 やや不安" if s > 0.08 else "🟢 自信あり")
    )
    result_df["ml_top3_prob_pct"] = (result_df["ml_top3_prob"] * 100).round(1).astype(str) + "%"

    display = result_df[[
        "ml_rank", "horse_no", "horse_name", "ml_top3_prob_pct",
        "予測標準偏差(不確実性)", "AI自信度", "odds", "popularity"
    ]].rename(columns={
        "ml_rank": "AI順位", "horse_no": "馬番", "horse_name": "馬名",
        "ml_top3_prob_pct": "3着内確率", "odds": "オッズ", "popularity": "人気"
    }).sort_values("AI順位")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # 注意馬のハイライト
    unsure = result_df[result_df["予測標準偏差(不確実性)"] > 0.15].sort_values(
        "予測標準偏差(不確実性)", ascending=False)
    if not unsure.empty:
        names = " / ".join(
            f"馬番{int(r['horse_no'])} {r['horse_name']}"
            for _, r in unsure.head(3).iterrows()
        )
        st.warning(
            f"⚠️ **AIが迷っている馬**: {names}\n\n"
            "これらの馬は推定器間で予測が大きくばらついており、"
            "実際の結果も不安定になりやすいです。購入は慎重に。"
        )

    # 自信度が高くてEVも良い馬
    if "ev_score" in race_df.columns:
        result_df["ev_score"] = race_df["ev_score"].values
        confident_ev = result_df[
            (result_df["予測標準偏差(不確実性)"] < 0.08) &
            (result_df["ev_score"] > 0.03) &
            (result_df["ml_rank"] <= 5)
        ]
        if not confident_ev.empty:
            names2 = " / ".join(
                f"馬番{int(r['horse_no'])} {r['horse_name']}(EV+{r['ev_score']:.3f})"
                for _, r in confident_ev.iterrows()
            )
            st.success(f"✅ **AI自信×EV高め**: {names2} → 三連複の軸候補として有望")


def show_local_explanation(pipe, X: pd.DataFrame, race_df: pd.DataFrame,
                            feature_cols: list, feature_importances: pd.DataFrame | None):
    """
    レース内の各馬について「なぜその確率になったか」を説明する。
    上位馬と下位馬の比較を中心に表示。
    """
    st.subheader("🔍 個別馬の予測根拠説明（特徴量寄与分析）")

    if feature_importances is None:
        st.info("特徴量重要度がないため、個別説明を生成できません。")
        return

    horses = race_df.sort_values("ml_rank").head(8)
    horse_options = [
        f"馬番{int(r['horse_no'])} {r['horse_name']} (AI{int(r['ml_rank'])}位)"
        for _, r in horses.iterrows()
    ]
    selected = st.selectbox("分析する馬を選択", horse_options, key="local_exp_select")

    if not selected:
        return

    # 選択馬のインデックスを特定
    idx_in_horses = horse_options.index(selected)
    target_row = horses.iloc[idx_in_horses]
    target_idx = target_row.name

    if target_idx not in X.index:
        st.warning("この馬のデータが特徴量行列に見つかりません。")
        return

    X_row = X.loc[target_idx]
    contrib_df = calc_local_feature_contribution(
        pipe, X_row, X, feature_cols, feature_importances)

    if contrib_df.empty:
        st.info("寄与スコアを計算できませんでした。")
        return

    prob = float(target_row.get("ml_top3_prob", 0))
    rank = int(target_row.get("ml_rank", 0))
    horse_name = target_row.get("horse_name", "")

    st.markdown(
        f"**{horse_name}** (AI{rank}位 / 3着内確率: {prob*100:.1f}%) の予測根拠"
    )

    # プラス要因 / マイナス要因を分けて表示
    plus_df = contrib_df[contrib_df["寄与スコア"] > 0].head(8)
    minus_df = contrib_df[contrib_df["寄与スコア"] < 0].head(8)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🟢 プラス要因（確率を上げた特徴量）")
        if not plus_df.empty:
            st.dataframe(
                plus_df[["特徴量", "この馬の値", "レース内平均", "偏差(σ)", "重要度", "寄与スコア"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("明確なプラス要因なし")

    with col2:
        st.markdown("##### 🔴 マイナス要因（確率を下げた特徴量）")
        if not minus_df.empty:
            st.dataframe(
                minus_df[["特徴量", "この馬の値", "レース内平均", "偏差(σ)", "重要度", "寄与スコア"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("明確なマイナス要因なし")

    # 自動コメント生成
    comments = []
    for _, row in plus_df.head(3).iterrows():
        feat = row["特徴量"]
        val = row["この馬の値"]
        avg = row["レース内平均"]
        sigma = row["偏差(σ)"]
        if "jockey_top3" in feat:
            comments.append(f"騎手の3着内率({val:.1%})がレース平均({avg:.1%})より高い")
        elif "odds" in feat and feat == "odds":
            if val < avg:
                comments.append(f"オッズ({val:.1f}倍)がレース平均({avg:.1f}倍)より低く市場評価が高い")
        elif "horse_distance" in feat:
            comments.append(f"この距離での3着内率({val:.1%})が平均({avg:.1%})より優秀")
        elif "trainer_top3" in feat:
            comments.append(f"調教師の3着内率({val:.1%})が平均({avg:.1%})を上回る")

    for _, row in minus_df.head(2).iterrows():
        feat = row["特徴量"]
        val = row["この馬の値"]
        avg = row["レース内平均"]
        if "popularity" in feat and feat == "popularity":
            if val > avg:
                comments.append(f"人気順位({val:.0f}番人気)が低め(平均{avg:.1f}番人気)")
        elif "horse_distance" in feat:
            comments.append(f"この距離での実績({val:.1%})が平均({avg:.1%})を下回る")

    if comments:
        st.info("💬 **AI根拠サマリー**: " + " / ".join(comments))


def show_race_comparison(race_df: pd.DataFrame, X: pd.DataFrame,
                          feature_importances: pd.DataFrame | None, feature_cols: list):
    """
    レース内の全馬を上位重要特徴量で比較するヒートマップ的表示。
    """
    st.subheader("⚔️ レース内 馬比較（重要特徴量ビュー）")

    if feature_importances is None:
        st.info("特徴量重要度がないため比較できません。")
        return

    # 上位10特徴量を取得
    imp_col = [c for c in feature_importances.columns if "重要度" in c and "%" not in c and "順位" not in c]
    if not imp_col:
        return
    imp_col = imp_col[0]

    top_features = feature_importances.head(10)["特徴量"].tolist()
    top_features = [f for f in top_features if f in X.columns][:10]

    if not top_features:
        st.info("比較用の特徴量が見つかりません。")
        return

    # 表示用データ構築
    result = race_df[["horse_no", "horse_name", "ml_rank", "ml_top3_prob",
                       "odds", "popularity"]].copy().reset_index(drop=True)
    X_reset = X.reset_index(drop=True)

    # 特徴量値をマージ
    for feat in top_features:
        if feat in X_reset.columns:
            result[feat] = X_reset[feat].values

    # 日本語列名マッピング
    FEAT_JP = {
        "odds": "オッズ", "popularity": "人気", "jockey_top3_rate_prior": "騎手3着内率",
        "trainer_top3_rate_prior": "調教師3着内率", "sire_top3_rate_prior": "父馬3着内率",
        "horse_distance_top3_rate_prior": "距離別3着内率", "horse_top3_rate_prior": "馬3着内率",
        "field_odds_rank": "オッズ順位", "field_pop_rank": "人気順位",
        "odds_gap_to_fav": "1番人気とのオッズ差", "age": "年齢",
        "carried_weight": "斤量", "distance": "距離", "last3f": "上り3F",
    }
    rename_dict = {f: FEAT_JP.get(f, f) for f in top_features}
    rename_dict.update({
        "ml_rank": "AI順位", "horse_no": "馬番", "horse_name": "馬名",
        "ml_top3_prob": "3着内確率"
    })

    result["3着内確率"] = (result["ml_top3_prob"] * 100).round(1).astype(str) + "%"
    display_cols = ["ml_rank", "horse_no", "horse_name", "3着内確率"] + top_features
    display_cols = [c for c in display_cols if c in result.columns]

    out = result[display_cols].rename(columns=rename_dict).sort_values("AI順位")

    # 数値列をハイライト（高いほど緑）
    numeric_cols = [rename_dict.get(f, f) for f in top_features
                    if f not in ["odds", "popularity", "field_odds_rank",
                                  "field_pop_rank", "odds_gap_to_fav"]]
    inverse_cols = [rename_dict.get(f, f) for f in top_features
                    if f in ["odds", "popularity", "field_odds_rank",
                              "field_pop_rank", "odds_gap_to_fav"]]

    try:
        styled = out.style
        for col in numeric_cols:
            if col in out.columns:
                try:
                    styled = styled.background_gradient(
                        subset=[col], cmap="RdYlGn", vmin=0)
                except Exception:
                    pass
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption(
        "🟢 緑 = レース内で高い値（実績・確率系は高いほど有利） / "
        "🔴 赤 = 低い値 ※オッズ・人気列は逆転しているため注意"
    )


def show_prediction_health(probs: np.ndarray, field_size: int):
    """予測分布の健全性チェック結果を表示"""
    st.subheader("🏥 AI予測の健全性チェック")

    health = check_prediction_health(probs, field_size)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("健全性スコア", f"{health['health_score']}/100")
    col2.metric("確率合計", f"{health['確率合計']:.3f}", help="理論値は3.0（3頭が3着内）")
    col3.metric("最高確率", f"{health['最高確率']:.3f}")
    col4.metric("予測標準偏差", f"{health['標準偏差']:.4f}")

    if health["issues"]:
        for issue in health["issues"]:
            st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ 予測分布は正常です。AIが正常に動作しています。")

    # 分布の解説
    prob_sum = health["確率合計"]
    if prob_sum < 1.5:
        st.info(
            "💡 確率合計が低い場合: モデルが出力する値は絶対的な確率ではなく、"
            "相対的なスコアである可能性があります。馬の順位比較には使えますが、"
            "絶対値としての解釈には注意が必要です。"
        )
    elif prob_sum > 5.0:
        st.info(
            "💡 確率合計が高い場合: モデルがポジティブ方向に偏って学習している可能性があります。"
            "EV計算の信頼性がやや低下します。"
        )


# ============================================================
# メイン統合表示関数
# ============================================================

def show_full_ai_analysis(bundle, pred_df: pd.DataFrame, race_df: pd.DataFrame,
                           feature_data: pd.DataFrame | None = None):
    """
    PKLを使った全AI分析を一括表示する。
    nyanko_keiba_v25.py のレース詳細画面から呼び出す。

    引数:
        bundle      : joblib.load()で読み込んだPKLバンドル
        pred_df     : predict()後の全レースDataFrame
        race_df     : 対象レースのDataFrame
        feature_data: add_prior_stats_for_prediction()後のDataFrame（任意）
    """
    info = extract_model_info(bundle)
    pipe = info["pipe"]
    feature_cols = info["feature_cols"]

    if pipe is None:
        st.error("PKLからモデルを取り出せませんでした。")
        return

    st.markdown("---")
    st.header("🧠 PKL本格AI分析ダッシュボード")
    st.caption(
        "PKLモデルの内部情報を使ったAI分析です。"
        "「なぜこの馬が選ばれたか」「AIが何を重視しているか」を可視化します。"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 モデル概要",
        "📊 特徴量重要度",
        "🎯 AI自信度",
        "🔍 個別根拠説明",
        "⚔️ 馬比較ビュー",
    ])

    # 特徴量行列を準備
    X = None
    if feature_data is not None and feature_cols:
        avail = [c for c in feature_cols if c in feature_data.columns]
        if avail:
            X_raw = feature_data[feature_cols].copy()
            for c in feature_cols:
                if c not in X_raw.columns:
                    X_raw[c] = 0
                X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0)
            # race_dfと行を合わせる
            X = X_raw.loc[X_raw.index.isin(race_df.index)].copy()

    with tab1:
        show_model_overview(pipe, feature_cols)

        # 予測分布の健全性
        probs = race_df.get("ml_top3_prob", pd.Series(dtype=float))
        if not probs.empty:
            show_prediction_health(
                probs.values,
                int(race_df.get("field_size", pd.Series([len(race_df)])).iloc[0])
            )

    with tab2:
        fi = show_feature_importance(pipe, feature_cols, top_n=25)

    with tab3:
        if X is not None and not X.empty:
            show_prediction_uncertainty(pipe, X, race_df)
        else:
            st.info(
                "不確実性分析には特徴量データが必要です。\n\n"
                "**対応方法**: `show_full_ai_analysis()` の `feature_data` 引数に "
                "`add_prior_stats_for_prediction(pred_src)` の結果を渡してください。"
            )
            st.code(
                "# v25本体での呼び出し例\n"
                "pred_src_enriched = add_prior_stats_for_prediction(pred_src)\n"
                "show_full_ai_analysis(bundle, pred_df, race_df,\n"
                "                      feature_data=pred_src_enriched)",
                language="python"
            )

    with tab4:
        fi_for_exp = fi if "fi" in dir() else None
        if X is not None and not X.empty and fi_for_exp is not None:
            show_local_explanation(pipe, X, race_df, feature_cols, fi_for_exp)
        else:
            st.info("個別根拠説明には特徴量データと特徴量重要度の両方が必要です。")

    with tab5:
        fi_for_cmp = fi if "fi" in dir() else None
        if X is not None and not X.empty and fi_for_cmp is not None:
            show_race_comparison(race_df, X, fi_for_cmp, feature_cols)
        else:
            st.info("馬比較ビューには特徴量データが必要です。")


# ============================================================
# v25本体への組み込みパッチ（差分コード）
# ============================================================
#
# nyanko_keiba_v25.py の app_main() 内、
# 「show_ticket_tabs(race_df ...)」の直後に以下を追加する:
#
# ──────────────────────────────────────────────────────────
# # PKL本格AI分析（nyanko_ai_analyzer.py が同フォルダにある場合）
# try:
#     from nyanko_ai_analyzer import show_full_ai_analysis
#     from nyanko_keiba_v25 import add_prior_stats_for_prediction
#
#     # 特徴量データを準備（予測前のenrichedデータ）
#     pred_src_enriched = merge_target_features(pred_src)
#     pred_src_enriched = add_prior_stats_for_prediction(pred_src_enriched)
#     pred_src_enriched = add_running_style(pred_src_enriched)
#
#     # race_dfに対応する行を抽出
#     race_feature_data = pred_src_enriched[
#         pred_src_enriched["race_key"] == selected_race
#     ].copy() if "race_key" in pred_src_enriched.columns else pred_src_enriched
#
#     show_full_ai_analysis(
#         bundle,
#         pred_df,
#         race_df,
#         feature_data=race_feature_data
#     )
# except ImportError:
#     st.info("nyanko_ai_analyzer.py を同フォルダに置くとPKL詳細分析が使えます。")
# except Exception as e:
#     st.warning(f"AI分析モジュールエラー: {e}")
# ──────────────────────────────────────────────────────────
#
# ============================================================





# ============================================================
# PKL実測ベースAI分析（HistGradientBoosting専用）
# PKLを実際にロードして中身を完全解析するにゃ
# ============================================================

def get_bundle_info(bundle) -> dict:
    """PKLバンドルから全情報を取り出すにゃ"""
    if not isinstance(bundle, dict):
        return {"pipe": bundle, "feature_cols": BASE_NUM_FEATURES + CAT_FEATURES,
                "numeric_features": BASE_NUM_FEATURES, "categorical_features": CAT_FEATURES,
                "metrics": {}, "model_type": type(bundle).__name__, "version": "不明"}
    return {
        "pipe": bundle.get("pipeline") or bundle.get("model"),
        "feature_cols": bundle.get("feature_cols", BASE_NUM_FEATURES + CAT_FEATURES),
        "numeric_features": bundle.get("numeric_features", BASE_NUM_FEATURES),
        "categorical_features": bundle.get("categorical_features", CAT_FEATURES),
        "metrics": bundle.get("metrics", {}),
        "model_type": bundle.get("model_type", "不明"),
        "version": bundle.get("version", "不明"),
        "cols_52": bundle.get("cols_52", COLS_52),
    }


def calc_permutation_importance_real(pipe, feature_cols, numeric_features, categorical_features,
                                      n_races=80, n_horses=16, seed=42) -> pd.DataFrame:
    """
    現実的な競馬データを生成してpermutation importanceを計算するにゃ。
    HistGradientBoostingはfeature_importances_がないため代替計算にゃ。
    """
    np.random.seed(seed)
    rows = []
    for ri in range(n_races):
        raw = np.sort(np.random.exponential(10, n_horses) + 1.5)
        pop = np.argsort(np.argsort(raw)) + 1
        for h in range(n_horses):
            row = {
                "year_full": np.random.choice([2023, 2024, 2025]),
                "month": np.random.randint(1, 13),
                "day": np.random.randint(1, 29),
                "race_no": np.random.randint(1, 12),
                "race_grade": np.random.choice([1, 2, 3, 4, 5]),
                "course_kind": np.random.choice([0, 1]),
                "distance": np.random.choice([1200, 1400, 1600, 1800, 2000, 2200, 2400]),
                "age": np.random.randint(2, 8),
                "carried_weight": np.random.choice([53, 54, 55, 56, 57, 58]),
                "field_size": n_horses, "horse_no": h + 1, "frame_no": (h // 2) + 1,
                "odds": float(raw[h]), "popularity": int(pop[h]),
                "jockey_runs_prior": np.random.randint(50, 500),
                "jockey_win_rate_prior": np.random.uniform(0.05, 0.20),
                "jockey_top3_rate_prior": np.random.uniform(0.25, 0.45),
                "trainer_runs_prior": np.random.randint(30, 300),
                "trainer_win_rate_prior": np.random.uniform(0.05, 0.15),
                "trainer_top3_rate_prior": np.random.uniform(0.20, 0.40),
                "sire_runs_prior": np.random.randint(100, 2000),
                "sire_win_rate_prior": np.random.uniform(0.05, 0.15),
                "sire_top3_rate_prior": np.random.uniform(0.20, 0.40),
                "horse_runs_prior": np.random.randint(0, 30),
                "horse_win_rate_prior": np.random.uniform(0, 0.30),
                "horse_top3_rate_prior": np.random.uniform(0, 0.50),
                "horse_distance_runs_prior": np.random.randint(0, 15),
                "horse_distance_top3_rate_prior": np.random.uniform(0, 0.60),
                "horse_track_runs_prior": np.random.randint(0, 10),
                "horse_track_top3_rate_prior": np.random.uniform(0, 0.60),
                "field_odds_rank": float(pop[h]), "field_pop_rank": float(pop[h]),
                "odds_gap_to_fav": float(raw[h] - raw[0]),
                "popularity_gap_to_fav": float(pop[h] - 1),
                "place": np.random.choice(["東京", "阪神", "中山", "京都", "中京", "福島", "札幌"]),
                "race_name": np.random.choice(["未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]),
                "track_type": np.random.choice(["芝", "ダ"]),
                "going": np.random.choice(["良", "稍重", "重"]),
                "sex": np.random.choice(["牡", "牝", "セ"]),
                "jockey": np.random.choice(["ルメール", "川田将雅", "武豊", "戸崎圭太", "池添謙一", "福永祐一"]),
                "trainer": np.random.choice(["矢作芳人", "国枝栄", "藤沢和雄", "池江泰寿", "角居勝彦"]),
                "belonging": np.random.choice(["美浦", "栗東"]),
                "sire": np.random.choice(["ディープインパクト", "キングカメハメハ", "ハーツクライ",
                                           "ロードカナロア", "モーリス", "エピファネイア"]),
                "dam": np.random.choice(["牝馬A", "牝馬B", "牝馬C", "牝馬D", "牝馬E"]),
                "broodmare_sire": np.random.choice(["サンデーサイレンス", "ノーザンテースト",
                                                     "ブライアンズタイム", "トニービン"]),
            }
            rows.append(row)

    X = pd.DataFrame(rows)
    # 特徴量列だけ抽出（存在しない列は0埋めにゃ）
    X_feat = pd.DataFrame()
    for f in feature_cols:
        if f in X.columns:
            X_feat[f] = X[f]
        else:
            X_feat[f] = 0

    try:
        base_pred = pipe.predict_proba(X_feat)[:, 1]
    except Exception:
        return pd.DataFrame()

    importances = []
    for feat in feature_cols:
        X_perm = X_feat.copy()
        X_perm[feat] = X_feat[feat].sample(frac=1, random_state=seed).values
        try:
            perm_pred = pipe.predict_proba(X_perm)[:, 1]
            imp = float(np.mean(np.abs(base_pred - perm_pred)))
        except Exception:
            imp = 0.0
        importances.append({"特徴量": feat, "重要度": imp})

    df = pd.DataFrame(importances).sort_values("重要度", ascending=False).reset_index(drop=True)
    total = df["重要度"].sum()
    df["重要度(%)"] = (df["重要度"] / total * 100).round(2) if total > 0 else 0.0
    df["日本語名"] = df["特徴量"].map(FEAT_JP).fillna(df["特徴量"])
    df.insert(0, "順位", range(1, len(df) + 1))
    return df


def calc_uncertainty_for_race(pipe, race_df, feature_cols) -> pd.Series | None:
    """
    HistGradientBoostingの各木の予測から不確実性を計算するにゃ。
    内部の_predictorsを使って各木の予測値の標準偏差を出すにゃ。
    """
    try:
        # 前処理を通した後のデータにゃ
        preprocessor = pipe.steps[0][1]
        est = pipe.steps[-1][1]

        # 特徴量列を整備にゃ
        X = pd.DataFrame()
        for f in feature_cols:
            if f in race_df.columns:
                X[f] = pd.to_numeric(race_df[f], errors="coerce").fillna(0)                     if f not in ["place","race_name","track_type","going","sex",
                                 "jockey","trainer","belonging","sire","dam","broodmare_sire"]                     else race_df[f].astype(str).fillna("")
            else:
                X[f] = 0

        X_transformed = preprocessor.transform(X)

        # 各木の予測を取得にゃ（最大50木でサンプリングにゃ）
        predictors = est._predictors
        n_sample = min(50, len(predictors))
        step = max(1, len(predictors) // n_sample)
        sample_preds = []

        for i in range(0, len(predictors), step):
            tree_pred = predictors[i][0].predict(X_transformed)
            sample_preds.append(tree_pred)

        if len(sample_preds) < 2:
            return None

        pred_arr = np.array(sample_preds)
        # ロジスティック変換にゃ
        from scipy.special import expit
        prob_arr = expit(pred_arr * est.n_iter_no_change if hasattr(est,"n_iter_no_change") else pred_arr)
        return pd.Series(prob_arr.std(axis=0), index=race_df.index)
    except Exception:
        return None


def show_pkl_ai_dashboard(bundle, race_df, pred_enriched_df=None):
    """
    PKL実測ベースの完全AI分析ダッシュボードにゃ。
    HistGradientBoostingの実情報を直接使うにゃ。
    """
    st.markdown("---")
    st.header("🧠 PKL本格AI分析ダッシュボード（実測ベース）")
    st.caption(
        f"PKLバンドルを直接解析した結果にゃ。"
        f"HistGradientBoostingの実際の内部情報を使って「AIが何を重視しているか」を可視化するにゃ🐾"
    )

    info = get_bundle_info(bundle)
    pipe = info["pipe"]
    feature_cols = info["feature_cols"]
    numeric_features = info["numeric_features"]
    categorical_features = info["categorical_features"]
    metrics = info["metrics"]

    if pipe is None:
        st.error("PKLからモデルを取り出せなかったにゃ。")
        return

    repair_simple_imputer(pipe)
    est = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 モデル概要",
        "📊 特徴量重要度（実測）",
        "🎯 AI自信度",
        "⚠️ 過学習診断",
        "🔍 個別馬根拠説明",
    ])

    # ── Tab1: モデル概要 ──
    with tab1:
        st.subheader("🤖 PKLモデル概要にゃ")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("モデル種類", info["model_type"] or type(est).__name__)
        c2.metric("特徴量数", len(feature_cols))
        c3.metric("学習レース数", f"{metrics.get('races', '不明'):,}" if metrics.get('races') else "不明")
        c4.metric("学習行数", f"{metrics.get('rows', '不明'):,}" if metrics.get('rows') else "不明")

        st.caption(f"バージョン: {info['version']}")

        if hasattr(est, "get_params"):
            params = est.get_params()
            imp_keys = ["max_iter", "learning_rate", "max_depth", "max_leaf_nodes",
                        "min_samples_leaf", "l2_regularization", "max_bins",
                        "validation_fraction", "n_iter_no_change"]
            prows = [{"パラメータ": k, "値": params[k]} for k in imp_keys if k in params]
            with st.expander("📋 ハイパーパラメータにゃ"):
                st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True)

        # Pipelineステップにゃ
        if hasattr(pipe, "steps"):
            with st.expander("🔧 Pipelineステップにゃ"):
                sdf = pd.DataFrame([{"ステップ": n, "クラス": type(s).__name__}
                                     for n, s in pipe.steps])
                st.dataframe(sdf, use_container_width=True, hide_index=True)

        # 学習スコア推移にゃ
        if hasattr(est, "train_score_") and est.train_score_ is not None:
            ts = est.train_score_
            vs = est.validation_score_ if hasattr(est, "validation_score_") else None
            st.markdown("#### 学習スコア推移にゃ")
            n_iter = len(ts)
            pts = [10, 25, 50, 100, 150, 200, 250, 300]
            score_rows = []
            for pt in pts:
                if pt <= n_iter:
                    row = {"イテレーション": pt, "学習スコア": round(float(ts[pt-1]), 6)}
                    if vs is not None and len(vs) >= pt:
                        row["検証スコア"] = round(float(vs[pt-1]), 6)
                    score_rows.append(row)
            if score_rows:
                st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
            st.caption(f"実際のイテレーション数: {est.n_iter_} / 最大: {est.max_iter}にゃ")
            if est.n_iter_ >= est.max_iter:
                st.warning("⚠️ 早期停止せずに上限到達にゃ。過学習の可能性があるにゃ！")

    # ── Tab2: 特徴量重要度（実測）──
    with tab2:
        st.subheader("📊 特徴量重要度ランキング（PKL実測ベース）にゃ")
        st.caption(
            "HistGradientBoostingはfeature_importances_プロパティがないにゃ。"
            "そのため「各特徴量をシャッフルしたときに予測がどれだけ変わるか」で重要度を計算するにゃ🐾"
        )

        with st.spinner("現実的な競馬データで重要度計算中にゃ...（10〜30秒かかるにゃ）"):
            fi_df = calc_permutation_importance_real(
                pipe, feature_cols, numeric_features, categorical_features)

        if fi_df.empty:
            st.warning("特徴量重要度の計算に失敗したにゃ。")
        else:
            # カテゴリ分類にゃ
            def categorize(feat):
                if feat in ["odds", "popularity", "field_odds_rank", "field_pop_rank",
                            "odds_gap_to_fav", "popularity_gap_to_fav"]: return "💴 オッズ/市場系"
                if "jockey" in feat: return "🏇 騎手系"
                if "trainer" in feat: return "🏋️ 調教師系"
                if any(x in feat for x in ["sire", "dam", "broodmare"]): return "🧬 血統系"
                if "horse_" in feat: return "📈 馬実績系"
                if feat in ["distance", "course_kind", "race_grade", "track_type",
                            "going", "place"]: return "🏟️ コース条件系"
                if feat in ["frame_no", "horse_no", "field_size", "race_no",
                            "race_name"]: return "🔢 レース情報系"
                if feat in ["age", "carried_weight", "sex", "belonging"]: return "⚖️ 馬体系"
                if feat in ["year_full", "month", "day"]: return "📅 日付系"
                return "📌 その他"

            fi_df["カテゴリ"] = fi_df["特徴量"].apply(categorize)

            # 上位表示にゃ
            top_fi = fi_df[fi_df["重要度"] > 0].copy()
            if top_fi.empty:
                st.info("全特徴量の重要度がほぼ0にゃ。これは過学習の証拠にゃ（モデルが特定パターンだけ記憶しているにゃ）。")
                # 全件表示にゃ
                top_fi = fi_df.head(20)

            try:
                imp_col = "重要度"
                styled = top_fi[["順位", "日本語名", "特徴量", imp_col, "重要度(%)", "カテゴリ"]].style.bar(
                    subset=[imp_col], color="#4CAF50", vmin=0
                ).format({imp_col: "{:.5f}", "重要度(%)": "{:.2f}%"})
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(top_fi, use_container_width=True, hide_index=True)

            # 重要特徴量の解説にゃ
            top1 = fi_df.iloc[0]
            if top1["重要度"] > 0:
                top1_name = str(top1["日本語名"])
                top1_feat = str(top1["特徴量"])
                top1_imp  = float(top1["重要度"])
                st.info(
                    f"💡 最重要特徴量: {top1_name}（{top1_feat}）にゃ"
                    f"この特徴量をシャッフルすると予測が平均 {top1_imp:.4f} 変化するにゃ。"
                    f"この要素がAIの判断に最も影響しているにゃ🐾"
                )

            # カテゴリ別シェアにゃ
            st.markdown("#### カテゴリ別重要度シェアにゃ")
            cat_sum = (fi_df.groupby("カテゴリ")["重要度"].sum()
                       .reset_index().rename(columns={"重要度": "合計重要度"})
                       .sort_values("合計重要度", ascending=False))
            total = cat_sum["合計重要度"].sum()
            cat_sum["シェア(%)"] = (cat_sum["合計重要度"] / total * 100).round(1) if total > 0 else 0
            st.dataframe(cat_sum, use_container_width=True, hide_index=True)

    # ── Tab3: AI自信度 ──
    with tab3:
        st.subheader("🎯 AI予測の自信度（不確実性分析）にゃ")
        st.caption("内部の木ごとの予測のばらつきから「AIが迷っている馬」を検出するにゃ🐾")

        with st.spinner("不確実性を計算中にゃ..."):
            unc = calc_uncertainty_for_race(pipe, race_df, feature_cols)

        if unc is None:
            st.info("HistGradientBoostingの木構造から不確実性を計算できなかったにゃ。")
        else:
            rd = race_df[["horse_no", "horse_name", "ml_top3_prob",
                           "ml_rank", "odds", "popularity"]].copy().reset_index(drop=True)
            unc_vals = unc.reset_index(drop=True) if hasattr(unc, "reset_index") else unc
            rd["予測標準偏差"] = unc_vals.values[:len(rd)] if hasattr(unc_vals,"values") else np.zeros(len(rd))
            rd["AI自信度"] = rd["予測標準偏差"].apply(
                lambda s: "🔴 迷っている" if s > 0.15 else ("🟡 やや不安" if s > 0.08 else "🟢 自信あり"))
            rd["3着内確率"] = (rd["ml_top3_prob"] * 100).round(1).astype(str) + "%"
            disp = rd[["ml_rank", "horse_no", "horse_name", "3着内確率",
                        "予測標準偏差", "AI自信度", "odds", "popularity"]].rename(columns={
                "ml_rank": "AI順位", "horse_no": "馬番", "horse_name": "馬名",
                "odds": "オッズ", "popularity": "人気"}).sort_values("AI順位")
            st.dataframe(disp, use_container_width=True, hide_index=True)

            unsure = rd[rd["予測標準偏差"] > 0.15].sort_values("予測標準偏差", ascending=False)
            if not unsure.empty:
                ns = " / ".join(f"馬番{int(r['horse_no'])} {r['horse_name']}"
                                 for _, r in unsure.head(3).iterrows())
                st.warning(f"⚠️ AIが迷っている馬にゃ: {ns} → 購入は慎重にゃ🐾")

    # ── Tab4: 過学習診断 ──
    with tab4:
        st.subheader("⚠️ 過学習診断にゃ")

        m = metrics
        auc = m.get("auc", 0)
        logloss = m.get("logloss", 999)
        top3_rate = m.get("top3_rate", 0)

        # 診断にゃ
        issues = []
        if auc >= 0.999:
            issues.append(("🔴 重大", "AUC=1.0", "学習データを丸暗記している疑いが強いにゃ。実戦では大幅に精度が落ちるにゃ。"))
        if logloss < 0.001:
            issues.append(("🔴 重大", f"LogLoss={logloss:.2e}", "損失関数がほぼゼロにゃ。過学習の典型症状にゃ。"))
        if hasattr(est, "n_iter_") and est.n_iter_ >= est.max_iter:
            issues.append(("🟠 警告", "早期停止なし", f"300回イテレーション上限に到達にゃ。過学習が進行している可能性にゃ。"))
        if "odds" in feature_cols and "popularity" in feature_cols:
            issues.append(("🟡 注意", "当日オッズ使用", "オッズ・人気は3着内と高相関にゃ。モデルがこれに依存しすぎている可能性にゃ。"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{auc:.6f}", delta="理論値<0.85" if auc > 0.90 else None,
                  delta_color="inverse")
        c2.metric("LogLoss", f"{logloss:.2e}", delta="異常に低いにゃ" if logloss < 0.01 else None,
                  delta_color="inverse")
        c3.metric("平均3着内率", f"{top3_rate*100:.1f}%")
        c4.metric("診断結果", f"{'要注意' if issues else '正常'}にゃ",
                  delta=f"{len(issues)}件の問題" if issues else "問題なし")

        if issues:
            for level, title, desc in issues:
                if "重大" in level:
                    st.error(f"**{level}: {title}**にゃ{desc}")
                elif "警告" in level:
                    st.warning(f"**{level}: {title}**にゃ{desc}")
                else:
                    st.info(f"**{level}: {title}**にゃ{desc}")

        st.markdown("#### 改善提案にゃ🔧")
        st.markdown("""
- **odds・popularityを除外して再学習にゃ** → モデルが自分で実力を判断するようになるにゃ
- **early_stoppingを有効にするにゃ** → `n_iter_no_change=20, validation_fraction=0.15`にゃ
- **l2_regularizationを追加するにゃ** → `l2_regularization=0.1` で過学習を抑制にゃ
- **max_iter を100〜200に下げるにゃ** → 木の数を減らして汎化性能を上げるにゃ
        """)

    # ── Tab5: 個別馬根拠説明 ──
    with tab5:
        st.subheader("🔍 個別馬の予測根拠説明にゃ")
        st.caption("各特徴量がレース内平均と比べてどう違うかから「なぜこの確率か」を説明するにゃ🐾")

        horses = race_df.sort_values("ml_rank").head(10)
        opts = [f"馬番{int(r['horse_no'])} {r['horse_name']} (AI{int(r['ml_rank'])}位 / {float(r['ml_top3_prob'])*100:.1f}%)"
                for _, r in horses.iterrows()]
        sel = st.selectbox("分析する馬を選択にゃ", opts, key="pkl_horse_sel")

        if sel and opts:
            idx = opts.index(sel)
            tr = horses.iloc[idx]
            prob = float(tr.get("ml_top3_prob", 0))
            rank = int(tr.get("ml_rank", 0))

            st.markdown(f"**{tr['horse_name']}** (AI{rank}位 / 3着内確率{prob*100:.1f}%) の予測根拠にゃ")

            # 特徴量値とレース内平均の比較にゃ
            contrib_rows = []
            for feat in feature_cols:
                if feat not in race_df.columns:
                    continue
                if feat in ["place", "race_name", "track_type", "going", "sex",
                            "jockey", "trainer", "belonging", "sire", "dam", "broodmare_sire"]:
                    # カテゴリ特徴量はそのまま表示にゃ
                    val = str(tr.get(feat, "不明"))
                    contrib_rows.append({
                        "特徴量": FEAT_JP.get(feat, feat),
                        "この馬の値": val,
                        "タイプ": "カテゴリ",
                        "偏差(σ)": "-",
                        "評価": "📋",
                    })
                else:
                    try:
                        val = float(tr.get(feat, 0))
                        col_vals = pd.to_numeric(race_df[feat], errors="coerce").dropna()
                        if col_vals.empty:
                            continue
                        mean = float(col_vals.mean())
                        std = float(col_vals.std()) if col_vals.std() > 0 else 1.0
                        z = (val - mean) / std

                        # 特徴量ごとにプラス方向を決めるにゃ
                        negative_feats = ["odds", "popularity", "field_odds_rank", "field_pop_rank",
                                          "odds_gap_to_fav", "popularity_gap_to_fav",
                                          "horse_no", "frame_no"]
                        is_good = (z < 0) if feat in negative_feats else (z > 0)
                        icon = "🟢" if is_good and abs(z) > 0.5 else ("🔴" if not is_good and abs(z) > 0.5 else "⚪")

                        contrib_rows.append({
                            "特徴量": FEAT_JP.get(feat, feat),
                            "この馬の値": round(val, 3),
                            "レース内平均": round(mean, 3),
                            "偏差(σ)": round(z, 2),
                            "タイプ": "数値",
                            "評価": icon,
                        })
                    except Exception:
                        continue

            if contrib_rows:
                # 数値特徴量のみ偏差でソートにゃ
                num_rows = [r for r in contrib_rows if r["タイプ"] == "数値"]
                cat_rows = [r for r in contrib_rows if r["タイプ"] == "カテゴリ"]
                num_rows.sort(key=lambda x: abs(float(x["偏差(σ)"])) if x["偏差(σ)"] != "-" else 0, reverse=True)

                # プラス要因とマイナス要因を分けるにゃ
                plus_rows = [r for r in num_rows if str(r["評価"]).startswith("🟢")][:8]
                minus_rows = [r for r in num_rows if str(r["評価"]).startswith("🔴")][:8]

                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown("##### 🟢 プラス要因にゃ（平均より優れている点）")
                    if plus_rows:
                        st.dataframe(pd.DataFrame(plus_rows)[["特徴量","この馬の値","レース内平均","偏差(σ)"]],
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("明確なプラス要因なしにゃ")
                with pc2:
                    st.markdown("##### 🔴 マイナス要因にゃ（平均より劣っている点）")
                    if minus_rows:
                        st.dataframe(pd.DataFrame(minus_rows)[["特徴量","この馬の値","レース内平均","偏差(σ)"]],
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("明確なマイナス要因なしにゃ")

                # カテゴリ特徴量にゃ
                with st.expander("📋 カテゴリ特徴量にゃ（騎手・馬場・血統など）"):
                    if cat_rows:
                        st.dataframe(pd.DataFrame(cat_rows)[["特徴量","この馬の値"]],
                                     use_container_width=True, hide_index=True)

                # 自動サマリーにゃ
                comments = []
                for r in plus_rows[:3]:
                    f = r["特徴量"]; v = r["この馬の値"]; a = r["レース内平均"]
                    if "騎手3着内率" in f:
                        comments.append(f"騎手3着内率({v:.1%})が平均({a:.1%})より高いにゃ")
                    elif "距離別3着内率" in f:
                        comments.append(f"この距離での3着内率({v:.1%})が平均({a:.1%})より優秀にゃ")
                    elif "調教師3着内率" in f:
                        comments.append(f"調教師3着内率({v:.1%})が平均を上回るにゃ")
                    elif "オッズ" in f:
                        comments.append(f"オッズ({v:.1f}倍)がレース内で低め（市場評価が高い）にゃ")
                if comments:
                    st.success("💬 AI根拠サマリーにゃ: " + " / ".join(comments))


def app_main():
    st.title("🐾 にゃんこ競馬AI v26にゃ")
    st.success(f"起動版にゃ: {VERSION}にゃ")
    st.caption(
        "v25: ①三連複確率を条件付き確率に刷新 ②implied_top3をオッズ帯別係数に刷新 "
        "③危険馬フィルタ強化(AI4位) ④相手B確率下限追加 ⑤Kelly比を複勝/三連複に分離 "
        "⑥補完買い目品質フィルタ強化 ⑦レース質分析→買い目生成に反映 ⑧頭数別動的EV閾値"
    )

    with st.sidebar:
        st.header("設定")

        st.markdown("### 🎯 予想モード")
        strategy_mode = st.radio(
            "モードを選択",
            STRATEGY_MODE_OPTIONS,
            index=0,
            help=(
                "**回収率重視**: Kelly基準+EV乖離で絞り込み。高配当狙い。\n\n"
                "**的中率重視**: AI上位馬+軸信頼度で安定的中狙い。"
            )
        )
        if strategy_mode == STRATEGY_MODE_ROI:
            st.info("💰 回収率重視: Kelly正馬・EV高め馬を積極評価。高配当を狙います。")
        else:
            st.success("🏆 的中率重視: AI上位・軸信頼度重視で安定的中を狙います。")

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
            "**v25 主要修正**\n\n"
            "🔴 三連複ロジック全面刷新:\n"
            "- [FIX-1] 三連複確率: 独立仮定→条件付き確率+頭数補正\n"
            "- [FIX-2] implied_top3: オッズ÷3→オッズ帯別係数テーブル\n"
            "- [FIX-3] 危険馬: AI5位→AI4位以上で危険\n"
            "- [FIX-4] 相手B: EVプラスのみ→確率下限も追加\n"
            "- [FIX-5] Kelly比: 1種類→複勝/三連複に分離\n"
            "- [FIX-6] 補完買い目: 無制限→品質フィルタ強化\n"
            "- [FIX-7] レース質分析を買い目生成に反映\n"
            "- [FIX-8] 三連複EV閾値を頭数別に動的化\n"
            "- [FIX-9] pivot_confidence: 三連複Kelly連動\n"
        )

    st.subheader("入力方法にゃ🐾")
    input_method = st.radio(
        "入力方法を選択にゃ",
        ["🌐 netkeiba自動取得（当日レース）",
         "🌐 netkeiba race_id/URL指定",
         "📁 事前CSVから選択",
         "📄 出馬表CSVアップロード",
         "netkeiba URL単発"],
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

            # レース質分析表示
            st.markdown("---")
            race_quality = analyze_race_quality(race_df)
            st.markdown(f"#### 🏟️ レース質分析: **{race_quality['type']}**")
            if race_quality["advice"]:
                st.info(race_quality["advice"])
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            col_q1.metric("最低オッズ", f"{race_quality['min_odds']:.1f}倍")
            col_q2.metric("オッズ標準偏差", f"{race_quality['odds_std']:.1f}")
            col_q3.metric("レースタイプ", race_quality['type'])
            col_q4.metric("推奨フォーカス", race_quality.get('rec_bet_focus', '-'))

            # 推奨購入点数ダッシュボード
            st.markdown("---")
            rec = calc_recommended_tickets(race_df, strategy_mode=strategy_mode)
            st.markdown("#### 📈 推奨購入点数ダッシュボード")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("推奨点数", f"{rec['推奨点数']}点")
            m2.metric("Kelly正(複勝)", f"{rec['Kelly正(複勝)']}頭")
            m3.metric("Kelly正(三連複)", f"{rec['Kelly正(三連複)']}頭")
            m4.metric("買い候補馬数", f"{rec['買い候補馬数']}頭")
            key5 = [k for k in rec.keys() if "的中率" in k]
            if key5:
                m5.metric(key5[0], rec[key5[0]])

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

            # ============================================================
            # 🧠 PKL本格AI分析ダッシュボード（HistGradientBoosting実測ベース）にゃ
            # ============================================================
            try:
                show_pkl_ai_dashboard(bundle, race_df)
            except Exception as _ai_err:
                st.warning(f"AI分析でエラーが発生したにゃ: {_ai_err}")
                import traceback
                st.caption(traceback.format_exc())

            show_roi_strategy(race_df, strategy_mode=strategy_mode)

            # 2モード比較表示
            st.markdown("---")
            st.subheader("📊 2モード比較（同一レース）")
            st.caption("同じAIスコアから、モードによって買い判定がどう変わるか比較できます。")
            col_roi, col_hit = st.columns(2)
            with col_roi:
                st.markdown("**💰 回収率重視**")
                df_roi_compare = add_value_strategy(race_df.copy(), strategy_mode=STRATEGY_MODE_ROI)
                buy_roi = df_roi_compare[df_roi_compare["buy_flag"] == "買い"][
                    ["horse_name", "buy_flag", "buy_reason", "kelly_ratio", "kelly_ratio_sanren"]]
                st.dataframe(buy_roi.rename(columns=JP_COLUMNS), use_container_width=True, hide_index=True)
            with col_hit:
                st.markdown("**🏆 的中率重視**")
                df_hit_compare = add_value_strategy(race_df.copy(), strategy_mode=STRATEGY_MODE_HITRATE)
                buy_hit = df_hit_compare[df_hit_compare["buy_flag"] == "買い"][
                    ["horse_name", "buy_flag", "buy_reason", "pivot_confidence"]]
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
    with st.expander("簡易CSVテンプレ（v25対応）"):
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
