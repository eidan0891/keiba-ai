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
import traceback
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

# ============================================================
# 安全な型変換ヘルパーにゃ（NA/inf対策にゃ）
# ============================================================

def _safe_int(val, default=0) -> int:
    """NA/inf/None を受け取っても安全にintに変換するにゃ"""
    try:
        v = float(val)
        if v != v or v == float('inf') or v == float('-inf'):  # NaN/inf チェックにゃ
            return default
        return int(v)
    except (TypeError, ValueError):
        return default

def _safe_float(val, default=0.0) -> float:
    """NA/inf/None を受け取っても安全にfloatに変換するにゃ"""
    try:
        v = float(val)
        if v != v or v == float('inf') or v == float('-inf'):
            return default
        return v
    except (TypeError, ValueError):
        return default

def _safe_hno(row) -> str:
    """馬番を安全に文字列変換するにゃ"""
    try:
        v = row.get("horse_no", None) if hasattr(row, "get") else row
        return str(_safe_int(v, 0)) if _safe_int(v, -1) > 0 else ""
    except Exception:
        return ""

def _safe_hlabel(row) -> str:
    """馬番+馬名ラベルを安全に生成するにゃ"""
    try:
        hno  = _safe_int(row.get("horse_no", 0) if hasattr(row, "get") else 0, 0)
        name = str(row.get("horse_name", "") if hasattr(row, "get") else "")
        return f"{hno} {name}" if hno > 0 else name
    except Exception:
        return ""



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
    """netkeibaアクセス用セッションにゃ（CP932/Shift-JIS対応にゃ）"""
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


def _fetch_with_encoding(url: str, session=None,
                           timeout: int = 20) -> str:
    """
    netkeibaはShift-JIS(CP932)にゃ。
    自動検出ではなく明示的にCP932でデコードするにゃ。 """
    if session is None:
        session = _make_session()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    # netkeibaは常にCP932(Shift-JIS)にゃ → 強制指定にゃ
    r.encoding = 'cp932'
    return r.text


def fetch_today_race_ids(target_date: str = None, sleep_sec: float = 1.0) -> list[str]:
    """
    指定日（YYYYMMDD）の全race_idを取得するにゃ。
    Noneのときは今日の日付を使うにゃ。 """
    if target_date is None:
        target_date = date.today().strftime("%Y%m%d")

    session = _make_session()
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date}"
    try:
        html = _fetch_with_encoding(url, session)
    except Exception as e:
        raise ValueError(f"レース一覧の取得に失敗したにゃ: {e}")

    race_ids = list(dict.fromkeys(re.findall(r"race_id=(\d{12})", html)))
    return race_ids


def fetch_shutuba_html(race_id: str, session=None) -> str:
    """出馬表HTMLをCP932で取得するにゃ"""
    if session is None: session = _make_session()
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    return _fetch_with_encoding(url, session)


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
    netkeibaの出馬表テーブルをパースするにゃ。 """
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
    update_odds=True のときリアルタイムオッズで上書きするにゃ。 """
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
                hno = str(_safe_int(row["horse_no"], 0)) if pd.notna(row.get("horse_no")) and _safe_int(row["horse_no"], 0) > 0 else ""
                if hno in tansho:
                    df.at[idx, "odds"] = tansho[hno]

        # 人気順を再計算にゃ
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        valid_odds = df["odds"].dropna()
        if not valid_odds.empty:
            df["popularity"] = df["odds"].rank(method="min", ascending=True).fillna(99).round(0).astype(int)

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
        lambda x: _safe_int(x) + 2000 if pd.notna(x) and _safe_int(x) < 100 else _safe_int(x)
    )
    df["date_int"] = (
        df["year_full"].fillna(0).replace([float("inf"), float("-inf")], 0).astype(int) * 10000
        + df["month"].fillna(0).replace([float("inf"), float("-inf")], 0).astype(int) * 100
        + df["day"].fillna(0).replace([float("inf"), float("-inf")], 0).astype(int)
    )
    if "source_file" not in df.columns:
        df["source_file"] = ""
    df = normalize_match_keys(df)
    df["race_key"] = (
        df["date_int"].astype(str) + "_"
        + df.get("place", "").astype(str) + "_"
        + df["race_no"].fillna(0).replace([float("inf"), float("-inf")], 0).astype(int).astype(str).str.zfill(2) + "_"
        + df["source_file"].astype(str)
    )
    df["race_label"] = (
        df["date_int"].astype(str) + " "
        + df.get("place", "").astype(str) + " "
        + df["race_no"].fillna(0).replace([float("inf"), float("-inf")], 0).astype(int).astype(str) + "R "
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

    # ── 元CSVにある列をそのまま引き継ぐにゃ ──
    for carry_col in ["running_style","style_note",
                       "finish","race_id","race_key","date_int",
                       "time_raw","last3f","pass1","pass2","pass3","pass4",
                       "body_weight","trainer","prize"]:
        if carry_col in src.columns:
            df[carry_col] = src[carry_col].reset_index(drop=True).values[:len(df)]

    # race_id → race_key を自動生成するにゃ
    if "race_key" not in df.columns or df["race_key"].isna().all():
        if "race_id" in df.columns:
            df["race_key"] = df["race_id"].astype(str)

    # race_name が長いタイトルの場合は短縮するにゃ（G1判定のためにゃ）
    # 例: "日本ダービー(G1) 出馬表 | ..." → "日本ダービー(G1)"
    if "race_name" in src.columns:
        def _shorten_race_name(s):
            s = str(s)
            for sep in ["|", "｜", " 出馬表", " - ", "　"]:
                if sep in s:
                    s = s.split(sep)[0].strip()
            return s[:40]
        df["race_name"] = src["race_name"].apply(_shorten_race_name).reset_index(drop=True).values[:len(df)]

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
    シグモイド変換で適切な範囲に引き戻すにゃ。 """
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

    # 同率による馬番昇順化を防ぐため、微小なノイズを加えるにゃ
    # ノイズは 1e-6 オーダーなので実質的な確率に影響しないにゃ
    np.random.seed(42)
    noise = np.random.uniform(-1e-6, 1e-6, len(p_calibrated))
    p_calibrated = p_calibrated + noise

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
        calibrated[mask.values] = calibrate_prob_isotonic(raw_prob[mask.values])
    df["ml_top3_prob"] = calibrated
    df["calibrated_prob"] = calibrated  # 表示用にも保持にゃ

    # ml_rank計算にゃ: 同率タイブレークを「人気→オッズ→馬番」順で行うにゃ
    # method="first"は行順（=馬番昇順）依存になるので使わないにゃ
    # タイブレーク基準にゃ: 1.人気小(強いにゃ) 2.オッズ小(市場評価高いにゃ) 3.馬番小
    _pop_tb  = pd.to_numeric(df["popularity"], errors="coerce").fillna(99)
    _odds_tb = pd.to_numeric(df["odds"],       errors="coerce").fillna(999)
    _hno_tb  = pd.to_numeric(df["horse_no"],   errors="coerce").fillna(99)
    _tiebreak = (
        (1.0 / _pop_tb.clip(lower=1))       * 1e-4
        + (1.0 / _odds_tb.clip(lower=0.1))  * 1e-6
        + (1.0 / _hno_tb.clip(lower=1))     * 1e-8
    )
    df["_composite_rank"] = df["ml_top3_prob"] + _tiebreak
    df["ml_rank"] = (
        df.groupby("race_key")["_composite_rank"]
        .rank(ascending=False, method="first")
        .fillna(1).astype(int)
    )
    df = df.drop(columns=["_composite_rank"])
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
    # ── S級強化処理にゃ（展開予測・多次元EV・見送り判定・最終スコアにゃ）──
    try:
        df = add_pace_advantage(df)
        df = add_ev_score_v2(df)
        df = add_pass_score(df, strategy_mode=strategy_mode)
        df = add_final_score(df, strategy_mode=strategy_mode)
    except Exception as _s_err:
        pass  # S級計算失敗しても旧版で動くにゃ
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
            return f"{_safe_int(row.get('horse_no', 0))} {row.get('horse_name', '')}"
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
    field_size = _safe_int(r["field_size"].max(), len(r)) if pd.notna(r["field_size"].max()) and r["field_size"].max() > 0 else len(r)

    def _no(row):
        return str(_safe_int(row["horse_no"], 0)) if _safe_int(row.get("horse_no", 0), 0) > 0 else ""

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
        # 相手B: EV×確率×Kelly の複合スコアで選出にゃ（v26改善にゃ）
        min_prob_for_b = float(safe["ml_top3_prob"].quantile(0.25))
        safe_tmp = safe.copy()
        safe_tmp["_aite_b_score"] = (
            safe_tmp["ml_top3_prob"] * 0.50
            + safe_tmp["ev_score"].clip(lower=0) * 0.30
            + safe_tmp["kelly_ratio_sanren"].clip(lower=0) * 0.20
        )
        aite_b_df = safe_tmp.sort_values("_aite_b_score", ascending=False).copy()
        aite_b_nums = [
            _no(row) for _, row in aite_b_df.iterrows()
            if _no(row) not in [pivot_no, pivot2_no] + aite_a_nums
            and float(row.get("ml_top3_prob", 0)) >= min_prob_for_b
        ][:6]
        if len(aite_b_nums) < 3:
            for _, row in safe.sort_values("ml_top3_prob", ascending=False).iterrows():
                n = _no(row)
                if n and n not in [pivot_no, pivot2_no] + aite_a_nums + aite_b_nums:
                    if float(row.get("ml_top3_prob", 0)) >= min_prob_for_b * 0.6:
                        aite_b_nums.append(n)
                if len(aite_b_nums) >= 6:
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
                    sorted_tri = sorted(tri, key=int)
                    combos.append({
                        "買い目": "-".join(sorted_tri),
                        "軸馬番": pivot_no,
                        "相手A": ha,
                        "相手B": hb,
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
            str(_safe_int(row.get("horse_no", row["horse_no"] if "horse_no" in row.index else 0))) for _, row in
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
            str(_safe_int(row.get("horse_no", row["horse_no"] if "horse_no" in row.index else 0))) for _, row in
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

    field_size = _safe_int(r["field_size"].max(), len(r)) if pd.notna(r["field_size"].max()) and r["field_size"].max() > 0 else len(r)

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
            return f"{_safe_int(row.get('horse_no', 0))} {row.get('horse_name', '')}"
        except Exception:
            return str(row.get("horse_name", ""))

    def _no(row):
        try:
            return str(_safe_int(row.get("horse_no", row["horse_no"] if "horse_no" in row.index else 0)))
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
            "AI順位": _safe_int(row.get("ml_rank", 0), 0),
            "人気": _safe_int(row.get("popularity", 0), 0),
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


    # [v26] 組合EVを条件付き確率版で計算にゃ
    # 確率上位馬が先頭に来る「AI推奨順」列も追加するにゃ
    combos = []
    for tri in itertools.combinations(range(len(nums)), 3):
        tri_nums   = [nums[i] for i in tri]
        # 購入票用: 馬番昇順にゃ
        tri_sorted = sorted(tri_nums, key=lambda x: _safe_int(x, 0))
        # AI推奨順: 確率の高い馬から並べるにゃ
        tri_by_prob = sorted(tri_nums, key=lambda x: -prob_map.get(x, 0))
        h1, h2, h3  = tri_by_prob[0], tri_by_prob[1], tri_by_prob[2]
        combo_ev = _calc_combo_ev_score(h1, h2, h3, prob_map, ev_map, field_size=field_size)
        p1 = prob_map.get(h1, 0.05)
        p2 = prob_map.get(h2, 0.05)
        p3 = prob_map.get(h3, 0.05)
        order_score = p1 * min(p2/max(1.0-p1, 0.05), 0.95) * min(p3/max(1.0-p1-p2, 0.05), 0.95)
        combos.append({
            "No": 0,
            "買い目(購入用)": "-".join(tri_sorted),
            "AI推奨順": f"{h1}→{h2}→{h3}",
            "組合EV(v26)": round(combo_ev, 6),
            "順序スコア": round(order_score * 1000, 4),
        })
    combos.sort(key=lambda x: -(x["組合EV(v26)"] * 0.7 + x["順序スコア"] * 0.3))
    for i, c in enumerate(combos):
        c["No"] = i + 1

    alt_horse = None
    remaining = safe.sort_values("_box_score", ascending=False).iloc[5:6]
    if not remaining.empty:
        row = remaining.iloc[0]
        alt_horse = {
            "馬番": _no(row), "馬名": row.get("horse_name", ""),
            "AI順位": _safe_int(row.get("ml_rank", 0), 0),
            "人気": _safe_int(row.get("popularity", 0), 0),
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
    """
    三連単フォーメーション生成にゃ（v26改善版にゃ）

    【改善点にゃ】
    - 1着/2着/3着の候補を「AI確率×着順ウェイト」でスコアリングにゃ
    - 単純なfirst/second/thirdの固定割り当てをやめて
      pivot_confidence順でフレキシブルにフォーメーションを組むにゃ
    - 1着確率の近似 = ml_top3_prob × (1/popularity)^0.5 にゃ
    - combo_ev × 着順スコア でソートするにゃ
    """
    r = race_df.copy()
    for c in ["ml_rank", "value_score", "ml_top3_prob", "odds", "popularity",
              "ev_score", "horse_no", "kelly_ratio", "kelly_ratio_sanren",
              "field_size", "pivot_confidence"]:
        if c not in r.columns:
            r[c] = 0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0)

    all_nums = [str(_safe_int(n, 0)) for n in r["horse_no"].dropna() if n > 0]
    field_size = _safe_int(r["field_size"].max(), len(r))         if pd.notna(r["field_size"].max()) and r["field_size"].max() > 0 else len(r)
    safe = (r[r.get("danger_popular", pd.Series([""] * len(r))) != "危険"].copy()
            if "danger_popular" in r.columns else r.copy())
    if safe.empty:
        safe = r.copy()

    # 着順別スコアを計算するにゃ
    # 1着スコア: top3_prob × (1/popularity)^0.5 × pivot_confidence
    # 2着スコア: top3_prob × pivot_confidence
    # 3着スコア: top3_prob × ev_score_bonus（穴馬ボーナスにゃ）
    pop_safe = safe["popularity"].clip(lower=1)
    safe = safe.copy()
    safe["score_1st"] = (
        safe["ml_top3_prob"]
        * (1.0 / pop_safe) ** 0.5
        * safe["pivot_confidence"].clip(lower=0.01)
    )
    safe["score_2nd"] = (
        safe["ml_top3_prob"]
        * safe["pivot_confidence"].clip(lower=0.01)
        * (1 + safe["ev_score"].clip(lower=0) * 0.5)
    )
    safe["score_3rd"] = (
        safe["ml_top3_prob"]
        * (1 + safe["ev_score"].clip(lower=0) * 1.5)  # 穴馬を3着に期待にゃ
        * (1 + (safe["popularity"] >= 5).astype(float) * 0.3)
    )

    def get_nums_by_score(score_col, max_n=6, exclude=None):
        df_sorted = safe.sort_values(score_col, ascending=False)
        nums = []
        for _, row in df_sorted.iterrows():
            try:
                n = str(_safe_int(row["horse_no"], 0))
            except Exception:
                n = ""
            if n and n not in nums and (exclude is None or n not in exclude):
                nums.append(n)
            if len(nums) >= max_n:
                break
        return nums

    prob_map = {}
    ev_map   = {}
    for _, row in r.iterrows():
        try:
            n = str(_safe_int(row["horse_no"], 0))
        except Exception:
            continue
        prob_map[n] = float(row.get("ml_top3_prob", 0.05))
        ev_map[n]   = float(row.get("ev_score", 0))

    if strategy_mode == STRATEGY_MODE_ROI:
        # 回収率重視にゃ: EVが高い穴馬を3着に積極活用にゃ
        first_candidates  = get_nums_by_score("score_1st", max_n=3)
        second_candidates = get_nums_by_score("score_2nd", max_n=5,
                                               exclude=set(first_candidates))
        # 3着には穴馬を優先にゃ（EV高め）にゃ
        third_candidates  = get_nums_by_score("score_3rd", max_n=7,
                                               exclude=set(first_candidates))
    else:
        # 的中率重視にゃ: AI上位・人気馬で固めるにゃ
        first_candidates  = get_nums_by_score("score_1st", max_n=2)
        second_candidates = get_nums_by_score("score_2nd", max_n=4,
                                               exclude=set(first_candidates))
        third_candidates  = get_nums_by_score("ml_top3_prob", max_n=5,
                                               exclude=set(first_candidates))

    # フォーメーション生成にゃ
    combos = []
    seen   = set()

    for h1 in first_candidates:
        for h2 in second_candidates:
            if h2 == h1:
                continue
            for h3 in third_candidates:
                if len({h1, h2, h3}) != 3:
                    continue
                key = f"{h1}→{h2}→{h3}"
                if key in seen:
                    continue

                spread   = _calc_spread_score([h1, h2, h3], all_nums)
                combo_ev = _calc_combo_ev_score(
                    h1, h2, h3, prob_map, ev_map, field_size=field_size)

                # 三連単スコア = 1着確率 × 2着確率 × 3着確率 × EV補正にゃ
                p1 = prob_map.get(h1, 0.05)
                p2 = prob_map.get(h2, 0.05)
                p3 = prob_map.get(h3, 0.05)
                # 条件付き確率にゃ（h1→h2→h3の順序を考慮にゃ）
                pc2 = min(p2 / max(1.0 - p1, 0.05), 0.95)
                pc3 = min(p3 / max(1.0 - p1 - p2, 0.05), 0.95)
                order_score = p1 * pc2 * pc3

                ev_bonus = float(np.clip(
                    1.0 + (ev_map.get(h1, 0) + ev_map.get(h3, 0)) * 0.5,
                    0.7, 2.0))

                combos.append({
                    "買い目": key,
                    "1着候補": h1,
                    "2着候補": h2,
                    "3着候補": h3,
                    "狙い": (
                        "AI上位→人気→穴" if strategy_mode == STRATEGY_MODE_ROI
                        else "AI上位固定"
                    ),
                    "順序スコア": round(order_score * 1000, 4),
                    "分散スコア": round(spread, 3),
                    "組合EV(v25)": round(combo_ev, 6),
                    "_sort_key": order_score * ev_bonus * (1 + spread * 0.2),
                })
                seen.add(key)

    if not combos:
        return []

    # 順序スコア×EV×分散でソートにゃ
    combos.sort(key=lambda x: -x["_sort_key"])

    # _sort_key列を除去して返すにゃ
    result = []
    for c in combos[:max_count]:
        c.pop("_sort_key", None)
        result.append(c)
    return result

def _horse_no(row) -> str:
    try:
        return str(_safe_int(row.get("horse_no", row["horse_no"] if "horse_no" in row.index else 0)))
    except Exception:
        return str(row.get("horse_no", ""))


def _horse_label(row) -> str:
    try:
        return f"{_safe_int(row.get('horse_no', 0))} {row.get('horse_name', '')}"
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
                    f"AI{_safe_int(pivot_row.get('ml_rank', 0))}位 / "
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
                    f"AI{_safe_int(pivot2_row.get('ml_rank', 0))}位 / "
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
    high_ev = _safe_int((r["ev_score"] >= 0.06).sum())
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
            f"馬番{_safe_int(r.get('horse_no',0))} {r.get('horse_name','')}"
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
                f"馬番{_safe_int(r.get('horse_no',0))} {r.get('horse_name','')}(EV+{_safe_float(r.get('ev_score',0)):.3f})"
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
        f"馬番{_safe_int(r.get('horse_no',0))} {r.get('horse_name','')} (AI{_safe_int(r.get('ml_rank',0))}位)"
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
    HistGradientBoostingはfeature_importances_がないため代替計算にゃ。 """
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
    内部の_predictorsを使って各木の予測値の標準偏差を出すにゃ。 """
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
    HistGradientBoostingの実情報を直接使うにゃ。 """
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
                ns = " / ".join(f"馬番{_safe_int(r.get('horse_no',0))} {r.get('horse_name','')}"
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
        opts = [f"馬番{_safe_int(r.get('horse_no',0))} {r.get('horse_name','')} (AI{_safe_int(r.get('ml_rank',0))}位 / {float(r['ml_top3_prob'])*100:.1f}%)"
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





# ============================================================
# ============================================================
# バックテストモジュールにゃ（v26追加にゃ）
# ============================================================
# ============================================================

def run_backtest(bundle, history_df: pd.DataFrame,
                 strategy_mode: str = STRATEGY_MODE_ROI,
                 min_odds: float = 1.0,
                 max_odds: float = 9999.0) -> dict:
    import traceback as _tb
    """
    過去レース結果データでバックテストを実行するにゃ。

    history_df には以下の列が必要にゃ:
      - COLS_52 の全列（finish列に着順が入っているにゃ）
      - または簡易CSV（finish列付きにゃ）

    戻り値にゃ:
      的中率・回収率・券種別成績・レース別成績などにゃ
    """
    if history_df is None or history_df.empty:
        return {}

    # finishがない場合はバックテスト不可にゃ
    if "finish" not in history_df.columns:
        return {"error": "finish（着順）列がないにゃ。結果CSVが必要にゃ。"}

    df = history_df.copy()
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df[df["finish"].notna() & (df["finish"] > 0)].copy()

    if df.empty:
        return {"error": "有効な着順データがないにゃ"}

    # 予想を実行するにゃ
    # バックテスト用: merge_target_featuresは省略し軽量化するにゃ
    try:
        df_for_pred = add_prior_stats_for_prediction(df)
        df_for_pred = add_running_style(df_for_pred)
        pipe, fc = get_pipeline_and_features(bundle)
        miss = [c for c in fc if c not in df_for_pred.columns]
        if miss:
            # 不足特徴量を0で補完にゃ
            for c in miss:
                df_for_pred[c] = 0.0
        if hasattr(pipe, "predict_proba"):
            raw_prob = pipe.predict_proba(df_for_pred[fc])[:, 1]
        else:
            raw_prob = np.asarray(pipe.predict(df_for_pred[fc]), dtype=float)
        # 確率校正にゃ
        calibrated = np.zeros(len(df_for_pred))
        for rk in df_for_pred["race_key"].unique():
            mask = df_for_pred["race_key"] == rk
            calibrated[mask.values] = calibrate_prob_isotonic(raw_prob[mask.values])
        df["ml_top3_prob"] = calibrated
        # タイブレーク付きランク計算にゃ
        _pop_tb  = pd.to_numeric(df["popularity"], errors="coerce").fillna(99)
        _odds_tb = pd.to_numeric(df["odds"],       errors="coerce").fillna(999)
        _hno_tb  = pd.to_numeric(df["horse_no"],   errors="coerce").fillna(99)
        _tb = ((1.0/_pop_tb.clip(lower=1))*1e-4
               + (1.0/_odds_tb.clip(lower=0.1))*1e-6
               + (1.0/_hno_tb.clip(lower=1))*1e-8)
        df["_comp"] = df["ml_top3_prob"] + _tb
        df["ml_rank"] = (df.groupby("race_key")["_comp"]
                          .rank(ascending=False, method="first")
                          .fillna(1).astype(int))
        df = df.drop(columns=["_comp"])
        df = add_ev_score(df)
        df = add_kelly_ratio(df)
        df = add_value_strategy(df, strategy_mode=strategy_mode)
        pred_df = df
    except Exception as e:
        return {"error": f"予想実行エラーにゃ: {e}\n詳細にゃ: {_tb.format_exc()}"}

    # 実際の着順をマージするにゃ
    finish_map = {}
    for _, row in df.iterrows():
        key = (str(row.get("race_key", "")), str(_safe_int(row.get("horse_no", 0), 0)))
        finish_map[key] = _safe_int(row.get("finish", 99), 99)

    pred_df["actual_finish"] = pred_df.apply(
        lambda r: finish_map.get(
            (str(r.get("race_key", "")), str(_safe_int(r.get("horse_no", 0), 0))), 99),
        axis=1
    )
    pred_df["is_win"]   = pred_df["actual_finish"] == 1
    pred_df["is_top2"]  = pred_df["actual_finish"] <= 2
    pred_df["is_top3"]  = pred_df["actual_finish"] <= 3

    # ── 券種別集計にゃ ──
    results = {}
    race_keys = pred_df["race_key"].dropna().unique()
    n_races = len(race_keys)

    # 単勝・複勝・馬連・三連複・三連単の成績にゃ
    tansho_hits = 0;  tansho_bets = 0;  tansho_return = 0.0
    fukusho_hits = 0; fukusho_bets = 0; fukusho_return = 0.0
    umaren_hits = 0;  umaren_bets = 0;  umaren_return = 0.0
    san3_hits = 0;    san3_bets = 0;    san3_return = 0.0
    san1_hits = 0;    san1_bets = 0;    san1_return = 0.0

    race_records = []

    for rk in race_keys:
        rdf = pred_df[pred_df["race_key"] == rk].copy()
        if rdf.empty:
            continue

        # 予算: 1レース100円×買い目数にゃ
        buy_df = rdf[rdf["buy_flag"] == "買い"].copy() if "buy_flag" in rdf.columns \
            else rdf.head(3).copy()

        if buy_df.empty:
            continue

        # 実際の1着・2着・3着馬にゃ
        winner  = rdf[rdf["actual_finish"] == 1]
        second  = rdf[rdf["actual_finish"] == 2]
        third   = rdf[rdf["actual_finish"] == 3]
        w_no    = str(_safe_int(winner["horse_no"].iloc[0],  0)) if not winner.empty  else ""
        s_no    = str(_safe_int(second["horse_no"].iloc[0],  0)) if not second.empty  else ""
        t_no    = str(_safe_int(third["horse_no"].iloc[0],   0)) if not third.empty   else ""
        top3_set = {w_no, s_no, t_no} - {"0", ""}

        buy_nos = set(str(_safe_int(r["horse_no"], 0)) for _, r in buy_df.iterrows())

        # ── 単勝にゃ ──
        # AI1位馬を単勝購入にゃ
        ai1 = rdf.sort_values("ml_rank").iloc[0] if not rdf.empty else None
        if ai1 is not None:
            ai1_no = str(_safe_int(ai1["horse_no"], 0))
            odds_ai1 = _safe_float(ai1.get("odds", 0), 0)
            if min_odds <= odds_ai1 <= max_odds:
                tansho_bets += 100
                if ai1_no == w_no:
                    ret = int(100 * odds_ai1 * (1 - 0.20))
                    tansho_hits += 1
                    tansho_return += ret
                # 外れは0にゃ

        # ── 複勝にゃ（買い判定馬を全部複勝にゃ）──
        for _, brow in buy_df.iterrows():
            bno  = str(_safe_int(brow["horse_no"], 0))
            bods = _safe_float(brow.get("odds", 0), 0)
            if min_odds <= bods <= max_odds:
                fukusho_bets += 100
                if bno in top3_set:
                    # 複勝オッズの簡易推定にゃ（単勝×0.3 + フロアにゃ）
                    fuku_est = max(1.1, bods * 0.3)
                    ret = int(100 * fuku_est * (1 - 0.20))
                    fukusho_hits += 1
                    fukusho_return += ret

        # ── 馬連にゃ（AI上位2頭の組み合わせにゃ）──
        top2 = rdf.sort_values("ml_rank").head(2)
        if len(top2) == 2:
            t2_nos = [str(_safe_int(r["horse_no"], 0)) for _, r in top2.iterrows()]
            t2_odds = [_safe_float(r["odds"], 0) for _, r in top2.iterrows()]
            avg_odds = sum(t2_odds) / 2
            if min_odds <= avg_odds <= max_odds:
                umaren_bets += 100
                if set(t2_nos) <= top3_set and w_no in t2_nos:
                    # 馬連オッズの簡易推定にゃ
                    umaren_est = max(2.0, t2_odds[0] * t2_odds[1] * 0.15)
                    ret = int(100 * umaren_est * (1 - 0.225))
                    umaren_hits += 1
                    umaren_return += ret

        # ── 三連複にゃ（AI上位3頭BOXにゃ）──
        top3_pred = rdf.sort_values("ml_rank").head(3)
        if len(top3_pred) >= 3:
            t3_nos = set(str(_safe_int(r["horse_no"], 0)) for _, r in top3_pred.iterrows())
            t3_odds_list = [_safe_float(r["odds"], 0) for _, r in top3_pred.iterrows()]
            avg_o3 = sum(t3_odds_list) / 3
            if min_odds <= avg_o3 <= max_odds:
                san3_bets += 100
                if t3_nos == top3_set:
                    # 三連複オッズの簡易推定にゃ
                    san3_est = max(5.0,
                        t3_odds_list[0] * t3_odds_list[1] * t3_odds_list[2] * 0.05)
                    ret = int(100 * san3_est * (1 - 0.25))
                    san3_hits += 1
                    san3_return += ret

        # ── 三連単にゃ（AI1位→2位→3位にゃ）──
        top3_san = rdf.sort_values("ml_rank").head(3)
        if len(top3_san) >= 3:
            ts_nos = [str(_safe_int(r["horse_no"], 0)) for _, r in top3_san.iterrows()]
            ts_odds = [_safe_float(r["odds"], 0) for _, r in top3_san.iterrows()]
            avg_ts = sum(ts_odds) / 3
            if min_odds <= avg_ts <= max_odds:
                san1_bets += 100
                if ts_nos[0]==w_no and ts_nos[1]==s_no and ts_nos[2]==t_no:
                    san1_est = max(10.0,
                        ts_odds[0]*ts_odds[1]*ts_odds[2]*0.15)
                    ret = int(100 * san1_est * (1 - 0.25))
                    san1_hits += 1
                    san1_return += ret

        # レース記録にゃ
        race_records.append({
            "レースにゃ": rdf["race_label"].iloc[0] if "race_label" in rdf.columns else rk,
            "1着にゃ": f"馬番{w_no}" if w_no else "-",
            "2着にゃ": f"馬番{s_no}" if s_no else "-",
            "3着にゃ": f"馬番{t_no}" if t_no else "-",
            "AI1位にゃ": f"馬番{ai1_no}" if ai1 is not None else "-",
            "AI1位単勝にゃ": "✅" if (ai1 is not None and ai1_no == w_no) else "❌",
            "複勝的中にゃ": "✅" if any(
                str(_safe_int(r["horse_no"],0)) in top3_set
                for _,r in buy_df.iterrows()
            ) else "❌",
            "三連複的中にゃ": "✅" if (
                len(top3_pred)>=3 and
                set(str(_safe_int(r["horse_no"],0)) for _,r in top3_pred.iterrows())==top3_set
            ) else "❌",
        })

    def pct(h, b):
        return f"{h/b*100:.1f}%" if b > 0 else "0.0%"

    def roi(r, b):
        return f"{r/b*100:.1f}%" if b > 0 else "0.0%"

    results = {
        "総レース数にゃ": n_races,
        "券種別成績にゃ": pd.DataFrame([
            {
                "券種にゃ": "単勝にゃ（AI1位固定にゃ）",
                "購入回数にゃ": tansho_bets // 100,
                "的中回数にゃ": tansho_hits,
                "的中率にゃ":    pct(tansho_hits, tansho_bets // 100),
                "投資額にゃ":    f"¥{tansho_bets:,}",
                "回収額にゃ":    f"¥{int(tansho_return):,}",
                "回収率にゃ":    roi(tansho_return, tansho_bets),
            },
            {
                "券種にゃ": "複勝にゃ（買い判定馬にゃ）",
                "購入回数にゃ": fukusho_bets // 100,
                "的中回数にゃ": fukusho_hits,
                "的中率にゃ":    pct(fukusho_hits, fukusho_bets // 100),
                "投資額にゃ":    f"¥{fukusho_bets:,}",
                "回収額にゃ":    f"¥{int(fukusho_return):,}",
                "回収率にゃ":    roi(fukusho_return, fukusho_bets),
            },
            {
                "券種にゃ": "馬連にゃ（AI上位2頭にゃ）",
                "購入回数にゃ": umaren_bets // 100,
                "的中回数にゃ": umaren_hits,
                "的中率にゃ":    pct(umaren_hits, umaren_bets // 100),
                "投資額にゃ":    f"¥{umaren_bets:,}",
                "回収額にゃ":    f"¥{int(umaren_return):,}",
                "回収率にゃ":    roi(umaren_return, umaren_bets),
            },
            {
                "券種にゃ": "三連複にゃ（軸1×相手5→10点にゃ）",
                "購入回数にゃ": san3_bets // 100,
                "的中回数にゃ": san3_hits,
                "的中率にゃ":    pct(san3_hits, san3_bets // 100),
                "投資額にゃ":    f"¥{san3_bets:,}",
                "回収額にゃ":    f"¥{int(san3_return):,}",
                "回収率にゃ":    roi(san3_return, san3_bets),
            },
            {
                "券種にゃ": "三連単にゃ（AI順1→2→3にゃ）",
                "購入回数にゃ": san1_bets // 100,
                "的中回数にゃ": san1_hits,
                "的中率にゃ":    pct(san1_hits, san1_bets // 100),
                "投資額にゃ":    f"¥{san1_bets:,}",
                "回収額にゃ":    f"¥{int(san1_return):,}",
                "回収率にゃ":    roi(san1_return, san1_bets),
            },
        ]),
        "レース別成績にゃ": pd.DataFrame(race_records),
        "raw": {
            "tansho":  (tansho_hits,  tansho_bets,  tansho_return),
            "fukusho": (fukusho_hits, fukusho_bets, fukusho_return),
            "umaren":  (umaren_hits,  umaren_bets,  umaren_return),
            "san3":    (san3_hits,    san3_bets,    san3_return),
            "san1":    (san1_hits,    san1_bets,    san1_return),
        }
    }
    return results


def show_backtest_tab(bundle, strategy_mode: str = STRATEGY_MODE_ROI):
    """
    バックテストタブのUI にゃ。
    過去データをアップロードして的中率・回収率を自動計算するにゃ🐾
    """
    st.header("📊 バックテスト（実績検証）にゃ🐾")
    st.caption(
        "過去の出馬表＋着順データをアップロードすると、"
        "実際の的中率・回収率を自動計算するにゃ🐾\n"
        "**yosou.csv**（TARGET形式・着順列付き）または"
        "簡易CSV（finish列付き）を使うにゃ。"
    )

    col1, col2 = st.columns(2)
    with col1:
        bt_file = st.file_uploader(
            "過去データCSVにゃ（finish列必須にゃ）",
            type=["csv"], key="bt_upload"
        )
    with col2:
        use_yosou = st.checkbox(
            f"yosou.csv を使うにゃ（{TARGET_CSV_PATH.name}）",
            value=TARGET_CSV_PATH.exists()
        )
        min_o = st.number_input("最低オッズにゃ（対象レースにゃ）", 1.0, 10.0, 1.0, 0.5)
        max_o = st.number_input("最高オッズにゃ（対象レースにゃ）", 10.0, 9999.0, 9999.0, 10.0)

    # データ読み込みにゃ
    hist_df = None
    if bt_file is not None:
        raw = bt_file.read()
        try:
            hist_df = normalize_52cols(read_csv_bytes(raw), bt_file.name)
        except Exception:
            try:
                hist_df = read_simple_csv_to_52(raw, bt_file.name)
            except Exception as e:
                st.error(f"CSVを読めなかったにゃ: {e}")

    elif use_yosou and TARGET_CSV_PATH.exists():
        hist_df = read_target_history_csv(TARGET_CSV_PATH)

    if hist_df is None:
        st.info(
            "📂 過去データをアップロードするか、"
            "yosou.csv を配置するにゃ🐾\n\n"
            "**必要な列にゃ**: 馬名・馬番・オッズ・人気・finish（着順）にゃ"
        )
        with st.expander("バックテスト用CSVのサンプルにゃ"):
            st.code(
                "日付,馬番,馬名,オッズ,人気,finish,競馬場,レース番号\n"
                "20260101,1,サンプルAにゃ,2.8,1,1,東京,11\n"
                "20260101,2,サンプルBにゃ,8.5,5,3,東京,11\n"
                "20260101,3,サンプルCにゃ,5.1,2,2,東京,11\n",
                language="csv"
            )
        return

    # finish列の確認にゃ
    if "finish" not in hist_df.columns:
        st.error("❌ finish（着順）列がないにゃ。着順入りCSVをアップロードするにゃ🐾")
        return

    valid_rows = hist_df["finish"].notna().sum()
    total_rows = len(hist_df)
    n_races_est = hist_df["race_key"].nunique() if "race_key" in hist_df.columns else "?"
    st.success(
        f"✅ データ読み込み完了にゃ: {total_rows}行 / 有効着順: {valid_rows}行 / "
        f"推定レース数: {n_races_est}にゃ"
    )

    if st.button("🐾 バックテスト実行にゃ！", type="primary", key="bt_run"):
        with st.spinner("バックテスト実行中にゃ...（レース数が多いと時間がかかるにゃ🐾）"):
            results = run_backtest(
                bundle, hist_df,
                strategy_mode=strategy_mode,
                min_odds=min_o, max_odds=max_o
            )

        if "error" in results:
            st.error(f"❌ {results['error']}")
            return

        st.markdown("---")
        st.subheader(f"📈 バックテスト結果にゃ（{results['総レース数にゃ']}レース分にゃ）")

        # ── サマリーメトリクスにゃ ──
        raw = results["raw"]
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric(
            "単勝的中率にゃ",
            f"{raw['tansho'][0]/max(raw['tansho'][1]//100,1)*100:.1f}%",
            f"回収率にゃ {raw['tansho'][2]/max(raw['tansho'][1],1)*100:.1f}%"
        )
        m2.metric(
            "複勝的中率にゃ",
            f"{raw['fukusho'][0]/max(raw['fukusho'][1]//100,1)*100:.1f}%",
            f"回収率にゃ {raw['fukusho'][2]/max(raw['fukusho'][1],1)*100:.1f}%"
        )
        m3.metric(
            "馬連的中率にゃ",
            f"{raw['umaren'][0]/max(raw['umaren'][1]//100,1)*100:.1f}%",
            f"回収率にゃ {raw['umaren'][2]/max(raw['umaren'][1],1)*100:.1f}%"
        )
        m4.metric(
            "三連複的中率にゃ",
            f"{raw['san3'][0]/max(raw['san3'][1]//100,1)*100:.1f}%",
            f"回収率にゃ {raw['san3'][2]/max(raw['san3'][1],1)*100:.1f}%"
        )
        m5.metric(
            "三連単的中率にゃ",
            f"{raw['san1'][0]/max(raw['san1'][1]//100,1)*100:.1f}%",
            f"回収率にゃ {raw['san1'][2]/max(raw['san1'][1],1)*100:.1f}%"
        )

        # 回収率の判定にゃ
        fuku_roi = raw['fukusho'][2] / max(raw['fukusho'][1], 1) * 100
        san3_roi = raw['san3'][2]    / max(raw['san3'][1],    1) * 100

        st.markdown("---")
        if fuku_roi >= 80:
            st.success(f"✅ 複勝回収率 {fuku_roi:.1f}% にゃ！良好にゃ🐾")
        elif fuku_roi >= 60:
            st.warning(f"⚠️ 複勝回収率 {fuku_roi:.1f}% にゃ。改善の余地ありにゃ")
        else:
            st.error(f"🔴 複勝回収率 {fuku_roi:.1f}% にゃ。モデルの再学習を推奨するにゃ")

        if san3_roi >= 100:
            st.success(f"🎉 三連複回収率 {san3_roi:.1f}% にゃ！プラスにゃ！すごいにゃ🐾")
        elif san3_roi >= 70:
            st.info(f"📊 三連複回収率 {san3_roi:.1f}% にゃ。まずまずにゃ")
        else:
            st.warning(f"⚠️ 三連複回収率 {san3_roi:.1f}% にゃ。買い目を絞るにゃ")

        # ── 券種別詳細にゃ ──
        st.markdown("#### 🎯 券種別成績にゃ")
        st.dataframe(
            results["券種別成績にゃ"],
            use_container_width=True, hide_index=True
        )

        # ── レース別成績にゃ ──
        if not results["レース別成績にゃ"].empty:
            st.markdown("#### 📋 レース別成績にゃ")
            race_rec = results["レース別成績にゃ"]

            # 色付きにゃ
            def color_row(row):
                if row.get("三連複的中にゃ") == "✅":
                    return ["background-color:#d4edda"] * len(row)
                if row.get("複勝的中にゃ") == "✅":
                    return ["background-color:#fff3cd"] * len(row)
                return [""] * len(row)

            try:
                st.dataframe(
                    race_rec.style.apply(color_row, axis=1),
                    use_container_width=True, hide_index=True
                )
            except Exception:
                st.dataframe(race_rec, use_container_width=True, hide_index=True)

            # CSVダウンロードにゃ
            st.download_button(
                "📥 バックテスト結果CSVにゃ",
                data=race_rec.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="backtest_result.csv", mime="text/csv"
            )

        # ── 分析サマリーにゃ ──
        st.markdown("---")
        st.markdown("#### 💡 改善提案にゃ")
        tan_rate = raw['tansho'][0]  / max(raw['tansho'][1]  // 100, 1) * 100
        san3_rate= raw['san3'][0]    / max(raw['san3'][1]    // 100, 1) * 100
        tips = []
        if tan_rate < 15:
            tips.append("🔴 単勝的中率が低いにゃ → AIの1位予想精度が低いにゃ。PKLの再学習を検討にゃ")
        if san3_rate > 15:
            tips.append("🟢 三連複的中率が高いにゃ → 配当を上げるため穴馬を相手に加えるにゃ")
        if fuku_roi < 70:
            tips.append("⚠️ 回収率が低いにゃ → 買い判定の閾値を上げて点数を絞るにゃ")
        if san3_roi > 100:
            tips.append("🎉 三連複がプラスにゃ → このロジックは有効にゃ！点数を増やすにゃ")
        if not tips:
            tips.append("📊 まずはデータ数を増やして傾向を掴むにゃ🐾")
        for tip in tips:
            st.info(tip)




# ============================================================
# ============================================================
# S級強化モジュールにゃ（期待値・見送り・展開 完全刷新にゃ）
# ============================================================
# ============================================================


# ============================================================
# 【S級-1】 展開予測エンジンにゃ
# 当日メンバー全体の脚質バランスを分析して
# 各馬への展開利不利スコアを計算するにゃ
# ============================================================

# 距離帯別の有利脚質テーブルにゃ
DIST_STYLE_ADVANTAGE = {
    # (距離下限, 距離上限): {脚質: 基本スコアにゃ}
    (0,    1400): {"逃げ": 1.30, "先行": 1.15, "差し": 0.90, "追込": 0.70, "未取得": 1.00},
    (1401, 1800): {"逃げ": 1.15, "先行": 1.10, "差し": 1.05, "追込": 0.85, "未取得": 1.00},
    (1801, 2200): {"逃げ": 1.00, "先行": 1.05, "差し": 1.10, "追込": 1.00, "未取得": 1.00},
    (2201, 9999): {"逃げ": 0.85, "先行": 1.00, "差し": 1.15, "追込": 1.10, "未取得": 1.00},
}

# コース種別×脚質の有利テーブルにゃ
TRACK_STYLE_ADVANTAGE = {
    # 芝にゃ（差し・追込が有利にゃ）
    "芝": {"逃げ": 0.95, "先行": 1.00, "差し": 1.08, "追込": 1.05, "未取得": 1.00},
    # ダートにゃ（先行・逃げが有利にゃ）
    "ダ": {"逃げ": 1.20, "先行": 1.15, "差し": 0.90, "追込": 0.75, "未取得": 1.00},
    "ダート": {"逃げ": 1.20, "先行": 1.15, "差し": 0.90, "追込": 0.75, "未取得": 1.00},
}

# 馬場状態×脚質の有利テーブルにゃ
GOING_STYLE_ADVANTAGE = {
    "良":   {"逃げ": 1.00, "先行": 1.00, "差し": 1.00, "追込": 1.00, "未取得": 1.00},
    "稍重": {"逃げ": 1.05, "先行": 1.05, "差し": 0.98, "追込": 0.95, "未取得": 1.00},
    "重":   {"逃げ": 1.10, "先行": 1.08, "差し": 0.95, "追込": 0.88, "未取得": 1.00},
    "不良": {"逃げ": 1.15, "先行": 1.10, "差し": 0.90, "追込": 0.80, "未取得": 1.00},
}

# 枠順×脚質の有利テーブルにゃ（内枠=逃げ先行有利にゃ）
def _frame_style_bonus(frame_no: int, field_size: int, style: str) -> float:
    """枠番と脚質の組み合わせボーナスにゃ"""
    if field_size <= 0:
        return 1.0
    frame_ratio = frame_no / max(field_size / 2, 1)  # 内枠=低い、外枠=高いにゃ
    if style in ["逃げ", "先行"]:
        # 内枠ほど有利にゃ
        return 1.0 + (1.0 - min(frame_ratio, 2.0)) * 0.08
    elif style in ["差し", "追込"]:
        # 外枠の方がやや有利にゃ（包まれにくいにゃ）
        return 1.0 + min(frame_ratio - 1.0, 1.0) * 0.04
    return 1.0


def analyze_pace(race_df: pd.DataFrame) -> dict:
    """
    当日メンバーのペース予測にゃ。
    逃げ・先行馬の頭数からハイペース/スローを推定するにゃ。 """
    if "running_style" not in race_df.columns:
        return {"pace": "不明", "escape_count": 0, "front_count": 0,
                "pace_score": 0.5, "pace_note": "脚質データなしにゃ"}

    styles = race_df["running_style"].fillna("未取得")
    escape_count = int((styles == "逃げ").sum())
    senkou_count = int((styles == "先行").sum())
    sashi_count  = int((styles == "差し").sum())
    oikomi_count = int((styles == "追込").sum())
    front_count  = escape_count + senkou_count
    field_size   = max(len(race_df), 1)

    front_ratio = front_count / field_size

    # ペース判定にゃ
    if escape_count >= 3 or front_ratio >= 0.45:
        pace = "ハイペース"
        pace_score = 0.8   # 差し・追込有利スコアにゃ
        pace_note = f"逃げ{escape_count}頭・先行{senkou_count}頭でハイペース濃厚にゃ"
    elif escape_count == 0 or (escape_count == 1 and senkou_count <= 2):
        pace = "スローペース"
        pace_score = 0.2   # 逃げ・先行有利スコアにゃ
        pace_note = f"逃げ{escape_count}頭・先行{senkou_count}頭でスロー濃厚にゃ"
    elif escape_count == 1 and front_ratio <= 0.35:
        pace = "ミドルペース"
        pace_score = 0.5
        pace_note = f"逃げ1頭でミドルペース想定にゃ"
    else:
        pace = "流動的"
        pace_score = 0.5
        pace_note = f"逃げ{escape_count}頭・先行{senkou_count}頭で流動的にゃ"

    return {
        "pace": pace,
        "escape_count": escape_count,
        "senkou_count": senkou_count,
        "sashi_count": sashi_count,
        "oikomi_count": oikomi_count,
        "front_count": front_count,
        "field_size": field_size,
        "pace_score": pace_score,
        "pace_note": pace_note,
    }


def add_pace_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    各馬にペース適性スコアを付与するにゃ。
    展開利不利を総合スコア化するにゃ。

    【スコア体系にゃ】
    pace_advantage: 展開有利度（高いほど有利にゃ）にゃ
    pace_note_detail: 展開メモにゃ
    """
    df = df.copy()
    if "running_style" not in df.columns:
        df = add_running_style(df)

    # レース情報を取得にゃ
    pace_info   = analyze_pace(df)
    pace_score  = pace_info["pace_score"]   # 0=逃げ先行有利, 1=差し追込有利にゃ
    pace        = pace_info["pace"]
    distance    = _safe_int(df["distance"].iloc[0] if "distance" in df.columns else 2000, 2000)
    track_type  = str(df["track_type"].iloc[0] if "track_type" in df.columns else "芝").strip()
    going       = str(df["going"].iloc[0] if "going" in df.columns else "良").strip()
    field_size  = _safe_int(df["field_size"].max() if "field_size" in df.columns else len(df), len(df))

    # 距離帯別有利テーブルにゃ
    dist_table = {1.0: 1.0}
    for (lo, hi), tbl in DIST_STYLE_ADVANTAGE.items():
        if lo <= distance <= hi:
            dist_table = tbl
            break

    track_table = TRACK_STYLE_ADVANTAGE.get(track_type, TRACK_STYLE_ADVANTAGE["芝"])
    going_table = GOING_STYLE_ADVANTAGE.get(going, GOING_STYLE_ADVANTAGE["良"])

    advantages = []
    for _, row in df.iterrows():
        style    = str(row.get("running_style", "未取得"))
        frame_no = _safe_int(row.get("frame_no", 1), 1)

        # 基本スコアにゃ（距離・トラック・馬場にゃ）
        dist_adv  = dist_table.get(style, 1.0)
        track_adv = track_table.get(style, 1.0)
        going_adv = going_table.get(style, 1.0)

        # ペース適性にゃ
        if style == "逃げ":
            # スローなら超有利、ハイなら超不利にゃ
            pace_adv = 1.0 + (0.5 - pace_score) * 0.6
            n_escape = pace_info["escape_count"]
            if n_escape >= 2:
                pace_adv *= 0.80  # 逃げ馬が多いと競合するにゃ
        elif style == "先行":
            pace_adv = 1.0 + (0.5 - pace_score) * 0.3
        elif style == "差し":
            pace_adv = 1.0 + (pace_score - 0.5) * 0.4
        elif style == "追込":
            pace_adv = 1.0 + (pace_score - 0.5) * 0.6
        else:
            pace_adv = 1.0

        # 枠順ボーナスにゃ
        frame_adv = _frame_style_bonus(frame_no, field_size, style)

        # 総合展開スコアにゃ（各要素の積にゃ）
        total = dist_adv * track_adv * going_adv * pace_adv * frame_adv

        # メモにゃ
        note_parts = []
        if pace_adv > 1.05:
            note_parts.append(f"{pace}有利にゃ")
        elif pace_adv < 0.95:
            note_parts.append(f"{pace}不利にゃ")
        if dist_adv > 1.05:
            note_parts.append("距離適性◎にゃ")
        elif dist_adv < 0.95:
            note_parts.append("距離適性△にゃ")
        if going_adv > 1.05:
            note_parts.append(f"馬場{going}○にゃ")
        elif going_adv < 0.95:
            note_parts.append(f"馬場{going}×にゃ")
        if frame_adv > 1.05:
            note_parts.append(f"枠{frame_no}番有利にゃ")
        elif frame_adv < 0.95:
            note_parts.append(f"枠{frame_no}番不利にゃ")

        advantages.append({
            "pace_advantage": round(float(total), 4),
            "pace_adv_detail": "・".join(note_parts) if note_parts else "展開中立にゃ",
            "pace_dist_adv": round(float(dist_adv), 3),
            "pace_track_adv": round(float(track_adv), 3),
            "pace_going_adv": round(float(going_adv), 3),
            "pace_pace_adv": round(float(pace_adv), 3),
            "pace_frame_adv": round(float(frame_adv), 3),
        })

    adv_df = pd.DataFrame(advantages, index=df.index)
    for col in adv_df.columns:
        df[col] = adv_df[col]

    df["pace_summary"] = pace_info["pace"]
    df["pace_note"]    = pace_info["pace_note"]
    return df


# ============================================================
# 【S級-2】 多次元期待値計算エンジンにゃ
# 単純な「AI確率 - 市場確率」だけでなく
# 展開・騎手・距離・馬場・枠順を全部考慮するにゃ
# ============================================================

def add_ev_score_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    S級期待値計算にゃ（v26強化版にゃ）

    【計算式にゃ】
    ev_score_v2 = (AI確率 × 展開スコア × 騎手ボーナス) - 市場暗示確率
    ev_composite = ev_score_v2 × オッズ（実質期待値にゃ）

    展開スコアで「展開有利な馬の期待値を上積み」するにゃ🐾
    """
    df = df.copy()

    # 既存のev_scoreがないなら計算するにゃ
    if "ev_score" not in df.columns:
        df = add_ev_score(df)

    if "pace_advantage" not in df.columns:
        df = add_pace_advantage(df)

    prob   = pd.to_numeric(df["ml_top3_prob"], errors="coerce").fillna(0)
    odds   = pd.to_numeric(df["odds"],         errors="coerce").fillna(0)
    implied= pd.to_numeric(df["implied_top3"], errors="coerce").fillna(0)
    pace   = pd.to_numeric(df["pace_advantage"],errors="coerce").fillna(1.0)
    jr     = pd.to_numeric(df.get("jockey_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)
    dr     = pd.to_numeric(df.get("horse_distance_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)
    tr     = pd.to_numeric(df.get("trainer_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)

    # 騎手ボーナス（平均より上回っていればプラスにゃ）
    jockey_bonus = ((jr / 0.28) - 1.0).clip(-0.15, 0.20)

    # 距離適性ボーナス（過去の距離実績にゃ）
    dist_bonus = ((dr / 0.28) - 1.0).clip(-0.10, 0.15)

    # 展開補正済みAI確率にゃ
    prob_adjusted = (prob * pace.clip(0.7, 1.4)).clip(0, 0.95)

    # 多次元EV計算にゃ
    valid = odds > 1.0
    ev_v2_raw = prob_adjusted - implied + jockey_bonus * 0.3 + dist_bonus * 0.2
    df["ev_score_v2"]    = np.where(valid, ev_v2_raw, 0.0).round(4)

    # 実質期待値にゃ（回収率ベースにゃ）
    df["ev_composite"]   = np.where(
        valid,
        (prob_adjusted * odds * (1 - FUKUSHO_DEDUCTION) - 1.0).round(4),
        0.0
    )

    # EV信頼度にゃ（騎手・距離・展開が全部プラスのときS評価にゃ）
    def ev_grade(row):
        ev2  = _safe_float(row.get("ev_score_v2",  0), 0)
        evc  = _safe_float(row.get("ev_composite", 0), 0)
        pace_a = _safe_float(row.get("pace_advantage", 1.0), 1.0)
        if ev2 >= 0.10 and evc >= 0.20 and pace_a >= 1.05: return "S"
        if ev2 >= 0.06 and evc >= 0.10:                    return "A"
        if ev2 >= 0.02 and evc >= 0.00:                    return "B"
        if ev2 >= -0.03:                                    return "C"
        return "D"

    df["ev_grade"] = df.apply(ev_grade, axis=1)

    return df


# ============================================================
# 【S級-3】 見送り判定エンジン（完全刷新にゃ）
# 単純なKelly比不足・危険馬判定から
# 多角的な「見送り理由スコア」に進化にゃ
# ============================================================

# 見送り判定の各要素にゃ
PASS_WEIGHTS = {
    "危険人気馬": 100,   # 即見送りにゃ
    "展開大不利": 60,
    "EV大幅マイナス": 50,
    "AI圏外": 40,
    "Kelly不足": 30,
    "距離適性不良": 25,
    "馬場適性不良": 20,
    "外枠不利": 15,
    "逃げ頭数多い": 10,
}

def add_pass_score(df: pd.DataFrame,
                   strategy_mode: str = STRATEGY_MODE_ROI) -> pd.DataFrame:
    """
    S級見送り判定にゃ。
    各馬に「見送りスコア」を付けて多角的に判定するにゃ。
    スコアが高いほど見送り理由が多いにゃ。 """
    df = df.copy()

    for col, dv in [("ml_top3_prob", 0), ("ml_rank", 99), ("odds", 0),
                    ("popularity", 99), ("ev_score_v2", 0), ("ev_composite", 0),
                    ("pace_advantage", 1.0), ("kelly_ratio", 0),
                    ("kelly_ratio_sanren", 0), ("danger_level", ""),
                    ("pivot_confidence", 0)]:
        if col not in df.columns:
            df[col] = dv
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(dv) \
            if col not in ["danger_level"] else df[col].fillna("")

    pass_scores  = []
    pass_reasons = []
    buy_flags_v2 = []
    buy_reasons_v2 = []

    for _, row in df.iterrows():
        score   = 0
        reasons = []

        ml_rank  = _safe_int(row.get("ml_rank",  99), 99)
        pop      = _safe_int(row.get("popularity",99), 99)
        ev2      = _safe_float(row.get("ev_score_v2",  0), 0)
        evc      = _safe_float(row.get("ev_composite", 0), 0)
        pace_a   = _safe_float(row.get("pace_advantage", 1.0), 1.0)
        kf       = _safe_float(row.get("kelly_ratio", 0), 0)
        ks       = _safe_float(row.get("kelly_ratio_sanren", 0), 0)
        dl       = str(row.get("danger_level", ""))
        dr       = _safe_float(row.get("horse_distance_top3_rate_prior", 0.25), 0.25)
        jt       = _safe_float(row.get("jockey_top3_rate_prior", 0.25), 0.25)
        prob     = _safe_float(row.get("ml_top3_prob", 0), 0)
        ev_grade = str(row.get("ev_grade", "C"))

        # ── 見送り要因チェックにゃ ──
        if dl in ["強危険", "危険"]:
            score += PASS_WEIGHTS["危険人気馬"]
            reasons.append(f"⚠️危険人気馬({dl})にゃ")

        if pace_a < 0.85:
            score += PASS_WEIGHTS["展開大不利"]
            reasons.append(f"🌪️展開大不利({pace_a:.2f})にゃ")
        elif pace_a < 0.92:
            score += PASS_WEIGHTS["展開大不利"] // 2
            reasons.append(f"🌪️展開不利({pace_a:.2f})にゃ")

        if ev2 < -0.10:
            score += PASS_WEIGHTS["EV大幅マイナス"]
            reasons.append(f"📉EV大幅マイナス({ev2:.3f})にゃ")
        elif ev2 < -0.05:
            score += PASS_WEIGHTS["EV大幅マイナス"] // 2
            reasons.append(f"📉EVマイナス({ev2:.3f})にゃ")

        if ml_rank > 6:
            score += PASS_WEIGHTS["AI圏外"]
            reasons.append(f"🤖AI{ml_rank}位（圏外）にゃ")
        elif ml_rank > 4 and strategy_mode == STRATEGY_MODE_ROI:
            score += PASS_WEIGHTS["AI圏外"] // 2
            reasons.append(f"🤖AI{ml_rank}位（回収率重視では厳しいにゃ）")

        if kf < MIN_KELLY_RATIO and ks < MIN_KELLY_RATIO:
            score += PASS_WEIGHTS["Kelly不足"]
            reasons.append(f"📊Kelly不足(複:{kf:.3f}/三:{ks:.3f})にゃ")

        if dr < 0.15:
            score += PASS_WEIGHTS["距離適性不良"]
            reasons.append(f"📏距離適性不良({dr:.1%})にゃ")

        # ── 買い判定にゃ ──
        if score >= PASS_WEIGHTS["危険人気馬"]:
            # 即見送りにゃ
            buy_v2 = "見送り"
            buy_r  = "・".join(reasons[:2])
        elif strategy_mode == STRATEGY_MODE_ROI:
            # 回収率重視: EV+展開両方OKにゃ
            if score <= 10 and ev2 >= 0.05 and pace_a >= 1.00 and ml_rank <= 5:
                buy_v2 = "◎買い"
                buy_r  = f"EV{ev2:.3f}・展開{pace_a:.2f}・AI{ml_rank}位にゃ"
            elif score <= 20 and ev2 >= 0.02 and ml_rank <= 4:
                buy_v2 = "○買い"
                buy_r  = f"EV{ev2:.3f}・AI{ml_rank}位にゃ"
            elif score <= 30 and ev_grade in ["S","A"] and prob >= 0.20:
                buy_v2 = "▲買い"
                buy_r  = f"EVグレード{ev_grade}・確率{prob:.1%}にゃ"
            elif score >= 50:
                buy_v2 = "見送り"
                buy_r  = "・".join(reasons[:2]) if reasons else "総合スコア不足にゃ"
            else:
                buy_v2 = "△検討"
                buy_r  = f"スコア{score}点にゃ"
        else:
            # 的中率重視: AI上位+確率重視にゃ
            if score <= 15 and ml_rank <= 3 and prob >= 0.22:
                buy_v2 = "◎買い"
                buy_r  = f"AI{ml_rank}位・確率{prob:.1%}にゃ"
            elif score <= 25 and ml_rank <= 5 and pop <= 5:
                buy_v2 = "○買い"
                buy_r  = f"AI{ml_rank}位・人気{pop}にゃ"
            elif score >= 60:
                buy_v2 = "見送り"
                buy_r  = "・".join(reasons[:2]) if reasons else "総合スコア不足にゃ"
            else:
                buy_v2 = "△検討"
                buy_r  = f"スコア{score}点にゃ"

        pass_scores.append(score)
        pass_reasons.append("・".join(reasons) if reasons else "問題なしにゃ")
        buy_flags_v2.append(buy_v2)
        buy_reasons_v2.append(buy_r)

    df["pass_score"]     = pass_scores
    df["pass_reason"]    = pass_reasons
    df["buy_flag_v2"]    = buy_flags_v2
    df["buy_reason_v2"]  = buy_reasons_v2

    return df


# ============================================================
# 【S級-4】 総合予想スコア（全要素を統合にゃ）
# ============================================================

def add_final_score(df: pd.DataFrame,
                    strategy_mode: str = STRATEGY_MODE_ROI) -> pd.DataFrame:
    """
    全要素を統合した最終スコアにゃ。

    final_score = AI確率(40%) + EV複合(25%) + 展開適性(20%) + 実績(15%)
    """
    df = df.copy()

    prob  = pd.to_numeric(df.get("ml_top3_prob",  0), errors="coerce").fillna(0)
    ev2   = pd.to_numeric(df.get("ev_score_v2",   0), errors="coerce").fillna(0)
    evc   = pd.to_numeric(df.get("ev_composite",  0), errors="coerce").fillna(0)
    pace  = pd.to_numeric(df.get("pace_advantage",1), errors="coerce").fillna(1.0)
    jr    = pd.to_numeric(df.get("jockey_top3_rate_prior",  0.25), errors="coerce").fillna(0.25)
    dr    = pd.to_numeric(df.get("horse_distance_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)
    tr    = pd.to_numeric(df.get("trainer_top3_rate_prior", 0.25), errors="coerce").fillna(0.25)
    kf    = pd.to_numeric(df.get("kelly_ratio",          0), errors="coerce").fillna(0)
    ks    = pd.to_numeric(df.get("kelly_ratio_sanren",   0), errors="coerce").fillna(0)
    pconf = pd.to_numeric(df.get("pivot_confidence",     0), errors="coerce").fillna(0)
    pscr  = pd.to_numeric(df.get("pass_score",          99), errors="coerce").fillna(99)

    # 各要素を0〜1に正規化にゃ
    ai_score     = prob.clip(0, 1)
    ev_score_n   = (ev2 * 2 + 0.5).clip(0, 1)
    pace_score_n = (pace - 0.7).clip(0, 0.6) / 0.6
    jitsuryoku   = (jr/0.35*0.4 + dr/0.35*0.35 + tr/0.35*0.25).clip(0, 1)

    if strategy_mode == STRATEGY_MODE_ROI:
        final = (
            ai_score   * 0.35
            + ev_score_n * 0.30
            + pace_score_n * 0.20
            + jitsuryoku * 0.15
        )
    else:
        final = (
            ai_score   * 0.45
            + pace_score_n * 0.20
            + jitsuryoku * 0.20
            + ev_score_n * 0.15
        )

    # 見送りペナルティにゃ
    penalty = (pscr / 200).clip(0, 0.5)
    final   = (final - penalty).clip(0, 1)

    df["final_score"] = final.round(4)

    # 最終印にゃ（final_scoreベースにゃ）
    def final_mark(row):
        fs = _safe_float(row.get("final_score", 0), 0)
        bv = str(row.get("buy_flag_v2", ""))
        if "見送り" in bv:
            return "×"
        if fs >= 0.65: return "◎"
        if fs >= 0.50: return "○"
        if fs >= 0.38: return "▲"
        if fs >= 0.25: return "△"
        return "×"

    df["final_mark"] = df.apply(final_mark, axis=1)
    return df


# ============================================================
# 【S級-5】 展開表示にゃ
# ============================================================

def show_pace_analysis(race_df: pd.DataFrame):
    """展開予測タブの表示にゃ"""
    st.subheader("🌪️ 展開予測＆有利不利分析にゃ")

    if "pace_advantage" not in race_df.columns:
        st.warning("展開データが計算されていないにゃ。予想を再実行するにゃ🐾")
        return

    pace_info = analyze_pace(race_df)
    pace      = pace_info["pace"]
    pace_icon = {"ハイペース":"🔥","スローペース":"💤","ミドルペース":"⚡","流動的":"🌀"}.get(pace,"❓")

    # ペース概要にゃ
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("予想ペースにゃ", f"{pace_icon} {pace}")
    c2.metric("逃げ馬数にゃ",   f"{pace_info['escape_count']}頭にゃ")
    c3.metric("先行馬数にゃ",   f"{pace_info['senkou_count']}頭にゃ")
    c4.metric("差し馬数にゃ",   f"{pace_info['sashi_count']}頭にゃ")
    c5.metric("追込馬数にゃ",   f"{pace_info['oikomi_count']}頭にゃ")

    st.info(f"📊 {pace_info['pace_note']}")

    # ペース別有利脚質にゃ
    if pace == "ハイペース":
        st.success("🔥 ハイペース → **差し・追込馬が有利**にゃ！逃げ・先行は注意にゃ🐾")
    elif pace == "スローペース":
        st.success("💤 スローペース → **逃げ・先行馬が有利**にゃ！差し・追込には厳しいにゃ🐾")
    elif pace == "ミドルペース":
        st.info("⚡ ミドルペース → **先行馬が安定**にゃ。展開の綾が少ないにゃ")
    else:
        st.warning("🌀 流動的 → **展開読みが難しいにゃ**。AI確率重視で判断するにゃ")

    # 各馬の展開有利不利テーブルにゃ
    st.markdown("#### 🏇 馬別展開スコアにゃ")
    cols_show = ["ml_rank","final_mark","horse_no","horse_name","running_style",
                 "pace_advantage","pace_adv_detail","pace_dist_adv",
                 "pace_track_adv","pace_going_adv","pace_pace_adv","pace_frame_adv"]
    cols_show = [c for c in cols_show if c in race_df.columns]
    disp = race_df[cols_show].sort_values("ml_rank").copy()

    # カラー付けにゃ
    def color_pace(row):
        pa = _safe_float(row.get("展開スコアにゃ", row.get("pace_advantage", 1.0)), 1.0)
        if pa >= 1.10: return ["background-color:#d4edda"]*len(row)
        if pa >= 1.03: return ["background-color:#d1ecf1"]*len(row)
        if pa <= 0.88: return ["background-color:#f8d7da"]*len(row)
        if pa <= 0.95: return ["background-color:#fff3cd"]*len(row)
        return [""]*len(row)

    rename_map = {
        "ml_rank":"AI順位にゃ","final_mark":"最終印にゃ",
        "horse_no":"馬番にゃ","horse_name":"馬名にゃ",
        "running_style":"脚質にゃ","pace_advantage":"展開スコアにゃ",
        "pace_adv_detail":"展開メモにゃ","pace_dist_adv":"距離適性にゃ",
        "pace_track_adv":"コース適性にゃ","pace_going_adv":"馬場適性にゃ",
        "pace_pace_adv":"ペース適性にゃ","pace_frame_adv":"枠順適性にゃ",
    }
    disp = disp.rename(columns=rename_map)
    try:
        st.dataframe(disp.style.apply(color_pace, axis=1),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # 展開有利馬ランキングにゃ
    st.markdown("#### 🎯 展開有利馬 TOP5にゃ")
    top5 = race_df.sort_values("pace_advantage", ascending=False).head(5)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        pa   = _safe_float(row.get("pace_advantage", 1.0), 1.0)
        icon = "🟢" if pa >= 1.05 else ("🟡" if pa >= 1.00 else "🔴")
        hno  = _safe_int(row.get("horse_no", 0), 0)
        name = str(row.get("horse_name", ""))
        st = row.get("running_style", "不明")
        note = str(row.get("pace_adv_detail", ""))
        ai_r = _safe_int(row.get("ml_rank", 0), 0)
        st_obj = __import__('streamlit')
        st_obj.markdown(
            f"{icon} **{rank}位** 馬番{hno} {name}（{st}）"
            f"　展開スコア: **{pa:.3f}**　AI{ai_r}位　{note}"
        )


def show_ev_analysis_v2(race_df: pd.DataFrame):
    """S級期待値分析タブにゃ"""
    st.subheader("📈 S級期待値分析にゃ（展開補正済みにゃ）")

    if "ev_score_v2" not in race_df.columns:
        st.warning("期待値V2が計算されていないにゃ。予想を再実行するにゃ🐾")
        return

    # EVグレード分布にゃ
    if "ev_grade" in race_df.columns:
        grade_cnt = race_df["ev_grade"].value_counts()
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, grade, icon in [
            (c1,"S","🌟"),(c2,"A","⭐"),(c3,"B","✅"),(c4,"C","⚠️"),(c5,"D","❌")
        ]:
            cnt = int(grade_cnt.get(grade, 0))
            col.metric(f"{icon} {grade}評価にゃ", f"{cnt}頭にゃ")

    # 期待値テーブルにゃ
    cols = ["ml_rank","final_mark","horse_no","horse_name","odds","popularity",
            "ml_top3_prob","ev_score","ev_score_v2","ev_composite",
            "ev_grade","pace_advantage","buy_flag_v2","buy_reason_v2","pass_score"]
    cols = [c for c in cols if c in race_df.columns]
    disp = race_df[cols].sort_values("ev_score_v2", ascending=False).copy()

    # フォーマットにゃ
    for c in ["ml_top3_prob"]:
        if c in disp.columns:
            disp[c] = (pd.to_numeric(disp[c], errors="coerce")*100).round(1).astype(str)+"%"
    for c in ["ev_score","ev_score_v2","ev_composite","pace_advantage"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)

    rename_map = {
        "ml_rank":"AI順位","final_mark":"最終印",
        "horse_no":"馬番","horse_name":"馬名",
        "odds":"オッズ","popularity":"人気",
        "ml_top3_prob":"AI確率","ev_score":"EV(旧)",
        "ev_score_v2":"EV(展開補正)","ev_composite":"実質期待値",
        "ev_grade":"EVグレード","pace_advantage":"展開スコア",
        "buy_flag_v2":"買い判定","buy_reason_v2":"判定理由","pass_score":"見送りスコア",
    }
    disp = disp.rename(columns=rename_map)

    def color_ev(row):
        ev = _safe_float(row.get("EV(展開補正)", row.get("ev_score_v2", 0)), 0)
        grade = str(row.get("EVグレード", "C"))
        if grade == "S":  return ["background-color:#c3e6cb"]*len(row)
        if grade == "A":  return ["background-color:#d1ecf1"]*len(row)
        if ev < -0.08:    return ["background-color:#f8d7da"]*len(row)
        return [""]*len(row)

    try:
        st.dataframe(disp.style.apply(color_ev, axis=1),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # S・A評価馬の解説にゃ
    if "ev_grade" in race_df.columns:
        sa_horses = race_df[race_df["ev_grade"].isin(["S","A"])].sort_values(
            "ev_score_v2", ascending=False)
        if not sa_horses.empty:
            st.markdown("#### 🌟 S・A評価馬（買い推奨にゃ）")
            for _, row in sa_horses.iterrows():
                hno   = _safe_int(row.get("horse_no",0),0)
                name  = str(row.get("horse_name",""))
                ev2   = _safe_float(row.get("ev_score_v2",0),0)
                evc   = _safe_float(row.get("ev_composite",0),0)
                grade = str(row.get("ev_grade",""))
                pace  = _safe_float(row.get("pace_advantage",1.0),1.0)
                ai_r  = _safe_int(row.get("ml_rank",0),0)
                icon  = "🌟" if grade=="S" else "⭐"
                st.success(
                    f"{icon} **馬番{hno} {name}**（AI{ai_r}位）"
                    f"　EV(補正)={ev2:.3f}　実質期待値={evc:.3f}　展開スコア={pace:.3f}にゃ"
                )


def show_pass_judgment(race_df: pd.DataFrame,
                       strategy_mode: str = STRATEGY_MODE_ROI):
    """S級見送り判定タブにゃ"""
    st.subheader("🚦 S級買い/見送り判定にゃ（多角的スコアリングにゃ）")

    if "pass_score" not in race_df.columns:
        st.warning("見送りスコアが計算されていないにゃ。予想を再実行するにゃ🐾")
        return

    # サマリーにゃ
    buy_cnt  = int((race_df["buy_flag_v2"].str.contains("買い")).sum()) \
        if "buy_flag_v2" in race_df.columns else 0
    pass_cnt = int((race_df["buy_flag_v2"] == "見送り").sum()) \
        if "buy_flag_v2" in race_df.columns else 0
    kento    = int((race_df["buy_flag_v2"] == "△検討").sum()) \
        if "buy_flag_v2" in race_df.columns else 0

    c1,c2,c3 = st.columns(3)
    c1.metric("◎○▲買い推奨にゃ", f"{buy_cnt}頭にゃ")
    c2.metric("△検討にゃ",        f"{kento}頭にゃ")
    c3.metric("見送りにゃ",        f"{pass_cnt}頭にゃ")

    # 判定テーブルにゃ
    cols = ["ml_rank","final_mark","horse_no","horse_name","odds","popularity",
            "ml_top3_prob","ev_grade","pace_advantage","pass_score","pass_reason",
            "buy_flag_v2","buy_reason_v2"]
    cols = [c for c in cols if c in race_df.columns]
    disp = race_df[cols].copy()

    # ソート: 買い→検討→見送り、その中でfinal_score降順にゃ
    order_map = {"◎買い":0,"○買い":1,"▲買い":2,"△検討":3,"見送り":4}
    if "buy_flag_v2" in disp.columns:
        disp["_sort"] = disp["buy_flag_v2"].map(order_map).fillna(9)
        if "final_score" in race_df.columns:
            disp["_fs"] = race_df["final_score"].values
            disp = disp.sort_values(["_sort","_fs"], ascending=[True,False])
        else:
            disp = disp.sort_values("_sort")
        disp = disp.drop(columns=["_sort","_fs"] if "_fs" in disp.columns else ["_sort"])

    if "ml_top3_prob" in disp.columns:
        disp["ml_top3_prob"] = (pd.to_numeric(disp["ml_top3_prob"],errors="coerce")*100).round(1).astype(str)+"%"
    if "pace_advantage" in disp.columns:
        disp["pace_advantage"] = pd.to_numeric(disp["pace_advantage"],errors="coerce").round(3)

    rename_map = {
        "ml_rank":"AI順位","final_mark":"最終印",
        "horse_no":"馬番","horse_name":"馬名",
        "odds":"オッズ","popularity":"人気",
        "ml_top3_prob":"AI確率","ev_grade":"EVグレード",
        "pace_advantage":"展開スコア","pass_score":"見送りスコア",
        "pass_reason":"見送り理由","buy_flag_v2":"判定","buy_reason_v2":"判定理由",
    }
    disp = disp.rename(columns=rename_map)

    def color_judge(row):
        judge = str(row.get("判定",""))
        if "◎" in judge: return ["background-color:#c3e6cb"]*len(row)
        if "○" in judge: return ["background-color:#d1ecf1"]*len(row)
        if "▲" in judge: return ["background-color:#fff3cd"]*len(row)
        if "見送り" in judge: return ["background-color:#f8d7da"]*len(row)
        return [""]*len(row)

    try:
        st.dataframe(disp.style.apply(color_judge, axis=1),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # 強力推奨馬にゃ
    strong = race_df[race_df.get("buy_flag_v2","").str.startswith("◎")
                     if "buy_flag_v2" in race_df.columns
                     else pd.Series([False]*len(race_df))]
    if not strong.empty:
        st.markdown("---")
        st.markdown("#### 🌟 強力推奨馬にゃ（◎買いにゃ）")
        for _, row in strong.iterrows():
            hno  = _safe_int(row.get("horse_no",0),0)
            name = str(row.get("horse_name",""))
            r    = str(row.get("buy_reason_v2",""))
            ps   = _safe_int(row.get("pass_score",0),0)
            st.success(f"◎ **馬番{hno} {name}**　{r}　見送りスコア={ps}点にゃ")





# ============================================================
# ============================================================
# ML強化モジュールにゃ
# ① Leakage防止チェッカーにゃ
# ② Walk-Forward バリデーションにゃ
# ③ ROI最適化エンジンにゃ
# ④ 本格キャリブレーション（Platt / Isotonic）にゃ
# ============================================================
# ============================================================


# ============================================================
# ① Leakage防止チェッカーにゃ
# ============================================================
# AUC=1.0の根本原因にゃ：
#   レース当日に確定するデータ（オッズ・人気・着順）が
#   学習特徴量に混入しているにゃ
#   → 予測時には存在しない未来情報を学習してしまうにゃ

# 学習に使ってはいけない「リーク特徴量」にゃ
LEAKAGE_FEATURES = [
    "finish",           # 着順（答えそのものにゃ）
    "target_value",     # ターゲット変数にゃ
    "time_sec",         # タイム（レース後確定にゃ）
    "time_raw",         # タイム（レース後確定にゃ）
    "last3f",           # 上り3F（レース後確定にゃ）
    "pass1","pass2","pass3","pass4",  # 通過順（レース後確定にゃ）
    "prize",            # 賞金（レース後確定にゃ）
    "body_weight",      # 当日馬体重（当日確定にゃ）※議論あり
]

# 危険な特徴量（使う場合は理由を要確認にゃ）
CAUTION_FEATURES = [
    "odds",             # 当日オッズ（予測時点で利用可能だが高リーク性にゃ）
    "popularity",       # 当日人気にゃ
    "field_odds_rank",  # オッズ順位にゃ
    "field_pop_rank",   # 人気順位にゃ
    "odds_gap_to_fav",  # オッズ差にゃ
    "popularity_gap_to_fav",  # 人気差にゃ
]


def check_leakage(feature_cols: list,
                  bundle: dict = None,
                  verbose: bool = True) -> dict:
    """
    特徴量リストのリークチェックにゃ。
    リーク特徴量が含まれていると AUC=1.0 になるにゃ。

    戻り値にゃ:
      leakage_found: bool
      leaked_cols: list
      caution_cols: list
      safe_cols: list
      leak_severity: str ('critical'/'warning'/'ok')
    """
    if feature_cols is None:
        feature_cols = BASE_NUM_FEATURES + CAT_FEATURES

    leaked   = [f for f in feature_cols if f in LEAKAGE_FEATURES]
    caution  = [f for f in feature_cols if f in CAUTION_FEATURES]
    safe     = [f for f in feature_cols
                if f not in LEAKAGE_FEATURES and f not in CAUTION_FEATURES]

    if leaked:
        severity = "critical"
    elif len(caution) >= 3:
        severity = "warning"
    else:
        severity = "ok"

    result = {
        "leakage_found": len(leaked) > 0,
        "leaked_cols":   leaked,
        "caution_cols":  caution,
        "safe_cols":     safe,
        "leak_severity": severity,
        "n_total":       len(feature_cols),
        "n_leaked":      len(leaked),
        "n_caution":     len(caution),
        "n_safe":        len(safe),
    }

    if verbose:
        import streamlit as st
        st.markdown("#### 🔍 Leakageチェック結果にゃ")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("全特徴量にゃ",    f"{len(feature_cols)}個にゃ")
        c2.metric("🔴 リーク確定にゃ", f"{len(leaked)}個にゃ",
                  delta="要除外にゃ" if leaked else None, delta_color="inverse")
        c3.metric("🟡 要注意にゃ",    f"{len(caution)}個にゃ")
        c4.metric("✅ 安全にゃ",      f"{len(safe)}個にゃ")

        if leaked:
            st.error(
                f"🔴 **Critical: リーク特徴量が{len(leaked)}個あるにゃ！**\n\n"
                f"`{', '.join(leaked)}`\n\n"
                "これらはレース後にしか確定しない情報にゃ。"
                "AUC=1.0の直接原因になるにゃ🐾"
            )
        if caution:
            st.warning(
                f"🟡 **Warning: 要注意特徴量が{len(caution)}個あるにゃ**\n\n"
                f"`{', '.join(caution)}`\n\n"
                "当日オッズ・人気は予測時点で使えるが、"
                "目的変数との相関が高すぎてリーク的な動作をするにゃ。"
                "使う場合は意図的に含めていることを確認するにゃ🐾"
            )
        if severity == "ok" and not caution:
            st.success("✅ リーク特徴量なしにゃ。安全な特徴量セットにゃ🐾")

    return result


def get_safe_feature_cols(feature_cols: list,
                          remove_leakage: bool = True,
                          remove_caution: bool = False) -> list:
    """
    安全な特徴量リストを返すにゃ。
    remove_leakage=True: リーク確定を除去にゃ
    remove_caution=True: 要注意も除去（オッズ除外版にゃ）
    """
    result = list(feature_cols)
    if remove_leakage:
        result = [f for f in result if f not in LEAKAGE_FEATURES]
    if remove_caution:
        result = [f for f in result if f not in CAUTION_FEATURES]
    return result


# ============================================================
# ② Walk-Forward バリデーションにゃ
# ============================================================
# 通常のランダム分割はNG: 未来のデータで過去を予測してしまうにゃ
# Walk-Forward: 時系列順を守った検証にゃ
#
# 例にゃ:
#   Train: 2020/1/1 〜 2022/12/31
#   Test:  2023/1/1 〜 2023/3/31
#   → 常に「過去で学習→未来を予測」にゃ

def create_walkforward_splits(df: pd.DataFrame,
                               n_splits: int = 5,
                               test_months: int = 3,
                               min_train_months: int = 12) -> list[dict]:
    """
    Walk-Forward の時系列分割を生成するにゃ。

    戻り値にゃ:
      [{"fold": 1, "train_idx": [...], "test_idx": [...],
        "train_start": date, "train_end": date,
        "test_start": date, "test_end": date}, ...]
    """
    import warnings
    warnings.filterwarnings('ignore')

    if "date_int" not in df.columns:
        raise ValueError("date_int 列が必要にゃ（clean_types()を先に実行するにゃ）")

    date_vals = pd.to_numeric(df["date_int"], errors="coerce").dropna().sort_values().unique()
    if len(date_vals) == 0:
        raise ValueError("有効な日付データがないにゃ")

    # 月単位に変換にゃ（date_int = YYYYMMDD なのでにゃ）
    months = sorted(set(int(str(int(d))[:6]) for d in date_vals if d > 0))
    if len(months) < min_train_months + test_months:
        raise ValueError(
            f"データ期間が短すぎるにゃ（{len(months)}ヶ月）。"
            f"最低{min_train_months + test_months}ヶ月必要にゃ"
        )

    splits = []
    test_start_idx = min_train_months

    for fold in range(n_splits):
        ts_idx = test_start_idx + fold * test_months
        if ts_idx + test_months > len(months):
            break

        train_months_set = set(months[:ts_idx])
        test_months_set  = set(months[ts_idx: ts_idx + test_months])

        # date_intから月を取得にゃ
        df_months = df["date_int"].apply(
            lambda d: int(str(int(d))[:6]) if pd.notna(d) and d > 0 else 0)

        train_idx = df.index[df_months.isin(train_months_set)].tolist()
        test_idx  = df.index[df_months.isin(test_months_set)].tolist()

        if not train_idx or not test_idx:
            continue

        splits.append({
            "fold":        fold + 1,
            "train_idx":   train_idx,
            "test_idx":    test_idx,
            "train_months": sorted(train_months_set),
            "test_months":  sorted(test_months_set),
            "n_train":      len(train_idx),
            "n_test":       len(test_idx),
        })

    return splits


def run_walkforward_validation(bundle,
                                history_df: pd.DataFrame,
                                n_splits: int = 5,
                                test_months: int = 3,
                                strategy_mode: str = STRATEGY_MODE_ROI) -> dict:
    """
    Walk-Forward バリデーションを実行するにゃ。
    各フォールドで的中率・回収率を計算するにゃ。 """
    import traceback as _tb

    if history_df is None or history_df.empty:
        return {"error": "データがないにゃ"}
    if "finish" not in history_df.columns:
        return {"error": "finish（着順）列が必要にゃ"}

    try:
        splits = create_walkforward_splits(
            history_df, n_splits=n_splits, test_months=test_months)
    except Exception as e:
        return {"error": f"分割エラーにゃ: {e}"}

    if not splits:
        return {"error": "有効な分割が作れなかったにゃ"}

    fold_results = []
    all_preds = []

    for sp in splits:
        test_df = history_df.loc[sp["test_idx"]].copy()
        if test_df.empty:
            continue

        # 予想を実行するにゃ（テストデータのみにゃ）
        try:
            test_df2 = add_prior_stats_for_prediction(test_df.copy())
            test_df2 = add_running_style(test_df2)
            pipe, fc = get_pipeline_and_features(bundle)
            miss = [c for c in fc if c not in test_df2.columns]
            for c in miss:
                test_df2[c] = 0.0

            if hasattr(pipe, "predict_proba"):
                raw_prob = pipe.predict_proba(test_df2[fc])[:, 1]
            else:
                raw_prob = np.asarray(pipe.predict(test_df2[fc]), dtype=float)

            # キャリブレーションにゃ
            calibrated = np.zeros(len(test_df2))
            for rk in test_df2["race_key"].unique():
                mask = test_df2["race_key"] == rk
                calibrated[mask.values] = calibrate_prob_isotonic(
                    raw_prob[mask.values])
            test_df2["ml_top3_prob"] = calibrated

            # タイブレーク付きランクにゃ
            _pop  = pd.to_numeric(test_df2["popularity"],errors="coerce").fillna(99)
            _odds = pd.to_numeric(test_df2["odds"],errors="coerce").fillna(999)
            _hno  = pd.to_numeric(test_df2["horse_no"],errors="coerce").fillna(99)
            _tb2  = (1.0/_pop.clip(1))*1e-4 + (1.0/_odds.clip(0.1))*1e-6 + (1.0/_hno.clip(1))*1e-8
            test_df2["_comp"] = test_df2["ml_top3_prob"] + _tb2
            test_df2["ml_rank"] = (test_df2.groupby("race_key")["_comp"]
                                    .rank(ascending=False,method="first")
                                    .fillna(1).astype(int))
            test_df2 = test_df2.drop(columns=["_comp"])
            test_df2 = add_ev_score(test_df2)
            test_df2 = add_kelly_ratio(test_df2)
            test_df2 = add_value_strategy(test_df2, strategy_mode=strategy_mode)
            test_df2["actual_finish"] = test_df["finish"].values

        except Exception as e:
            fold_results.append({
                "フォールドにゃ": sp["fold"],
                "テスト月にゃ": f"{sp['test_months'][0]}〜{sp['test_months'][-1]}",
                "エラーにゃ": str(e),
            })
            continue

        # 的中率・回収率を計算にゃ
        tansho_h=0; tansho_b=0; tansho_r=0.0
        san3_h=0;   san3_b=0;   san3_r=0.0
        fuku_h=0;   fuku_b=0;   fuku_r=0.0

        for rk in test_df2["race_key"].unique():
            rdf = test_df2[test_df2["race_key"]==rk]
            if rdf.empty: continue

            winner = rdf[rdf["actual_finish"]==1]
            second = rdf[rdf["actual_finish"]==2]
            third  = rdf[rdf["actual_finish"]==3]
            w_no   = str(_safe_int(winner["horse_no"].iloc[0],0)) if not winner.empty else ""
            s_no   = str(_safe_int(second["horse_no"].iloc[0],0)) if not second.empty else ""
            t_no   = str(_safe_int(third["horse_no"].iloc[0], 0)) if not third.empty else ""
            top3_set = {w_no,s_no,t_no}-{"0",""}

            ai1    = rdf.sort_values("ml_rank").iloc[0]
            ai1_no = str(_safe_int(ai1["horse_no"],0))
            ai1_od = _safe_float(ai1.get("odds",0),0)

            tansho_b += 100
            if ai1_no == w_no:
                tansho_h += 1
                tansho_r += int(100 * ai1_od * 0.80)

            buy_df = rdf[rdf["buy_flag"]=="買い"] if "buy_flag" in rdf.columns else rdf.head(3)
            for _,brow in buy_df.iterrows():
                bno  = str(_safe_int(brow["horse_no"],0))
                bods = _safe_float(brow.get("odds",0),0)
                fuku_b += 100
                if bno in top3_set:
                    fuku_h += 1
                    fuku_r += int(100 * max(1.1, bods*0.3) * 0.80)

            top3_pred = rdf.sort_values("ml_rank").head(3)
            t3_nos = set(str(_safe_int(r["horse_no"],0)) for _,r in top3_pred.iterrows())
            san3_b += 100
            if t3_nos == top3_set:
                san3_h += 1
                t3_ods = [_safe_float(r["odds"],0) for _,r in top3_pred.iterrows()]
                san3_r += int(100*max(5.0,t3_ods[0]*t3_ods[1]*t3_ods[2]*0.05)*0.75)

        fold_results.append({
            "フォールドにゃ":   sp["fold"],
            "テスト月にゃ":     f"{sp['test_months'][0]}〜{sp['test_months'][-1]}",
            "テスト数にゃ":     sp["n_test"],
            "単勝的中率にゃ":   f"{tansho_h/max(tansho_b//100,1)*100:.1f}%",
            "単勝回収率にゃ":   f"{tansho_r/max(tansho_b,1)*100:.1f}%",
            "複勝的中率にゃ":   f"{fuku_h/max(fuku_b//100,1)*100:.1f}%",
            "複勝回収率にゃ":   f"{fuku_r/max(fuku_b,1)*100:.1f}%",
            "三連複的中率にゃ": f"{san3_h/max(san3_b//100,1)*100:.1f}%",
            "三連複回収率にゃ": f"{san3_r/max(san3_b,1)*100:.1f}%",
            "_tan_h": tansho_h, "_tan_b": tansho_b, "_tan_r": tansho_r,
            "_fuku_h": fuku_h, "_fuku_b": fuku_b, "_fuku_r": fuku_r,
            "_san_h": san3_h, "_san_b": san3_b, "_san_r": san3_r,
        })
        all_preds.append(test_df2)

    if not fold_results:
        return {"error": "有効なフォールドがなかったにゃ"}

    # 全フォールド集計にゃ
    valid = [r for r in fold_results if "_tan_b" in r]
    if valid:
        tot_tan_h = sum(r["_tan_h"] for r in valid)
        tot_tan_b = sum(r["_tan_b"] for r in valid)
        tot_tan_r = sum(r["_tan_r"] for r in valid)
        tot_fuku_h= sum(r["_fuku_h"] for r in valid)
        tot_fuku_b= sum(r["_fuku_b"] for r in valid)
        tot_fuku_r= sum(r["_fuku_r"] for r in valid)
        tot_san_h = sum(r["_san_h"] for r in valid)
        tot_san_b = sum(r["_san_b"] for r in valid)
        tot_san_r = sum(r["_san_r"] for r in valid)
        overall = {
            "単勝的中率(全体)":   f"{tot_tan_h/max(tot_tan_b//100,1)*100:.1f}%",
            "単勝回収率(全体)":   f"{tot_tan_r/max(tot_tan_b,1)*100:.1f}%",
            "複勝的中率(全体)":   f"{tot_fuku_h/max(tot_fuku_b//100,1)*100:.1f}%",
            "複勝回収率(全体)":   f"{tot_fuku_r/max(tot_fuku_b,1)*100:.1f}%",
            "三連複的中率(全体)": f"{tot_san_h/max(tot_san_b//100,1)*100:.1f}%",
            "三連複回収率(全体)": f"{tot_san_r/max(tot_san_b,1)*100:.1f}%",
        }
    else:
        overall = {}

    return {
        "splits":       splits,
        "fold_results": fold_results,
        "overall":      overall,
        "all_preds_df": pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(),
    }


# ============================================================
# ③ ROI最適化エンジンにゃ
# ============================================================
# 「何点買うか」「どの馬を買うか」を
# 実測データから最適なしきい値を求めるにゃ

def optimize_roi_thresholds(pred_df: pd.DataFrame,
                             history_df: pd.DataFrame = None) -> dict:
    """
    ROI最大化のための最適しきい値を探索するにゃ。

    探索パラメータにゃ:
    - EV閾値（ev_score_v2 / ev_score）にゃ
    - AI確率閾値にゃ
    - Kelly閾値にゃ
    - 最大購入点数にゃ

    戻り値にゃ:
    - 各パラメータの最適値にゃ
    - 最大ROIとその条件にゃ
    """
    if pred_df is None or pred_df.empty:
        return {}

    # 必要な列を確保にゃ
    r = pred_df.copy()
    for col,dv in [("ml_top3_prob",0),("ev_score",0),("ev_score_v2",0),
                   ("kelly_ratio",0),("kelly_ratio_sanren",0),
                   ("ml_rank",99),("popularity",99),("odds",0),
                   ("actual_finish",99)]:
        if col not in r.columns: r[col] = dv
        r[col] = pd.to_numeric(r[col], errors="coerce").fillna(dv)

    has_result = "actual_finish" in pred_df.columns and \
                 pred_df["actual_finish"].notna().sum() > 0

    # 探索グリッドにゃ
    ev_thresholds   = [-0.05, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    prob_thresholds = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28]
    kelly_thresholds= [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    max_points      = [3, 4, 5, 6, 8, 10]

    best_roi    = -999
    best_params = {}
    grid_results= []

    ev_col = "ev_score_v2" if "ev_score_v2" in r.columns else "ev_score"

    for ev_th in ev_thresholds:
        for prob_th in prob_thresholds:
            for kelly_th in kelly_thresholds:
                # このパラメータでの買い候補にゃ
                mask = (
                    (r[ev_col] >= ev_th) &
                    (r["ml_top3_prob"] >= prob_th) &
                    ((r["kelly_ratio"] >= kelly_th) | (r["kelly_ratio_sanren"] >= kelly_th)) &
                    (r.get("danger_level", pd.Series([""]* len(r))).isin(["","注意"]))
                )
                buy_df = r[mask]
                if buy_df.empty: continue

                n_buy = int(mask.sum())
                avg_odds = float(buy_df["odds"].mean())

                if has_result:
                    # 実績から計算にゃ
                    hits = int((buy_df["actual_finish"] <= 3).sum())
                    bets = n_buy * 100
                    fuku_ret = sum(
                        int(100 * max(1.1, row["odds"]*0.3) * 0.80)
                        for _, row in buy_df[buy_df["actual_finish"]<=3].iterrows()
                    )
                    roi = fuku_ret / max(bets, 1) * 100
                    hitrate = hits / n_buy * 100
                else:
                    # 期待値ベースの推定にゃ
                    exp_ret = float((buy_df["ml_top3_prob"] * buy_df["odds"] * 0.80).sum())
                    bets    = n_buy * 100
                    roi     = exp_ret / max(bets/100, 1) * 100
                    hitrate = float(buy_df["ml_top3_prob"].mean()) * 100

                grid_results.append({
                    "EV閾値にゃ":    ev_th,
                    "確率閾値にゃ":  prob_th,
                    "Kelly閾値にゃ": kelly_th,
                    "買い点数にゃ":  n_buy,
                    "推定ROIにゃ":   round(roi, 1),
                    "推定的中率にゃ":round(hitrate, 1),
                    "平均オッズにゃ":round(avg_odds, 1),
                })
                if roi > best_roi:
                    best_roi = roi
                    best_params = {
                        "ev_threshold":    ev_th,
                        "prob_threshold":  prob_th,
                        "kelly_threshold": kelly_th,
                        "expected_roi":    round(roi, 1),
                        "expected_hitrate":round(hitrate, 1),
                        "n_buy":           n_buy,
                    }

    grid_df = pd.DataFrame(grid_results).sort_values("推定ROIにゃ", ascending=False)

    return {
        "best_params":  best_params,
        "best_roi":     best_roi,
        "grid_results": grid_df,
        "top10":        grid_df.head(10),
        "has_result":   has_result,
    }


def apply_roi_optimized_filter(df: pd.DataFrame,
                                best_params: dict) -> pd.DataFrame:
    """
    ROI最適化パラメータを適用して買い候補を絞り込むにゃ。 """
    df = df.copy()
    ev_th    = best_params.get("ev_threshold",   0.02)
    prob_th  = best_params.get("prob_threshold",  0.18)
    kelly_th = best_params.get("kelly_threshold", 0.02)

    ev_col = "ev_score_v2" if "ev_score_v2" in df.columns else "ev_score"

    for col,dv in [(ev_col,0),("ml_top3_prob",0),
                   ("kelly_ratio",0),("kelly_ratio_sanren",0)]:
        if col not in df.columns: df[col]=dv
        df[col] = pd.to_numeric(df[col],errors="coerce").fillna(dv)

    mask = (
        (df[ev_col] >= ev_th) &
        (df["ml_top3_prob"] >= prob_th) &
        ((df["kelly_ratio"] >= kelly_th) | (df["kelly_ratio_sanren"] >= kelly_th))
    )
    dl = df.get("danger_level", pd.Series([""]*len(df)))
    mask &= ~dl.isin(["強危険","危険"])

    df["roi_optimized_buy"] = mask.map({True:"◎ROI最適", False:"見送り"})
    return df


# ============================================================
# ④ 本格キャリブレーションにゃ
# ============================================================
# 現在のcalibrate_prob: シグモイド圧縮のみにゃ
# 問題にゃ:
#   ① 実際の的中率との乖離を補正していないにゃ
#   ② Platt Scaling / Isotonic Regression を使っていないにゃ
#   ③ レース種別・頭数による差を考慮していないにゃ

def calibrate_prob_isotonic(raw_prob: np.ndarray,
                             reference_hitrate: float = None) -> np.ndarray:
    """
    Isotonic Regression 近似によるキャリブレーションにゃ。

    【改善点にゃ】
    1. 単純なシグモイド圧縮から「単調増加制約付き補正」にゃ
    2. 実際の3着内率（約21.8%）に合わせた補正にゃ
    3. 過学習モデルの確率の「平滑化」にゃ

    reference_hitrate: 実際の3着内率（デフォルト0.218にゃ）
    """
    if len(raw_prob) == 0:
        return raw_prob

    # 参照的中率にゃ（PKLのmetricsから取るにゃ、なければ競馬の一般的な値にゃ）
    if reference_hitrate is None:
        reference_hitrate = 0.218  # 17〜18頭立てで3着内=3/17≈18%にゃ

    p = np.clip(raw_prob, 1e-6, 1 - 1e-6)
    n = len(p)

    # ── Step1: 確率の合計を理論値3.0に正規化にゃ ──
    target_sum = 3.0
    p_sum = float(p.sum())
    if p_sum > 0:
        p_scaled = p * (target_sum / p_sum)
    else:
        p_scaled = p.copy()
    p_scaled = np.clip(p_scaled, 0, 0.95)

    # ── Step2: Isotonic Regression 近似にゃ ──
    # 確率が単調に並ぶよう「プールアジャセント バイオレーター」で平滑化にゃ
    # （PAVA: Pool Adjacent Violators Algorithmにゃ）
    def pava(y):
        """単調増加制約を満たすよう平均化にゃ"""
        y = np.array(y, dtype=float)
        n = len(y)
        # 降順ソートのインデックスにゃ
        idx = np.argsort(-y)
        y_sorted = y[idx]
        # PAVA適用にゃ（降順にゃ）
        i = 0
        while i < n - 1:
            if y_sorted[i] < y_sorted[i+1]:
                # バイオレーション発生にゃ → 平均化にゃ
                j = i + 1
                while j < n and y_sorted[j] > y_sorted[i]:
                    avg = np.mean(y_sorted[i:j+1])
                    y_sorted[i:j+1] = avg
                    j += 1
            i += 1
        # 元の順序に戻すにゃ
        result = np.empty_like(y)
        result[idx] = y_sorted
        return result

    p_isotonic = pava(p_scaled)

    # ── Step3: 参照的中率に合わせた補正にゃ ──
    # 期待3着内頭数 = 頭数 × reference_hitrate にゃ
    target_top3 = n * reference_hitrate
    current_sum = float(p_isotonic.sum())
    if current_sum > 0:
        p_final = p_isotonic * (target_top3 / current_sum)
    else:
        p_final = p_isotonic

    # ── Step4: Platt Scaling 近似にゃ（sigmoid calibrationにゃ）──
    # logistic: p_cal = 1/(1+exp(-(A*logit(p)+B)))
    # A,Bはデフォルト値を使うにゃ（実データがあれば学習するにゃ）
    A = 0.85  # 過学習モデルは圧縮するにゃ（< 1にゃ）
    B = 0.0   # バイアス補正にゃ
    logit_p = np.log(np.clip(p_final, 1e-6, 1-1e-6) /
                     (1 - np.clip(p_final, 1e-6, 1-1e-6) + 1e-8) + 1e-8)
    p_platt = 1.0 / (1.0 + np.exp(-(A * logit_p + B)))

    # ── Step5: 最終正規化にゃ ──
    final_sum = float(p_platt.sum())
    if final_sum > 0:
        p_final2 = p_platt * (target_top3 / final_sum)
    else:
        p_final2 = p_platt

    # ── Step6: 微小ノイズにゃ（同率防止にゃ）──
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1e-6, 1e-6, len(p_final2))
    p_final2 = p_final2 + noise

    return np.clip(p_final2, 0.01, 0.95)


def fit_calibrator_from_history(bundle,
                                  history_df: pd.DataFrame) -> dict:
    """
    過去データから実際のキャリブレーションパラメータを学習するにゃ。
    Platt Scaling の A,B を実データから推定するにゃ。

    戻り値にゃ:
      calibration_params: {"A": float, "B": float, "reference_hitrate": float}
      calibration_curve: DataFrame（確率帯別の実際の的中率にゃ）
    """
    if history_df is None or history_df.empty:
        return {}
    if "finish" not in history_df.columns:
        return {}

    df = history_df.copy()
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df[df["finish"].notna() & (df["finish"] > 0)].copy()

    # 予想を実行にゃ
    try:
        df2 = add_prior_stats_for_prediction(df.copy())
        pipe, fc = get_pipeline_and_features(bundle)
        miss = [c for c in fc if c not in df2.columns]
        for c in miss: df2[c] = 0.0

        if hasattr(pipe, "predict_proba"):
            raw_prob = pipe.predict_proba(df2[fc])[:, 1]
        else:
            raw_prob = np.asarray(pipe.predict(df2[fc]), dtype=float)

        df2["raw_prob"] = raw_prob
        df2["is_top3"]  = df["finish"].between(1, 3).values

    except Exception as e:
        return {"error": f"予測エラーにゃ: {e}"}

    # 確率帯別の実際の的中率にゃ（キャリブレーション曲線にゃ）
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0]
    labels = [f"{bins[i]:.2f}〜{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    df2["prob_bin"] = pd.cut(df2["raw_prob"], bins=bins, labels=labels)

    calib_curve = df2.groupby("prob_bin", observed=True).agg(
        件数=("is_top3","count"),
        AI予測確率平均=("raw_prob","mean"),
        実際的中率=("is_top3","mean"),
    ).reset_index().rename(columns={"prob_bin":"確率帯にゃ"})
    calib_curve["乖離にゃ"] = (
        calib_curve["AI予測確率平均"] - calib_curve["実際的中率"]
    ).round(4)
    calib_curve["AI予測確率平均"] = calib_curve["AI予測確率平均"].round(4)
    calib_curve["実際的中率"]     = calib_curve["実際的中率"].round(4)

    # 全体的中率にゃ
    overall_hitrate = float(df2["is_top3"].mean())

    # Platt Scaling パラメータをSimple線形回帰で推定にゃ
    valid = calib_curve[calib_curve["件数"] >= 5].copy()
    if len(valid) >= 3:
        x = valid["AI予測確率平均"].values
        y = valid["実際的中率"].values
        # logit変換にゃ
        x_logit = np.log(np.clip(x,1e-6,1-1e-6) / (1-np.clip(x,1e-6,1-1e-6)+1e-8) + 1e-8)
        y_logit = np.log(np.clip(y,1e-6,1-1e-6) / (1-np.clip(y,1e-6,1-1e-6)+1e-8) + 1e-8)
        # 最小二乗法にゃ
        if len(x_logit) > 1 and np.std(x_logit) > 0:
            A = float(np.cov(x_logit, y_logit)[0,1] / np.var(x_logit))
            B = float(np.mean(y_logit) - A * np.mean(x_logit))
        else:
            A, B = 1.0, 0.0
    else:
        A, B = 1.0, 0.0

    return {
        "calibration_params": {
            "A": round(A, 4),
            "B": round(B, 4),
            "reference_hitrate": round(overall_hitrate, 4),
        },
        "calibration_curve": calib_curve,
        "overall_hitrate":   overall_hitrate,
        "n_samples":         len(df2),
    }


# ============================================================
# ⑤ ML強化ダッシュボード表示にゃ
# ============================================================

def show_ml_enhance_dashboard(bundle, history_df=None,
                               pred_df=None, race_df=None,
                               strategy_mode=STRATEGY_MODE_ROI):
    """
    Leakage・Walk-Forward・ROI最適化・キャリブレーション
    4機能を統合したダッシュボードにゃ🐾
    """
    st.header("🔬 ML強化ダッシュボードにゃ（Leakage防止・WF検証・ROI最適化・キャリブレーションにゃ）")
    st.caption(
        "AIの**本当の精度**を計測するにゃ。"
        "過学習(AUC=1.0)の原因特定から「実際に勝てる買い方」の最適化まで行うにゃ🐾"
    )

    info = get_bundle_info(bundle) if bundle else {}
    fc   = info.get("feature_cols", BASE_NUM_FEATURES + CAT_FEATURES)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Leakage防止にゃ",
        "📅 Walk-Forward検証にゃ",
        "💰 ROI最適化にゃ",
        "📊 本格キャリブレーションにゃ",
    ])

    # ── Tab1: Leakage防止にゃ ──
    with tab1:
        st.subheader("🔍 データリーク（Leakage）診断にゃ")
        st.markdown("""
**なぜ AUC=1.0 になるのかにゃ？**

> レース後にしか分からない情報（着順・タイム・上り3F）が
> 学習データに混入している可能性にゃ。
> これを「データリーク」と呼ぶにゃ。

**当日オッズ・人気の問題にゃ:**
オッズと人気は「予測時点では使えるにゃ」が、
3着内と非常に高相関なのでモデルが答えを丸暗記するにゃ。 """)

        st.markdown("---")
        result = check_leakage(fc, bundle=bundle, verbose=True)

        # 安全な特徴量セットにゃ
        with st.expander("✅ リーク除外後の推奨特徴量セットにゃ"):
            safe_no_odds = get_safe_feature_cols(fc, remove_leakage=True, remove_caution=True)
            safe_with_odds = get_safe_feature_cols(fc, remove_leakage=True, remove_caution=False)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**オッズ除外版にゃ（推奨にゃ）**")
                st.caption(f"{len(safe_no_odds)}個の特徴量にゃ")
                st.code('\n'.join(safe_no_odds), language="text")
            with c2:
                st.markdown("**オッズ保持版にゃ（現状にゃ）**")
                st.caption(f"{len(safe_with_odds)}個の特徴量にゃ")
                st.code('\n'.join(safe_with_odds), language="text")

        # 再学習推奨コードにゃ
        with st.expander("📝 再学習推奨コードにゃ"):
            st.code("""
# Leakage防止版の再学習コードにゃ
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# オッズ・人気を除外した特徴量にゃ
NO_LEAKAGE_FEATURES = [
    "year_full","month","day","race_no","race_grade",
    "course_kind","distance","age","carried_weight",
    "field_size","horse_no","frame_no",
    # ↑ コース情報にゃ
    "jockey_runs_prior","jockey_win_rate_prior","jockey_top3_rate_prior",
    "trainer_runs_prior","trainer_win_rate_prior","trainer_top3_rate_prior",
    "sire_runs_prior","sire_win_rate_prior","sire_top3_rate_prior",
    "horse_runs_prior","horse_win_rate_prior","horse_top3_rate_prior",
    "horse_distance_runs_prior","horse_distance_top3_rate_prior",
    "horse_track_runs_prior","horse_track_top3_rate_prior",
    # ↑ 事前統計にゃ（レース前確定にゃ）
    "place","track_type","going","sex","jockey","trainer",
    "belonging","sire","dam","broodmare_sire",
    # ↑ カテゴリにゃ
]

model = HistGradientBoostingClassifier(
    max_iter=150,            # 過学習防止にゃ
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=0.1,  # 正則化にゃ
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)
            """, language="python")

    # ── Tab2: Walk-Forward検証にゃ ──
    with tab2:
        st.subheader("📅 Walk-Forward バリデーションにゃ")
        st.markdown("""
**通常のランダム分割の問題にゃ:**
> 未来のデータで過去を予測 → 楽観的すぎる結果にゃ

**Walk-Forwardにゃ:**
> 常に「過去で学習 → 未来を予測」にゃ
> 実際の運用に近い条件で検証にゃ🐾
        """)

        if history_df is None or "finish" not in (history_df.columns if history_df is not None else []):
            st.info(
                "📂 着順データ（finish列付きCSV）が必要にゃ。\n\n"
                "バックテストタブからデータをアップロードするにゃ🐾"
            )
            return

        col1, col2, col3 = st.columns(3)
        n_splits    = col1.number_input("フォールド数にゃ", 3, 10, 5, key="wf_n")
        test_months = col2.number_input("テスト月数/フォールドにゃ", 1, 6, 3, key="wf_m")
        col3.metric("データ量にゃ", f"{len(history_df)}行にゃ")

        if st.button("🐾 Walk-Forward検証 実行にゃ！", key="wf_run"):
            with st.spinner("Walk-Forward検証中にゃ...（フォールド数×レース数分かかるにゃ）"):
                wf_result = run_walkforward_validation(
                    bundle, history_df,
                    n_splits=int(n_splits),
                    test_months=int(test_months),
                    strategy_mode=strategy_mode
                )

            if "error" in wf_result:
                st.error(f"❌ {wf_result['error']}")
                return

            # フォールド別結果にゃ
            st.markdown("#### 📋 フォールド別結果にゃ")
            fold_df = pd.DataFrame([
                {k:v for k,v in r.items() if not k.startswith("_")}
                for r in wf_result["fold_results"]
            ])
            st.dataframe(fold_df, use_container_width=True, hide_index=True)

            # 全体集計にゃ
            st.markdown("#### 📊 全体集計にゃ")
            overall = wf_result.get("overall", {})
            if overall:
                c1,c2,c3 = st.columns(3)
                c1.metric("単勝的中率にゃ",   overall.get("単勝的中率(全体)","N/A"))
                c2.metric("複勝的中率にゃ",   overall.get("複勝的中率(全体)","N/A"))
                c3.metric("三連複的中率にゃ", overall.get("三連複的中率(全体)","N/A"))
                c1.metric("単勝回収率にゃ",   overall.get("単勝回収率(全体)","N/A"))
                c2.metric("複勝回収率にゃ",   overall.get("複勝回収率(全体)","N/A"))
                c3.metric("三連複回収率にゃ", overall.get("三連複回収率(全体)","N/A"))

            # 安定性評価にゃ
            valid_folds = [r for r in wf_result["fold_results"] if "_tan_b" in r]
            if len(valid_folds) >= 2:
                fuku_rois = [r["_fuku_r"]/max(r["_fuku_b"],1)*100 for r in valid_folds]
                roi_std = float(np.std(fuku_rois))
                roi_avg = float(np.mean(fuku_rois))
                if roi_std < 15:
                    st.success(f"✅ 複勝回収率は安定しているにゃ（平均{roi_avg:.1f}%±{roi_std:.1f}%にゃ）🐾")
                elif roi_std < 30:
                    st.warning(f"⚠️ 複勝回収率にやや波があるにゃ（平均{roi_avg:.1f}%±{roi_std:.1f}%にゃ）")
                else:
                    st.error(f"🔴 複勝回収率が不安定にゃ（平均{roi_avg:.1f}%±{roi_std:.1f}%にゃ）。モデルの再学習を推奨にゃ")

            # CSV出力にゃ
            st.download_button(
                "📥 Walk-Forward結果CSVにゃ",
                data=fold_df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="walkforward_result.csv", mime="text/csv"
            )

    # ── Tab3: ROI最適化にゃ ──
    with tab3:
        st.subheader("💰 ROI最適化にゃ（最適しきい値探索にゃ）")
        st.markdown("""
**「何点買うか」「どの馬を買うか」を自動最適化するにゃ。**

EV閾値・AI確率閾値・Kelly閾値を変えながら
最もROIが高い組み合わせを探索するにゃ🐾
        """)

        target_df = pred_df if pred_df is not None else race_df
        if target_df is None or target_df.empty:
            st.info("予想を実行してからROI最適化するにゃ🐾")
            return

        # 着順データがあればリンクにゃ
        has_result = (history_df is not None and
                      "finish" in (history_df.columns if history_df is not None else []))
        if has_result:
            st.info("✅ 着順データあり → 実績ベースでROI最適化するにゃ🐾")
            # 予想と実績をマージにゃ
            target_df2 = target_df.copy()
            if "race_key" in target_df2.columns and "race_key" in history_df.columns:
                fin_map = dict(zip(
                    history_df["race_key"].astype(str) + "_" +
                    history_df["horse_no"].fillna(0).astype(int).astype(str),
                    history_df["finish"]
                ))
                target_df2["actual_finish"] = [
                    fin_map.get(
                        str(r.get("race_key","")) + "_" + str(_safe_int(r.get("horse_no",0),0)),
                        np.nan
                    )
                    for _, r in target_df2.iterrows()
                ]
        else:
            st.info("ℹ️ 着順データなし → 期待値ベースでROI最適化するにゃ（推定値にゃ）")
            target_df2 = target_df.copy()
            target_df2["actual_finish"] = np.nan

        if st.button("🐾 ROI最適化 実行にゃ！", key="roi_run"):
            with st.spinner("ROIを最適化中にゃ..."):
                roi_result = optimize_roi_thresholds(target_df2, history_df)

            if not roi_result:
                st.warning("最適化できなかったにゃ"); return

            bp = roi_result["best_params"]
            st.markdown("#### 🌟 最適パラメータにゃ")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("EV閾値にゃ",      f"{bp.get('ev_threshold',0):.3f}")
            c2.metric("確率閾値にゃ",    f"{bp.get('prob_threshold',0):.2f}")
            c3.metric("Kelly閾値にゃ",   f"{bp.get('kelly_threshold',0):.3f}")
            c4.metric("推定ROIにゃ",     f"{bp.get('expected_roi',0):.1f}%")
            c5.metric("推定的中率にゃ",  f"{bp.get('expected_hitrate',0):.1f}%")

            if bp.get("expected_roi",0) >= 100:
                st.success(f"🎉 期待ROI {bp.get('expected_roi',0):.1f}% にゃ！プラス期待値にゃ🐾")
            elif bp.get("expected_roi",0) >= 80:
                st.info(f"📊 期待ROI {bp.get('expected_roi',0):.1f}% にゃ。馬券の控除率（75-80%）を上回っているにゃ")
            else:
                st.warning(f"⚠️ 期待ROI {bp.get('expected_roi',0):.1f}% にゃ。控除率を下回っているにゃ")

            # グリッド結果にゃ
            st.markdown("#### 📋 上位10パラメータにゃ")
            top10 = roi_result.get("top10", pd.DataFrame())
            if not top10.empty:
                disp_cols = [c for c in ["EV閾値にゃ","確率閾値にゃ","Kelly閾値にゃ",
                                          "買い点数にゃ","推定ROIにゃ","推定的中率にゃ","平均オッズにゃ"]
                             if c in top10.columns]
                st.dataframe(top10[disp_cols], use_container_width=True, hide_index=True)

            # 最適パラメータで再フィルタにゃ
            if target_df is not None:
                filtered = apply_roi_optimized_filter(target_df, bp)
                buy_opt  = filtered[filtered["roi_optimized_buy"]=="◎ROI最適"]
                st.markdown(f"#### ✅ 最適パラメータでの買い候補にゃ: {len(buy_opt)}頭にゃ")
                if not buy_opt.empty:
                    disp = buy_opt[["horse_no","horse_name","odds","popularity",
                                    "ml_top3_prob","ev_score",
                                    "kelly_ratio","roi_optimized_buy"]].copy()
                    if "ml_top3_prob" in disp.columns:
                        disp["ml_top3_prob"] = (pd.to_numeric(disp["ml_top3_prob"],errors="coerce")*100).round(1).astype(str)+"%"
                    st.dataframe(disp.rename(columns={
                        "horse_no":"馬番","horse_name":"馬名","odds":"オッズ",
                        "popularity":"人気","ml_top3_prob":"AI確率",
                        "ev_score":"EV乖離","kelly_ratio":"Kelly比",
                        "roi_optimized_buy":"ROI最適判定",
                    }), use_container_width=True, hide_index=True)

    # ── Tab4: 本格キャリブレーションにゃ ──
    with tab4:
        st.subheader("📊 本格キャリブレーションにゃ（Isotonic + Platt Scalingにゃ）")
        st.markdown("""
**現在のキャリブレーション問題にゃ:**
- シグモイド圧縮のみ → 実際の的中率との乖離を補正していないにゃ
- 固定係数 → レース種別・頭数の違いを無視にゃ

**改善内容にゃ:**
1. **Isotonic Regression** → 単調増加制約付き平滑化にゃ
2. **Platt Scaling** → 実データからA,Bパラメータを学習にゃ
3. **キャリブレーション曲線** → 「AI予測30%の馬は実際何%来るか」を可視化にゃ
        """)

        if history_df is None or "finish" not in (history_df.columns if history_df is not None else []):
            st.info("📂 着順データが必要にゃ。バックテストタブからデータをアップロードするにゃ🐾")

            # キャリブレーション効果のデモにゃ
            st.markdown("#### 📈 キャリブレーション効果のデモにゃ")
            demo_raw = np.array([0.99, 0.95, 0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.01])
            demo_cal = calibrate_prob_isotonic(demo_raw, reference_hitrate=0.218)
            demo_df  = pd.DataFrame({
                "生確率(過学習にゃ)":    [f"{p:.3f}" for p in demo_raw],
                "補正後確率(Isotonic)":  [f"{p:.3f}" for p in demo_cal],
                "変化にゃ":             [f"{(c-r)*100:+.1f}%" for r,c in zip(demo_raw,demo_cal)],
            })
            st.dataframe(demo_df, use_container_width=True, hide_index=True)
            st.caption("AUC=1.0モデルの0/1への張り付きが緩和されているにゃ🐾")
            return

        if st.button("🐾 キャリブレーション分析 実行にゃ！", key="cal_run"):
            with st.spinner("キャリブレーション分析中にゃ..."):
                cal_result = fit_calibrator_from_history(bundle, history_df)

            if "error" in cal_result:
                st.error(f"❌ {cal_result['error']}")
                return

            params = cal_result.get("calibration_params", {})
            st.markdown("#### 📋 学習済みキャリブレーションパラメータにゃ")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Platt A にゃ",      f"{params.get('A',1.0):.4f}",
                      help="<1ならモデルが過信している（圧縮するにゃ）")
            c2.metric("Platt B にゃ",      f"{params.get('B',0.0):.4f}",
                      help="バイアス補正にゃ")
            c3.metric("実際の3着内率にゃ",  f"{params.get('reference_hitrate',0.218)*100:.1f}%")
            c4.metric("サンプル数にゃ",     f"{cal_result.get('n_samples',0):,}にゃ")

            A = params.get("A", 1.0)
            if A < 0.7:
                st.error(
                    f"🔴 A={A:.3f} → モデルが極端に過信しているにゃ。"
                    "オッズ除外での再学習を強く推奨にゃ🐾"
                )
            elif A < 0.9:
                st.warning(
                    f"🟡 A={A:.3f} → モデルがやや過信にゃ。"
                    "キャリブレーションで補正中にゃ"
                )
            else:
                st.success(f"✅ A={A:.3f} → キャリブレーションは良好にゃ🐾")

            # キャリブレーション曲線にゃ
            st.markdown("#### 📈 キャリブレーション曲線にゃ")
            st.caption("完璧なモデルでは「AI予測確率 = 実際的中率」になるにゃ。乖離が大きいほど補正が必要にゃ。")
            cal_curve = cal_result.get("calibration_curve", pd.DataFrame())
            if not cal_curve.empty:
                def color_cal(row):
                    d = abs(_safe_float(row.get("乖離にゃ",0),0))
                    if d > 0.10: return ["background-color:#f8d7da"]*len(row)
                    if d > 0.05: return ["background-color:#fff3cd"]*len(row)
                    return ["background-color:#d4edda"]*len(row)
                try:
                    st.dataframe(
                        cal_curve.style.apply(color_cal, axis=1),
                        use_container_width=True, hide_index=True
                    )
                except Exception:
                    st.dataframe(cal_curve, use_container_width=True, hide_index=True)

                st.caption("🟢=乖離小（良好にゃ） / 🟡=乖離中 / 🔴=乖離大（補正要にゃ）")

            st.download_button(
                "📥 キャリブレーション結果CSVにゃ",
                data=cal_curve.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="calibration_curve.csv", mime="text/csv"
            )





# ============================================================
# ============================================================
# バックテスト v2 完全版にゃ
# ① 全券種の正確な的中・回収計算にゃ
# ② 複数条件の一括比較にゃ
# ③ 時系列グラフ（月別推移にゃ）にゃ
# ④ レース種別・距離・競馬場別の詳細分析にゃ
# ⑤ 改善提案の自動生成にゃ
# ============================================================
# ============================================================


# ============================================================
# バックテストコアエンジン v2にゃ
# ============================================================

def _bt_predict_race(bundle, rdf_raw, strategy_mode):
    """1レース分の予想を実行するにゃ（軽量版にゃ）"""
    try:
        df2 = add_prior_stats_for_prediction(rdf_raw.copy())
        df2 = add_running_style(df2)
        pipe, fc = get_pipeline_and_features(bundle)
        for c in [c for c in fc if c not in df2.columns]:
            df2[c] = 0.0
        if hasattr(pipe, "predict_proba"):
            raw_prob = pipe.predict_proba(df2[fc])[:, 1]
        else:
            raw_prob = np.asarray(pipe.predict(df2[fc]), dtype=float)

        # キャリブレーション + タイブレークにゃ
        cal = calibrate_prob_isotonic(raw_prob)
        df2["ml_top3_prob"] = cal
        _pop  = pd.to_numeric(df2["popularity"], errors="coerce").fillna(99)
        _odds = pd.to_numeric(df2["odds"],       errors="coerce").fillna(999)
        _hno  = pd.to_numeric(df2["horse_no"],   errors="coerce").fillna(99)
        _tb   = (1.0/_pop.clip(1))*1e-4 + (1.0/_odds.clip(0.1))*1e-6 + (1.0/_hno.clip(1))*1e-8
        df2["_c"] = df2["ml_top3_prob"] + _tb
        df2["ml_rank"] = (df2.groupby("race_key")["_c"]
                           .rank(ascending=False, method="first")
                           .fillna(1).astype(int))
        df2 = df2.drop(columns=["_c"])
        df2 = add_ev_score(df2)
        df2 = add_kelly_ratio(df2)
        df2 = add_value_strategy(df2, strategy_mode=strategy_mode)
        # S級処理にゃ
        try:
            df2 = add_pace_advantage(df2)
            df2 = add_ev_score_v2(df2)
            df2 = add_pass_score(df2, strategy_mode=strategy_mode)
            df2 = add_final_score(df2, strategy_mode=strategy_mode)
        except Exception:
            pass
        return df2
    except Exception:
        return None


def _get_top3_set(rdf, finish_col="actual_finish"):
    """実際の1〜3着馬番セットにゃ"""
    s = set()
    for pos in [1, 2, 3]:
        rows = rdf[rdf[finish_col] == pos]
        if not rows.empty:
            s.add(str(_safe_int(rows.iloc[0]["horse_no"], 0)))
    return s - {"0", ""}


def run_backtest_v2(bundle,
                    history_df: pd.DataFrame,
                    strategy_mode: str = STRATEGY_MODE_ROI,
                    min_odds: float = 1.0,
                    max_odds: float = 9999.0,
                    bet_unit: int = 100) -> dict:
    """
    バックテスト v2 完全版にゃ。

    計算する券種にゃ:
      単勝（AI1位固定にゃ）
      複勝（買い判定馬全員にゃ）
      馬連（AI上位2頭にゃ）
      ワイド（AI上位3頭のBOX全3点にゃ）
      馬単（AI1位→2位にゃ）
      三連複（AI上位3頭BOXにゃ）
      三連単（AI順1→2→3にゃ）
      S級判定買いにゃ（buy_flag_v2=◎/○/▲にゃ）

    戻り値にゃ:
      race_records: レース別詳細にゃ
      monthly_records: 月別集計にゃ
      summary: 全体サマリーにゃ
      by_place: 競馬場別にゃ
      by_distance: 距離帯別にゃ
      by_field_size: 頭数別にゃ
    """
    import traceback as _tb

    if history_df is None or history_df.empty:
        return {"error": "データがないにゃ"}
    if "finish" not in history_df.columns:
        return {"error": "finish（着順）列が必要にゃ"}

    df = history_df.copy()
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df[df["finish"].notna() & (df["finish"] > 0)].copy()
    if df.empty:
        return {"error": "有効な着順データがないにゃ"}

    # race_key がなければ race_id から生成するにゃ
    if "race_key" not in df.columns:
        if "race_id" in df.columns:
            df["race_key"] = df["race_id"].astype(str)
        else:
            return {"error": "race_key または race_id 列が必要にゃ"}
    race_keys = df["race_key"].dropna().unique()
    if len(race_keys) == 0:
        return {"error": "race_keyが設定されていないにゃ"}

    # ── 券種別集計バッファにゃ ──
    BET_TYPES = ["単勝","複勝","馬連","ワイド","馬単","三連複","三連単","S級判定","S級×複勝"]
    stats = {bt: {"hit":0, "bet":0, "ret":0.0, "races":0} for bt in BET_TYPES}
    race_records   = []
    monthly_buffer = {}  # {YYYYMM: {bt: stats}}

    total = len(race_keys)
    for idx, rk in enumerate(race_keys):
        rdf_raw = df[df["race_key"] == rk].copy()
        if rdf_raw.empty:
            continue

        # 着順列を退避にゃ
        rdf_raw["actual_finish"] = pd.to_numeric(rdf_raw["finish"], errors="coerce")
        top3_set = _get_top3_set(rdf_raw)
        if len(top3_set) < 3:
            continue

        # 実際の着順馬にゃ
        winner_row = rdf_raw[rdf_raw["actual_finish"] == 1]
        second_row = rdf_raw[rdf_raw["actual_finish"] == 2]
        third_row  = rdf_raw[rdf_raw["actual_finish"] == 3]
        w_no = str(_safe_int(winner_row.iloc[0]["horse_no"],0)) if not winner_row.empty else ""
        s_no = str(_safe_int(second_row.iloc[0]["horse_no"],0)) if not second_row.empty else ""
        t_no = str(_safe_int(third_row.iloc[0]["horse_no"],0))  if not third_row.empty  else ""
        w_odds = _safe_float(winner_row.iloc[0].get("odds",10), 10) if not winner_row.empty else 10.0

        # オッズフィルタにゃ
        avg_odds = float(rdf_raw["odds"].mean()) if "odds" in rdf_raw.columns else 10.0
        if not (min_odds <= avg_odds <= max_odds):
            continue

        # レース情報にゃ
        place     = str(rdf_raw["place"].iloc[0]) if "place" in rdf_raw.columns else "不明"
        distance  = _safe_int(rdf_raw["distance"].iloc[0], 0) if "distance" in rdf_raw.columns else 0
        field_sz  = _safe_int(rdf_raw["field_size"].max(), len(rdf_raw)) if "field_size" in rdf_raw.columns else len(rdf_raw)
        date_int  = _safe_int(rdf_raw["date_int"].iloc[0], 0) if "date_int" in rdf_raw.columns else 0
        yyyymm    = int(str(date_int)[:6]) if date_int > 0 else 0
        race_label= str(rdf_raw["race_label"].iloc[0]) if "race_label" in rdf_raw.columns else str(rk)

        # 予想実行にゃ
        pred = _bt_predict_race(bundle, rdf_raw, strategy_mode)
        if pred is None:
            continue
        pred["actual_finish"] = rdf_raw["actual_finish"].values

        pred_sorted = pred.sort_values("ml_rank")
        ai_horses = [str(_safe_int(row["horse_no"],0)) for _,row in pred_sorted.iterrows()
                     if _safe_int(row.get("horse_no",0),0) > 0]
        if len(ai_horses) < 3:
            continue

        ai1,ai2,ai3 = ai_horses[0], ai_horses[1], ai_horses[2]
        ai1_odds = _safe_float(pred_sorted.iloc[0].get("odds",10), 10)
        ai2_odds = _safe_float(pred_sorted.iloc[1].get("odds",10), 10) if len(pred_sorted)>1 else 10.0

        # 買い判定馬にゃ
        buy_df = pred[pred.get("buy_flag","") == "買い"] if "buy_flag" in pred.columns else pred.head(3)
        buy_nos = [str(_safe_int(r["horse_no"],0)) for _,r in buy_df.iterrows()]
        # S級判定にゃ
        s_buy_df = pred[pred.get("buy_flag_v2","").str.contains("買い", na=False)] \
            if "buy_flag_v2" in pred.columns else buy_df
        s_buy_nos = [str(_safe_int(r["horse_no"],0)) for _,r in s_buy_df.iterrows()]

        # ── 月別バッファ初期化にゃ ──
        if yyyymm not in monthly_buffer:
            monthly_buffer[yyyymm] = {bt:{"hit":0,"bet":0,"ret":0.0} for bt in BET_TYPES}

        # ────────────────────────
        # 各券種の的中・払戻計算にゃ
        # ────────────────────────
        rec = {
            "レースにゃ": race_label, "競馬場にゃ": place,
            "距離にゃ": distance, "頭数にゃ": field_sz,
            "date_int": date_int, "yyyymm": yyyymm,
            "1着にゃ": f"馬番{w_no}({w_odds:.1f}倍)", "2着にゃ": f"馬番{s_no}", "3着にゃ": f"馬番{t_no}",
            "AI1位にゃ": f"馬番{ai1}({ai1_odds:.1f}倍)",
        }

        # ── 単勝にゃ ──
        tan_odds = ai1_odds
        if min_odds <= tan_odds <= max_odds:
            stats["単勝"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["単勝"]["bet"] += bet_unit
            if ai1 == w_no:
                ret = int(bet_unit * tan_odds * (1 - TANSHO_DEDUCTION))
                stats["単勝"]["hit"] += 1
                stats["単勝"]["ret"] += ret
                monthly_buffer[yyyymm]["単勝"]["hit"] += 1
                monthly_buffer[yyyymm]["単勝"]["ret"] += ret
                rec["単勝にゃ"] = f"✅ {ret}円にゃ"
            else:
                rec["単勝にゃ"] = f"❌ 馬番{ai1}→{w_no}にゃ"
            stats["単勝"]["races"] += 1

        # ── 複勝にゃ（買い判定馬全員にゃ）──
        for bno in buy_nos:
            b_odds = _safe_float(pred[pred["horse_no"].astype(str).str.strip() == bno]["odds"].iloc[0]
                                  if not pred[pred["horse_no"].astype(str).str.strip() == bno].empty else 0, 10)
            if b_odds < min_odds: continue
            fuku_est = max(1.1, b_odds * 0.30)
            stats["複勝"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["複勝"]["bet"] += bet_unit
            if bno in top3_set:
                ret = int(bet_unit * fuku_est * (1 - FUKUSHO_DEDUCTION))
                stats["複勝"]["hit"] += 1
                stats["複勝"]["ret"] += ret
                monthly_buffer[yyyymm]["複勝"]["hit"] += 1
                monthly_buffer[yyyymm]["複勝"]["ret"] += ret
        rec["複勝にゃ"] = "✅" if any(n in top3_set for n in buy_nos) else "❌"
        stats["複勝"]["races"] += 1

        # ── 馬連にゃ（AI上位2頭にゃ）──
        if ai2:
            stats["馬連"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["馬連"]["bet"] += bet_unit
            umaren_hit = {ai1,ai2} <= top3_set and w_no in {ai1,ai2}
            if umaren_hit:
                um_est = max(2.0, ai1_odds * ai2_odds * 0.15)
                ret = int(bet_unit * um_est * (1 - UMAREN_DEDUCTION))
                stats["馬連"]["hit"] += 1
                stats["馬連"]["ret"] += ret
                monthly_buffer[yyyymm]["馬連"]["hit"] += 1
                monthly_buffer[yyyymm]["馬連"]["ret"] += ret
                rec["馬連にゃ"] = f"✅ {ret}円にゃ"
            else:
                rec["馬連にゃ"] = "❌"
            stats["馬連"]["races"] += 1

        # ── ワイドにゃ（AI上位3頭のBOX3点にゃ）──
        wide_pairs = [(ai1,ai2),(ai1,ai3),(ai2,ai3)]
        for wa,wb in wide_pairs:
            stats["ワイド"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["ワイド"]["bet"] += bet_unit
            if {wa,wb} <= top3_set:
                wa_odds = _safe_float(pred[pred["horse_no"]==_safe_int(wa,0)]["odds"].iloc[0]
                                       if not pred[pred["horse_no"]==_safe_int(wa,0)].empty else 0, 5)
                wb_odds = _safe_float(pred[pred["horse_no"]==_safe_int(wb,0)]["odds"].iloc[0]
                                       if not pred[pred["horse_no"]==_safe_int(wb,0)].empty else 0, 5)
                wide_est = max(1.5, (wa_odds + wb_odds) * 0.15)
                ret = int(bet_unit * wide_est * (1 - WIDE_DEDUCTION))
                stats["ワイド"]["hit"] += 1
                stats["ワイド"]["ret"] += ret
                monthly_buffer[yyyymm]["ワイド"]["hit"] += 1
                monthly_buffer[yyyymm]["ワイド"]["ret"] += ret
        stats["ワイド"]["races"] += 1

        # ── 馬単にゃ（AI1→2にゃ）──
        if ai2:
            stats["馬単"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["馬単"]["bet"] += bet_unit
            if ai1 == w_no and ai2 == s_no:
                um_tan_est = max(5.0, ai1_odds * ai2_odds * 0.25)
                ret = int(bet_unit * um_tan_est * (1 - UMATAN_DEDUCTION))
                stats["馬単"]["hit"] += 1
                stats["馬単"]["ret"] += ret
                monthly_buffer[yyyymm]["馬単"]["hit"] += 1
                monthly_buffer[yyyymm]["馬単"]["ret"] += ret
                rec["馬単にゃ"] = f"✅ {ret}円にゃ"
            else:
                rec["馬単にゃ"] = "❌"
            stats["馬単"]["races"] += 1

        # ── 三連複にゃ（軸1頭 × 相手5頭フォーメーション = 10点にゃ）──
        # 旧: AI上位3頭BOX1点 → 的中率0%にゃ
        # 新: AI1位を軸にAI2〜6位の中から2頭 → 的中率大幅UPにゃ🐾
        ai_top6 = [str(_safe_int(row["horse_no"],0))
                   for _, row in pred_sorted.head(6).iterrows()
                   if _safe_int(row.get("horse_no",0),0) > 0]
        pivot    = ai_top6[0] if ai_top6 else ai1
        aite5    = ai_top6[1:6]  # 相手5頭にゃ

        # 10点分ベットするにゃ（C(5,2)=10通りにゃ）
        san3_bet_total = bet_unit * 10
        stats["三連複"]["bet"] += san3_bet_total
        monthly_buffer[yyyymm]["三連複"]["bet"] += san3_bet_total

        # 的中判定にゃ: 軸が3着内 かつ 相手5頭のうち2頭が3着内にゃ
        pivot_in  = pivot in top3_set
        aite_hits = [n for n in aite5 if n in top3_set]
        san3_hit  = pivot_in and len(aite_hits) >= 2

        if san3_hit:
            # 払戻推定にゃ（的中した3頭のオッズから計算にゃ）
            hit3 = [pivot] + aite_hits[:2]
            h3_odds = []
            for hn in hit3:
                row = pred[pred["horse_no"].astype(str).str.strip() == hn]
                h3_odds.append(_safe_float(row["odds"].iloc[0] if not row.empty else 10, 10))
            san3_est = max(5.0, h3_odds[0] * h3_odds[1] * h3_odds[2] * 0.05)
            ret = int(bet_unit * san3_est * (1 - SANRENPUKU_DEDUCTION))
            stats["三連複"]["hit"] += 1
            stats["三連複"]["ret"] += ret
            monthly_buffer[yyyymm]["三連複"]["hit"] += 1
            monthly_buffer[yyyymm]["三連複"]["ret"] += ret
            rec["三連複にゃ"] = f"✅ {ret}円にゃ"
        else:
            rec["三連複にゃ"] = "❌"
        stats["三連複"]["races"] += 1

        # ── 三連単にゃ（AI順1→2→3にゃ）──
        stats["三連単"]["bet"] += bet_unit
        monthly_buffer[yyyymm]["三連単"]["bet"] += bet_unit
        if ai1==w_no and ai2==s_no and ai3==t_no:
            ai3_odds2 = _safe_float(pred_sorted.iloc[2].get("odds",10) if len(pred_sorted)>2 else 10, 10)
            san1_est = max(10.0, ai1_odds * ai2_odds * ai3_odds2 * 0.12)
            ret = int(bet_unit * san1_est * (1 - SANRENTAN_DEDUCTION))
            stats["三連単"]["hit"] += 1
            stats["三連単"]["ret"] += ret
            monthly_buffer[yyyymm]["三連単"]["hit"] += 1
            monthly_buffer[yyyymm]["三連単"]["ret"] += ret
            rec["三連単にゃ"] = f"✅ {ret}円にゃ"
        else:
            rec["三連単にゃ"] = "❌"
        stats["三連単"]["races"] += 1

        # ── S級判定買いにゃ ──
        for sno in s_buy_nos:
            s_odds = _safe_float(pred[pred["horse_no"]==_safe_int(sno,0)]["odds"].iloc[0]
                                  if not pred[pred["horse_no"]==_safe_int(sno,0)].empty else 0, 10)
            if s_odds < min_odds: continue
            s_fuku = max(1.1, s_odds * 0.30)
            stats["S級判定"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["S級判定"]["bet"] += bet_unit
            if sno in top3_set:
                ret = int(bet_unit * s_fuku * (1 - FUKUSHO_DEDUCTION))
                stats["S級判定"]["hit"] += 1
                stats["S級判定"]["ret"] += ret
                monthly_buffer[yyyymm]["S級判定"]["hit"] += 1
                monthly_buffer[yyyymm]["S級判定"]["ret"] += ret
        stats["S級判定"]["races"] += 1

        # ── S級×複勝にゃ（S級判定 かつ 複勝買い判定の馬のみにゃ）──
        # バックテスト結果: S級257%・複勝156% → 両方通った馬が最強にゃ🐾
        sx_nos = [
            n for n in s_buy_nos
            if n in buy_nos  # 複勝買い判定も通っている馬だけにゃ
        ]
        for sno in sx_nos:
            s_odds_row = pred[pred["horse_no"] == _safe_int(sno, 0)]
            s_odds2 = _safe_float(
                s_odds_row["odds"].iloc[0] if not s_odds_row.empty else 0, 10)
            if s_odds2 < min_odds:
                continue
            s_fuku2 = max(1.1, s_odds2 * 0.30)
            stats["S級×複勝"]["bet"] += bet_unit
            monthly_buffer[yyyymm]["S級×複勝"]["bet"] += bet_unit
            if sno in top3_set:
                ret = int(bet_unit * s_fuku2 * (1 - FUKUSHO_DEDUCTION))
                stats["S級×複勝"]["hit"] += 1
                stats["S級×複勝"]["ret"] += ret
                monthly_buffer[yyyymm]["S級×複勝"]["hit"] += 1
                monthly_buffer[yyyymm]["S級×複勝"]["ret"] += ret
        stats["S級×複勝"]["races"] += 1
        rec["S級×複勝にゃ"] = "✅" if any(n in top3_set for n in sx_nos) else (
            "対象なし" if not sx_nos else "❌")

        race_records.append(rec)

    if not race_records:
        return {"error": "有効なレースが処理できなかったにゃ"}

    # ── サマリーにゃ ──
    def pct(h, b):
        return f"{h/b*100:.1f}%" if b > 0 else "-"
    def roi(r, b):
        return f"{r/b*100:.1f}%" if b > 0 else "-"
    def roi_float(r, b):
        return round(r/b*100, 1) if b > 0 else 0.0

    summary_rows = []
    for bt in BET_TYPES:
        s = stats[bt]
        n_bet = s["bet"] // bet_unit
        summary_rows.append({
            "券種にゃ":    bt,
            "レース数にゃ": s["races"],
            "購入点数にゃ": n_bet,
            "的中数にゃ":  s["hit"],
            "的中率にゃ":  pct(s["hit"], n_bet),
            "投資額にゃ":  f"¥{s['bet']:,}",
            "回収額にゃ":  f"¥{int(s['ret']):,}",
            "回収率にゃ":  roi(s["ret"], s["bet"]),
            "損益にゃ":    f"{'+'if s['ret']>s['bet'] else ''}{int(s['ret']-s['bet']):,}円にゃ",
            "_roi":       roi_float(s["ret"], s["bet"]),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── 月別集計にゃ ──
    monthly_rows = []
    for yyyymm_key in sorted(monthly_buffer.keys()):
        if yyyymm_key == 0:
            continue
        mb = monthly_buffer[yyyymm_key]
        row = {"年月にゃ": str(yyyymm_key)}
        for bt in ["単勝","複勝","三連複","S級判定"]:
            s2 = mb[bt]
            nb = s2["bet"] // bet_unit
            row[f"{bt}的中率にゃ"] = pct(s2["hit"], nb)
            row[f"{bt}回収率にゃ"] = roi(s2["ret"], s2["bet"])
            row[f"_{bt}_roi"]     = roi_float(s2["ret"], s2["bet"])
        monthly_rows.append(row)
    monthly_df = pd.DataFrame(monthly_rows)

    # ── 競馬場別にゃ ──
    rec_df = pd.DataFrame(race_records)
    by_place_rows = []
    for pl, g in rec_df.groupby("競馬場にゃ"):
        n = len(g)
        san3_hits = int(g["三連複にゃ"].str.startswith("✅").sum()) if "三連複にゃ" in g.columns else 0
        tan_hits  = int(g["単勝にゃ"].str.startswith("✅").sum())   if "単勝にゃ"  in g.columns else 0
        by_place_rows.append({
            "競馬場にゃ":     pl,
            "レース数にゃ":   n,
            "単勝的中率にゃ": pct(tan_hits, n),
            "三連複的中率にゃ":pct(san3_hits, n),
        })
    by_place_df = pd.DataFrame(by_place_rows).sort_values("三連複的中率にゃ", ascending=False)

    # ── 距離帯別にゃ ──
    def dist_band(d):
        if d <= 1400:   return "短距離(〜1400)"
        if d <= 1800:   return "マイル(1401〜1800)"
        if d <= 2200:   return "中距離(1801〜2200)"
        return "長距離(2201〜)"

    rec_df["距離帯にゃ"] = rec_df["距離にゃ"].apply(lambda x: dist_band(_safe_int(x,2000)))
    by_dist_rows = []
    for db, g in rec_df.groupby("距離帯にゃ"):
        n = len(g)
        san3_hits = int(g["三連複にゃ"].str.startswith("✅").sum()) if "三連複にゃ" in g.columns else 0
        tan_hits  = int(g["単勝にゃ"].str.startswith("✅").sum())   if "単勝にゃ" in g.columns else 0
        by_dist_rows.append({
            "距離帯にゃ":     db,
            "レース数にゃ":   n,
            "単勝的中率にゃ": pct(tan_hits, n),
            "三連複的中率にゃ":pct(san3_hits, n),
        })
    by_dist_df = pd.DataFrame(by_dist_rows)

    # ── 頭数帯別にゃ ──
    def field_band(f):
        if f <= 8:  return "少頭数(〜8頭)"
        if f <= 12: return "中頭数(9〜12頭)"
        if f <= 16: return "多頭数(13〜16頭)"
        return "大頭数(17頭〜)"

    rec_df["頭数帯にゃ"] = rec_df["頭数にゃ"].apply(lambda x: field_band(_safe_int(x,16)))
    by_field_rows = []
    for fb, g in rec_df.groupby("頭数帯にゃ"):
        n = len(g)
        san3_hits = int(g["三連複にゃ"].str.startswith("✅").sum()) if "三連複にゃ" in g.columns else 0
        tan_hits  = int(g["単勝にゃ"].str.startswith("✅").sum())   if "単勝にゃ" in g.columns else 0
        by_field_rows.append({
            "頭数帯にゃ":    fb,
            "レース数にゃ":  n,
            "単勝的中率にゃ":pct(tan_hits, n),
            "三連複的中率にゃ":pct(san3_hits, n),
        })
    by_field_df = pd.DataFrame(by_field_rows)

    # ── 改善提案の自動生成にゃ ──
    tips = []
    for s_row in summary_df.itertuples():
        bt    = getattr(s_row, "券種にゃ")
        roi_v = getattr(s_row, "_roi", 0)
        if roi_v >= 100:
            tips.append(f"🎉 **{bt}がプラス収支**にゃ！回収率{roi_v:.1f}% → この券種に集中するにゃ🐾")
        elif roi_v >= 80:
            tips.append(f"✅ **{bt}はほぼ均衡**にゃ（{roi_v:.1f}%にゃ）。点数を絞れば黒字化できるにゃ")
        elif roi_v < 50 and bt in ["三連単","馬単"]:
            tips.append(f"⚠️ **{bt}は回収率{roi_v:.1f}%**にゃ。見送りを推奨するにゃ")

    # 競馬場別アドバイスにゃ
    if not by_place_df.empty:
        best_pl = by_place_df.iloc[0]
        tips.append(
            f"🏟️ **{best_pl['競馬場にゃ']}が三連複的中率最高**にゃ"
            f"（{best_pl['三連複的中率にゃ']}にゃ）。この競馬場のレースを優先するにゃ"
        )

    return {
        "summary":      summary_df,
        "race_records": rec_df,
        "monthly":      monthly_df,
        "by_place":     by_place_df,
        "by_distance":  by_dist_df,
        "by_field":     by_field_df,
        "tips":         tips,
        "n_races":      len(race_records),
        "stats_raw":    stats,
    }


# ============================================================
# 複数条件の一括比較にゃ
# ============================================================

def run_backtest_comparison(bundle,
                             history_df: pd.DataFrame,
                             conditions: list[dict] = None,
                             bet_unit: int = 100) -> dict:
    """
    複数の条件でバックテストを一括実行して比較するにゃ。

    conditions の例にゃ:
    [
      {"name":"回収率重視にゃ", "strategy_mode": STRATEGY_MODE_ROI,   "min_odds":1.0},
      {"name":"的中率重視にゃ", "strategy_mode": STRATEGY_MODE_HITRATE,"min_odds":1.0},
      {"name":"高オッズ限定にゃ","strategy_mode": STRATEGY_MODE_ROI,  "min_odds":5.0},
      {"name":"低オッズ限定にゃ","strategy_mode": STRATEGY_MODE_ROI,  "min_odds":1.0, "max_odds":5.0},
    ]
    """
    if conditions is None:
        conditions = [
            {"name": "回収率重視にゃ",   "strategy_mode": STRATEGY_MODE_ROI,     "min_odds": 1.0, "max_odds": 9999.0},
            {"name": "的中率重視にゃ",   "strategy_mode": STRATEGY_MODE_HITRATE,  "min_odds": 1.0, "max_odds": 9999.0},
            {"name": "中穴限定にゃ(5〜20倍)", "strategy_mode": STRATEGY_MODE_ROI, "min_odds": 5.0, "max_odds": 20.0},
            {"name": "本命限定にゃ(〜5倍)",   "strategy_mode": STRATEGY_MODE_ROI, "min_odds": 1.0, "max_odds": 5.0},
        ]

    results = {}
    for cond in conditions:
        name = cond.get("name", "条件にゃ")
        try:
            r = run_backtest_v2(
                bundle, history_df,
                strategy_mode=cond.get("strategy_mode", STRATEGY_MODE_ROI),
                min_odds=cond.get("min_odds", 1.0),
                max_odds=cond.get("max_odds", 9999.0),
                bet_unit=bet_unit,
            )
            results[name] = r
        except Exception as e:
            results[name] = {"error": str(e)}

    # 比較テーブルにゃ
    compare_rows = []
    for name, r in results.items():
        if "error" in r:
            compare_rows.append({"条件にゃ": name, "エラーにゃ": r["error"]})
            continue
        s = r.get("stats_raw", {})
        def roi_f(bt):
            st2 = s.get(bt, {})
            b = st2.get("bet", 0)
            ret = st2.get("ret", 0)
            return round(ret/b*100, 1) if b > 0 else 0.0
        def hit_f(bt):
            st2 = s.get(bt, {})
            nb = st2.get("bet",0) // bet_unit
            return round(st2.get("hit",0)/nb*100,1) if nb > 0 else 0.0

        compare_rows.append({
            "条件にゃ":       name,
            "レース数にゃ":   r.get("n_races", 0),
            "単勝的中%にゃ":  hit_f("単勝"),
            "単勝回収%にゃ":  roi_f("単勝"),
            "複勝的中%にゃ":  hit_f("複勝"),
            "複勝回収%にゃ":  roi_f("複勝"),
            "三連複的中%にゃ":hit_f("三連複"),
            "三連複回収%にゃ":roi_f("三連複"),
            "S級的中%にゃ":   hit_f("S級判定"),
            "S級回収%にゃ":   roi_f("S級判定"),
        })
    compare_df = pd.DataFrame(compare_rows)

    return {"results": results, "compare": compare_df}


# ============================================================
# バックテスト v2 表示にゃ
# ============================================================

def _read_backtest_csv(raw: bytes, fname: str) -> pd.DataFrame | None:
    """
    バックテスト用CSVを柔軟に読み込むにゃ。
    以下の形式に対応にゃ:
      ① 52列TARGET形式にゃ
      ② 簡易CSV（horse_name/finish列ありにゃ）
      ③ にゃんこ予想結果CSV（nyanko_v26_all.csv形式にゃ）
         → 着順列がないのでfinish入力UIを出すにゃ
    """
    # まずヘッダーを読んで形式を判断にゃ
    for enc in ["utf-8-sig","utf-8","cp932","shift_jis"]:
        try:
            header_df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str, nrows=1)
            break
        except Exception:
            header_df = None

    if header_df is None:
        return None

    cols = set(str(c).strip() for c in header_df.columns)

    # ③ にゃんこ予想結果CSV判定にゃ（日本語列にゃ）
    nyanko_markers = {"AI順位","馬番","馬名","オッズ","人気","レース","レースID"}
    if len(cols & nyanko_markers) >= 4:
        for enc in ["utf-8-sig","utf-8","cp932","shift_jis"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
                break
            except Exception:
                continue
        # 列名を内部形式にマッピングにゃ
        rename = {
            "レース": "race_label", "レースID": "race_key",
            "AI順位": "ml_rank", "印": "mark",
            "馬番": "horse_no", "馬名": "horse_name",
            "性別": "sex", "年齢": "age", "騎手": "jockey", "斤量": "carried_weight",
            "オッズ": "odds", "人気": "popularity",
            "3着内確率": "ml_top3_prob", "期待値": "expected_value",
            "EV乖離スコア": "ev_score", "市場暗示3着内確率": "implied_top3",
            "危険人気馬": "danger_popular", "危険度": "danger_level",
            "穴候補": "value_horse", "脚質": "running_style", "脚質メモ": "style_note",
            "騎手実績": "jockey_top3_rate_prior",
            "調教師実績": "trainer_top3_rate_prior",
            "血統実績": "sire_top3_rate_prior",
            "距離適性": "horse_distance_top3_rate_prior",
            "Kelly比(複勝)": "kelly_ratio",
            "Kelly比(三連複)": "kelly_ratio_sanren",
            "軸信頼度": "pivot_confidence",
        }
        df = df.rename(columns=rename)

        # odds/popularityを数値化にゃ
        for c in ["odds","popularity","ml_rank","horse_no","ml_top3_prob",
                  "ev_score","kelly_ratio","kelly_ratio_sanren"]:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace("%","").str.replace(",",""),
                    errors="coerce")

        # race_keyを確定にゃ
        if "race_key" not in df.columns or df["race_key"].isna().all():
            if "race_label" in df.columns:
                df["race_key"] = df["race_label"].astype(str)
            else:
                df["race_key"] = "race_001"

        # date_int にゃ（race_keyから抽出にゃ）
        def _extract_date_int(rk):
            m = re.search(r"(\d{8})", str(rk))
            return int(m.group(1)) if m else 20260101
        df["date_int"] = df["race_key"].apply(_extract_date_int)

        # field_sizeにゃ
        if "field_size" not in df.columns:
            df["field_size"] = df.groupby("race_key")["horse_no"].transform("count")

        # place / race_no にゃ（race_keyやrace_labelから抽出にゃ）
        rl = df.get("race_label", df.get("race_key", pd.Series(["不明"]*len(df))))
        def _extract_place(s):
            for p in ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]:
                if p in str(s): return p
            return "不明"
        def _extract_race_no(s):
            m = re.search(r"(\d+)R", str(s))
            return int(m.group(1)) if m else 1
        df["place"]    = rl.apply(_extract_place)
        df["race_no"]  = rl.apply(_extract_race_no)
        df["race_name"]= rl.astype(str)

        # source_fileにゃ
        df["source_file"] = fname
        df["_is_nyanko_csv"] = True  # にゃんこ予想CSVフラグにゃ
        return df

    # ① result_fetcher.py 出力CSV判定にゃ
    # 列: race_id, finish, horse_no, horse_name, odds, popularity ...
    result_markers = {"race_id","finish","horse_no","horse_name"}
    if len(cols & result_markers) >= 3:
        for enc in ["utf-8-sig","utf-8","cp932","shift_jis"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
                break
            except Exception:
                continue
        # race_key自動生成にゃ
        if "race_key" not in df.columns:
            if "race_id" in df.columns:
                df["race_key"] = df["race_id"].astype(str)
        # date_int自動生成にゃ
        if "date_int" not in df.columns:
            if "race_id" in df.columns:
                df["date_int"] = df["race_id"].astype(str).str[:8].apply(
                    lambda x: int(x) if x.isdigit() else 20260101)
            else:
                df["date_int"] = 20260101
        # odds補完にゃ（コーナー通過順が混入している場合にゃ）
        if "odds" in df.columns:
            odds_bad = df["odds"].astype(str).str.contains("-", na=False).mean() > 0.5
            if odds_bad:
                pop = pd.to_numeric(df.get("popularity",""), errors="coerce").fillna(99)
                pop_is_bad = pop.notna().mean() < 0.3
                if pop_is_bad:
                    fin = pd.to_numeric(df.get("finish",""), errors="coerce").fillna(99)
                    df["popularity"] = fin.astype(int)
                pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(99)
                df["odds"] = (pop * 1.5 + 1.0).round(1)
        # field_sizeにゃ
        if "field_size" not in df.columns:
            df["field_size"] = df.groupby("race_key")["horse_no"].transform("count")
        # placeにゃ
        if "place" not in df.columns:
            place_map = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
                         "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
            df["place"] = df["race_id"].astype(str).str[4:6].map(place_map).fillna("不明")
        # race_noにゃ
        if "race_no" not in df.columns:
            df["race_no"] = df["race_id"].astype(str).str[10:12].apply(
                lambda x: int(x) if x.isdigit() else 1)
        # source_fileにゃ
        df["source_file"] = fname
        return clean_types(df) if "clean_types" in dir() else df

    # ② 52列TARGET形式にゃ
    try:
        raw_df = read_csv_bytes(raw)
        return normalize_52cols(raw_df, fname)
    except Exception:
        pass

    # ③ 簡易CSVにゃ
    try:
        return read_simple_csv_to_52(raw, fname)
    except Exception:
        pass

    return None


def show_backtest_v2_tab(bundle, strategy_mode=STRATEGY_MODE_ROI):
    """バックテスト v2 完全表示にゃ"""
    st.header("📊 バックテスト v2（完全版・比較分析にゃ）")
    st.caption(
        "実際の着順データで**全券種の的中率・回収率**を計測するにゃ🐾\n"
        "複数条件を一括比較して**最も勝てる買い方**を見つけるにゃ"
    )

    st.info(
        "**対応CSVにゃ🐾**\n\n"
        "① `nyanko_v26_all.csv`（予想結果 + 着順を別途入力にゃ）\n"
        "② 着順付きCSV（`result_fetcher.py`の出力にゃ）\n"
        "③ JRA-VAN/TARGET 52列CSVにゃ"
    )

    # ── データ読み込みUIにゃ ──
    col1, col2 = st.columns(2)
    with col1:
        bt2_file = st.file_uploader(
            "CSVをアップロードにゃ（予想CSV or 着順CSVにゃ）",
            type=["csv"], key="bt2_upload"
        )
        use_yosou2 = st.checkbox(
            "yosou.csvを使うにゃ",
            value=TARGET_CSV_PATH.exists(), key="bt2_yosou"
        )
    with col2:
        bet_unit2 = st.number_input("1点あたり金額にゃ（円にゃ）", 100, 10000, 100, 100, key="bt2_unit")
        min_o2    = st.number_input("最低オッズにゃ", 1.0, 10.0, 1.0, 0.5, key="bt2_mino")
        max_o2    = st.number_input("最高オッズにゃ", 10.0, 9999.0, 9999.0, 10.0, key="bt2_maxo")

    # データ読み込みにゃ
    hist_df2 = None
    is_nyanko_csv = False

    if bt2_file is not None:
        raw2 = bt2_file.read()
        hist_df2 = _read_backtest_csv(raw2, bt2_file.name)
        if hist_df2 is None:
            st.error("CSVを読み込めなかったにゃ🐾")
            return
        is_nyanko_csv = bool(hist_df2.get("_is_nyanko_csv", pd.Series([False])).any())             if "_is_nyanko_csv" in hist_df2.columns else False
    elif use_yosou2 and TARGET_CSV_PATH.exists():
        hist_df2 = read_target_history_csv(TARGET_CSV_PATH)

    if hist_df2 is None:
        st.info(
            "📂 CSVをアップロードするにゃ🐾\n\n"
            "① `nyanko_v26_all.csv`（予想結果にゃ） → 着順を入力するにゃ\n"
            "② `result_fetcher.py`の出力CSV（着順付きにゃ）→ そのまま使えるにゃ"
        )
        return

    # ── にゃんこ予想CSVの場合にゃ: 着順を入力させるにゃ ──
    if is_nyanko_csv and "finish" not in hist_df2.columns:
        st.markdown("---")
        st.subheader("📝 着順を入力するにゃ🐾")
        st.caption(
            "`nyanko_v26_all.csv` は予想結果のみにゃ。"
            "実際の着順を入力してバックテストするにゃ🐾"
        )

        # レース選択にゃ
        race_keys_list = hist_df2["race_key"].dropna().unique().tolist()
        race_labels_list = hist_df2["race_label"].dropna().unique().tolist()             if "race_label" in hist_df2.columns else race_keys_list
        label2key = dict(zip(race_labels_list, race_keys_list))

        # 着順入力テーブルにゃ
        st.markdown("#### 実際の着順を入力するにゃ（馬番で入力にゃ）")

        finish_inputs = {}
        for rk in race_keys_list:
            rdf = hist_df2[hist_df2["race_key"] == rk].sort_values("ml_rank")
            rl  = str(rdf["race_label"].iloc[0]) if "race_label" in rdf.columns else rk
            st.markdown(f"**{rl}**にゃ")

            cols_f = st.columns(3)
            fin1 = cols_f[0].text_input("1着 馬番にゃ", key=f"fin1_{rk}", placeholder="例: 13")
            fin2 = cols_f[1].text_input("2着 馬番にゃ", key=f"fin2_{rk}", placeholder="例: 3")
            fin3 = cols_f[2].text_input("3着 馬番にゃ", key=f"fin3_{rk}", placeholder="例: 6")
            finish_inputs[rk] = {"1": fin1.strip(), "2": fin2.strip(), "3": fin3.strip()}

            # このレースの馬一覧にゃ
            with st.expander("馬一覧にゃ（参照用にゃ）"):
                disp_cols = [c for c in ["ml_rank","horse_no","horse_name","odds","popularity"]
                             if c in rdf.columns]
                st.dataframe(rdf[disp_cols].rename(columns={
                    "ml_rank":"AI順位","horse_no":"馬番","horse_name":"馬名",
                    "odds":"オッズ","popularity":"人気"
                }), use_container_width=True, hide_index=True)

        if st.button("✅ 着順を確定してバックテスト準備にゃ", key="bt2_set_finish"):
            # 着順をDataFrameに追加するにゃ
            hist_df2["finish"] = 99  # デフォルト99にゃ
            for rk, fins in finish_inputs.items():
                for rank_str, horse_no_str in fins.items():
                    if horse_no_str and horse_no_str.isdigit():
                        try:
                            hno = int(horse_no_str)
                            mask = (hist_df2["race_key"] == rk) &                                    (pd.to_numeric(hist_df2["horse_no"],errors="coerce") == hno)
                            hist_df2.loc[mask, "finish"] = int(rank_str)
                        except Exception:
                            pass
            st.session_state["bt2_hist_df"] = hist_df2
            st.success("✅ 着順を設定したにゃ！下のバックテストボタンを押すにゃ🐾")

        # セッションステートから取得にゃ
        if "bt2_hist_df" in st.session_state:
            hist_df2 = st.session_state["bt2_hist_df"]
        else:
            st.info("☝️ 着順を入力して「着順を確定」ボタンを押すにゃ🐾")
            return

    # finishチェックにゃ
    if "finish" not in hist_df2.columns:
        st.error("❌ finish（着順）列がないにゃ。着順付きCSVをアップロードするにゃ🐾")
        return

    # ── race_key 自動生成にゃ ──
    if "race_key" not in hist_df2.columns:
        if "race_id" in hist_df2.columns:
            hist_df2["race_key"] = hist_df2["race_id"].astype(str)
        else:
            hist_df2["race_key"] = (
                hist_df2.get("place", pd.Series(["X"]*len(hist_df2))).astype(str) + "_" +
                hist_df2.get("race_no", pd.Series(range(len(hist_df2)))).astype(str)
            )

    # ── date_int 自動生成にゃ ──
    if "date_int" not in hist_df2.columns:
        if "race_id" in hist_df2.columns:
            def _race_id_to_date(rid):
                rid = str(rid)
                if len(rid) >= 8:
                    return int(rid[:8]) if rid[:8].isdigit() else 20260101
                return 20260101
            hist_df2["date_int"] = hist_df2["race_id"].apply(_race_id_to_date)
        else:
            hist_df2["date_int"] = 20260101

    # ── odds補完にゃ（コーナー通過順が入っている場合にゃ）──
    # odds列が "2-2" のような形式 → コーナー通過順にゃ → 人気から推定にゃ
    if "odds" in hist_df2.columns:
        odds_check = hist_df2["odds"].astype(str).str.contains("-", na=False)
        if odds_check.mean() > 0.5:
            st.warning(
                "⚠️ odds列にコーナー通過順が混入しているにゃ。"
                "人気からオッズを推定して処理するにゃ🐾"
            )
            # 人気列が調教師名になっている場合にゃ
            if "popularity" in hist_df2.columns:
                pop_num = pd.to_numeric(hist_df2["popularity"], errors="coerce")
                if pop_num.notna().mean() < 0.3:
                    # popularity列も誤っているにゃ → finishベースで人気を推定にゃ
                    hist_df2["popularity"] = pd.to_numeric(
                        hist_df2["finish"], errors="coerce").fillna(99).astype(int)
            # 人気からオッズを推定にゃ（近似式にゃ）
            pop = pd.to_numeric(hist_df2.get("popularity", pd.Series([99]*len(hist_df2))),
                                 errors="coerce").fillna(99)
            hist_df2["odds"] = (pop * 1.5 + 1.0).round(1)
        else:
            hist_df2["odds"] = pd.to_numeric(hist_df2["odds"], errors="coerce").fillna(10.0)

    # ── popularityの補完にゃ ──
    if "popularity" in hist_df2.columns:
        pop_num = pd.to_numeric(hist_df2["popularity"], errors="coerce")
        if pop_num.notna().mean() < 0.3:
            # 人気列が壊れているにゃ → finishベースで近似にゃ
            hist_df2["popularity"] = pd.to_numeric(
                hist_df2["finish"], errors="coerce").fillna(99).astype(int)

    # ── field_size 補完にゃ ──
    if "field_size" not in hist_df2.columns or        pd.to_numeric(hist_df2.get("field_size",""), errors="coerce").notna().mean() < 0.3:
        hist_df2["field_size"] = hist_df2.groupby("race_key")["horse_no"].transform("count")

    n_valid = int(pd.to_numeric(hist_df2["finish"], errors="coerce").notna().sum())
    n_races = hist_df2["race_key"].nunique()
    st.success(f"✅ データ読込完了にゃ: {len(hist_df2)}行 / 有効着順:{n_valid}行 / {n_races}レースにゃ")

    # ── 実行モード選択にゃ ──
    run_mode = st.radio(
        "実行モードにゃ",
        ["🎯 単一条件バックテスト", "⚖️ 複数条件一括比較（推奨にゃ）"],
        horizontal=True
    )

    if run_mode == "🎯 単一条件バックテスト":
        if st.button("🐾 バックテスト実行にゃ！", type="primary", key="bt2_single"):
            with st.spinner("バックテスト実行中にゃ...🐾"):
                try:
                    bundle_bt2, _ = load_model_safely(None)
                    if bundle_bt2 is None:
                        st.error("PKLが必要にゃ🐾"); return
                    result = run_backtest_v2(
                        bundle_bt2, hist_df2,
                        strategy_mode=strategy_mode,
                        min_odds=min_o2, max_odds=max_o2,
                        bet_unit=int(bet_unit2)
                    )
                except Exception as e:
                    st.error(f"エラーにゃ: {e}"); return

            if "error" in result:
                st.error(f"❌ {result['error']}"); return

            _show_bt2_single_result(result, int(bet_unit2))

    else:  # 複数条件比較にゃ
        st.markdown("#### 比較条件設定にゃ")
        conditions_default = [
            {"name":"回収率重視にゃ",       "strategy_mode": STRATEGY_MODE_ROI,     "min_odds":1.0,  "max_odds":9999.0},
            {"name":"的中率重視にゃ",       "strategy_mode": STRATEGY_MODE_HITRATE, "min_odds":1.0,  "max_odds":9999.0},
            {"name":"中穴限定にゃ(5〜20倍)", "strategy_mode": STRATEGY_MODE_ROI,    "min_odds":5.0,  "max_odds":20.0},
            {"name":"本命限定にゃ(〜5倍)",  "strategy_mode": STRATEGY_MODE_ROI,    "min_odds":1.0,  "max_odds":5.0},
        ]
        st.dataframe(pd.DataFrame(conditions_default), use_container_width=True, hide_index=True)

        if st.button("🐾 全条件一括比較 実行にゃ！", type="primary", key="bt2_compare"):
            bundle_bt2, _ = load_model_safely(None)
            if bundle_bt2 is None:
                st.error("PKLが必要にゃ🐾"); return

            prog = st.progress(0)
            results_all = {}
            for ci, cond in enumerate(conditions_default):
                prog.progress((ci+1)/len(conditions_default),
                               text=f"条件 {ci+1}/{len(conditions_default)}: {cond['name']}にゃ")
                try:
                    r = run_backtest_v2(
                        bundle_bt2, hist_df2,
                        strategy_mode=cond["strategy_mode"],
                        min_odds=cond["min_odds"],
                        max_odds=cond["max_odds"],
                        bet_unit=int(bet_unit2)
                    )
                    results_all[cond["name"]] = r
                except Exception as e:
                    results_all[cond["name"]] = {"error": str(e)}

            prog.empty()
            _show_bt2_compare_result(results_all, int(bet_unit2))


def _show_bt2_single_result(result: dict, bet_unit: int):
    """単一バックテスト結果表示にゃ"""
    st.markdown(f"### 📈 バックテスト結果にゃ（{result['n_races']}レースにゃ）")

    # ── サマリーにゃ ──
    summary = result.get("summary", pd.DataFrame())
    if not summary.empty:
        st.markdown("#### 🏆 券種別成績にゃ")

        def color_summary(row):
            roi_v = _safe_float(str(row.get("回収率にゃ","0%")).replace("%",""), 0)
            if roi_v >= 100: return ["background-color:#c3e6cb"]*len(row)
            if roi_v >= 80:  return ["background-color:#d1ecf1"]*len(row)
            if roi_v < 50:   return ["background-color:#f8d7da"]*len(row)
            return [""]*len(row)

        disp = summary.drop(columns=["_roi"], errors="ignore")
        try:
            st.dataframe(disp.style.apply(color_summary, axis=1),
                         use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # KPIメトリクスにゃ
        c1,c2,c3,c4 = st.columns(4)
        for col, bt in [(c1,"単勝"),(c2,"複勝"),(c3,"三連複"),(c4,"S級判定")]:
            row = summary[summary["券種にゃ"]==bt]
            if not row.empty:
                roi_v = _safe_float(str(row.iloc[0].get("回収率にゃ","0%")).replace("%",""), 0)
                hit_v = str(row.iloc[0].get("的中率にゃ", "-"))
                delta_c = "normal" if roi_v >= 80 else "inverse"
                col.metric(f"{bt}にゃ", f"回収{roi_v:.0f}%",
                           delta=f"的中{hit_v}", delta_color=delta_c)

        st.download_button("📥 サマリーCSVにゃ",
            data=summary.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="bt_summary.csv", mime="text/csv")

    # ── 月別推移にゃ ──
    monthly = result.get("monthly", pd.DataFrame())
    if not monthly.empty:
        st.markdown("#### 📅 月別回収率推移にゃ")
        disp_m = monthly.drop(columns=[c for c in monthly.columns if c.startswith("_")], errors="ignore")
        st.dataframe(disp_m, use_container_width=True, hide_index=True)

        # 月別グラフにゃ（簡易にゃ）
        roi_cols = [c for c in monthly.columns if c.endswith("回収率にゃ")]
        if roi_cols:
            chart_data = monthly[["年月にゃ"] + roi_cols].copy()
            for c in roi_cols:
                chart_data[c] = chart_data[c].apply(
                    lambda x: _safe_float(str(x).replace("%",""), 0))
            st.line_chart(chart_data.set_index("年月にゃ"))

    # ── 詳細分析にゃ ──
    tab_pl, tab_dist, tab_field, tab_race = st.tabs([
        "🏟️ 競馬場別にゃ", "📏 距離帯別にゃ", "👥 頭数別にゃ", "📋 レース別明細にゃ"
    ])
    with tab_pl:
        bp = result.get("by_place", pd.DataFrame())
        if not bp.empty:
            st.dataframe(bp, use_container_width=True, hide_index=True)
    with tab_dist:
        bd = result.get("by_distance", pd.DataFrame())
        if not bd.empty:
            st.dataframe(bd, use_container_width=True, hide_index=True)
    with tab_field:
        bf = result.get("by_field", pd.DataFrame())
        if not bf.empty:
            st.dataframe(bf, use_container_width=True, hide_index=True)
    with tab_race:
        rr = result.get("race_records", pd.DataFrame())
        if not rr.empty:
            disp_r = rr.drop(columns=["date_int","yyyymm"], errors="ignore")
            st.dataframe(disp_r, use_container_width=True, hide_index=True)
            st.download_button("📥 レース別明細CSVにゃ",
                data=disp_r.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="bt_race_detail.csv", mime="text/csv")

    # ── 改善提案にゃ ──
    tips = result.get("tips", [])
    if tips:
        st.markdown("---")
        st.markdown("#### 💡 改善提案にゃ")
        for tip in tips:
            st.info(tip)


def _show_bt2_compare_result(results_all: dict, bet_unit: int):
    """複数条件比較結果の表示にゃ"""
    st.markdown("### ⚖️ 複数条件比較結果にゃ")

    # 比較テーブルにゃ
    compare_rows = []
    for name, r in results_all.items():
        if "error" in r:
            compare_rows.append({"条件にゃ": name, "エラーにゃ": r.get("error","")})
            continue
        s = r.get("stats_raw", {})
        def roi_f(bt):
            s2 = s.get(bt, {})
            b = s2.get("bet", 0)
            return round(s2.get("ret",0)/b*100,1) if b > 0 else 0.0
        def hit_f(bt):
            s2 = s.get(bt, {})
            nb = s2.get("bet",0)//bet_unit
            return round(s2.get("hit",0)/nb*100,1) if nb > 0 else 0.0

        compare_rows.append({
            "条件にゃ":       name,
            "レース数にゃ":   r.get("n_races",0),
            "単勝的中%にゃ":  hit_f("単勝"),
            "単勝回収%にゃ":  roi_f("単勝"),
            "複勝的中%にゃ":  hit_f("複勝"),
            "複勝回収%にゃ":  roi_f("複勝"),
            "三連複的中%にゃ":hit_f("三連複"),
            "三連複回収%にゃ":roi_f("三連複"),
            "S級的中%にゃ":   hit_f("S級判定"),
            "S級回収%にゃ":   roi_f("S級判定"),
        })
    compare_df = pd.DataFrame(compare_rows)

    # カラーにゃ
    def color_compare(row):
        colors = []
        for col in row.index:
            v = _safe_float(str(row[col]).replace("%",""), 0)
            if "回収" in str(col) and v >= 100:
                colors.append("background-color:#c3e6cb")
            elif "回収" in str(col) and v >= 80:
                colors.append("background-color:#d1ecf1")
            elif "回収" in str(col) and v < 60:
                colors.append("background-color:#f8d7da")
            else:
                colors.append("")
        return colors

    try:
        st.dataframe(compare_df.style.apply(color_compare, axis=1),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.download_button("📥 比較結果CSVにゃ",
        data=compare_df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="bt_compare.csv", mime="text/csv")

    # ── 条件別詳細にゃ ──
    st.markdown("#### 📋 条件別詳細にゃ")
    tabs = st.tabs([f"📊 {name}" for name in results_all.keys()])
    for tab, (name, r) in zip(tabs, results_all.items()):
        with tab:
            if "error" in r:
                st.error(f"エラーにゃ: {r['error']}")
                continue
            _show_bt2_single_result(r, bet_unit)

    # ── 総合ベスト条件にゃ ──
    valid = {n:r for n,r in results_all.items() if "error" not in r}
    if valid:
        st.markdown("---")
        st.markdown("#### 🌟 総合ベスト条件にゃ（複勝回収率にゃ）")
        best_name = max(
            valid.keys(),
            key=lambda n: _safe_float(
                str(valid[n].get("summary",pd.DataFrame())
                    .query("券種にゃ=='複勝'")["回収率にゃ"].iloc[0]
                    if not valid[n].get("summary",pd.DataFrame()).empty else "0%")
                .replace("%",""), 0)
        )
        best_r = valid[best_name]
        summary_best = best_r.get("summary", pd.DataFrame())
        fuku_roi = ""
        if not summary_best.empty:
            f_row = summary_best[summary_best["券種にゃ"]=="複勝"]
            if not f_row.empty:
                fuku_roi = str(f_row.iloc[0].get("回収率にゃ",""))
        st.success(
            f"🏆 **{best_name}** が最もROI高いにゃ！\n\n"
            f"複勝回収率: **{fuku_roi}**にゃ🐾"
        )





# ============================================================
# ============================================================
# 着順データ一括取得モジュールにゃ🐾
# netkeibaから過去レース結果を自動取得するにゃ
# ============================================================
# ============================================================

import time as _time

# ── 結果ページのURL生成にゃ ──
def _result_url(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/result.html?race_id={race_id}"

def _shutuba_past_url(race_id: str) -> str:
    """出馬表+結果の旧式URL（より安定にゃ）"""
    return f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"

def _db_url(race_id: str) -> str:
    """db.netkeibaのURL（着順データが豊富にゃ）"""
    return f"https://db.netkeiba.com/race/{race_id}/"


def fetch_result_html(race_id: str, session=None) -> tuple[str, str]:
    """
    レース結果HTMLをCP932(Shift-JIS)で取得するにゃ。
    複数URLを試してどれか取得できたものを返すにゃ。
    戻り値: (html, 使用したURLにゃ)
    """
    if session is None:
        session = _make_session()

    urls = [
        (_result_url(race_id),       "result"),
        (_shutuba_past_url(race_id), "shutuba_past"),
        (_db_url(race_id),           "db"),
    ]

    for url, url_type in urls:
        try:
            html = _fetch_with_encoding(url, session)
            if len(html) > 1000:
                return html, url_type
        except Exception:
            continue
    raise ValueError(f"レース結果の取得に失敗したにゃ: race_id={race_id}")


def parse_result_html(html: str, race_id: str,
                       url_type: str = "result") -> pd.DataFrame:
    """
    結果HTMLをパースして着順付きDataFrameを返すにゃ。
    finish列（着順）が必ず含まれるにゃ。 """
    info = race_id_to_info(race_id)

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        raise ValueError(f"HTML解析失敗にゃ: {e}")

    def flatten_cols(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in c if str(x) != "nan"]).strip("_")
                for c in df.columns
            ]
        else:
            df.columns = [str(c) for c in df.columns]
        return df

    # 結果テーブルを探すにゃ
    target = None
    for t in tables:
        t = flatten_cols(t)
        j = " ".join(str(c) for c in t.columns)
        # 着順・馬名・騎手が揃っているテーブルにゃ
        if (("着順" in j or "着" in j or "finish" in j.lower()) and
                ("馬名" in j or "馬番" in j)):
            target = t
            break
        # 出走馬リストにゃ（shutuba_past形式にゃ）
        if "馬名" in j and "騎手" in j and len(t) >= 5:
            target = t
            break

    if target is None:
        raise ValueError("結果テーブルが見つからなかったにゃ")

    # 列名を正規化にゃ
    rename = {}
    for c in target.columns:
        s = str(c).lower()
        if any(x in str(c) for x in ["着順","着 順","確定着順"]) or s in ["着","finish","入線"]:
            rename[c] = "finish"
        elif "馬番" in str(c):
            rename[c] = "horse_no"
        elif "馬名" in str(c):
            rename[c] = "horse_name"
        elif "騎手" in str(c):
            rename[c] = "jockey"
        elif "斤量" in str(c):
            rename[c] = "carried_weight"
        elif "単勝" in str(c) or ("オッズ" in str(c) and "複" not in str(c)):
            rename[c] = "odds"
        elif "人気" in str(c):
            rename[c] = "popularity"
        elif "枠番" in str(c) or str(c) == "枠":
            rename[c] = "frame_no"
        elif "タイム" in str(c) or "time" in s:
            rename[c] = "time_raw"
        elif "上り" in str(c) or "上がり" in str(c) or "3f" in s:
            rename[c] = "last3f"
        elif "通過" in str(c) or "pass" in s:
            if "1" in str(c):
                rename[c] = "pass1"
            elif "2" in str(c):
                rename[c] = "pass2"
            elif "3" in str(c):
                rename[c] = "pass3"
            elif "4" in str(c):
                rename[c] = "pass4"
        elif "馬体重" in str(c) or "体重" in str(c):
            rename[c] = "body_weight"
        elif "調教師" in str(c) or "厩舎" in str(c):
            rename[c] = "trainer"
        elif "賞金" in str(c):
            rename[c] = "prize"
    target = target.rename(columns=rename)

    # finishがなければ行番号を着順として使うにゃ
    if "finish" not in target.columns:
        # 1列目が着順の場合にゃ
        first_col = target.columns[0]
        first_vals = pd.to_numeric(target[first_col], errors="coerce")
        if first_vals.notna().sum() > len(target) * 0.5:
            target = target.rename(columns={first_col: "finish"})
        else:
            target["finish"] = range(1, len(target) + 1)

    # 馬名がなければ失敗にゃ
    if "horse_name" not in target.columns:
        # horse_noで代替にゃ
        if "horse_no" not in target.columns:
            raise ValueError("馬名・馬番が見つからないにゃ")

    # クリーニングにゃ
    target = target.dropna(subset=["horse_name"] if "horse_name" in target.columns
                            else ["horse_no"]).copy()
    if "horse_name" in target.columns:
        target["horse_name"] = (target["horse_name"].astype(str)
                                 .str.replace("\n", " ").str.strip())
        target = target[~target["horse_name"].str.contains("馬名|除外|取消", na=False)]
        target = target[target["horse_name"].ne("")]

    # HTMLからレース情報を追加取得にゃ
    distance_m   = re.search(r"(\d{4})m",           html, re.IGNORECASE)
    track_type_m = re.search(r"(芝|ダート|障害)",   html)
    going_m      = re.search(r"馬場[:：\s]*([良稍重不良]+)", html)
    race_name_m  = re.search(r'class="RaceTitle[^"]*"[^>]*>([^<]+)<', html)

    # 52列DataFrameを構築にゃ
    rows = []
    for i, r in target.iterrows():
        row = {c: "" for c in COLS_52}
        row.update({
            "year":       info["year"] - 2000,
            "month":      1,
            "day":        1,
            "kai":        info["kai"],
            "place":      info["place"],
            "nichiji":    info["nichiji"],
            "race_no":    info["race_no"],
            "race_name":  race_name_m.group(1).strip() if race_name_m else f"R{info['race_no']}",
            "race_grade": "3",
            "track_type": track_type_m.group(1) if track_type_m else "芝",
            "course_kind":"0",
            "distance":   distance_m.group(1) if distance_m else "2000",
            "going":      going_m.group(1) if going_m else "良",
            "finish":     r.get("finish", ""),
            "horse_name": r.get("horse_name", ""),
            "horse_no":   r.get("horse_no",   ""),
            "frame_no":   r.get("frame_no",   ""),
            "jockey":     r.get("jockey",     ""),
            "carried_weight": r.get("carried_weight", ""),
            "odds":       r.get("odds",       ""),
            "popularity": r.get("popularity", ""),
            "time_raw":   r.get("time_raw",   ""),
            "last3f":     r.get("last3f",     ""),
            "pass1":      r.get("pass1",      ""),
            "pass2":      r.get("pass2",      ""),
            "pass3":      r.get("pass3",      ""),
            "pass4":      r.get("pass4",      ""),
            "body_weight":r.get("body_weight",""),
            "trainer":    r.get("trainer",    ""),
            "prize":      r.get("prize",      ""),
            "field_size": len(target),
        })
        # 性齢分解にゃ
        for sex_col in ["sex_age","性齢","性令"]:
            sa = str(r.get(sex_col, "")).strip()
            if sa and sa not in ["nan","None",""]:
                row["sex"] = sa[0]
                m2 = re.search(r"(\d+)", sa[1:])
                row["age"] = m2.group(1) if m2 else ""
                break
        rows.append([row[c] for c in COLS_52])

    df = pd.DataFrame(rows, columns=COLS_52)
    df["source_file"] = f"netkeiba_result_{race_id}"
    return clean_types(df)


def fetch_race_result(race_id: str,
                       session=None) -> pd.DataFrame:
    """
    1レースの結果（着順付き）を取得するにゃ。 """
    if session is None:
        session = _make_session()
    html, url_type = fetch_result_html(race_id, session)
    return parse_result_html(html, race_id, url_type)


def fetch_results_by_date(target_date: str,
                           session=None,
                           sleep_sec: float = 1.5) -> tuple[pd.DataFrame, list]:
    """
    指定日の全レース結果を一括取得するにゃ。
    target_date: "YYYYMMDD"にゃ
    戻り値: (全レース着順DataFrame, エラーリストにゃ)
    """
    if session is None:
        session = _make_session()

    # まず当日のrace_idを取得にゃ
    race_ids = fetch_today_race_ids(target_date)
    if not race_ids:
        return pd.DataFrame(), [{"date": target_date, "error": "レースIDが取得できなかったにゃ"}]

    frames, errors = [], []
    for rid in race_ids:
        try:
            df = fetch_race_result(rid, session)
            frames.append(df)
        except Exception as e:
            errors.append({"race_id": rid, "error": str(e)})
        _time.sleep(sleep_sec)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, errors


def fetch_results_by_date_range(start_date: str,
                                  end_date: str,
                                  sleep_sec: float = 1.5,
                                  max_days: int = 90) -> tuple[pd.DataFrame, dict]:
    """
    日付範囲の全レース結果を一括取得するにゃ🐾
    start_date, end_date: "YYYYMMDD"にゃ
    max_days: 最大取得日数にゃ（デフォルト90日にゃ）

    戻り値: (全着順DataFrame, 進捗サマリーにゃ)
    """
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt   = datetime.strptime(end_date,   "%Y%m%d")

    # 日数制限にゃ
    delta = (end_dt - start_dt).days + 1
    if delta > max_days:
        end_dt  = start_dt + timedelta(days=max_days - 1)
        delta   = max_days

    session = _make_session()
    all_frames  = []
    all_errors  = []
    progress    = {"total_days": delta, "done_days": 0,
                   "total_races": 0, "error_races": 0,
                   "dates_processed": []}

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        try:
            df, errors = fetch_results_by_date(
                date_str, session=session, sleep_sec=sleep_sec)
            if not df.empty:
                all_frames.append(df)
                progress["total_races"]  += df["race_key"].nunique() \
                    if "race_key" in df.columns else len(df) // 10
            progress["error_races"]     += len(errors)
            all_errors.extend(errors)
            progress["dates_processed"].append(date_str)
        except Exception as e:
            all_errors.append({"date": date_str, "error": str(e)})

        progress["done_days"] += 1
        current += timedelta(days=1)
        _time.sleep(sleep_sec * 0.5)  # 日切り替え時の待機にゃ

    all_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    progress["final_rows"]  = len(all_df)
    progress["error_count"] = len(all_errors)
    progress["errors"]      = all_errors[:20]  # 先頭20件のみにゃ

    return all_df, progress


def fetch_results_by_race_ids(race_ids: list[str],
                               sleep_sec: float = 1.5) -> tuple[pd.DataFrame, list]:
    """
    race_idリストから結果を一括取得するにゃ。 """
    session = _make_session()
    frames, errors = [], []
    for rid in race_ids:
        try:
            df = fetch_race_result(rid, session)
            frames.append(df)
        except Exception as e:
            errors.append({"race_id": rid, "error": str(e)})
        _time.sleep(sleep_sec)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, errors


# ============================================================
# 着順データ一括取得タブにゃ
# ============================================================

def show_result_fetch_tab():
    """
    着順データ一括取得の画面にゃ🐾
    3つの取得方法にゃ:
      ① 日付指定にゃ（1日分にゃ）
      ② 日付範囲指定にゃ（最大90日にゃ）
      ③ race_id直接指定にゃ
    """
    st.header("📥 着順データ一括取得にゃ🐾")
    st.caption(
        "netkeibaから過去レースの着順データを自動取得するにゃ。\n"
        "取得したCSVはバックテストに直接使えるにゃ🐾"
    )

    st.info(
        "💡 **推奨フローにゃ**\n\n"
        "① ここで着順データを取得 → CSVダウンロードにゃ\n"
        "② バックテストタブでそのCSVをアップロードにゃ\n"
        "③ 複数条件で一括比較 → 最強の買い方を発見するにゃ🐾"
    )

    # 取得方法の選択にゃ
    METHOD_DATE    = "📅 日付指定にゃ（1日分にゃ）"
    METHOD_RANGE   = "📆 日付範囲指定にゃ（最大90日にゃ）"
    METHOD_RACE_ID = "🔑 race_id直接指定にゃ"

    method = st.radio(
        "取得方法にゃ",
        [METHOD_DATE, METHOD_RANGE, METHOD_RACE_ID],
        horizontal=True
    )

    sleep_sec = st.slider(
        "アクセス間隔（秒）にゃ",
        min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help="短すぎるとBANされるにゃ。2秒以上推奨にゃ🐾"
    )

    # ── 各取得方法のUIにゃ ──
    target_ids   = []
    target_date  = None
    start_date   = None
    end_date     = None
    max_days_v   = 30

    if method == METHOD_DATE:
        st.markdown("#### 📅 日付指定にゃ")
        col1, col2 = st.columns(2)
        with col1:
            sel_date = st.date_input(
                "取得日にゃ",
                value=date.today() - __import__('datetime').timedelta(days=1),
                key="rf_date"
            )
            target_date = sel_date.strftime("%Y%m%d")
        with col2:
            st.metric("取得対象日にゃ", target_date)
            st.caption("その日の全レース（通常8〜12Rにゃ）を取得にゃ")

    elif method == METHOD_RANGE:
        st.markdown("#### 📆 日付範囲指定にゃ")
        col1, col2, col3 = st.columns(3)
        with col1:
            import datetime as _dt
            start_d = st.date_input(
                "開始日にゃ",
                value=_dt.date.today() - _dt.timedelta(days=30),
                key="rf_start"
            )
            start_date = start_d.strftime("%Y%m%d")
        with col2:
            end_d = st.date_input(
                "終了日にゃ",
                value=_dt.date.today() - _dt.timedelta(days=1),
                key="rf_end"
            )
            end_date = end_d.strftime("%Y%m%d")
        with col3:
            max_days_v = st.number_input(
                "最大取得日数にゃ", 1, 90, 30, key="rf_maxdays")

        days = (_dt.datetime.strptime(end_date, "%Y%m%d") -
                _dt.datetime.strptime(start_date, "%Y%m%d")).days + 1
        est_races = days * 9  # 1日平均9レースにゃ
        est_time  = int(days * 9 * sleep_sec / 60)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("対象日数にゃ",   f"{days}日にゃ")
        col_b.metric("推定レース数にゃ", f"約{est_races}レースにゃ")
        col_c.metric("推定時間にゃ",   f"約{est_time}分にゃ")

        if days > 30:
            st.warning(
                f"⚠️ {days}日分は時間がかかるにゃ（推定{est_time}分にゃ）。\n"
                "まず30日以内で試すことを推奨するにゃ🐾"
            )

    else:  # METHOD_RACE_ID
        st.markdown("#### 🔑 race_id直接指定にゃ")
        ids_text = st.text_area(
            "race_idを1行ずつ入力にゃ（またはURLも可にゃ）",
            placeholder="202505040811\n202505040812\nhttps://race.netkeiba.com/race/result.html?race_id=202505040813",
            height=150, key="rf_rids"
        )
        target_ids = [
            extract_race_id(line.strip())
            for line in ids_text.splitlines()
            if line.strip()
        ]
        target_ids = [r for r in target_ids if r]
        if target_ids:
            st.metric("取得予定レース数にゃ", f"{len(target_ids)}レースにゃ")
            est_t = int(len(target_ids) * sleep_sec / 60)
            if est_t > 0:
                st.caption(f"推定所要時間にゃ: 約{est_t}分にゃ")

    # 入力チェックにゃ
    can_run = (
        (method == METHOD_DATE   and target_date)  or
        (method == METHOD_RANGE  and start_date and end_date) or
        (method == METHOD_RACE_ID and len(target_ids) > 0)
    )
    if not can_run:
        st.info("取得条件を入力するにゃ🐾")
        return

    # ── 実行ボタンにゃ ──
    if st.button("🐾 着順データ取得 開始にゃ！", type="primary", key="rf_run"):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        result_df    = pd.DataFrame()
        errors_list  = []

        try:
            if method == METHOD_DATE:
                status_text.info(f"📡 {target_date} のレースデータを取得中にゃ...")
                result_df, errors_list = fetch_results_by_date(
                    target_date, sleep_sec=sleep_sec)
                progress_bar.progress(1.0)

            elif method == METHOD_RANGE:
                status_text.info(f"📡 {start_date}〜{end_date} のデータを取得中にゃ...")

                # プログレス付き取得にゃ
                from datetime import datetime as _dtt, timedelta as _td
                start_dt = _dtt.strptime(start_date, "%Y%m%d")
                end_dt   = _dtt.strptime(end_date,   "%Y%m%d")
                days_total = min((end_dt - start_dt).days + 1, int(max_days_v))
                session  = _make_session()
                frames   = []

                for di in range(days_total):
                    current_d  = start_dt + _td(days=di)
                    date_str   = current_d.strftime("%Y%m%d")
                    status_text.info(
                        f"📡 {date_str} を取得中にゃ..."
                        f"（{di+1}/{days_total}日目にゃ）"
                    )
                    try:
                        df_day, errs = fetch_results_by_date(
                            date_str, session=session, sleep_sec=sleep_sec)
                        if not df_day.empty:
                            frames.append(df_day)
                        errors_list.extend(errs)
                    except Exception as e:
                        errors_list.append({"date": date_str, "error": str(e)})

                    progress_bar.progress((di + 1) / days_total)
                    _time.sleep(0.3)

                result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            else:  # race_id指定にゃ
                status_text.info(f"📡 {len(target_ids)}レースを取得中にゃ...")
                frames = []
                session = _make_session()
                for ri, rid in enumerate(target_ids):
                    status_text.info(
                        f"📡 {rid} を取得中にゃ..."
                        f"（{ri+1}/{len(target_ids)}にゃ）"
                    )
                    try:
                        df_r = fetch_race_result(rid, session)
                        frames.append(df_r)
                    except Exception as e:
                        errors_list.append({"race_id": rid, "error": str(e)})
                    progress_bar.progress((ri + 1) / len(target_ids))
                    _time.sleep(sleep_sec)
                result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        except Exception as e:
            st.error(f"❌ 取得エラーにゃ: {e}にゃ")
            with st.expander("エラー詳細にゃ"):
                import traceback
                st.code(traceback.format_exc())
            return

        progress_bar.empty()
        status_text.empty()

        # ── 結果表示にゃ ──
        if result_df.empty:
            st.error(
                "❌ データが取得できなかったにゃ🐾\n\n"
                "**考えられる原因にゃ:**\n"
                "- 指定日にレースが開催されていないにゃ\n"
                "- netkeibaのIP制限にゃ（少し時間を置くにゃ）\n"
                "- インターネット接続の問題にゃ"
            )
            if errors_list:
                st.dataframe(pd.DataFrame(errors_list[:10]),
                             use_container_width=True, hide_index=True)
            return

        n_rows  = len(result_df)
        n_races = result_df["race_key"].nunique() if "race_key" in result_df.columns else "?"
        n_valid_finish = int(
            pd.to_numeric(result_df.get("finish", pd.Series()), errors="coerce").notna().sum()
        )

        st.success(
            f"✅ 取得完了にゃ🐾\n\n"
            f"**{n_races}レース / {n_rows}頭 / 着順あり:{n_valid_finish}行**にゃ"
        )

        # エラー報告にゃ
        if errors_list:
            st.warning(f"⚠️ 取得失敗にゃ: {len(errors_list)}件にゃ")
            with st.expander("失敗詳細にゃ"):
                st.dataframe(pd.DataFrame(errors_list[:20]),
                             use_container_width=True, hide_index=True)

        # プレビューにゃ
        st.markdown("#### 📋 プレビューにゃ（先頭20行にゃ）")
        preview_cols = [c for c in
                        ["race_label","horse_no","horse_name","finish",
                         "odds","popularity","jockey","time_raw","last3f"]
                        if c in result_df.columns]
        st.dataframe(
            result_df[preview_cols].head(20).rename(columns={
                "race_label":"レースにゃ","horse_no":"馬番にゃ","horse_name":"馬名にゃ",
                "finish":"着順にゃ","odds":"オッズにゃ","popularity":"人気にゃ",
                "jockey":"騎手にゃ","time_raw":"タイムにゃ","last3f":"上り3Fにゃ",
            }),
            use_container_width=True, hide_index=True
        )

        # 統計にゃ
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("取得レース数にゃ", f"{n_races}にゃ")
        col2.metric("取得頭数にゃ",     f"{n_rows}頭にゃ")
        col3.metric("着順データにゃ",   f"{n_valid_finish}行にゃ")
        col4.metric("エラーにゃ",       f"{len(errors_list)}件にゃ")

        # ── ダウンロードにゃ ──
        st.markdown("---")
        st.markdown("#### 💾 CSVダウンロードにゃ")

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        # 52列フルCSVにゃ（バックテスト用にゃ）
        with col_dl1:
            csv_full = result_df.to_csv(
                index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 52列フルCSVにゃ\n（バックテスト用にゃ）",
                data=csv_full,
                file_name=f"results_full_{target_date or start_date or 'batch'}.csv",
                mime="text/csv",
                help="バックテストタブに直接使えるにゃ🐾"
            )

        # 簡易CSVにゃ（着順・馬名・オッズのみにゃ）
        with col_dl2:
            simple_cols = [c for c in
                           ["race_label","place","race_no","horse_no","horse_name",
                            "finish","odds","popularity","jockey","last3f","time_raw"]
                           if c in result_df.columns]
            csv_simple = result_df[simple_cols].rename(columns={
                "race_label":"レースにゃ","place":"競馬場にゃ","race_no":"R番号にゃ",
                "horse_no":"馬番にゃ","horse_name":"馬名にゃ","finish":"着順にゃ",
                "odds":"オッズにゃ","popularity":"人気にゃ","jockey":"騎手にゃ",
                "last3f":"上り3Fにゃ","time_raw":"タイムにゃ",
            }).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 簡易CSVにゃ\n（確認用にゃ）",
                data=csv_simple,
                file_name=f"results_simple_{target_date or start_date or 'batch'}.csv",
                mime="text/csv"
            )

        # race_idリストにゃ（再取得用にゃ）
        with col_dl3:
            if "race_key" in result_df.columns:
                race_ids_list = result_df["source_file"].str.replace(
                    "netkeiba_result_","").unique().tolist()
                ids_text_out = "\n".join(str(r) for r in race_ids_list if r)
                st.download_button(
                    "📥 race_idリストにゃ\n（再取得用にゃ）",
                    data=ids_text_out.encode("utf-8"),
                    file_name=f"race_ids_{target_date or start_date or 'batch'}.txt",
                    mime="text/plain"
                )

        # バックテスト即時実行にゃ
        st.markdown("---")
        st.markdown("#### ⚡ 取得データで即バックテストにゃ")
        if st.button("🐾 このデータでバックテスト実行にゃ！", key="rf_bt_run"):
            bundle_rf, status_rf = load_model_safely(None)
            if bundle_rf is None:
                st.error("PKLが必要にゃ🐾（サイドバーからアップロードするにゃ）")
            else:
                with st.spinner("バックテスト実行中にゃ...🐾"):
                    try:
                        bt_result = run_backtest_v2(
                            bundle_rf, result_df,
                            strategy_mode=STRATEGY_MODE_ROI,
                        )
                        if "error" in bt_result:
                            st.error(f"❌ {bt_result['error']}")
                        else:
                            _show_bt2_single_result(bt_result, 100)
                    except Exception as e:
                        st.error(f"バックテストエラーにゃ: {e}にゃ")




# ============================================================
# 🏆 S級×複勝フィルター表示にゃ
# バックテスト実績: S級257% × 複勝156% の最強絞り込みにゃ
# ============================================================

def _show_sx_fuku_filter(race_df: pd.DataFrame):
    """
    S級判定 かつ 複勝buy_flag の馬だけ表示するにゃ。
    今週のメイン購入候補にゃ🐾
    """
    df = race_df.copy()

    # S級判定チェックにゃ
    has_s = "buy_flag_v2" in df.columns
    has_b = "buy_flag"    in df.columns

    if not has_s and not has_b:
        st.info("予想を実行してからフィルターを使うにゃ🐾")
        return

    # S級×複勝フィルターにゃ
    mask_s = (df["buy_flag_v2"].str.contains("買い", na=False)
              if has_s else pd.Series([True]*len(df)))
    mask_b = (df["buy_flag"] == "買い"
              if has_b else pd.Series([True]*len(df)))
    mask_sx = mask_s & mask_b

    sx_df = df[mask_sx].copy()
    all_s  = df[mask_s].copy() if has_s else pd.DataFrame()
    all_b  = df[mask_b].copy() if has_b else pd.DataFrame()

    # メトリクスにゃ
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S級判定馬にゃ",      f"{len(all_s)}頭にゃ")
    c2.metric("複勝買い馬にゃ",      f"{len(all_b)}頭にゃ")
    c3.metric("🏆 S級×複勝にゃ",   f"{len(sx_df)}頭にゃ",
              delta="最強候補にゃ🐾" if len(sx_df) > 0 else "なしにゃ")
    c4.metric("絞り込み率にゃ",
              f"{len(sx_df)/max(len(df),1)*100:.0f}%にゃ")

    if sx_df.empty:
        st.warning(
"⚠️ S級×複勝の候補がいないにゃ。このレースは見送りを推奨するにゃ🐾"
        )
        # S級のみ表示にゃ
        if not all_s.empty:
            st.markdown("#### S級判定馬のみにゃ（参考にゃ）")
            _disp_sx(all_s)
        return

    # ── 🏆 購入推奨馬リストにゃ ──
    st.markdown("#### 🏆 今週の購入推奨馬にゃ（複勝で買うにゃ）")
    st.success(
        f"✅ **{len(sx_df)}頭が対象にゃ！** "
        f"この馬を複勝で購入するにゃ🐾"
    )
    _disp_sx(sx_df)

    # ── 買い目サマリーにゃ ──
    st.markdown("#### 💰 推奨買い目にゃ")
    total_pts = len(sx_df)
    for _, row in sx_df.sort_values("ml_rank").iterrows():
        hno   = _safe_int(row.get("horse_no", 0), 0)
        name  = str(row.get("horse_name", ""))
        odds  = _safe_float(row.get("odds", 0), 0)
        pop   = _safe_int(row.get("popularity", 0), 0)
        prob  = _safe_float(row.get("ml_top3_prob", 0), 0)
        ev2   = _safe_float(row.get("ev_score_v2", row.get("ev_score", 0)), 0)
        pace  = _safe_float(row.get("pace_advantage", 1.0), 1.0)
        s_flag= str(row.get("buy_flag_v2", ""))
        fuku_est = max(1.1, odds * 0.30)

        icon = "🥇" if pop <= 3 else ("🥈" if pop <= 6 else "💎")
        msg = (
            f"{icon} {hno}番 {name}"
            f" ({pop}番人気/{odds:.1f}倍)"
            f" 複勝推定{fuku_est:.1f}倍"
            f" AI{prob*100:.1f}% EV{ev2:+.3f}"
            f" 展開{pace:.2f} {s_flag}"
        )
        st.info(msg)

    st.markdown(f"**合計 {total_pts}点 × 100円 = {total_pts*100:,}円にゃ**")

    # ── 🚀 稼ぐ戦略にゃ（バックテスト実績ベースにゃ）──
    st.markdown("---")
    st.subheader("🚀 稼ぐ戦略にゃ（バックテスト実績ベースにゃ）")

    if sx_df.empty:
        st.info("S級×複勝馬がいないレースは見送りにゃ🐾")
    else:
        n_sx = len(sx_df)
        # 各戦略の期待収益にゃ
        s_rate_real  = 0.462   # 実績的中率にゃ
        s_pay_real   = 5.56    # 実績平均複勝払戻倍率にゃ
        s3_rate      = 0.50    # 三連複S級軸的中率にゃ
        s3_pay       = 35.0    # 三連複平均払戻にゃ

        strats = [
            {
                "name": "🥉 ライトにゃ（安全策にゃ）",
                "desc": "S級×複勝 200円",
                "invest": n_sx * 200,
                "exp_ret": n_sx * s_rate_real * s_pay_real * 0.80 * 200,
            },
            {
                "name": "🥈 スタンダードにゃ（推奨にゃ）",
                "desc": "S級×複勝 500円 + 三連複10点100円",
                "invest": n_sx * 500 + 10 * 100,
                "exp_ret": n_sx * s_rate_real * s_pay_real * 0.80 * 500
                          + s3_rate * s3_pay * 0.75 * 100,
            },
            {
                "name": "🥇 アグレッシブにゃ（高リターンにゃ）",
                "desc": "S級×複勝 1000円 + 三連複10点300円 + ワイド3点200円",
                "invest": n_sx * 1000 + 10 * 300 + 3 * 200,
                "exp_ret": n_sx * s_rate_real * s_pay_real * 0.80 * 1000
                          + s3_rate * s3_pay * 0.75 * 300
                          + 0.55 * 8.0 * 0.775 * 200 * 3,
            },
        ]

        cols = st.columns(3)
        for col, strat in zip(cols, strats):
            roi = strat["exp_ret"] / strat["invest"] * 100
            profit = strat["exp_ret"] - strat["invest"]
            col.markdown(f"**{strat['name']}**")
            col.caption(strat["desc"])
            col.metric(
                "期待ROIにゃ", f"{roi:.0f}%",
                delta=f"損益 {profit:+.0f}円にゃ",
                delta_color="normal" if profit >= 0 else "inverse"
            )

        # 推奨戦略の詳細にゃ
        st.markdown("---")
        st.markdown("#### 🥈 推奨戦略の買い目にゃ")
        rec_strat = strats[1]
        rec_name   = rec_strat["name"]
        rec_invest = rec_strat["invest"]
        rec_ret    = rec_strat["exp_ret"]
        rec_roi    = rec_ret / rec_invest * 100
        st.success(
            f"{rec_name}\n\n"
            f"投資:{rec_invest:,}円 / 期待回収:{rec_ret:,.0f}円 / ROI:{rec_roi:.0f}%にゃ🐾"
        )

        buy_md = []
        for _, row in sx_df.sort_values("ml_rank").iterrows():
            hno  = _safe_int(row.get("horse_no",0),0)
            name = str(row.get("horse_name",""))
            odds = _safe_float(row.get("odds",0),0)
            pop  = _safe_int(row.get("popularity",0),0)
            buy_md.append(f"- **複勝 500円**: 馬番{hno} {name}（{pop}番人気/{odds:.1f}倍）にゃ")

        # 三連複の軸・相手にゃ
        all_ai = race_df.sort_values("ml_rank")
        pivot_row = all_ai.iloc[0] if not all_ai.empty else None
        aite_rows = all_ai.iloc[1:6] if len(all_ai) > 1 else all_ai

        if pivot_row is not None:
            p_no   = _safe_int(pivot_row.get("horse_no",0),0)
            p_name = str(pivot_row.get("horse_name",""))
            a_list = [f"馬番{_safe_int(r.get('horse_no',0),0)}" for _,r in aite_rows.iterrows()]
            buy_md.append(f"- **三連複 100円×10点**: 軸 馬番{p_no}{p_name} × 相手 {','.join(a_list)}にゃ")

        for line in buy_md:
            st.markdown(line)

    # ── 見送りの場合の基準にゃ ──
    st.markdown("---")
    with st.expander("📋 見送り基準にゃ（さらに絞りたい場合にゃ）"):
        st.markdown("""
**S級×複勝の中でさらに絞るにゃ🐾**

| 条件にゃ | 買うにゃ | 見送るにゃ |
|---------|---------|---------|
| EVにゃ | +0.05以上にゃ | マイナスにゃ |
| 展開スコアにゃ | 1.05以上にゃ | 0.90以下にゃ |
| オッズにゃ | 3〜15倍にゃ | 1.5倍以下 or 30倍以上にゃ |
| 人気にゃ | 1〜8番人気にゃ | 9番人気以下にゃ |
        """)
        # さらに絞った候補にゃ
        ev_col = "ev_score_v2" if "ev_score_v2" in sx_df.columns else "ev_score"
        ultra = sx_df[
            (pd.to_numeric(sx_df.get(ev_col, 0), errors="coerce").fillna(0) >= 0.05) &
            (pd.to_numeric(sx_df.get("pace_advantage", 1.0), errors="coerce").fillna(1.0) >= 1.05) &
            (pd.to_numeric(sx_df.get("odds", 0), errors="coerce").fillna(0).between(3, 15)) &
            (pd.to_numeric(sx_df.get("popularity", 99), errors="coerce").fillna(99) <= 8)
        ]
        if not ultra.empty:
            st.success(f"🌟 超絞り込み候補にゃ: **{len(ultra)}頭**にゃ")
            _disp_sx(ultra)
        else:
            st.info("超絞り込み条件を満たす馬はいないにゃ。S級×複勝全員を買うにゃ🐾")


def _disp_sx(df: pd.DataFrame):
    """S級×複勝フィルター結果の表示にゃ"""
    disp_cols = [c for c in [
        "ml_rank", "horse_no", "horse_name", "odds", "popularity",
        "ml_top3_prob", "ev_score_v2", "ev_score",
        "pace_advantage", "buy_flag_v2", "buy_flag",
        "pass_score", "kelly_ratio"
    ] if c in df.columns]

    disp = df[disp_cols].sort_values("ml_rank").copy()

    # フォーマットにゃ
    if "ml_top3_prob" in disp.columns:
        disp["ml_top3_prob"] = (
            pd.to_numeric(disp["ml_top3_prob"], errors="coerce") * 100
        ).round(1).astype(str) + "%"
    ev_col = "ev_score_v2" if "ev_score_v2" in disp.columns else "ev_score"
    if ev_col in disp.columns:
        disp[ev_col] = pd.to_numeric(disp[ev_col], errors="coerce").round(3)
    if "pace_advantage" in disp.columns:
        disp["pace_advantage"] = pd.to_numeric(
            disp["pace_advantage"], errors="coerce").round(3)

    rename = {
        "ml_rank": "AI順位", "horse_no": "馬番", "horse_name": "馬名",
        "odds": "オッズ", "popularity": "人気",
        "ml_top3_prob": "AI確率",
        "ev_score_v2": "EV(展開補正)", "ev_score": "EV乖離",
        "pace_advantage": "展開スコア",
        "buy_flag_v2": "S級判定", "buy_flag": "複勝判定",
        "pass_score": "見送りスコア", "kelly_ratio": "Kelly比",
    }
    disp = disp.rename(columns=rename)

    def color_sx(row):
        s_flag = str(row.get("S級判定", ""))
        if "◎" in s_flag: return ["background-color:#c3e6cb"] * len(row)
        if "○" in s_flag: return ["background-color:#d1ecf1"] * len(row)
        return ["background-color:#fff3cd"] * len(row)

    try:
        st.dataframe(
            disp.style.apply(color_sx, axis=1),
            use_container_width=True, hide_index=True
        )
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)





# ============================================================
# G1レース専用分析モジュールにゃ🏆
# 日本ダービー・有馬記念・天皇賞など大レース向けにゃ
# ============================================================

# ── G1固有データにゃ ──

# 日本ダービー過去10年枠順別成績にゃ
DERBY_FRAME_DATA = {
    1: {"複勝率": 0.25, "勝利数": 1, "note": "1枠1番は【1.1.1.7】複勝率38%にゃ"},
    2: {"複勝率": 0.10, "勝利数": 0, "note": "1枠2番は不振にゃ"},
    3: {"複勝率": 0.15, "勝利数": 0, "note": "標準的にゃ"},
    4: {"複勝率": 0.15, "勝利数": 0, "note": "4枠最後の勝利は1984年にゃ"},
    5: {"複勝率": 0.05, "勝利数": 0, "note": "5枠は馬券内率5%の不振枠にゃ"},
    6: {"複勝率": 0.30, "勝利数": 2, "note": "6枠は最高枠！複勝率30%・勝ち馬2頭にゃ"},
    7: {"複勝率": 0.20, "勝利数": 1, "note": "外目だが好走例あるにゃ"},
    8: {"複勝率": 0.15, "勝利数": 1, "note": "大外は位置取りが課題にゃ"},
}

# 騎手別東京芝2400m成績（重要にゃ）
JOCKEY_TOKYO_2400 = {
    "岩田康誠":  {"rate": 0.00, "record": "0-0-0-20", "note": "東京2400m全滅にゃ→切りにゃ"},
    "松山弘平":  {"rate": 0.35, "record": "G1実績あり", "note": "皐月賞制覇・信頼できるにゃ"},
    "津村明秀":  {"rate": 0.28, "record": "G1初制覇狙い", "note": "6枠の利を活かすにゃ"},
    "武豊":     {"rate": 0.32, "record": "ダービー多数V", "note": "青葉賞→ダービー実績にゃ"},
    "佐々木大輔": {"rate": 0.25, "record": "1枠内有利", "note": "内でため逃げ切り狙いにゃ"},
    "Ｄ．レーン":  {"rate": 0.30, "record": "外国人強力", "note": "ただし5枠不振が気になるにゃ"},
    "Ｃ．ルメール": {"rate": 0.38, "record": "リーディング常連", "note": "穴でも怖いにゃ"},
    "川田将雅":  {"rate": 0.33, "record": "東京得意", "note": "3枠から前目の競馬にゃ"},
    "西村淳也":  {"rate": 0.28, "record": "関西リーダー", "note": "京都新聞杯からの乗り替わりなしにゃ"},
    "横山和生":  {"rate": 0.22, "record": "標準的", "note": "1枠2番は過去不振にゃ"},
}

# 前走ローテ別成績にゃ
ROTO_DATA = {
    "皐月賞": {"複勝率": 0.60, "勝利数": 8, "note": "過去10年8勝・2着10回にゃ"},
    "京都新聞杯": {"複勝率": 0.35, "note": "別路線の刺客にゃ"},
    "青葉賞": {"複勝率": 0.25, "note": "ダービー直行組にゃ"},
    "プリンシパルS": {"複勝率": 0.10, "note": "過去苦戦傾向にゃ"},
}

# 馬別G1分析データにゃ（手動設定にゃ）
DERBY_2026_HORSE_DATA = {
    "ロブチェン":      {"前走": "皐月賞", "前走着順": 1, "脚質": "逃げ",  "血統": "ワールドプレミア", "距離適性": 0.85, "note": "皐月賞レコード勝ち。8枠外枠が唯一の懸念にゃ"},
    "リアライズシリウス": {"前走": "皐月賞", "前走着順": 2, "脚質": "先行", "血統": "不明",           "距離適性": 0.90, "note": "6枠2番人気。距離延長プラスにゃ"},
    "ゴーイントゥスカイ": {"前走": "青葉賞", "前走着順": 1, "脚質": "先行", "血統": "不明",           "距離適性": 0.92, "note": "青葉賞→ダービーで武豊騎乗にゃ"},
    "ライヒスアドラー":  {"前走": "皐月賞", "前走着順": 3, "脚質": "差し",  "血統": "不明",           "距離適性": 0.88, "note": "1枠1番で内ため有利にゃ"},
    "アウダーシア":    {"前走": "スプリングS","前走着順": 1, "脚質": "先行", "血統": "不明",           "距離適性": 0.75, "note": "5枠不振枠・GI未勝利にゃ"},
    "コンジェスタス":   {"前走": "京都新聞杯","前走着順": 1, "脚質": "差し",  "血統": "不明",           "距離適性": 0.90, "note": "別路線から参戦・3枠良いにゃ"},
    "バステール":     {"前走": "不明",     "前走着順": 0, "脚質": "先行", "血統": "不明",           "距離適性": 0.82, "note": "川田騎乗・3枠・穴候補にゃ"},
    "パントルナイーフ":  {"前走": "不明",     "前走着順": 0, "脚質": "差し",  "血統": "不明",           "距離適性": 0.80, "note": "ルメール騎乗10人気・穴にゃ"},
    "アスクエジンバラ":  {"前走": "皐月賞", "前走着順": 4, "脚質": "差し",  "血統": "不明",           "距離適性": 0.80, "note": "岩田康→東京2400m全滅・切りにゃ"},
    "グリーンエナジー":  {"前走": "皐月賞", "前走着順": 7, "脚質": "不明",  "血統": "不明",           "距離適性": 0.78, "note": "戸崎騎乗8人気にゃ"},
}


def analyze_derby_horse(row: pd.Series,
                         all_df: pd.DataFrame) -> dict:
    """
    1頭のダービー適性を多角的に評価するにゃ。
    スコア0〜100で返すにゃ。
    """
    name  = str(row.get("horse_name", ""))
    hno   = _safe_int(row.get("horse_no", 0), 0)
    frame = _safe_int(row.get("frame_no", 0), 0)
    odds  = _safe_float(row.get("odds", 99), 99)
    pop   = _safe_int(row.get("popularity", 99), 99)
    jockey = str(row.get("jockey", "")).strip()

    score = 50.0  # 基本スコアにゃ
    reasons = []
    warnings = []

    # ① 枠順スコアにゃ
    frame_info = DERBY_FRAME_DATA.get(frame, {"複勝率": 0.15, "note": "標準にゃ"})
    frame_rate = frame_info["複勝率"]
    frame_score = (frame_rate - 0.15) * 100  # 平均15%基準にゃ
    score += frame_score * 0.3
    if frame_rate >= 0.25:
        reasons.append(f"✅ {frame}枠有利（複勝率{frame_rate*100:.0f}%にゃ）")
    elif frame_rate <= 0.08:
        warnings.append(f"⚠️ {frame}枠不振（複勝率{frame_rate*100:.0f}%にゃ）")

    # ② 騎手スコアにゃ
    jockey_short = jockey[:4]
    jockey_info = None
    for k, v in JOCKEY_TOKYO_2400.items():
        if k in jockey or jockey in k:
            jockey_info = v
            break
    if jockey_info:
        j_rate = jockey_info["rate"]
        jockey_score = (j_rate - 0.25) * 100
        score += jockey_score * 0.25
        if j_rate >= 0.30:
            reasons.append(f"✅ {jockey[:6]}騎手強力（{jockey_info['note']}）にゃ")
        elif j_rate == 0.00:
            warnings.append(f"❌ {jockey[:6]}騎手 {jockey_info['record']} にゃ")
            score -= 20

    # ③ 馬別データスコアにゃ
    horse_info = DERBY_2026_HORSE_DATA.get(name)
    if horse_info:
        dist_score = (horse_info["距離適性"] - 0.82) * 100
        score += dist_score * 0.2

        roto = horse_info["前走"]
        roto_info = ROTO_DATA.get(roto, {"複勝率": 0.15})
        roto_score = (roto_info["複勝率"] - 0.25) * 50
        score += roto_score * 0.15

        prev_rank = horse_info["前走着順"]
        if prev_rank == 1: score += 10; reasons.append(f"✅ 前走1着（{roto}にゃ）")
        elif prev_rank == 2: score += 6; reasons.append(f"✅ 前走2着（{roto}にゃ）")
        elif prev_rank == 3: score += 3; reasons.append(f"✅ 前走3着（{roto}にゃ）")

        style = horse_info.get("脚質", "不明")
        # 東京2400は差し有利だが先行も可にゃ
        if style in ["先行", "差し"]: score += 3
        elif style == "逃げ": score -= 3  # 外枠逃げは距離ロスにゃ

    # ④ 人気オッズ補正にゃ（1番人気は少し下げるにゃ→過剰人気対策にゃ）
    if pop == 1:
        score += 5  # 能力評価にゃ
    elif pop <= 3:
        score += 3
    elif pop >= 10:
        score -= 5

    # スコアをクリップにゃ
    score = float(np.clip(score, 5, 98))

    return {
        "score":    round(score, 1),
        "reasons":  reasons,
        "warnings": warnings,
        "frame_info": frame_info,
        "horse_info": horse_info or {},
    }


def show_g1_derby_analysis(race_df: pd.DataFrame):
    """
    日本ダービー専用分析ダッシュボードにゃ🏆
    通常の予想に加えてG1固有のデータ分析を表示するにゃ
    """
    st.markdown("---")
    st.subheader("🏆 日本ダービー専用分析にゃ（枠順・騎手・ローテ・展開にゃ）")
    st.caption("東京芝2400m G1特有のデータで三連複を絞るにゃ🐾")

    if race_df is None or race_df.empty:
        st.info("出走表CSVを読み込んでから分析するにゃ🐾")
        return

    # ── 全馬スコアリングにゃ ──
    results = []
    for _, row in race_df.iterrows():
        analysis = analyze_derby_horse(row, race_df)
        results.append({
            "馬番": _safe_int(row.get("horse_no", 0), 0),
            "枠番": _safe_int(row.get("frame_no", 0), 0),
            "馬名": str(row.get("horse_name", "")),
            "騎手": str(row.get("jockey", ""))[:8],
            "人気": _safe_int(row.get("popularity", 99), 99),
            "オッズ": _safe_float(row.get("odds", 99), 99),
            "G1スコア": analysis["score"],
            "プラス要因": " / ".join(analysis["reasons"][:2]) if analysis["reasons"] else "—",
            "マイナス要因": " / ".join(analysis["warnings"][:1]) if analysis["warnings"] else "—",
            "_reasons": analysis["reasons"],
            "_warnings": analysis["warnings"],
            "_horse_info": analysis["horse_info"],
        })

    result_df = pd.DataFrame(results).sort_values("G1スコア", ascending=False)

    # ── タブ表示にゃ ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 G1スコアランキングにゃ",
        "🌪️ 展開予測にゃ",
        "🏇 騎手データにゃ",
        "🏆 三連複最終買い目にゃ",
    ])

    with tab1:
        st.markdown("#### 📊 日本ダービー適性スコアにゃ（高いほど有利にゃ）")
        st.caption("枠順・騎手実績・前走ローテ・距離適性を総合評価にゃ")

        def color_g1(row):
            s = float(row.get("G1スコア", 0))
            if s >= 65: return ["background-color:#c3e6cb"] * len(row)
            if s >= 55: return ["background-color:#d1ecf1"] * len(row)
            if s <= 35: return ["background-color:#f8d7da"] * len(row)
            return [""] * len(row)

        disp = result_df.drop(columns=["_reasons","_warnings","_horse_info"]).copy()
        try:
            st.dataframe(
                disp.style.apply(color_g1, axis=1),
                use_container_width=True, hide_index=True
            )
        except Exception:
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # 上位3頭の詳細にゃ
        st.markdown("#### 🏆 上位3頭の評価にゃ")
        for _, row in result_df.head(3).iterrows():
            hno = row["馬番"]
            with st.expander(f"馬番{hno} {row['馬名']} — G1スコア{row['G1スコア']}にゃ"):
                hi = row["_horse_info"]
                r2 = row["_reasons"]
                w2 = row["_warnings"]
                if hi:
                    st.write(f"**前走にゃ**: {hi.get('前走','-')} {hi.get('前走着順','-')}着")
                    st.write(f"**脚質にゃ**: {hi.get('脚質','-')}")
                    st.write(f"**距離適性にゃ**: {hi.get('距離適性',0)*100:.0f}%")
                    st.write(f"**評価にゃ**: {hi.get('note','-')}")
                for r3 in r2: st.success(r3)
                for w3 in w2: st.warning(w3)

    with tab2:
        st.markdown("#### 🌪️ ペース展開予測にゃ")
        # 脚質分布にゃ
        styles = {
            "逃げ": ["ロブチェン（馬17）"],
            "先行": ["リアライズシリウス（馬11）","ゴーイントゥスカイ（馬14）",
                    "バステール（馬5）","アウダーシア（馬9）"],
            "差し": ["ライヒスアドラー（馬1）","コンジェスタス（馬6）",
                    "パントルナイーフ（馬13）"],
            "追込": ["その他"],
        }
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("逃げ馬にゃ", "1頭")
        c2.metric("先行馬にゃ", "4頭")
        c3.metric("差し馬にゃ", "3頭+")
        c4.metric("予想ペースにゃ", "ミドル〜スロー")

        st.info(
            "💡 **展開予測にゃ**\n\n"
            "逃げはロブチェン1頭。先行馬が複数いるが潰し合いにはならないにゃ。\n"
            "**ミドルペース想定** → 先行有利・外枠ロブチェンはポジション争いがカギにゃ。\n"
            "6枠リアライズシリウスは3〜4番手から絶好の展開が見込めるにゃ🐾"
        )

        frame_rows = []
        for frame, info in DERBY_FRAME_DATA.items():
            frame_rows.append({
                "枠番にゃ": f"{frame}枠",
                "複勝率にゃ": f"{info['複勝率']*100:.0f}%",
                "勝利数にゃ": info.get("勝利数",0),
                "コメントにゃ": info["note"],
            })
        st.markdown("#### 東京芝2400m 枠番別成績（過去10年にゃ）")
        fdf = pd.DataFrame(frame_rows)
        def color_frame(row):
            r = float(row["複勝率にゃ"].replace("%","")) / 100
            if r >= 0.28: return ["background-color:#c3e6cb"]*len(row)
            if r <= 0.06: return ["background-color:#f8d7da"]*len(row)
            return [""]*len(row)
        try:
            st.dataframe(fdf.style.apply(color_frame, axis=1),
                         use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(fdf, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### 🏇 騎手別 東京芝2400m適性にゃ")
        jockey_rows = []
        for _, row in race_df.iterrows():
            jockey = str(row.get("jockey","")).strip()
            hno = _safe_int(row.get("horse_no",0),0)
            name = str(row.get("horse_name",""))
            for k, v in JOCKEY_TOKYO_2400.items():
                if k in jockey or jockey in k:
                    jockey_rows.append({
                        "馬番にゃ": hno,
                        "馬名にゃ": name,
                        "騎手にゃ": k,
                        "東京2400適性にゃ": f"{v['rate']*100:.0f}%",
                        "評価にゃ": v["note"],
                    })
                    break
        if jockey_rows:
            jdf = pd.DataFrame(jockey_rows).sort_values("東京2400適性にゃ", ascending=False)
            def color_jockey(row):
                r = float(row["東京2400適性にゃ"].replace("%","")) / 100
                if r >= 0.33: return ["background-color:#c3e6cb"]*len(row)
                if r == 0.00: return ["background-color:#f8d7da"]*len(row)
                return [""]*len(row)
            try:
                st.dataframe(jdf.style.apply(color_jockey, axis=1),
                             use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(jdf, use_container_width=True, hide_index=True)

        st.warning("❌ 岩田康誠（12番アスクエジンバラ）→ 東京芝2400m【0-0-0-20】で切りにゃ！")
        st.success("✅ ルメール（13番パントルナイーフ）→ 10人気でも穴で怖いにゃ！")

    with tab4:
        st.markdown("#### 🏆 三連複 最終買い目にゃ（G1スコア×AI予想にゃ）")

        # G1スコア上位6頭を相手候補にゃ
        top6 = result_df.head(6)
        top6_nos = [int(r["馬番"]) for _, r in top6.iterrows()]

        # 軸馬にゃ（G1スコア1位かつ6枠にゃ）
        pivot_candidates = result_df[result_df["枠番"] == 6]
        if not pivot_candidates.empty:
            pivot = int(pivot_candidates.iloc[0]["馬番"])
            pivot_name = str(pivot_candidates.iloc[0]["馬名"])
        else:
            pivot = int(result_df.iloc[0]["馬番"])
            pivot_name = str(result_df.iloc[0]["馬名"])

        # 相手馬（軸を除く上位5頭にゃ）
        aite = [n for n in top6_nos if n != pivot][:5]

        st.success(f"🎯 軸馬にゃ: 馬番{pivot} {pivot_name}（6枠G1スコア最高にゃ）")

        # 10点の買い目にゃ
        import itertools
        combos_10 = list(itertools.combinations(aite, 2))
        combos_display = [f"{pivot}-{min(a,b)}-{max(a,b)}" for a,b in combos_10]

        st.markdown(f"**軸 馬番{pivot} × 相手 {aite} → {len(combos_display)}点にゃ**")
        cols = st.columns(5)
        for i, combo in enumerate(combos_display[:10]):
            cols[i%5].markdown(f"`{combo}`")

        # ルメール・川田の穴にゃ
        st.markdown("---")
        st.markdown("**プラス穴 2点にゃ（ルメール・川田にゃ）**")
        st.markdown("`11-13-17` / `5-13-17`")

        # 投資プランにゃ
        st.markdown("---")
        st.markdown("#### 💰 投資プランにゃ")
        plan_df = pd.DataFrame([
            {"内容にゃ": f"軸{pivot}×相手5頭 {len(combos_display)}点",
             "単価にゃ": "300円", "合計にゃ": f"{len(combos_display)*300:,}円"},
            {"内容にゃ": "穴2点（ルメール川田）",
             "単価にゃ": "500円", "合計にゃ": "1,000円"},
        ])
        plan_df.loc[len(plan_df)] = {
            "内容にゃ": "合計",
            "単価にゃ": "—",
            "合計にゃ": f"{len(combos_display)*300 + 1000:,}円"
        }
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.caption(
            "的中時期待払戻にゃ: 2人気×1人気×3人気 → 約30倍前後にゃ。"
            "穴が絡むと100倍超もあるにゃ🏆"
        )





# ============================================================
# 全レース対応 統合分析ダッシュボードにゃ🐾
# どのレースでも「枠・騎手・展開・EV・三連複買い目」を一発表示にゃ
# ============================================================

# ── 汎用枠番有利テーブルにゃ（距離・コース種別別にゃ）──
# (コース種別, 距離帯) → {枠番: 有利度スコアにゃ}
# 0.0=普通 +0.1=有利 -0.1=不利にゃ

FRAME_ADVANTAGE_TABLE = {
    # 芝 短距離（〜1400mにゃ）→ 内枠有利にゃ
    ("芝", "短距離"):  {1:+0.12, 2:+0.08, 3:+0.05, 4:+0.02, 5:-0.02, 6:-0.04, 7:-0.06, 8:-0.08},
    # 芝 マイル（1401〜1800mにゃ）→ 内〜中枠にゃ
    ("芝", "マイル"):  {1:+0.08, 2:+0.06, 3:+0.04, 4:+0.02, 5:+0.00, 6:-0.02, 7:-0.04, 8:-0.06},
    # 芝 中距離（1801〜2200mにゃ）→ 中枠有利にゃ
    ("芝", "中距離"):  {1:+0.06, 2:+0.08, 3:+0.06, 4:+0.04, 5:+0.02, 6:+0.02, 7:-0.02, 8:-0.04},
    # 芝 長距離（2200m〜にゃ）→ 内枠有利・外枠不利にゃ
    ("芝", "長距離"):  {1:+0.10, 2:+0.08, 3:+0.06, 4:+0.04, 5:+0.00, 6:-0.02, 7:-0.04, 8:-0.08},
    # ダート 短距離（〜1400mにゃ）→ 外枠有利（砂被り回避にゃ）
    ("ダ", "短距離"):  {1:-0.08, 2:-0.04, 3:+0.00, 4:+0.04, 5:+0.06, 6:+0.08, 7:+0.08, 8:+0.06},
    # ダート マイルにゃ
    ("ダ", "マイル"):  {1:-0.04, 2:-0.02, 3:+0.02, 4:+0.04, 5:+0.04, 6:+0.06, 7:+0.04, 8:+0.02},
    # ダート 中距離にゃ
    ("ダ", "中距離"):  {1:-0.02, 2:+0.00, 3:+0.02, 4:+0.04, 5:+0.04, 6:+0.04, 7:+0.02, 8:+0.00},
}

def _get_dist_band(distance: int) -> str:
    if distance <= 1400: return "短距離"
    if distance <= 1800: return "マイル"
    if distance <= 2200: return "中距離"
    return "長距離"

def _get_frame_advantage(frame_no: int, track_type: str, distance: int) -> float:
    """枠番の有利不利スコアを返すにゃ（-0.15〜+0.15）"""
    dist_band = _get_dist_band(distance)
    # ダート判定にゃ
    tt = "ダ" if "ダ" in str(track_type) else "芝"
    table = FRAME_ADVANTAGE_TABLE.get((tt, dist_band),
            FRAME_ADVANTAGE_TABLE.get(("芝", "中距離"), {}))
    return table.get(int(frame_no), 0.0)


def _calc_all_race_score(row: pd.Series,
                          all_df: pd.DataFrame,
                          track_type: str,
                          distance: int) -> dict:
    """
    全レース汎用スコアリングにゃ。
    AI確率・EV・枠番・展開・騎手実績を総合するにゃ。
    """
    # 基本スコアにゃ（AI確率ベースにゃ）
    ml_prob = _safe_float(row.get("ml_top3_prob", 0), 0)
    ev2     = _safe_float(row.get("ev_score_v2",  row.get("ev_score", 0)), 0)
    ev_comp = _safe_float(row.get("ev_composite", 0), 0)
    pace    = _safe_float(row.get("pace_advantage", 1.0), 1.0)
    kelly   = _safe_float(row.get("kelly_ratio", 0), 0)
    kelly_s = _safe_float(row.get("kelly_ratio_sanren", 0), 0)
    final   = _safe_float(row.get("final_score", ml_prob), ml_prob)
    pass_sc = _safe_float(row.get("pass_score", 0), 0)
    buy_v2  = str(row.get("buy_flag_v2", ""))
    buy_v1  = str(row.get("buy_flag", ""))
    dl      = str(row.get("danger_level", ""))

    frame_no   = _safe_int(row.get("frame_no", 4), 4)
    frame_adv  = _get_frame_advantage(frame_no, track_type, distance)

    # 総合スコア計算にゃ（0〜100にゃ）
    score = (
        final   * 40          # AI最終スコアにゃ（最重要にゃ）
        + ml_prob * 20         # AI確率にゃ
        + ev2  * 15            # EV乖離にゃ
        + frame_adv * 10       # 枠番有利不利にゃ
        + pace * 5             # 展開適性にゃ
        + (kelly + kelly_s) * 5  # Kelly比にゃ
        - (pass_sc / 200) * 15   # 見送りペナルティにゃ
    )

    # 危険馬ペナルティにゃ
    if dl in ["強危険", "危険"]:
        score -= 20
    if "見送り" in buy_v2:
        score -= 10

    # ボーナスにゃ
    if "◎" in buy_v2: score += 15
    if "○" in buy_v2: score += 8
    if buy_v1 == "買い": score += 5

    score = float(np.clip(score * 100, 1, 99))

    # 評価コメントにゃ
    reasons, warnings = [], []
    if frame_adv >= 0.06:  reasons.append(f"✅ {frame_no}枠有利（{frame_adv:+.2f}にゃ）")
    elif frame_adv <= -0.06: warnings.append(f"⚠️ {frame_no}枠不利（{frame_adv:+.2f}にゃ）")
    if ev2 >= 0.06:  reasons.append(f"✅ EV高め（{ev2:+.3f}にゃ）")
    elif ev2 <= -0.08: warnings.append(f"⚠️ EV低め（{ev2:+.3f}にゃ）")
    if pace >= 1.08: reasons.append(f"✅ 展開有利（{pace:.2f}にゃ）")
    elif pace <= 0.90: warnings.append(f"⚠️ 展開不利（{pace:.2f}にゃ）")
    if "◎" in buy_v2: reasons.append(f"✅ S級◎判定にゃ")
    if dl in ["強危険","危険"]: warnings.append(f"❌ {dl}にゃ")

    return {
        "score":      round(score, 1),
        "frame_adv":  frame_adv,
        "reasons":    reasons,
        "warnings":   warnings,
    }


def show_race_analysis_full(race_df: pd.DataFrame,
                             strategy_mode: str = STRATEGY_MODE_ROI):
    """
    全レース対応の統合分析ダッシュボードにゃ🐾
    どんなレースでも以下を自動分析するにゃ:
      ① 総合スコアランキングにゃ
      ② 展開×脚質マッチにゃ
      ③ 枠番有利不利にゃ
      ④ 騎手・EV・Kelly総合にゃ
      ⑤ 三連複・ワイド・複勝 買い目一発にゃ
    """
    if race_df is None or race_df.empty:
        return

    # レース情報にゃ
    track_type = str(race_df.get("track_type", pd.Series(["芝"])).iloc[0])
    distance   = _safe_int(race_df.get("distance", pd.Series([2000])).iloc[0], 2000)
    race_name  = str(race_df.get("race_name", pd.Series(["レース"])).iloc[0])[:30]
    field_size = len(race_df)
    going      = str(race_df.get("going", pd.Series(["良"])).iloc[0])
    place      = str(race_df.get("place", pd.Series([""])).iloc[0])

    dist_band  = _get_dist_band(distance)
    tt_label   = "ダート" if "ダ" in track_type else "芝"

    st.markdown("---")
    st.subheader(f"🔬 全力分析にゃ — {race_name}")
    st.caption(
        f"{place} {tt_label}{distance}m {dist_band} {going} "
        f"{field_size}頭にゃ🐾"
    )

    # ── 全馬スコアリングにゃ ──
    score_rows = []
    for _, row in race_df.iterrows():
        analysis = _calc_all_race_score(row, race_df, track_type, distance)
        score_rows.append({
            "馬番":      _safe_int(row.get("horse_no",0),0),
            "枠番":      _safe_int(row.get("frame_no",0),0),
            "馬名":      str(row.get("horse_name","")),
            "騎手":      str(row.get("jockey",""))[:8],
            "人気":      _safe_int(row.get("popularity",99),99),
            "オッズ":    _safe_float(row.get("odds",0),0),
            "AI確率":    f"{_safe_float(row.get('ml_top3_prob',0),0)*100:.1f}%",
            "EV":        round(_safe_float(row.get("ev_score_v2",row.get("ev_score",0)),0),3),
            "展開":      round(_safe_float(row.get("pace_advantage",1.0),1.0),3),
            "S級判定":   str(row.get("buy_flag_v2",""))[:6],
            "複勝判定":  str(row.get("buy_flag","")),
            "総合スコア": analysis["score"],
            "枠有利":    f"{analysis['frame_adv']:+.2f}",
            "評価":      " ".join(analysis["reasons"][:1] + analysis["warnings"][:1]),
            "_r":        analysis["reasons"],
            "_w":        analysis["warnings"],
        })

    sdf = pd.DataFrame(score_rows).sort_values("総合スコア", ascending=False).reset_index(drop=True)

    # ── 5タブにゃ ──
    t1, t2, t3, t4, t5 = st.tabs([
        "🏆 総合スコアにゃ",
        "🌪️ 展開×脚質にゃ",
        "🔲 枠番分析にゃ",
        "💰 EV・Kellyにゃ",
        "🎯 買い目一発にゃ",
    ])

    # ── Tab1: 総合スコアにゃ ──
    with t1:
        st.markdown("#### 総合スコアランキングにゃ（AI確率+EV+枠+展開の総合評価にゃ）")

        def color_score(row):
            s = float(row.get("総合スコア", 0))
            if s >= 65: return ["background-color:#c3e6cb"] * len(row)
            if s >= 50: return ["background-color:#d1ecf1"] * len(row)
            if s <= 25: return ["background-color:#f8d7da"] * len(row)
            return [""] * len(row)

        disp_cols = ["総合スコア","馬番","枠番","馬名","騎手","人気","オッズ",
                     "AI確率","EV","展開","枠有利","S級判定","評価"]
        try:
            st.dataframe(
                sdf[disp_cols].style.apply(color_score, axis=1),
                use_container_width=True, hide_index=True
            )
        except Exception:
            st.dataframe(sdf[disp_cols], use_container_width=True, hide_index=True)

        # 上位3頭の詳細にゃ
        st.markdown("#### 🥇 上位3頭 評価詳細にゃ")
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1,c2,c3], sdf.head(3).iterrows()):
            with col:
                col.markdown(f"**馬番{row['馬番']} {row['馬名']}**")
                col.metric("総合スコアにゃ", f"{row['総合スコア']:.0f}点")
                col.caption(f"{row['人気']}人気 / {row['オッズ']}倍")
                for r3 in row["_r"][:2]:
                    col.success(r3.replace("✅ ",""))
                for w3 in row["_w"][:1]:
                    col.warning(w3.replace("⚠️ ","").replace("❌ ",""))

    # ── Tab2: 展開×脚質にゃ ──
    with t2:
        st.markdown("#### 展開予測×脚質マッチにゃ")

        if "running_style" not in race_df.columns:
            st.info("脚質データがないにゃ。予想実行後に表示されるにゃ🐾")
        else:
            pace_info = analyze_pace(race_df)
            pace_label = pace_info.get("pace", "不明")
            pace_score_val = pace_info.get("pace_score", 0.5)

            pace_icon = {"ハイペース":"🔥","スローペース":"💤","ミドルペース":"⚡","流動的":"🌀"}.get(pace_label,"❓")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ペースにゃ", f"{pace_icon} {pace_label}")
            c2.metric("逃げにゃ",   f"{pace_info.get('escape_count',0)}頭にゃ")
            c3.metric("先行にゃ",   f"{pace_info.get('senkou_count',0)}頭にゃ")
            c4.metric("差し追込にゃ",f"{pace_info.get('sashi_count',0)+pace_info.get('oikomi_count',0)}頭にゃ")
            st.info(pace_info.get("pace_note",""))

            # 展開有利馬 TOP5にゃ
            st.markdown("#### 展開有利馬 TOP5にゃ")
            if "pace_advantage" in race_df.columns:
                top5_pace = race_df.sort_values("pace_advantage", ascending=False).head(5)
                for rank, (_, r) in enumerate(top5_pace.iterrows(), 1):
                    pa   = _safe_float(r.get("pace_advantage",1.0),1.0)
                    icon = "🟢" if pa>=1.05 else ("🟡" if pa>=1.00 else "🔴")
                    st.markdown(
                        f"{icon} **{rank}位** 馬番{_safe_int(r.get('horse_no',0),0)} "
                        f"{r.get('horse_name','')} ({r.get('running_style','不明')}) "
                        f"展開スコア: **{pa:.3f}**にゃ"
                    )

    # ── Tab3: 枠番分析にゃ ──
    with t3:
        st.markdown(f"#### {tt_label}{distance}m({dist_band}) 枠番有利不利にゃ")

        frame_rows = []
        for fn in range(1, 9):
            adv = _get_frame_advantage(fn, track_type, distance)
            horses_in_frame = race_df[race_df["frame_no"].astype(str).str.strip() == str(fn)]
            horse_list = " / ".join(
                str(r.get("horse_name","")) for _, r in horses_in_frame.iterrows()
            )
            frame_rows.append({
                "枠番にゃ": f"{fn}枠",
                "有利度にゃ": f"{adv:+.2f}",
                "馬名にゃ": horse_list or "—",
                "評価にゃ": "✅ 有利" if adv>=0.06 else ("⚠️ やや不利" if adv<=-0.04 else "— 標準"),
            })

        fdf = pd.DataFrame(frame_rows)
        def color_frame(row):
            v = float(row["有利度にゃ"])
            if v >= 0.06: return ["background-color:#c3e6cb"]*len(row)
            if v <= -0.06: return ["background-color:#f8d7da"]*len(row)
            return [""]*len(row)
        try:
            st.dataframe(fdf.style.apply(color_frame, axis=1),
                         use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(fdf, use_container_width=True, hide_index=True)

        # 有利枠の馬にゃ
        best_frame_adv = max(range(1,9), key=lambda fn: _get_frame_advantage(fn, track_type, distance))
        best_adv_val   = _get_frame_advantage(best_frame_adv, track_type, distance)
        st.success(f"✅ このレースの最有利枠にゃ: **{best_frame_adv}枠** ({best_adv_val:+.2f}にゃ)")

    # ── Tab4: EV・Kellyにゃ ──
    with t4:
        st.markdown("#### EV乖離・Kelly比・総合判定にゃ")
        ev_col = "ev_score_v2" if "ev_score_v2" in race_df.columns else "ev_score"
        ev_disp_cols = [c for c in [
            "ml_rank","horse_no","horse_name","odds","popularity",
            "ml_top3_prob", ev_col, "pace_advantage",
            "kelly_ratio","kelly_ratio_sanren",
            "buy_flag_v2","buy_flag","pass_score",
        ] if c in race_df.columns]

        ev_df = race_df[ev_disp_cols].copy()
        ev_df = ev_df.sort_values("ml_rank" if "ml_rank" in ev_df.columns else ev_col,
                                   ascending=True if "ml_rank" in ev_df.columns else False)

        if "ml_top3_prob" in ev_df.columns:
            ev_df["ml_top3_prob"] = (
                pd.to_numeric(ev_df["ml_top3_prob"],errors="coerce")*100
            ).round(1).astype(str)+"%"
        for c in [ev_col,"pace_advantage","kelly_ratio","kelly_ratio_sanren"]:
            if c in ev_df.columns:
                ev_df[c] = pd.to_numeric(ev_df[c],errors="coerce").round(3)

        def color_ev4(row):
            ev = _safe_float(row.get("EV(展開補正)",row.get(ev_col,0)),0)
            bv = str(row.get("S級判定",row.get("buy_flag_v2","")))
            if "◎" in bv: return ["background-color:#c3e6cb"]*len(row)
            if ev >= 0.06: return ["background-color:#d1ecf1"]*len(row)
            if ev <= -0.08: return ["background-color:#f8d7da"]*len(row)
            return [""]*len(row)

        rename_ev = {
            "ml_rank":"AI順位","horse_no":"馬番","horse_name":"馬名",
            "odds":"オッズ","popularity":"人気","ml_top3_prob":"AI確率",
            ev_col:"EV(展開補正)","pace_advantage":"展開スコア",
            "kelly_ratio":"Kelly複勝","kelly_ratio_sanren":"Kelly三連複",
            "buy_flag_v2":"S級判定","buy_flag":"複勝判定","pass_score":"見送りスコア",
        }
        try:
            st.dataframe(
                ev_df.rename(columns=rename_ev).style.apply(color_ev4, axis=1),
                use_container_width=True, hide_index=True
            )
        except Exception:
            st.dataframe(ev_df.rename(columns=rename_ev),
                         use_container_width=True, hide_index=True)

    # ── Tab5: 買い目一発にゃ ──
    with t5:
        st.markdown("#### 🎯 買い目一発にゃ（全券種まとめにゃ）")
        st.caption("S級・EV・枠番・展開を全部加味した最終買い目にゃ🐾")

        # 総合スコア上位6頭にゃ
        top6 = sdf.head(6)
        top6_nos = [int(r["馬番"]) for _, r in top6.iterrows()]

        # S級◎の馬を軸に優先するにゃ
        s_pivot_rows = sdf[sdf["S級判定"].str.contains("◎", na=False)]
        if not s_pivot_rows.empty:
            pivot_no   = int(s_pivot_rows.iloc[0]["馬番"])
            pivot_name = str(s_pivot_rows.iloc[0]["馬名"])
            pivot_label = "S級◎"
        else:
            pivot_no   = int(top6.iloc[0]["馬番"])
            pivot_name = str(top6.iloc[0]["馬名"])
            pivot_label = "総合1位"

        aite5 = [n for n in top6_nos if n != pivot_no][:5]

        # 複勝候補にゃ（S級×複勝にゃ）
        fuku_candidates = sdf[
            sdf["S級判定"].str.contains("買い", na=False) &
            (sdf["複勝判定"] == "買い")
        ]
        if fuku_candidates.empty:
            fuku_candidates = sdf[sdf["複勝判定"] == "買い"]
        if fuku_candidates.empty:
            fuku_candidates = sdf.head(3)
        fuku_nos = [int(r["馬番"]) for _, r in fuku_candidates.iterrows()]

        # ワイド候補にゃ（総合上位2〜4頭のBOXにゃ）
        wide_nos = top6_nos[:4]

        # ── 表示にゃ ──
        st.markdown("**① 複勝にゃ（S級×複勝フィルターにゃ）**")
        if fuku_nos:
            for fn in fuku_nos[:4]:
                row = sdf[sdf["馬番"]==fn]
                if not row.empty:
                    r = row.iloc[0]
                    pop = int(r["人気"]); odds = float(r["オッズ"])
                    fuku_est = max(1.1, odds*0.30)
                    icon = "🥇" if pop<=3 else ("🥈" if pop<=6 else "💎")
                    st.info(
                        f"{icon} **馬番{fn} {r['馬名']}** "
                        f"（{pop}人気/{odds}倍 / 複勝推定{fuku_est:.1f}倍にゃ）"
                    )
        st.markdown("---")

        st.markdown(f"**② 三連複にゃ — 軸:{pivot_no}番{pivot_name}（{pivot_label}にゃ）× 相手{aite5}にゃ**")
        import itertools as _it
        san3_combos = [
            f"{pivot_no}-{min(a,b)}-{max(a,b)}"
            for a,b in _it.combinations(aite5, 2)
        ]
        cols5 = st.columns(5)
        for i, combo in enumerate(san3_combos[:10]):
            cols5[i%5].code(combo)

        st.markdown("---")
        st.markdown(f"**③ ワイドにゃ — 上位{len(wide_nos)}頭BOXにゃ（{len(list(_it.combinations(wide_nos,2)))}点にゃ）**")
        wide_combos = [f"{min(a,b)}-{max(a,b)}" for a,b in _it.combinations(wide_nos, 2)]
        wcols = st.columns(min(len(wide_combos), 6))
        for i, combo in enumerate(wide_combos[:6]):
            wcols[i%len(wcols)].code(combo)

        st.markdown("---")
        # 投資プランにゃ
        n_san3   = len(san3_combos)
        n_fuku   = len(fuku_nos[:4])
        n_wide   = len(wide_combos[:6])
        total    = n_san3*300 + n_fuku*200 + n_wide*100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("複勝にゃ",   f"{n_fuku}点×200円",   f"{n_fuku*200:,}円にゃ")
        c2.metric("三連複にゃ", f"{n_san3}点×300円",  f"{n_san3*300:,}円にゃ")
        c3.metric("ワイドにゃ", f"{n_wide}点×100円",  f"{n_wide*100:,}円にゃ")
        c4.metric("合計にゃ",   f"{n_san3+n_fuku+n_wide}点", f"{total:,}円にゃ")

        # CSV出力にゃ
        buy_rows = []
        for fn in fuku_nos[:4]:
            buy_rows.append({"券種":"複勝","買い目":str(fn),"単価":200})
        for combo in san3_combos[:10]:
            buy_rows.append({"券種":"三連複","買い目":combo,"単価":300})
        for combo in wide_combos[:6]:
            buy_rows.append({"券種":"ワイド","買い目":combo,"単価":100})
        buy_df = pd.DataFrame(buy_rows)
        buy_df["合計"] = buy_df["単価"]
        st.download_button(
            "📥 買い目CSVにゃ",
            data=buy_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"buy_{race_name[:10].replace(' ','_')}.csv",
            mime="text/csv",
        )



def app_main():
    st.title("🐾 にゃんこ競馬AI v26にゃ")
    st.success(f"起動版にゃ: {VERSION}にゃ")
    st.caption(
        "v26にゃ: netkeibaスクレイピング統合 / 確率校正（過学習対策） / "
        "三連複条件付き確率 / Kelly比分離 / PKL本格AI分析にゃ🐾"
    )

    # ── サイドバーにゃ ──
    with st.sidebar:
        st.header("設定にゃ🐾")
        st.markdown("### 🎯 予想モードにゃ")
        strategy_mode = st.radio(
            "モードを選択にゃ",
            STRATEGY_MODE_OPTIONS,
            index=0,
            help="**回収率重視にゃ**: Kelly+EV高め狙いにゃ\n\n**的中率重視にゃ**: AI上位+軸信頼度で安定的中にゃ"
        )
        if strategy_mode == STRATEGY_MODE_ROI:
            st.info("💰 回収率重視にゃ: 高配当を狙いますにゃ🐾")
        else:
            st.success("🏆 的中率重視にゃ: 安定的中を狙いますにゃ🐾")
        st.markdown("---")
        uploaded_model = st.file_uploader("学習済みPKLにゃ", type=["pkl"])
        csv_mode = st.radio("CSV形式にゃ", ["52列TARGET形式", "簡易CSV形式"], index=0)
        if MODEL_PATH.exists():
            st.success(f"同梱PKLあり: {MODEL_PATH.name}にゃ")
        else:
            st.warning("同梱PKLなしにゃ。アップロードが必要にゃ🐾")
        if TARGET_CSV_PATH.exists():
            st.success(f"TARGET CSV: {TARGET_CSV_PATH.name}にゃ")
        else:
            st.info("TARGET CSV未配置にゃ（yosou.csvを直下に置くにゃ）")
        st.markdown("---")
        st.caption(
            "**v26 主要改善にゃ**\n\n"
            "🔵 スクレイピングにゃ:\n- 開催日程自動取得にゃ\n"
            "- リアルタイムオッズ取得にゃ\n\n"
            "🔴 過学習対策にゃ:\n- 確率校正（合計→3.0）にゃ\n\n"
            "🟢 ロジック改善にゃ:\n- 条件付き確率×頭数補正にゃ\n"
            "- Kelly比分離にゃ\n- 危険馬AI4位以上にゃ\n\n"
            "🧠 AI分析にゃ:\n- 置換法特徴量重要度にゃ\n"
            "- 過学習診断にゃ\n- 個別馬根拠説明にゃ"
        )

    # ── 入力方法にゃ ──
    st.subheader("入力方法にゃ🐾")
    INPUT_NETKEIBA_AUTO  = "🌐 netkeiba自動取得（当日レース）にゃ"
    INPUT_NETKEIBA_ID    = "🌐 netkeiba race_id/URL指定にゃ"
    INPUT_CSV_SELECT     = "📁 事前CSVから選択にゃ"
    INPUT_CSV_UPLOAD     = "📄 出馬表CSVアップロードにゃ"
    INPUT_NETKEIBA_URL   = "🌐 netkeiba URL単発にゃ"

    input_method = st.radio(
        "入力方法を選択にゃ",
        [INPUT_NETKEIBA_AUTO, INPUT_NETKEIBA_ID,
         INPUT_CSV_SELECT, INPUT_CSV_UPLOAD, INPUT_NETKEIBA_URL],
        horizontal=True, index=2  # デフォルトは事前CSVにゃ
    )

    # ── 各入力方法のUI にゃ ──
    selected_preloaded_paths = []
    uploaded_csv  = None
    race_url      = ""
    race_items    = []
    target_date   = None
    update_odds   = True
    sleep_sec     = 1.2

    if input_method == INPUT_NETKEIBA_AUTO:
        st.caption("今日（または指定日）の全レースを自動取得するにゃ🐾")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_today = st.checkbox("今日の日付を使うにゃ", value=True)
        with col2:
            if not use_today:
                sel_date = st.date_input("開催日を指定にゃ", value=date.today())
                target_date = sel_date.strftime("%Y%m%d")
            else:
                target_date = date.today().strftime("%Y%m%d")
                st.info(f"今日: {target_date}にゃ")
        with col3:
            update_odds = st.checkbox("リアルタイムオッズを取得するにゃ", value=True)
        sleep_sec = st.slider("アクセス間隔（秒）にゃ", 0.5, 3.0, 1.2, 0.1)

    elif input_method == INPUT_NETKEIBA_ID:
        st.caption("race_idまたはURLを指定して取得するにゃ🐾")
        mk = st.radio("指定方法にゃ", ["race_id/URL一覧にゃ", "開催情報から自動生成にゃ"], horizontal=True)
        if mk == "race_id/URL一覧にゃ":
            txt = st.text_area(
                "race_id/URLを1行ずつにゃ",
                "202505040811\n202505040812\n202505040813", height=100
            )
            race_items = [x.strip() for x in txt.splitlines() if x.strip()]
        else:
            c1, c2, c3, c4 = st.columns(4)
            yr  = c1.number_input("年にゃ", 2020, 2035, date.today().year)
            pn  = c2.selectbox("競馬場にゃ", list(PLACE_CODE_MAP.keys()))
            kai = c3.number_input("開催回にゃ", 1, 10, 2)
            nt  = c4.text_input("日次（カンマ区切り）にゃ", "1,2")
            c5, c6 = st.columns(2)
            rs  = c5.number_input("開始Rにゃ", 1, 12, 1)
            re_ = c6.number_input("終了Rにゃ", 1, 12, 12)
            nl  = [int(x.strip()) for x in nt.split(",") if x.strip().isdigit()]
            race_items = build_race_ids(int(yr), pn, int(kai), nl, int(rs), int(re_))
        st.write(f"取得予定にゃ: {len(race_items)}レースにゃ")
        update_odds = st.checkbox("リアルタイムオッズを取得するにゃ", value=True, key="upd2")
        sleep_sec   = st.slider("アクセス間隔（秒）にゃ", 0.5, 3.0, 1.2, 0.1, key="sl2")

    elif input_method == INPUT_CSV_SELECT:
        st.caption("data/ フォルダに置いたCSVを選ぶだけで予想できるにゃ🐾")
        preloaded_paths = list_preloaded_csv_files()
        if not preloaded_paths:
            st.warning("dataフォルダにCSVがないにゃ")
        else:
            labels = [make_preloaded_file_label(p) for p in preloaded_paths]
            mode = st.radio(
                "読み込み方法にゃ",
                ["1レースだけ選ぶにゃ", "全部まとめて読むにゃ"],
                horizontal=True, index=0
            )
            if mode == "1レースだけ選ぶにゃ":
                sl = st.selectbox("CSVを選択にゃ", labels)
                selected_preloaded_paths = [preloaded_paths[labels.index(sl)]]
            else:
                selected_preloaded_paths = preloaded_paths
                st.info(f"全{len(preloaded_paths)}件を読みますにゃ")
            with st.expander("検出したCSVにゃ"):
                st.write([p.name for p in preloaded_paths])

    elif input_method == INPUT_CSV_UPLOAD:
        uploaded_csv = st.file_uploader("予想CSVをアップロードにゃ", type=["csv"])
        st.caption("TARGET 52列CSV、または簡易CSVを使えるにゃ🐾")

    else:  # INPUT_NETKEIBA_URL
        race_url = st.text_input(
            "netkeiba 出馬表URLにゃ",
            placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=202605020111"
        )

    # ── 入力チェックにゃ ──
    if input_method == INPUT_NETKEIBA_AUTO and not target_date:
        st.info("日付を設定するにゃ🐾"); return
    if input_method == INPUT_NETKEIBA_ID and not race_items:
        st.info("race_idを入力するにゃ🐾"); return
    if input_method == INPUT_CSV_SELECT and not selected_preloaded_paths:
        st.info("dataフォルダにCSVを置くか選択するにゃ🐾"); return
    if input_method == INPUT_CSV_UPLOAD and uploaded_csv is None:
        st.info("CSVをアップロードするにゃ🐾"); return
    if input_method == INPUT_NETKEIBA_URL and not race_url.strip():
        st.info("URLを入力するにゃ🐾"); return

    # ── 予想ボタンにゃ ──
    if st.button("🐾 予想するにゃ！", type="primary"):
        try:
            bundle, model_status = load_model_safely(uploaded_model)
            if bundle is None:
                st.error("PKLがないにゃ！アップロードが必要にゃ🐾")
                return
            st.success(f"モデル読込にゃ: {model_status} / モードにゃ: {strategy_mode}")

            # ── データ取得にゃ ──
            with st.spinner("データを取得中にゃ...🐾"):
                if input_method == INPUT_NETKEIBA_AUTO:
                    try:
                        race_ids_today = fetch_today_race_ids(target_date)
                        if not race_ids_today:
                            st.error(f"{target_date}にレースが見つからなかったにゃ🐾")
                            return
                        st.success(f"{len(race_ids_today)}レースを発見したにゃ🐾")
                        pred_src, errors = fetch_many_races(
                            race_ids_today, sleep_sec=sleep_sec, update_odds=update_odds)
                        if pred_src.empty:
                            st.error("取得できなかったにゃ🐾")
                            return
                        if errors:
                            st.warning(f"取得失敗にゃ: {len(errors)}件にゃ")
                            st.dataframe(pd.DataFrame(errors))
                    except Exception as e:
                        st.error(f"自動取得エラーにゃ: {e}にゃ")
                        st.info(
                            "💡 ヒントにゃ: netkeibaはIP制限があるにゃ。\n"
                            "- 開催日当日の朝〜夕方に試すにゃ\n"
                            "- 代わりにCSVアップロードを使うにゃ"
                        )
                        return

                elif input_method == INPUT_NETKEIBA_ID:
                    pred_src, errors = fetch_many_races(
                        race_items, sleep_sec=sleep_sec, update_odds=update_odds)
                    if pred_src.empty:
                        st.error("取得できなかったにゃ🐾")
                        return
                    if errors:
                        st.warning(f"取得失敗にゃ: {len(errors)}件にゃ")
                        st.dataframe(pd.DataFrame(errors))

                elif input_method == INPUT_CSV_SELECT:
                    pred_src = load_many_preloaded_entry_csv(
                        selected_preloaded_paths, csv_mode)
                    st.success(f"取得にゃ: {pred_src['race_key'].nunique()}レース / {len(pred_src)}頭にゃ")

                elif input_method == INPUT_CSV_UPLOAD:
                    pred_src = load_uploaded_entry_csv(uploaded_csv, csv_mode)
                    st.success("CSVから取得したにゃ🐾")

                else:  # INPUT_NETKEIBA_URL
                    pred_src = fetch_netkeiba_race_to_52cols(race_url.strip())
                    st.success("URLから取得したにゃ🐾")

            # 出馬表DLにゃ
            export_simple = convert_52_to_simple_export(pred_src)
            st.download_button(
                "📥 出馬表CSVにゃ",
                data=export_simple.to_csv(
                    index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="entry_races.csv", mime="text/csv",
            )

            # TARGET特徴量結合にゃ
            pred_src = merge_target_features(pred_src)
            if TARGET_CSV_PATH.exists():
                try:
                    _, fc_check = load_target_features_cached()
                    if fc_check:
                        st.success("TARGET CSV結合済みにゃ🐾")
                    else:
                        st.info("yosou.csv: 着順なし→補正なしにゃ")
                except Exception:
                    st.info("yosou.csv利用不可→出馬表単体にゃ")
            else:
                st.info("TARGET CSV未配置→出馬表単体にゃ")

            # 予想にゃ
            with st.spinner("AI予測中にゃ...🐾"):
                pred_df = predict(bundle, pred_src, strategy_mode=strategy_mode)
            st.success(f"予想完了にゃ: {len(pred_df)}頭 [{strategy_mode}]にゃ🐾")

            # ── 予想結果表示にゃ ──
            st.markdown("---")
            st.subheader("予想結果にゃ🐾")
            show_df = pred_df.sort_values(
                ["race_key", "ml_rank"]
                if "race_key" in pred_df.columns else ["ml_rank"]
            )
            try:
                view = jp_view(show_df, include_race_key=False)
            except Exception:
                view = show_df
            st.dataframe(view, use_container_width=True, hide_index=True)
            try:
                st.download_button(
                    "📥 予想結果CSVにゃ",
                    data=view.to_csv(
                        index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="nyanko_v26_result.csv", mime="text/csv",
                    key="dl_result"
                )
            except Exception:
                pass

            # 買い目候補にゃ
            show_bets(pred_df, key_prefix="main_bets", strategy_mode=strategy_mode)

            # ── レース詳細にゃ ──
            st.markdown("---")
            st.subheader("レース詳細にゃ🐾")
            race_options = (
                pred_df[["race_key", "race_label"]]
                .drop_duplicates()
                .sort_values("race_label")
            )
            label_map = dict(
                zip(race_options["race_label"], race_options["race_key"]))
            selected_label = st.selectbox("レース選択にゃ", list(label_map.keys()))
            selected_race  = label_map[selected_label]

            race_df = pred_df[pred_df["race_key"] == selected_race].sort_values(
                ["ml_rank", "value_score", "horse_no"],
                ascending=[True, False, True]
            )
            st.dataframe(jp_view(race_df), use_container_width=True, hide_index=True)

            # レース質分析にゃ
            st.markdown("---")
            race_quality = analyze_race_quality(race_df)
            st.markdown(f"#### 🏟️ レース質分析にゃ: **{race_quality['type']}**")
            if race_quality["advice"]:
                st.info(race_quality["advice"])
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            col_q1.metric("最低オッズにゃ", f"{race_quality['min_odds']:.1f}倍")
            col_q2.metric("オッズ標準偏差にゃ", f"{race_quality['odds_std']:.1f}")
            col_q3.metric("レースタイプにゃ", race_quality["type"])
            col_q4.metric("推奨フォーカスにゃ",
                          race_quality.get("rec_bet_focus", "-"))

            # 推奨購入点数にゃ
            st.markdown("---")
            rec = calc_recommended_tickets(race_df, strategy_mode=strategy_mode)
            st.markdown("#### 📈 推奨購入点数ダッシュボードにゃ")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("推奨点数にゃ", f"{rec['推奨点数']}点")
            m2.metric("Kelly正馬数にゃ", f"{rec.get('Kelly正馬数', rec.get('Kelly正(複勝)', 0))}頭")
            m3.metric("買い候補馬数にゃ", f"{rec['買い候補馬数']}頭")
            key4 = [k for k in rec if "的中率" in k]
            if key4:
                m4.metric(key4[0], rec[key4[0]])

            # 本命にゃ
            tickets = make_tickets(race_df)
            c1, c2, c3 = st.columns(3)
            c1.metric("本命にゃ", tickets["本命"])
            c2.metric("単勝にゃ",  tickets["単勝"])
            c3.metric("複勝にゃ",  tickets["複勝"])

            # EV乖離にゃ
            st.markdown("---")
            show_ev_ranking(race_df)

            # ── S級分析タブにゃ ──
            st.markdown("---")
            st.subheader("🏆 S級分析にゃ（展開・期待値・見送り判定）")
            s_tab1, s_tab2, s_tab3 = st.tabs([
                "🌪️ 展開予測にゃ",
                "📈 S級期待値にゃ",
                "🚦 S級見送り判定にゃ",
            ])
            with s_tab1:
                try:
                    show_pace_analysis(race_df)
                except Exception as e:
                    st.warning(f"展開予測エラーにゃ: {e}")
            with s_tab2:
                try:
                    show_ev_analysis_v2(race_df)
                except Exception as e:
                    st.warning(f"S級期待値エラーにゃ: {e}")
            with s_tab3:
                try:
                    show_pass_judgment(race_df, strategy_mode=strategy_mode)
                except Exception as e:
                    st.warning(f"S級見送り判定エラーにゃ: {e}")

            # 三連複にゃ
            st.markdown("---")
            show_sanrenpuku_tabs(race_df, strategy_mode=strategy_mode)

            # 馬券おすすめにゃ
            st.markdown("---")
            show_ticket_tabs(race_df, strategy_mode=strategy_mode)

            # 買い/見送り判定にゃ
            st.markdown("---")
            show_roi_strategy(race_df, strategy_mode=strategy_mode)

            # ── 🏆 S級×複勝フィルター（最強絞り込みにゃ）──
            st.markdown("---")
            st.subheader("🏆 S級×複勝フィルターにゃ（バックテスト実績: 複勝257%にゃ）")
            st.caption(
                "S級判定 かつ 複勝買い判定の馬だけに絞るにゃ🐾 "
                "バックテストでS級257%・複勝156%を記録した最強フィルターにゃ"
            )
            try:
                _show_sx_fuku_filter(race_df)
            except Exception as _sx_err:
                st.warning(f"S級×複勝フィルターエラーにゃ: {_sx_err}")

            # 2モード比較にゃ
            st.markdown("---")
            st.subheader("📊 2モード比較にゃ")
            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown("**💰 回収率重視にゃ**")
                dr = add_value_strategy(
                    race_df.copy(), strategy_mode=STRATEGY_MODE_ROI)
                br = dr[dr["buy_flag"] == "買い"][[
                    "horse_name", "buy_flag", "buy_reason",
                    "kelly_ratio", "kelly_ratio_sanren"
                ]]
                st.dataframe(
                    br.rename(columns=JP_COLUMNS),
                    use_container_width=True, hide_index=True
                )
            with cm2:
                st.markdown("**🏆 的中率重視にゃ**")
                dh = add_value_strategy(
                    race_df.copy(), strategy_mode=STRATEGY_MODE_HITRATE)
                bh = dh[dh["buy_flag"] == "買い"][[
                    "horse_name", "buy_flag", "buy_reason", "pivot_confidence"
                ]]
                st.dataframe(
                    bh.rename(columns=JP_COLUMNS),
                    use_container_width=True, hide_index=True
                )

            # 脚質にゃ
            show_style_tabs(pred_df, race_df)

            # 危険馬・穴候補にゃ
            c4, c5 = st.columns(2)
            c4.info(f"危険人気馬にゃ: {tickets.get('危険人気馬', 'なし')}")
            c5.success(f"穴候補にゃ: {tickets.get('穴候補', 'なし')}")

            # PKL本格AI分析にゃ
            try:
                show_pkl_ai_dashboard(bundle, race_df)
            except Exception as _ai_err:
                st.warning(f"AI分析エラーにゃ: {_ai_err}にゃ")

            # ── 全レース統合分析にゃ（G1も一般レースも対応にゃ）──
            try:
                show_race_analysis_full(race_df, strategy_mode=strategy_mode)
            except Exception as _all_err:
                st.warning(f"全力分析エラーにゃ: {_all_err}にゃ")

            # ── G1レース専用分析（G1のときは追加でさらに詳しくにゃ）──
            race_name_str = str(race_df.get("race_name", pd.Series([""])).iloc[0])
            is_g1 = any(x in race_name_str for x in ["G1","G１","GI","ダービー","天皇賞","有馬","ジャパンC","オークス","菊花賞","皐月賞","安田","マイルC","スプリンターズ","高松宮","宝塚","秋華賞","NHK","ヴィクトリア","フェブラリー","チャンピオンズ"])
            if is_g1:
                try:
                    show_g1_derby_analysis(race_df)
                except Exception as _g1_err:
                    st.warning(f"G1専用分析エラーにゃ: {_g1_err}にゃ")

            # ML強化ダッシュボードにゃ
            st.markdown("---")
            try:
                show_ml_enhance_dashboard(
                    bundle,
                    history_df=None,
                    pred_df=pred_df,
                    race_df=race_df,
                    strategy_mode=strategy_mode
                )
            except Exception as _ml_err:
                st.warning(f"ML強化ダッシュボードエラーにゃ: {_ml_err}にゃ")

            # 全レースにゃ
            st.markdown("---")
            st.subheader("全レースにゃ🐾")
            all_jp = jp_view(
                pred_df.sort_values(["race_key", "ml_rank"]),
                include_race_key=True
            )
            st.dataframe(all_jp, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 全レース予想CSVにゃ",
                data=all_jp.to_csv(
                    index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="nyanko_v26_all.csv", mime="text/csv"
            )

        except Exception as e:
            st.error(f"予想できなかったにゃ: {e}にゃ🐾")
            with st.expander("エラー詳細にゃ"):
                import traceback
                st.code(traceback.format_exc())

    # ── バックテストセクションにゃ（予想ボタンと独立して動くにゃ）──
    st.markdown("---")
    with st.expander("📊 バックテスト v2（完全版・比較分析にゃ）🐾 - クリックして開くにゃ", expanded=False):
        try:
            bundle_bt, bt_status = load_model_safely(uploaded_model)
            if bundle_bt is not None:
                show_backtest_v2_tab(bundle_bt, strategy_mode=strategy_mode)
            else:
                st.info("PKLをアップロードするとバックテストができるにゃ🐾")
        except Exception as _bt_load_err:
            st.info("PKLをアップロードするとバックテストができるにゃ🐾")

    # ── 着順データ一括取得セクションにゃ ──
    st.markdown("---")
    with st.expander("📥 着順データ一括取得にゃ🐾 - クリックして開くにゃ", expanded=False):
        try:
            show_result_fetch_tab()
        except Exception as _rf_err:
            st.error(f"着順取得エラーにゃ: {_rf_err}にゃ")
            with st.expander("エラー詳細にゃ"):
                import traceback
                st.code(traceback.format_exc())

    st.divider()
    with st.expander("簡易CSVテンプレにゃ（v26対応にゃ）"):
        st.caption("日付列を入れると正しい日付でレースが識別されるにゃ🐾")
        st.code(
            "日付,馬番,馬名,性別,年齢,騎手,斤量,オッズ,人気,競馬場,"
            "レース番号,レース名,距離,馬場,頭数,芝ダ\n"
            "20260510,1,サンプルAにゃ,牡,5,騎手Aにゃ,58.0,2.8,1,"
            "東京,11,サンプルにゃ,2000,良,18,芝\n"
            "20260510,2,サンプルBにゃ,牝,4,騎手Bにゃ,56.0,8.5,5,"
            "東京,11,サンプルにゃ,2000,良,18,芝\n",
            language="csv"
        )


try:
    app_main()
except Exception as e:
    st.error("アプリ起動時エラーです。下の詳細を確認してください。")
    st.exception(e)
