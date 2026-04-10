import io
import re
import time
import math
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False


# ============================================================
# 基本設定
# ============================================================

APP_TITLE = "競馬AI予想 完全版"
DEFAULT_RACE_URL = ""
REQUEST_TIMEOUT = 20
REQUEST_SLEEP = 0.5
TOP_FOR_TICKETS_DEFAULT = 10


def fetch_text_with_encoding(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    try:
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    except Exception:
        pass
    return resp.text


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

st.set_page_config(page_title=APP_TITLE, layout="wide")

if "race_df_store" not in st.session_state:
    st.session_state["race_df_store"] = None
if "hist_df_store" not in st.session_state:
    st.session_state["hist_df_store"] = None
if "prediction_ready" not in st.session_state:
    st.session_state["prediction_ready"] = False
if "loaded_from" not in st.session_state:
    st.session_state["loaded_from"] = ""


# ============================================================
# データクラス
# ============================================================

@dataclass
class RaceCardRow:
    race_id: str = ""
    race_date: str = ""
    race_name: str = ""
    course: str = ""
    track_type: str = ""
    distance: Optional[int] = None
    weather: str = ""
    ground: str = ""
    field_size: Optional[int] = None
    frame_no: Optional[int] = None
    horse_no: Optional[int] = None
    horse_name: str = ""
    sex_age: str = ""
    carried_weight: str = ""
    jockey: str = ""
    trainer: str = ""
    horse_url: str = ""
    jockey_url: str = ""
    odds: Optional[float] = None
    popularity: Optional[int] = None


@dataclass
class HorseHistoryRow:
    horse_name: str = ""
    horse_url: str = ""
    race_date: str = ""
    venue: str = ""
    race_name: str = ""
    class_name: str = ""
    track_type: str = ""
    distance: Optional[int] = None
    weather: str = ""
    ground: str = ""
    horse_no: Optional[int] = None
    frame_no: Optional[int] = None
    finish: Optional[int] = None
    jockey: str = ""
    carried_weight: str = ""
    time_str: str = ""
    margin: str = ""
    passing: str = ""
    last3f: Optional[float] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None
    body_weight: str = ""
    prize: str = ""


# ============================================================
# 汎用ユーティリティ
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm_text(value: Any) -> str:
    return str(value).replace(" ", "").replace("\u3000", "").strip()


def is_valid_horse_name(name: Any) -> bool:
    text = clean_text(name)
    if not text:
        return False
    ng_words = ["データベース", "のデータベース", "database", "http://", "https://"]
    if any(w.lower() in text.lower() for w in ng_words):
        return False
    if len(text) > 20:
        return False
    return True


def only_digits(value: Any) -> str:
    return re.sub(r"[^0-9-]", "", str(value))


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = clean_text(value).replace(",", "")
    if s in ["", "中止", "除外", "取消", "失格", "降着", "**"]:
        return None
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = clean_text(value).replace(",", "")
    if s in ["", "中止", "除外", "取消", "失格", "降着", "**"]:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def finish_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x)
    if s in ["中止", "除外", "取消", "失格", "降着"]:
        return np.nan
    m = re.search(r"\d+", s)
    return float(m.group()) if m else np.nan


def extract_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan


def parse_passing_positions(x: Any) -> List[int]:
    if pd.isna(x):
        return []
    return [int(n) for n in re.findall(r"\d+", str(x))]


def last_corner(x: Any) -> float:
    nums = parse_passing_positions(x)
    return float(nums[-1]) if nums else np.nan


def infer_running_style_from_history(passings: Iterable[Any], field_size: Any) -> str:
    if pd.isna(field_size) or field_size is None or field_size <= 0:
        return "不明"
    last_positions: List[float] = []
    for p in passings:
        pos = last_corner(p)
        if not pd.isna(pos):
            last_positions.append(float(pos))
    if not last_positions:
        return "不明"
    avg_last = float(np.mean(last_positions))
    rate = avg_last / float(field_size)
    if rate <= 0.33:
        return "先行"
    if rate <= 0.66:
        return "中団"
    return "差し"


def rank_score(s: pd.Series, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    return s.rank(ascending=ascending, pct=True)


def estimate_frame_no(horse_no: Any) -> float:
    try:
        return int((int(horse_no) + 1) // 2)
    except Exception:
        return np.nan


def normalize_track_type(value: Any) -> str:
    text = clean_text(value)
    if "障" in text:
        return "障害"
    if "ダ" in text:
        return "ダート"
    if "芝" in text:
        return "芝"
    return text


def normalize_ground(value: Any) -> str:
    text = clean_text(value)
    if "良" in text:
        return "良"
    if "稍" in text:
        return "稍重"
    if "重" in text and "不" not in text:
        return "重"
    if "不" in text:
        return "不良"
    return text


def estimate_race_pace(df: pd.DataFrame) -> str:
    if "running_style" not in df.columns or len(df) == 0:
        return "平均"
    styles = df["running_style"].fillna("不明").astype(str)
    front_ratio = (styles == "先行").mean()
    closer_ratio = (styles == "差し").mean()
    if front_ratio >= 0.38:
        return "ハイ"
    if front_ratio <= 0.18 and closer_ratio >= 0.45:
        return "スロー"
    return "平均"


def style_ground_pace_bonus(style: str, track_type: str, ground: str, pace: str) -> float:
    bonus = 0.0
    style = clean_text(style)
    track_type = normalize_track_type(track_type)
    ground = normalize_ground(ground)

    # ペース補正
    if pace == "ハイ":
        if style == "差し":
            bonus += 0.035
        elif style == "中団":
            bonus += 0.015
        elif style == "先行":
            bonus -= 0.015
    elif pace == "スロー":
        if style == "先行":
            bonus += 0.035
        elif style == "中団":
            bonus += 0.010
        elif style == "差し":
            bonus -= 0.015
    else:
        if style == "先行":
            bonus += 0.010
        elif style == "差し":
            bonus += 0.008

    # 馬場 × 脚質
    if track_type == "芝":
        if ground == "良":
            if style == "差し":
                bonus += 0.010
            elif style == "先行":
                bonus += 0.005
        elif ground in ["稍重", "重", "不良"]:
            if style == "先行":
                bonus += 0.020
            elif style == "差し":
                bonus -= 0.005
    elif track_type == "ダート":
        if ground == "良":
            if style == "先行":
                bonus += 0.015
        elif ground in ["稍重", "重", "不良"]:
            if style == "先行":
                bonus += 0.025
            elif style == "差し":
                bonus -= 0.010

    return bonus


def apply_style_ground_pace_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "running_style" not in out.columns:
        return out

    track_type = normalize_track_type(out["track_type"].dropna().astype(str).iloc[0]) if "track_type" in out.columns and out["track_type"].notna().any() else ""
    ground = normalize_ground(out["ground"].dropna().astype(str).iloc[0]) if "ground" in out.columns and out["ground"].notna().any() else ""
    pace = estimate_race_pace(out)

    out["pace"] = pace
    out["style_bonus"] = out["running_style"].map(lambda s: style_ground_pace_bonus(s, track_type, ground, pace))
    out["ai_score"] = pd.to_numeric(out["ai_score"], errors="coerce").fillna(0) + pd.to_numeric(out["style_bonus"], errors="coerce").fillna(0)

    expv = np.exp(out["ai_score"] - out["ai_score"].max())
    out["win_prob"] = expv / expv.sum()

    show_rate_col = pd.to_numeric(out["show_rate"], errors="coerce").fillna(0) if "show_rate" in out.columns else 0
    top2_rate_col = pd.to_numeric(out["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in out.columns else 0
    last3f_score_col = pd.to_numeric(out["s_last3f"], errors="coerce").fillna(0) if "s_last3f" in out.columns else 0
    place_raw = out["win_prob"] * 0.45 + show_rate_col * 0.30 + top2_rate_col * 0.15 + last3f_score_col * 0.10
    out["place_prob"] = place_raw / place_raw.sum()

    odds_valid = pd.to_numeric(out["odds_f"], errors="coerce").notna().sum() > 0 if "odds_f" in out.columns else False
    if odds_valid:
        out["market_prob"] = 1 / pd.to_numeric(out["odds_f"], errors="coerce")
        out["market_prob"] = out["market_prob"] / out["market_prob"].sum()
        out["ev_tansho"] = out["win_prob"] * out["odds_f"] * 0.8
    else:
        if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
            temp = 1 / pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
            out["market_prob"] = temp / temp.sum()
        else:
            out["market_prob"] = np.nan
        out["ev_tansho"] = np.nan

    out["gap"] = out["win_prob"] - out["market_prob"]
    if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        pop = pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        out["ana_score"] = out["gap"].fillna(0) * 0.6 + (1 / pop).fillna(0) * 0.4
    else:
        out["ana_score"] = out["gap"]

    out = out.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    out["mark"] = ""
    for i, m in enumerate(marks):
        if i < len(out):
            out.loc[i, "mark"] = m
    return out


def fmt_no(v: Any) -> Any:
    try:
        return int(v)
    except Exception:
        return v


def detect_track_type(course_text: str) -> str:
    text = clean_text(course_text)
    if "障" in text:
        return "障害"
    if "ダート" in text or text.startswith("ダ") or "ダ " in text or "ダ/" in text:
        return "ダート"
    if "芝" in text:
        return "芝"
    return text


def extract_distance(course_text: str) -> Optional[int]:
    m = re.search(r"(\d{3,4})m", clean_text(course_text))
    return int(m.group(1)) if m else None


def extract_race_id(url: str) -> str:
    m = re.search(r"race_id=(\d+)", url)
    return m.group(1) if m else ""


def extract_horse_id(url: str) -> str:
    m = re.search(r"/horse/(\d+)", url)
    return m.group(1) if m else ""


def convert_to_db_horse_url(horse_url: str) -> str:
    horse_id = extract_horse_id(horse_url)
    if not horse_id:
        return horse_url
    return f"https://db.netkeiba.com/horse/result/{horse_id}/"


def dataframe_from_dataclass_rows(rows: List[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in rows])


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            parts = [clean_text(x) for x in c if clean_text(x) and not str(x).startswith("Unnamed")]
            cols.append(" ".join(parts) if parts else "")
        else:
            cols.append(clean_text(c))
    out = df.copy()
    out.columns = cols
    return out


def safe_head(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    return df.head(min(n, len(df))).copy()


# ============================================================
# スクレイパー
# ============================================================

class Scraper:
    def __init__(self, timeout: int = REQUEST_TIMEOUT, sleep_sec: float = REQUEST_SLEEP) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self.sleep_sec = sleep_sec

    def get_html(self, url: str) -> str:
        if "nar.netkeiba.com" in url:
            html = self._try_requests_then_selenium(url)
            time.sleep(np.random.uniform(0.5, 1.2))
            return html
        try:
            html = self._requests_html(url)
            time.sleep(np.random.uniform(0.5, 1.2))
            return html
        except Exception:
            if not SELENIUM_AVAILABLE:
                raise
            html = self._selenium_html(url)
            time.sleep(np.random.uniform(0.5, 1.2))
            return html

    def _requests_html(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.text

    def _selenium_html(self, url: str) -> str:
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Seleniumが利用できません")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1600,2600")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)

            def ready(d):
                html = d.page_source
                return (
                    "<table" in html.lower()
                    and (
                        "/horse/" in html
                        or "着順" in html
                        or "上り" in html
                        or "上がり" in html
                        or "馬番" in html
                    )
                )

            WebDriverWait(driver, self.timeout).until(ready)
            return driver.page_source
        finally:
            driver.quit()

    def _try_requests_then_selenium(self, url: str) -> str:
        try:
            html = self._requests_html(url)
            if "<table" in html.lower():
                return html
        except Exception:
            pass
        return self._selenium_html(url)


# ============================================================
# 出馬表解析
# ============================================================

def parse_race_meta(soup: BeautifulSoup, race_url: str) -> Dict[str, Any]:
    race_id = extract_race_id(race_url)
    race_name = ""
    race_date = ""
    weather = ""
    ground = ""
    course = ""
    field_size = None

    title = soup.select_one(".RaceName") or soup.select_one("h1") or soup.select_one("title")
    if title:
        race_name = clean_text(title.get_text())

    data1 = soup.select_one(".RaceData01")
    data2 = soup.select_one(".RaceData02")
    if data1 is None and data2 is None:
        candidates = soup.select(".RaceData01, .RaceData02, .RaceData, .smalltxt, .racedata")
        meta_text = " ".join(x.get_text(" ", strip=True) for x in candidates)
        if not meta_text:
            meta_text = clean_text(soup.get_text(" ", strip=True))
    else:
        meta_text = " ".join(x.get_text(" ", strip=True) for x in [data1, data2] if x is not None)

    course = meta_text
    track_type = detect_track_type(meta_text)
    distance = extract_distance(meta_text)

    weather_m = re.search(r"天候\s*[:：]?\s*([^\s/]+)", meta_text)
    if weather_m:
        weather = weather_m.group(1)

    ground_m = re.search(r"馬場\s*[:：]?\s*([^\s/]+)", meta_text)
    if ground_m:
        ground = ground_m.group(1)

    date_m = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", meta_text)
    if date_m:
        race_date = date_m.group(1)

    count_m = re.search(r"(\d+)頭", meta_text)
    if count_m:
        field_size = int(count_m.group(1))

    return {
        "race_id": race_id,
        "race_name": race_name,
        "race_date": race_date,
        "course": course,
        "track_type": track_type,
        "distance": distance,
        "weather": weather,
        "ground": ground,
        "field_size": field_size,
    }



def dedupe_race_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "horse_name" in out.columns:
        out["horse_name_norm"] = out["horse_name"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    else:
        out["horse_name_norm"] = ""

    sort_cols = []
    if "horse_no" in out.columns:
        sort_cols.append("horse_no")
    if "odds" in out.columns:
        sort_cols.append("odds")
    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last")

    if "horse_url" in out.columns and out["horse_url"].notna().sum() > 0:
        out = out.drop_duplicates(subset=["horse_url"], keep="first")

    if "horse_no" in out.columns and out["horse_no"].notna().sum() > 0:
        out = out.drop_duplicates(subset=["horse_no"], keep="first")

    if "horse_name_norm" in out.columns:
        out = out.drop_duplicates(subset=["horse_name_norm"], keep="first")

    return out.drop(columns=["horse_name_norm"], errors="ignore").reset_index(drop=True)


def parse_race_card_jra(scraper: Scraper, race_url: str) -> List[RaceCardRow]:
    html = scraper.get_html(race_url)
    soup = BeautifulSoup(html, "lxml")
    meta = parse_race_meta(soup, race_url)

    table = None
    for selector in [
        "table.Shutuba_Table",
        "table.RaceTable01",
        "table[class*='Shutuba']",
        "table[class*='RaceTable']",
        "table",
    ]:
        cand = soup.select_one(selector)
        if cand is not None and cand.select("tr"):
            table = cand
            break

    if table is None:
        raise RuntimeError("出馬表テーブルが見つかりませんでした。")

    header_map: Dict[str, int] = {}
    header_found = False
    rows: List[RaceCardRow] = []

    for tr in table.select("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        text_cells = [clean_text(c.get_text(" ", strip=True)) for c in cells]

        if not header_found and any(x in "".join(text_cells) for x in ["馬番", "枠", "馬名"]):
            header_found = True
            for idx, val in enumerate(text_cells):
                if "枠" in val:
                    header_map["frame_no"] = idx
                elif "馬番" in val:
                    header_map["horse_no"] = idx
                elif "馬名" in val:
                    header_map["horse_name"] = idx
                elif "性齢" in val:
                    header_map["sex_age"] = idx
                elif "斤量" in val:
                    header_map["carried_weight"] = idx
                elif "騎手" in val:
                    header_map["jockey"] = idx
                elif "厩舎" in val or "調教師" in val:
                    header_map["trainer"] = idx
                elif "オッズ" in val or "単勝" in val:
                    header_map["odds"] = idx
                elif "人気" in val:
                    header_map["popularity"] = idx
            continue

        if len(cells) < 5:
            continue

        links = tr.find_all("a", href=True)
        horse_url = ""
        jockey_url = ""
        horse_name = ""
        jockey = ""
        trainer = ""
        for a in links:
            href = a.get("href", "")
            if "/horse/" in href and not horse_url:
                horse_url = urljoin(race_url, href)
                horse_name = clean_text(a.get_text())
            elif "/jockey/" in href and not jockey_url:
                jockey_url = urljoin(race_url, href)
                jockey = clean_text(a.get_text())

        def pick(name: str, fallback_idx: Optional[int] = None) -> str:
            idx = header_map.get(name, fallback_idx)
            if idx is None or idx >= len(text_cells):
                return ""
            return text_cells[idx]

        row = RaceCardRow(
            race_id=meta["race_id"],
            race_date=meta["race_date"],
            race_name=meta["race_name"],
            course=meta["course"],
            track_type=meta["track_type"],
            distance=meta["distance"],
            weather=meta["weather"],
            ground=meta["ground"],
            field_size=meta["field_size"],
            frame_no=to_int(pick("frame_no", 0)),
            horse_no=to_int(pick("horse_no", 1)),
            horse_name=horse_name or pick("horse_name", 3),
            sex_age=pick("sex_age", 4),
            carried_weight=pick("carried_weight", 5),
            jockey=jockey or pick("jockey", 6),
            trainer=trainer or pick("trainer", 7),
            horse_url=horse_url,
            jockey_url=jockey_url,
            odds=to_float(pick("odds", len(text_cells) - 2 if len(text_cells) >= 2 else None)),
            popularity=to_int(pick("popularity", len(text_cells) - 1 if len(text_cells) >= 1 else None)),
        )
        if row.horse_name and is_valid_horse_name(row.horse_name):
            rows.append(row)

    if not rows:
        raise RuntimeError("出馬表の解析結果が0件でした。")

    df = dedupe_race_df(dataframe_from_dataclass_rows(rows))
    return [RaceCardRow(**r) for r in df.to_dict(orient="records")]


def parse_race_card_nar(scraper: Scraper, race_url: str) -> List[RaceCardRow]:
    html = scraper.get_html(race_url)
    soup = BeautifulSoup(html, "lxml")
    meta = parse_race_meta(soup, race_url)
    tables = soup.select("table")
    if not tables:
        raise RuntimeError("NARの出馬表テーブルが見つかりませんでした。")

    rows: List[RaceCardRow] = []
    for table in tables:
        for tr in table.select("tr"):
            horse_a = tr.select_one('a[href*="/horse/"]')
            if horse_a is None:
                continue
            cells = tr.find_all(["td", "th"])
            text_cells = [clean_text(c.get_text(" ", strip=True)) for c in cells]
            if not text_cells:
                continue

            horse_url = urljoin(race_url, horse_a.get("href", ""))
            horse_name = clean_text(horse_a.get_text())
            jockey_a = tr.select_one('a[href*="/jockey/"]')
            jockey_url = urljoin(race_url, jockey_a.get("href", "")) if jockey_a else ""
            jockey = clean_text(jockey_a.get_text()) if jockey_a else ""

            numeric_cells = [to_int(x) for x in text_cells if to_int(x) is not None]
            frame_no = numeric_cells[0] if len(numeric_cells) >= 1 else None
            horse_no = numeric_cells[1] if len(numeric_cells) >= 2 else None
            sex_age = ""
            carried_weight = ""
            trainer = ""
            odds = None
            popularity = None

            for x in text_cells:
                if re.fullmatch(r"[牡牝セ]\d+", x):
                    sex_age = x
                    break
            for x in text_cells:
                v = to_float(x)
                if v is not None and 40 <= v <= 70:
                    carried_weight = x
                    break

            tail = text_cells[-8:]
            tail_f = [to_float(x) for x in tail]
            tail_i = [to_int(x) for x in tail]
            float_cands = [v for v in tail_f if v is not None and 1.0 <= v <= 9999]
            int_cands = [v for v in tail_i if v is not None and 1 <= v <= 18]
            if float_cands:
                odds = float_cands[0]
            if int_cands:
                popularity = int_cands[-1]

            row = RaceCardRow(
                race_id=meta["race_id"],
                race_date=meta["race_date"],
                race_name=meta["race_name"],
                course=meta["course"],
                track_type=meta["track_type"],
                distance=meta["distance"],
                weather=meta["weather"],
                ground=meta["ground"],
                field_size=meta["field_size"],
                frame_no=frame_no,
                horse_no=horse_no,
                horse_name=horse_name,
                sex_age=sex_age,
                carried_weight=carried_weight,
                jockey=jockey,
                trainer=trainer,
                horse_url=horse_url,
                jockey_url=jockey_url,
                odds=odds,
                popularity=popularity,
            )
            if row.horse_name and is_valid_horse_name(row.horse_name):
                rows.append(row)

    if not rows:
        raise RuntimeError("NARの出馬表の解析結果が0件でした。")

    df = dedupe_race_df(dataframe_from_dataclass_rows(rows))
    return [RaceCardRow(**r) for r in df.to_dict(orient="records")]


def parse_race_card(scraper: Scraper, race_url: str) -> List[RaceCardRow]:
    if "nar.netkeiba.com" in race_url:
        return parse_race_card_nar(scraper, race_url)
    return parse_race_card_jra(scraper, race_url)


# ============================================================
# 過去成績解析
# ============================================================

def normalize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df)
    mapping = {
        "日付": "race_date",
        "開催": "venue",
        "場名": "venue",
        "レース名": "race_name",
        "レース 名": "race_name",
        "レース": "race_name",
        "クラス": "class_name",
        "条件": "class_name",
        "距離": "distance_raw",
        "天気": "weather",
        "天候": "weather",
        "馬場": "ground",
        "枠番": "frame_no",
        "馬番": "horse_no",
        "着順": "finish",
        "着 順": "finish",
        "騎手": "jockey",
        "斤量": "carried_weight",
        "タイム": "time_str",
        "着差": "margin",
        "通過": "passing",
        "上り": "last3f",
        "上がり": "last3f",
        "上り3F": "last3f",
        "上がり3F": "last3f",
        "後3F": "last3f",
        "3F": "last3f",
        "単勝": "odds",
        "人気": "popularity",
        "馬体重": "body_weight",
        "馬体重(増減)": "body_weight",
        "賞金": "prize",
    }
    ren = {}
    for c in df.columns:
        key = clean_text(c)
        if key in mapping:
            ren[c] = mapping[key]
    df = df.rename(columns=ren).copy()

    if "distance_raw" not in df.columns:
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\d{3,4}m", regex=True, na=False).sum() > 0:
                df["distance_raw"] = df[c]
                break
        if "distance_raw" not in df.columns:
            df["distance_raw"] = None

    if "odds" not in df.columns or pd.Series(df["odds"]).isna().all():
        for c in df.columns:
            s = df[c].astype(str)
            cnt = s.str.fullmatch(r"\d+(?:\.\d+)?", na=False).sum()
            if cnt >= max(1, len(df) // 2) and s.str.contains(r"\.", regex=True, na=False).sum() >= max(1, len(df) // 3):
                df["odds"] = pd.to_numeric(df[c], errors="coerce")
                break

    if "popularity" not in df.columns or pd.Series(df["popularity"]).isna().all():
        for c in df.columns:
            nums = pd.to_numeric(df[c], errors="coerce")
            if nums.notna().sum() >= max(1, len(df) // 2):
                valid = nums.dropna()
                if len(valid) and valid.between(1, 18).mean() > 0.8:
                    df["popularity"] = nums
                    break

    if "last3f" not in df.columns or pd.Series(df["last3f"]).isna().all():
        for c in df.columns:
            cname = clean_text(c)
            if any(k in cname for k in ["上り", "上がり", "上り3F", "上がり3F", "後3F", "3F"]):
                nums = pd.to_numeric(df[c], errors="coerce")
                if nums.notna().sum() >= max(1, len(df) // 3):
                    valid = nums.dropna()
                    if len(valid) and valid.between(30, 50).mean() > 0.6:
                        df["last3f"] = nums
                        break

    if "last3f" not in df.columns:
        df["last3f"] = np.nan
    if "odds" not in df.columns:
        df["odds"] = np.nan
    if "popularity" not in df.columns:
        df["popularity"] = np.nan

    for col in [
        "race_date",
        "venue",
        "race_name",
        "class_name",
        "weather",
        "ground",
        "horse_no",
        "frame_no",
        "finish",
        "jockey",
        "carried_weight",
        "time_str",
        "margin",
        "passing",
        "body_weight",
        "prize",
    ]:
        if col not in df.columns:
            df[col] = "" if col not in ["horse_no", "frame_no", "finish"] else np.nan

    df["distance"] = df["distance_raw"].astype(str).str.extract(r"(\d{3,4})").iloc[:, 0]
    df["track_type"] = df["distance_raw"].astype(str).apply(detect_track_type)

    keep = [
        "race_date",
        "venue",
        "race_name",
        "class_name",
        "track_type",
        "distance",
        "weather",
        "ground",
        "horse_no",
        "frame_no",
        "finish",
        "jockey",
        "carried_weight",
        "time_str",
        "margin",
        "passing",
        "last3f",
        "odds",
        "popularity",
        "body_weight",
        "prize",
    ]
    return df[keep].copy()


def parse_horse_history(scraper: Scraper, horse_name: str, horse_url: str, max_rows: int = 10) -> List[HorseHistoryRow]:
    if not horse_url:
        return []
    db_url = convert_to_db_horse_url(horse_url)
    html = scraper.get_html(db_url)

    use_selenium = False
    if "<table" not in html.lower():
        use_selenium = True
    elif ("上り" not in html and "上がり" not in html and "上り3F" not in html and "上がり3F" not in html):
        use_selenium = True
    if use_selenium and SELENIUM_AVAILABLE:
        html = scraper._selenium_html(db_url)

    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []

    best = pd.DataFrame()
    best_score = -1
    for raw in tables:
        try:
            ndf = normalize_history_columns(raw).head(max_rows)
            finish_cnt = ndf["finish"].apply(to_int).notna().sum() if "finish" in ndf.columns else 0
            dist_cnt = ndf["distance"].apply(to_int).notna().sum() if "distance" in ndf.columns else 0
            last3f_cnt = ndf["last3f"].apply(to_float).notna().sum() if "last3f" in ndf.columns else 0
            odds_cnt = ndf["odds"].apply(to_float).notna().sum() if "odds" in ndf.columns else 0
            pop_cnt = ndf["popularity"].apply(to_int).notna().sum() if "popularity" in ndf.columns else 0
            passing_cnt = ndf["passing"].astype(str).str.contains(r"\d", regex=True, na=False).sum() if "passing" in ndf.columns else 0
            score = finish_cnt * 10 + dist_cnt * 5 + last3f_cnt * 5 + odds_cnt * 3 + pop_cnt * 3 + passing_cnt * 4
            if score > best_score:
                best = ndf
                best_score = score
        except Exception:
            continue

    if best.empty:
        return []

    out: List[HorseHistoryRow] = []
    for _, r in best.iterrows():
        out.append(
            HorseHistoryRow(
                horse_name=horse_name,
                horse_url=db_url,
                race_date=clean_text(r.get("race_date")),
                venue=clean_text(r.get("venue")),
                race_name=clean_text(r.get("race_name")),
                class_name=clean_text(r.get("class_name")),
                track_type=clean_text(r.get("track_type")),
                distance=to_int(r.get("distance")),
                weather=clean_text(r.get("weather")),
                ground=clean_text(r.get("ground")),
                horse_no=to_int(r.get("horse_no")),
                frame_no=to_int(r.get("frame_no")),
                finish=to_int(r.get("finish")),
                jockey=clean_text(r.get("jockey")),
                carried_weight=clean_text(r.get("carried_weight")),
                time_str=clean_text(r.get("time_str")),
                margin=clean_text(r.get("margin")),
                passing=clean_text(r.get("passing")),
                last3f=to_float(r.get("last3f")),
                odds=to_float(r.get("odds")),
                popularity=to_int(r.get("popularity")),
                body_weight=clean_text(r.get("body_weight")),
                prize=clean_text(r.get("prize")),
            )
        )
    return out


def fetch_race_and_history(race_url: str, max_horses: int, history_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scraper = Scraper()
    race_rows = parse_race_card(scraper, race_url)
    race_df = dataframe_from_dataclass_rows(race_rows)

    history_rows_out: List[HorseHistoryRow] = []
    total = min(len(race_rows), max_horses)
    prog = st.progress(0, text="過去成績を取得中...")
    for i, row in enumerate(race_rows[:total], start=1):
        histories = parse_horse_history(scraper, row.horse_name, row.horse_url, history_rows)
        history_rows_out.extend(histories)
        prog.progress(i / total if total else 1, text=f"過去成績を取得中... {i}/{total} {row.horse_name}")
    prog.empty()

    hist_df = dataframe_from_dataclass_rows(history_rows_out)
    if hist_df.empty:
        hist_df = pd.DataFrame(
            columns=[
                "horse_name",
                "horse_url",
                "race_date",
                "venue",
                "race_name",
                "class_name",
                "track_type",
                "distance",
                "weather",
                "ground",
                "horse_no",
                "frame_no",
                "finish",
                "jockey",
                "carried_weight",
                "time_str",
                "margin",
                "passing",
                "last3f",
                "odds",
                "popularity",
                "body_weight",
                "prize",
            ]
        )
    return race_df, hist_df


# ============================================================
# 予想ロジック
# ============================================================

def build_features(hist: pd.DataFrame, recent_n: int = 10) -> pd.DataFrame:
    if hist is None:
        hist = pd.DataFrame(columns=["horse_name", "finish"])
    hist = hist.copy()
    if "race_date" in hist.columns:
        hist["race_date"] = pd.to_datetime(hist["race_date"], errors="coerce")
        hist = hist.sort_values("race_date", ascending=False, na_position="last")
    if "key" not in hist.columns and "horse_name" in hist.columns:
        hist["key"] = hist["horse_name"].apply(norm_text)
    if "key" in hist.columns:
        hist = hist.groupby("key", group_keys=False).head(int(recent_n))

    rows = []
    for key, g in hist.groupby("key", sort=False):
        if "race_date" in g.columns:
            g = g.sort_values("race_date", ascending=False, na_position="last")
        r3 = g.head(3)
        r5 = g.head(5)
        f_all = g["finish_num"].dropna() if "finish_num" in g.columns else pd.Series(dtype=float)
        f3 = r3["finish_num"].dropna() if "finish_num" in r3.columns else pd.Series(dtype=float)
        f5 = r5["finish_num"].dropna() if "finish_num" in r5.columns else pd.Series(dtype=float)
        l3f3 = r3["last3f_num"].dropna() if "last3f_num" in r3.columns else pd.Series(dtype=float)
        c3 = r3["corner_num"].dropna() if "corner_num" in r3.columns else pd.Series(dtype=float)
        o3 = r3["odds_num"].dropna() if "odds_num" in r3.columns else pd.Series(dtype=float)
        p3 = r3["pop_num"].dropna() if "pop_num" in r3.columns else pd.Series(dtype=float)
        rows.append(
            {
                "key": key,
                "recent3_avg": f3.mean() if len(f3) else np.nan,
                "recent3_best": f3.min() if len(f3) else np.nan,
                "recent5_avg": f5.mean() if len(f5) else np.nan,
                "last_finish": f_all.iloc[0] if len(f_all) else np.nan,
                "show_rate": (f_all <= 3).mean() if len(f_all) else np.nan,
                "quinella_rate": (f_all <= 2).mean() if len(f_all) else np.nan,
                "win_rate": (f_all == 1).mean() if len(f_all) else np.nan,
                "top2_rate": (f_all <= 2).mean() if len(f_all) else np.nan,
                "std_finish": f_all.std() if len(f_all) else np.nan,
                "median_finish": f_all.median() if len(f_all) else np.nan,
                "last3f_avg": l3f3.mean() if len(l3f3) else np.nan,
                "corner_avg": c3.mean() if len(c3) else np.nan,
                "auto_running_style": infer_running_style_from_history(r3["passing"].tolist() if "passing" in r3.columns else [], 18),
                "hist_odds_avg": o3.mean() if len(o3) else np.nan,
                "hist_pop_avg": p3.mean() if len(p3) else np.nan,
            }
        )
    cols = [
        "key",
        "recent3_avg",
        "recent3_best",
        "recent5_avg",
        "last_finish",
        "show_rate",
        "quinella_rate",
        "win_rate",
        "top2_rate",
        "std_finish",
        "median_finish",
        "last3f_avg",
        "corner_avg",
        "auto_running_style",
        "hist_odds_avg",
        "hist_pop_avg",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def analyze_race(race: pd.DataFrame, hist: pd.DataFrame, recent_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    if hist is None or len(hist) == 0:
        hist = pd.DataFrame({"horse_name": [], "finish": []})
    if "horse_name" not in hist.columns:
        hist["horse_name"] = ""
    if "finish" not in hist.columns:
        hist["finish"] = np.nan
    if "horse_name" not in race.columns:
        raise ValueError("race_card に horse_name が必要です。")

    race = race.copy()
    hist = hist.copy()
    race["key"] = race["horse_name"].apply(norm_text)
    hist["key"] = hist["horse_name"].apply(norm_text)

    race["horse_no"] = pd.to_numeric(race["horse_no"], errors="coerce") if "horse_no" in race.columns else np.arange(1, len(race) + 1)
    race["frame_no"] = pd.to_numeric(race["frame_no"], errors="coerce") if "frame_no" in race.columns else race["horse_no"].apply(estimate_frame_no)
    race["odds_num"] = race["odds"].apply(extract_num) if "odds" in race.columns else np.nan
    race["pop_num"] = race["popularity"].apply(extract_num) if "popularity" in race.columns else np.nan

    hist["finish_num"] = hist["finish"].apply(finish_num)
    hist["last3f_num"] = hist["last3f"].apply(extract_num) if "last3f" in hist.columns else np.nan
    hist["corner_num"] = hist["passing"].apply(last_corner) if "passing" in hist.columns else np.nan
    hist["odds_num"] = hist["odds"].apply(extract_num) if "odds" in hist.columns else np.nan
    hist["pop_num"] = hist["popularity"].apply(extract_num) if "popularity" in hist.columns else np.nan
    if "race_date" in hist.columns:
        hist["race_date"] = pd.to_datetime(hist["race_date"], errors="coerce")

    feat = build_features(hist, recent_n=recent_n)
    if feat.empty or "key" not in feat.columns:
        feat = pd.DataFrame({"key": race["key"]}).drop_duplicates()

    df = race.merge(feat, on="key", how="left")
    df["odds_f"] = df["odds_num"].fillna(df["hist_odds_avg"] if "hist_odds_avg" in df.columns else np.nan)
    df["pop_f"] = df["pop_num"].fillna(df["hist_pop_avg"] if "hist_pop_avg" in df.columns else np.nan)
    field_size = max(len(df), 1)
    if "auto_running_style" in df.columns:
        df["running_style"] = df["auto_running_style"].fillna("不明")
    elif "corner_avg" in df.columns:
        df["running_style"] = df["corner_avg"].apply(lambda x: infer_running_style(x, field_size))
    else:
        df["running_style"] = "不明"

    df["s_recent3"] = 1 - rank_score(df["recent3_avg"], True) if "recent3_avg" in df.columns else 0
    df["s_recent5"] = 1 - rank_score(df["recent5_avg"], True) if "recent5_avg" in df.columns else 0
    df["s_best"] = 1 - rank_score(df["recent3_best"], True) if "recent3_best" in df.columns else 0
    df["s_last"] = 1 - rank_score(df["last_finish"], True) if "last_finish" in df.columns else 0
    df["s_show"] = rank_score(df["show_rate"], True) if "show_rate" in df.columns else 0
    df["s_quinella"] = rank_score(df["quinella_rate"], True) if "quinella_rate" in df.columns else 0
    df["s_winrate"] = pd.to_numeric(df["win_rate"], errors="coerce").fillna(0) if "win_rate" in df.columns else 0
    df["s_top2"] = pd.to_numeric(df["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in df.columns else 0
    df["s_std"] = rank_score(df["std_finish"], True) if "std_finish" in df.columns else 0
    df["s_median"] = 1 - rank_score(df["median_finish"], True) if "median_finish" in df.columns else 0
    df["s_last3f"] = 1 - rank_score(df["last3f_avg"], True) if "last3f_avg" in df.columns and pd.to_numeric(df["last3f_avg"], errors="coerce").notna().sum() > 0 else 0.0

    df["ai_score"] = (
        pd.to_numeric(df["s_recent3"], errors="coerce").fillna(0) * 0.18
        + pd.to_numeric(df["s_recent5"], errors="coerce").fillna(0) * 0.06
        + pd.to_numeric(df["s_best"], errors="coerce").fillna(0) * 0.11
        + pd.to_numeric(df["s_last"], errors="coerce").fillna(0) * 0.14
        + pd.to_numeric(df["s_show"], errors="coerce").fillna(0) * 0.08
        + pd.to_numeric(df["s_quinella"], errors="coerce").fillna(0) * 0.08
        + pd.to_numeric(df["s_winrate"], errors="coerce").fillna(0) * 0.08
        + pd.to_numeric(df["s_top2"], errors="coerce").fillna(0) * 0.08
        + pd.to_numeric(df["s_median"], errors="coerce").fillna(0) * 0.06
        + pd.to_numeric(df["s_std"], errors="coerce").fillna(0) * 0.03
        + pd.to_numeric(df["s_last3f"], errors="coerce").fillna(0) * 0.10
    )

    if "last3f_avg" in df.columns and pd.to_numeric(df["last3f_avg"], errors="coerce").notna().sum() >= 5:
        med = pd.to_numeric(df["last3f_avg"], errors="coerce").median()
        if pd.notna(med) and med <= 33.5:
            df.loc[df["running_style"] == "差し", "ai_score"] += 0.03
            df.loc[df["running_style"] == "中団", "ai_score"] += 0.01
    df.loc[df["running_style"] == "先行", "ai_score"] += 0.02

    if pd.to_numeric(df["pop_f"], errors="coerce").notna().sum() > 0:
        df.loc[(df["pop_f"] >= 5) & (df["pop_f"] <= 10), "ai_score"] += 0.02
    if pd.to_numeric(df["frame_no"], errors="coerce").notna().sum() > 0:
        df.loc[(df["running_style"] == "先行") & (df["frame_no"] <= 3), "ai_score"] += 0.02
        df.loc[(df["running_style"] == "先行") & (df["frame_no"] >= 6), "ai_score"] -= 0.015
        df.loc[(df["running_style"] == "差し") & (df["frame_no"] >= 6), "ai_score"] += 0.015

    expv = np.exp(df["ai_score"] - df["ai_score"].max())
    df["win_prob"] = expv / expv.sum()
    show_rate_col = pd.to_numeric(df["show_rate"], errors="coerce").fillna(0) if "show_rate" in df.columns else 0
    top2_rate_col = pd.to_numeric(df["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in df.columns else 0
    last3f_score_col = pd.to_numeric(df["s_last3f"], errors="coerce").fillna(0)
    place_raw = df["win_prob"] * 0.45 + show_rate_col * 0.30 + top2_rate_col * 0.15 + last3f_score_col * 0.10
    df["place_prob"] = place_raw / place_raw.sum()

    odds_valid = pd.to_numeric(df["odds_f"], errors="coerce").notna().sum() > 0
    if odds_valid:
        df["market_prob"] = 1 / pd.to_numeric(df["odds_f"], errors="coerce")
        df["market_prob"] = df["market_prob"] / df["market_prob"].sum()
        df["ev_tansho"] = df["win_prob"] * df["odds_f"] * 0.8
    else:
        if pd.to_numeric(df["pop_f"], errors="coerce").notna().sum() > 0:
            temp = 1 / pd.to_numeric(df["pop_f"], errors="coerce").replace(0, np.nan)
            df["market_prob"] = temp / temp.sum()
        else:
            df["market_prob"] = np.nan
        df["ev_tansho"] = np.nan

    df["gap"] = df["win_prob"] - df["market_prob"]
    if pd.to_numeric(df["pop_f"], errors="coerce").notna().sum() > 0:
        pop = pd.to_numeric(df["pop_f"], errors="coerce").replace(0, np.nan)
        df["ana_score"] = df["gap"].fillna(0) * 0.6 + (1 / pop).fillna(0) * 0.4
    else:
        df["ana_score"] = df["gap"]

    df = df.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    df["mark"] = ""
    for i, m in enumerate(marks):
        if i < len(df):
            df.loc[i, "mark"] = m

    top = df.head(min(12, len(df))).copy()
    honmei = df.sort_values(["win_prob", "ai_score"], ascending=False).head(3).copy()
    ana = df.sort_values(["ana_score", "gap"], ascending=False).head(8).copy()
    return df, top, honmei, ana, odds_valid


# ============================================================
# 馬券生成
# ============================================================

def build_pair_table(top_df: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    for i, j in combinations(range(len(top_df)), 2):
        a, b = top_df.iloc[i], top_df.iloc[j]
        rows.append(
            {
                "券種": label,
                "組み合わせ": f"{int(a['horse_no'])}-{int(b['horse_no'])}",
                "馬1": a["horse_name"],
                "馬2": b["horse_name"],
                "score": float(a[score_col]) * float(b[score_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame(columns=["券種", "組み合わせ", "馬1", "馬2", "score"])


def build_trio_table(top_df: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    for i, j, k in combinations(range(len(top_df)), 3):
        a, b, c = top_df.iloc[i], top_df.iloc[j], top_df.iloc[k]
        rows.append(
            {
                "券種": label,
                "組み合わせ": f"{int(a['horse_no'])}-{int(b['horse_no'])}-{int(c['horse_no'])}",
                "馬1": a["horse_name"],
                "馬2": b["horse_name"],
                "馬3": c["horse_name"],
                "score": float(a[score_col]) * float(b[score_col]) * float(c[score_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame(columns=["券種", "組み合わせ", "馬1", "馬2", "馬3", "score"])


def build_trifecta_table(top_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, j, k in permutations(range(len(top_df)), 3):
        if len({i, j, k}) < 3:
            continue
        a, b, c = top_df.iloc[i], top_df.iloc[j], top_df.iloc[k]
        rows.append(
            {
                "組み合わせ": f"{int(a['horse_no'])}→{int(b['horse_no'])}→{int(c['horse_no'])}",
                "1着": a["horse_name"],
                "2着": b["horse_name"],
                "3着": c["horse_name"],
                "score": float(a["win_prob"]) * float(b["place_prob"]) * float(c["place_prob"]),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame(columns=["組み合わせ", "1着", "2着", "3着", "score"])


def build_honmei_ana_tables(honmei: pd.DataFrame, ana: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide_rows, umaren_rows, trio_rows, trifecta_rows = [], [], [], []
    merged = pd.concat([honmei, ana]).drop_duplicates(subset=["horse_no"]) if len(honmei) or len(ana) else pd.DataFrame()

    for _, h in honmei.iterrows():
        for _, a in ana.iterrows():
            if h["horse_no"] == a["horse_no"]:
                continue
            umaren_rows.append(
                {
                    "組み合わせ": f"{fmt_no(h['horse_no'])}-{fmt_no(a['horse_no'])}",
                    "本命": h["horse_name"],
                    "穴": a["horse_name"],
                    "score": float(h["win_prob"]) * float(a["win_prob"]),
                }
            )
            wide_rows.append(
                {
                    "組み合わせ": f"{fmt_no(h['horse_no'])}-{fmt_no(a['horse_no'])}",
                    "本命": h["horse_name"],
                    "穴": a["horse_name"],
                    "score": float(h["place_prob"]) * float(a["place_prob"]),
                }
            )

    for i in range(len(honmei)):
        for j in range(i + 1, len(honmei)):
            h1 = honmei.iloc[i]
            h2 = honmei.iloc[j]
            for _, a in ana.iterrows():
                if len({h1["horse_no"], h2["horse_no"], a["horse_no"]}) < 3:
                    continue
                trio_rows.append(
                    {
                        "組み合わせ": f"{fmt_no(h1['horse_no'])}-{fmt_no(h2['horse_no'])}-{fmt_no(a['horse_no'])}",
                        "本命1": h1["horse_name"],
                        "本命2": h2["horse_name"],
                        "穴": a["horse_name"],
                        "score": float(h1["place_prob"]) * float(h2["place_prob"]) * float(a["place_prob"]),
                    }
                )

    if len(merged):
        for _, h1 in honmei.head(1).iterrows():
            for _, h2 in merged.iterrows():
                for _, h3 in merged.iterrows():
                    if len({h1["horse_no"], h2["horse_no"], h3["horse_no"]}) < 3:
                        continue
                    trifecta_rows.append(
                        {
                            "組み合わせ": f"{fmt_no(h1['horse_no'])}→{fmt_no(h2['horse_no'])}→{fmt_no(h3['horse_no'])}",
                            "1着": h1["horse_name"],
                            "2着": h2["horse_name"],
                            "3着": h3["horse_name"],
                            "score": float(h1["win_prob"]) * float(h2["place_prob"]) * float(h3["place_prob"]),
                        }
                    )

    return (
        pd.DataFrame(umaren_rows).sort_values("score", ascending=False) if umaren_rows else pd.DataFrame(columns=["組み合わせ", "本命", "穴", "score"]),
        pd.DataFrame(wide_rows).sort_values("score", ascending=False) if wide_rows else pd.DataFrame(columns=["組み合わせ", "本命", "穴", "score"]),
        pd.DataFrame(trio_rows).sort_values("score", ascending=False) if trio_rows else pd.DataFrame(columns=["組み合わせ", "本命1", "本命2", "穴", "score"]),
        pd.DataFrame(trifecta_rows).sort_values("score", ascending=False) if trifecta_rows else pd.DataFrame(columns=["組み合わせ", "1着", "2着", "3着", "score"]),
    )


# ============================================================
# Excel出力
# ============================================================

def make_download_file(tables: Dict[str, pd.DataFrame]) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            safe = re.sub(r'[\\/*?:\[\]]', '_', sheet_name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe)
    output.seek(0)
    return output


def load_csv_upload(file: Any) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            pass
    raise ValueError("CSVを読めませんでした。")


# ============================================================
# UI描画
# ============================================================


def apply_manual_running_style_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    return apply_style_ground_pace_adjustments(df)
    style_bonus = out["running_style"].map({"先行": 0.02, "中団": 0.0, "差し": 0.015, "不明": 0.0}).fillna(0.0)
    out["ai_score"] = pd.to_numeric(out["ai_score"], errors="coerce").fillna(0) + style_bonus
    expv = np.exp(out["ai_score"] - out["ai_score"].max())
    out["win_prob"] = expv / expv.sum()
    show_rate_col = pd.to_numeric(out["show_rate"], errors="coerce").fillna(0) if "show_rate" in out.columns else 0
    top2_rate_col = pd.to_numeric(out["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in out.columns else 0
    last3f_score_col = pd.to_numeric(out["s_last3f"], errors="coerce").fillna(0) if "s_last3f" in out.columns else 0
    place_raw = out["win_prob"] * 0.45 + show_rate_col * 0.30 + top2_rate_col * 0.15 + last3f_score_col * 0.10
    out["place_prob"] = place_raw / place_raw.sum()
    odds_valid = pd.to_numeric(out["odds_f"], errors="coerce").notna().sum() > 0 if "odds_f" in out.columns else False
    if odds_valid:
        out["market_prob"] = 1 / pd.to_numeric(out["odds_f"], errors="coerce")
        out["market_prob"] = out["market_prob"] / out["market_prob"].sum()
        out["ev_tansho"] = out["win_prob"] * out["odds_f"] * 0.8
    else:
        if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
            temp = 1 / pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
            out["market_prob"] = temp / temp.sum()
        else:
            out["market_prob"] = np.nan
        out["ev_tansho"] = np.nan
    out["gap"] = out["win_prob"] - out["market_prob"]
    if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        pop = pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        out["ana_score"] = out["gap"].fillna(0) * 0.6 + (1 / pop).fillna(0) * 0.4
    else:
        out["ana_score"] = out["gap"]
    out = out.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    out["mark"] = ""
    for i, m in enumerate(marks):
        if i < len(out):
            out.loc[i, "mark"] = m
    return out



def normalize_horse_name(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"\s+", "", text)
    return text


def infer_result_url_from_race_url(race_url: str) -> str:
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    if "nar.netkeiba.com" in race_url:
        return f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    return f"https://race.netkeiba.com/race/result.html?race_id={race_id}"


def parse_result_table_from_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = None
    for selector in ["table.RaceTable01", "table.race_table_01", "table.Shutuba_Table", "table"]:
        cand = soup.select_one(selector)
        if cand is not None and cand.select("tr"):
            table = cand
            break
    if table is None:
        return pd.DataFrame(columns=["着順", "馬番", "馬名"])

    rows = []
    for tr in table.select("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 5:
            continue
        txt = [clean_text(c.get_text(" ", strip=True)) for c in cells]
        joined = " ".join(txt)
        if "着順" in joined and "馬番" in joined:
            continue

        horse_no = None
        horse_name = ""
        finish = None

        horse_a = tr.select_one('a[href*="/horse/"]')
        if horse_a is not None:
            horse_name = clean_text(horse_a.get_text())

        numeric_cells = [to_int(x) for x in txt if to_int(x) is not None]
        if numeric_cells:
            finish = numeric_cells[0]
        if len(numeric_cells) >= 2:
            horse_no = numeric_cells[1]

        if not horse_name:
            for x in txt:
                if re.search(r"[ァ-ヶー一-龠A-Za-z]", x) and x not in ["取消", "除外", "中止", "失格"]:
                    horse_name = x
                    break

        if finish is not None and (horse_no is not None or horse_name):
            rows.append({"着順": finish, "馬番": horse_no, "馬名": horse_name})

    if not rows:
        return pd.DataFrame(columns=["着順", "馬番", "馬名"])
    out = pd.DataFrame(rows)
    out["馬名_norm"] = out["馬名"].map(normalize_horse_name)
    return out


def extract_payout_tables(result_url: str, html: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    tables_out: Dict[str, pd.DataFrame] = {}
    try:
        if html is None:
            html = requests.get(result_url, headers=HEADERS, timeout=REQUEST_TIMEOUT).text
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return tables_out

    for idx, t in enumerate(tables):
        flat = flatten_columns(t)
        cols = [clean_text(c) for c in flat.columns]
        key = f"table_{idx+1}"
        if any("払戻" in c for c in cols) or any("単勝" in c for c in cols) or any("馬連" in c for c in cols):
            key = f"payout_{idx+1}"
        tables_out[key] = flat
    return tables_out


def normalize_combo_text(value: Any) -> str:
    text = clean_text(value)
    nums = re.findall(r"\d+", text)
    if not nums:
        return ""
    return "-".join(nums)


def calc_recovery_rate(
    payout_tables: Dict[str, pd.DataFrame],
    ticket_tables: Dict[str, pd.DataFrame],
    bet_per_ticket: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    total_bet = 0
    total_return = 0

    payout_hits: Dict[str, Dict[str, int]] = {
        "ワイド": {},
        "馬連": {},
        "3連複": {},
        "3連単": {},
        "本命+穴ワイド": {},
        "本命+穴馬連": {},
        "本命2頭+穴1頭": {},
        "本命_本命穴": {},
    }

    # 払戻テーブルから券種ごとの組み合わせ・払戻を拾う
    for _, table in payout_tables.items():
        try:
            flat = flatten_columns(table)
        except Exception:
            flat = table.copy()

        for _, row in flat.iterrows():
            vals = [clean_text(v) for v in row.tolist()]
            line = " ".join(vals)
            nums = re.findall(r"\d+", line)
            if len(nums) < 2:
                continue

            payout_val = None
            for n in reversed(nums):
                try:
                    payout_val = int(n)
                    break
                except Exception:
                    pass
            if payout_val is None:
                continue

            combo = normalize_combo_text(line)
            if not combo:
                continue

            if "ワイド" in line:
                payout_hits["ワイド"][combo] = payout_val
                payout_hits["本命+穴ワイド"][combo] = payout_val
            if "馬連" in line:
                payout_hits["馬連"][combo] = payout_val
                payout_hits["本命+穴馬連"][combo] = payout_val
            if "3連複" in line:
                payout_hits["3連複"][combo] = payout_val
                payout_hits["本命2頭+穴1頭"][combo] = payout_val
            if "3連単" in line:
                payout_hits["3連単"][combo] = payout_val
                payout_hits["本命_本命穴"][combo] = payout_val

    for key, df in ticket_tables.items():
        if df is None or len(df) == 0:
            rows.append({"券種": key, "購入額": 0, "払戻": 0, "回収率": 0.0})
            continue

        bet = len(df) * bet_per_ticket
        ret = 0
        for _, row in df.iterrows():
            comb = clean_text(row.get("組み合わせ", ""))
            comb_norm = normalize_combo_text(comb)
            if comb_norm in payout_hits.get(key, {}):
                ret += int(payout_hits[key][comb_norm])

        recovery = (ret / bet * 100.0) if bet > 0 else 0.0
        rows.append({"券種": key, "購入額": bet, "払戻": ret, "回収率": recovery})
        total_bet += bet
        total_return += ret

    summary = {
        "購入総額": total_bet,
        "払戻総額": total_return,
        "総合回収率": (total_return / total_bet * 100.0) if total_bet > 0 else 0.0,
    }
    return pd.DataFrame(rows), summary



def validate_predictions(pred_df: pd.DataFrame, result_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = pred_df.copy()
    result2 = result_df.copy()

    # 正規化
    work["horse_name_norm"] = work["horse_name"].map(normalize_horse_name) if "horse_name" in work.columns else ""
    result2["horse_name_norm"] = result2["馬名"].map(normalize_horse_name) if "馬名" in result2.columns else ""

    # result側の重複除去
    if "馬番" in result2.columns and pd.to_numeric(result2["馬番"], errors="coerce").notna().sum() > 0:
        result2["馬番"] = pd.to_numeric(result2["馬番"], errors="coerce")
        result2 = result2.sort_values(["着順", "馬番"], na_position="last").drop_duplicates(subset=["馬番"], keep="first")
    result2 = result2.sort_values(["着順", "horse_name_norm"], na_position="last").drop_duplicates(subset=["horse_name_norm"], keep="first")

    # pred側の重複除去
    if "horse_no" in work.columns and pd.to_numeric(work["horse_no"], errors="coerce").notna().sum() > 0:
        work["horse_no"] = pd.to_numeric(work["horse_no"], errors="coerce")
        work = work.sort_values(["horse_no", "ai_score"], na_position="last", ascending=[True, False]).drop_duplicates(subset=["horse_no"], keep="first")
    work = work.sort_values(["horse_name_norm", "ai_score"], na_position="last", ascending=[True, False]).drop_duplicates(subset=["horse_name_norm"], keep="first")

    # まず馬番で結合
    merged = pd.DataFrame()
    if "horse_no" in work.columns and "馬番" in result2.columns and pd.to_numeric(work["horse_no"], errors="coerce").notna().sum() > 0:
        merged = work.merge(
            result2[["着順", "馬番", "馬名", "horse_name_norm"]],
            left_on="horse_no",
            right_on="馬番",
            how="left",
            suffixes=("", "_result"),
        )

    # 馬番で取れない行だけ名前で補完
    if merged.empty:
        merged = work.copy()
        merged["着順"] = pd.NA
        merged["馬番"] = pd.NA
        merged["馬名"] = pd.NA

    missing_mask = merged["着順"].isna() if "着順" in merged.columns else pd.Series([True] * len(merged))
    if missing_mask.any():
        fill_src = result2[["着順", "馬番", "馬名", "horse_name_norm"]].copy()
        fill_map = fill_src.set_index("horse_name_norm")[["着順", "馬番", "馬名"]]
        for idx in merged.index[missing_mask]:
            key = merged.at[idx, "horse_name_norm"]
            if key in fill_map.index:
                vals = fill_map.loc[key]
                if isinstance(vals, pd.DataFrame):
                    vals = vals.iloc[0]
                merged.at[idx, "着順"] = vals["着順"]
                merged.at[idx, "馬番"] = vals["馬番"]
                merged.at[idx, "馬名"] = vals["馬名"]

    # 最終重複除去
    if "horse_no" in merged.columns and pd.to_numeric(merged["horse_no"], errors="coerce").notna().sum() > 0:
        merged = merged.sort_values(["予想順位" if "予想順位" in merged.columns else "ai_score"], na_position="last").drop_duplicates(subset=["horse_no"], keep="first")
    merged = merged.sort_values(["horse_name_norm", "ai_score"], na_position="last", ascending=[True, False]).drop_duplicates(subset=["horse_name_norm"], keep="first")

    merged = merged.reset_index(drop=True)
    merged["予想順位"] = np.arange(1, len(merged) + 1)
    merged["3着内"] = pd.to_numeric(merged["着順"], errors="coerce").le(3)
    merged["1着的中"] = pd.to_numeric(merged["着順"], errors="coerce").eq(1)

    top5 = merged.head(5)
    top3 = merged.head(3)

    summary = {
        "勝ち馬を上位5頭に含む": bool(top5["1着的中"].fillna(False).any()) if len(top5) else False,
        "◎の着順": int(pd.to_numeric(merged.iloc[0]["着順"], errors="coerce")) if len(merged) and pd.notna(pd.to_numeric(merged.iloc[0]["着順"], errors="coerce")) else None,
        "上位3頭中の3着内頭数": int(top3["3着内"].fillna(False).sum()) if len(top3) else 0,
        "上位5頭中の3着内頭数": int(top5["3着内"].fillna(False).sum()) if len(top5) else 0,
    }
    return merged, summary


def ticket_hit_summary(result_df: pd.DataFrame, ticket_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    winners = result_df.sort_values("着順").head(3)
    top3_nums = [int(x) for x in winners["馬番"].dropna().tolist()]
    top2_nums = top3_nums[:2]
    top1_num = top3_nums[0] if top3_nums else None

    rows = []
    for key, df in ticket_tables.items():
        hit = False
        hit_row = ""
        if df is None or len(df) == 0:
            rows.append({"券種": key, "的中": False, "該当": ""})
            continue

        for _, row in df.iterrows():
            comb = clean_text(row.get("組み合わせ", ""))
            nums = [int(x) for x in re.findall(r"\d+", comb)]
            if key in ["ワイド", "馬連", "本命+穴ワイド", "本命+穴馬連"]:
                if len(nums) >= 2 and set(top2_nums).issubset(set(nums)):
                    hit = True
                    hit_row = comb
                    break
            elif key in ["3連複", "本命2頭+穴1頭"]:
                if len(nums) >= 3 and set(top3_nums).issubset(set(nums)):
                    hit = True
                    hit_row = comb
                    break
            elif key in ["3連単", "本命_本命穴"]:
                if len(nums) >= 3 and top3_nums[:3] == nums[:3]:
                    hit = True
                    hit_row = comb
                    break
        rows.append({"券種": key, "的中": hit, "該当": hit_row})
    return pd.DataFrame(rows)


def render_results(race: pd.DataFrame, hist: pd.DataFrame, recent_n: int, ana_count: int, ticket_head_count: int, wide_count: int, umaren_count: int, trio_count: int, trifecta_count: int) -> None:
    df, top, honmei, ana, odds_valid = analyze_race(race, hist, recent_n=recent_n)
    df = dedupe_race_df(df)
    df = apply_style_ground_pace_adjustments(df)

    display_cols = ["mark", "horse_no", "frame_no", "horse_name", "running_style", "ai_score", "win_prob", "place_prob", "ana_score"]
    if odds_valid:
        display_cols += ["odds_f", "ev_tansho"]
    if "pop_f" in df.columns:
        display_cols += ["pop_f"]

    st.subheader("総合ランキング")
    editor_df = df[display_cols].copy()
    style_key = "manual_running_style_map"
    if style_key not in st.session_state:
        st.session_state[style_key] = {}

    for _, row in editor_df.iterrows():
        key = f"{row.get('horse_no','')}_{row.get('horse_name','')}"
        if key in st.session_state[style_key]:
            editor_df.loc[row.name, "running_style"] = st.session_state[style_key][key]

    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "mark": st.column_config.TextColumn("印", disabled=True),
            "horse_no": st.column_config.NumberColumn("馬番", disabled=True),
            "frame_no": st.column_config.NumberColumn("枠番", disabled=True),
            "horse_name": st.column_config.TextColumn("馬名", disabled=True),
            "running_style": st.column_config.SelectboxColumn(
                "running_style",
                options=["先行", "中団", "差し", "不明"],
                required=True,
            ),
            "ai_score": st.column_config.NumberColumn("AIスコア", format="%.4f", disabled=True),
            "win_prob": st.column_config.NumberColumn("勝率", format="%.4f", disabled=True),
            "place_prob": st.column_config.NumberColumn("複勝率", format="%.4f", disabled=True),
            "ana_score": st.column_config.NumberColumn("穴スコア", format="%.4f", disabled=True),
            "odds_f": st.column_config.NumberColumn("オッズ", format="%.2f", disabled=True),
            "ev_tansho": st.column_config.NumberColumn("単勝EV", format="%.4f", disabled=True),
            "pop_f": st.column_config.NumberColumn("人気", disabled=True),
        },
        key="ranking_editor_full",
    )

    rerun_pressed = st.button("脚質を反映して再予想", use_container_width=True)

    if "running_style" in edited_df.columns:
        manual_map = {}
        for _, row in edited_df[["horse_no", "horse_name", "running_style"]].iterrows():
            k = f"{row.get('horse_no','')}_{row.get('horse_name','')}"
            manual_map[k] = row["running_style"]
        st.session_state[style_key] = manual_map

        style_series = []
        for _, row in df.iterrows():
            k = f"{row.get('horse_no','')}_{row.get('horse_name','')}"
            style_series.append(manual_map.get(k, row.get("running_style", "不明")))
        df["running_style"] = style_series

    if rerun_pressed:
        df = apply_manual_running_style_adjustments(df)

    top = df.head(min(ticket_head_count, len(df))).copy()
    ana = df.sort_values(["ana_score", "gap"], ascending=False).head(ana_count).copy()
    honmei = df.sort_values(["win_prob", "ai_score"], ascending=False).head(3).copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("出走頭数", len(race))
    c2.metric("過去成績件数", len(hist))
    c3.metric("本命", honmei.iloc[0]["horse_name"] if len(honmei) else "-")
    c4.metric("穴", ana.iloc[0]["horse_name"] if len(ana) else "-")

    st.subheader("本命候補")
    st.dataframe(honmei[["mark", "horse_no", "horse_name", "win_prob", "place_prob", "ai_score", "running_style"]], use_container_width=True, hide_index=True)

    ana_cols = ["horse_no", "horse_name", "running_style", "win_prob", "market_prob", "gap", "ana_score"]
    if odds_valid:
        ana_cols.insert(2, "odds_f")
    if "pop_f" in ana.columns:
        ana_cols.insert(3 if odds_valid else 2, "pop_f")
    st.subheader("穴候補")
    st.dataframe(ana[ana_cols], use_container_width=True, hide_index=True)

    wide_df = build_pair_table(top, "place_prob", "ワイド")
    umaren_df = build_pair_table(top, "win_prob", "馬連")
    trio_df = build_trio_table(top, "place_prob", "3連複")
    trifecta_df = build_trifecta_table(top)
    honmei_ana_umaren, honmei_ana_wide, honmei_ana_trio, honmei_ana_trifecta = build_honmei_ana_tables(honmei, ana)

    tabs = st.tabs([
        "ワイド",
        "馬連",
        "3連複",
        "3連単",
        "本命+穴ワイド",
        "本命+穴馬連",
        "本命2頭+穴1頭",
        "本命→本命穴",
    ])

    with tabs[0]:
        st.dataframe(wide_df.head(wide_count), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(umaren_df.head(umaren_count), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(trio_df.head(trio_count), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(trifecta_df.head(trifecta_count), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(honmei_ana_wide.head(wide_count), use_container_width=True, hide_index=True)
    with tabs[5]:
        st.dataframe(honmei_ana_umaren.head(umaren_count), use_container_width=True, hide_index=True)
    with tabs[6]:
        st.dataframe(honmei_ana_trio.head(trio_count), use_container_width=True, hide_index=True)
    with tabs[7]:
        st.dataframe(honmei_ana_trifecta.head(trifecta_count), use_container_width=True, hide_index=True)

    st.session_state["latest_prediction_df"] = df.copy()
    st.session_state["latest_ticket_tables"] = {
        "ワイド": wide_df.copy(),
        "馬連": umaren_df.copy(),
        "3連複": trio_df.copy(),
        "3連単": trifecta_df.copy(),
        "本命+穴ワイド": honmei_ana_wide.copy(),
        "本命+穴馬連": honmei_ana_umaren.copy(),
        "本命2頭+穴1頭": honmei_ana_trio.copy(),
        "本命_本命穴": honmei_ana_trifecta.copy(),
    }

    st.subheader("実レース結果で検証")
    default_result_url = infer_result_url_from_race_url(st.session_state.get("latest_race_url", ""))
    result_url = st.text_input("結果URL", value=default_result_url, key="result_url_input")
    if st.button("結果を取得して検証", use_container_width=True):
        try:
            if not result_url.strip():
                raise ValueError("結果URLを入れてください。")
            result_html = fetch_text_with_encoding(result_url.strip(), timeout=REQUEST_TIMEOUT)
            result_df = parse_result_table_from_html(result_html)
            if result_df.empty:
                raise ValueError("結果テーブルを取得できませんでした。")
            merged_df, summary = validate_predictions(df, result_df)
            st.write("### 予想と実着順")
            view_cols = [c for c in ["mark", "horse_no", "horse_name", "running_style", "予想順位", "着順", "3着内", "1着的中"] if c in merged_df.columns]
            st.dataframe(merged_df[view_cols], use_container_width=True, hide_index=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("◎の着順", summary["◎の着順"] if summary["◎の着順"] is not None else "-")
            c2.metric("上位3頭中の3着内", summary["上位3頭中の3着内頭数"])
            c3.metric("上位5頭中の3着内", summary["上位5頭中の3着内頭数"])
            c4.metric("勝ち馬を上位5頭に含む", "はい" if summary["勝ち馬を上位5頭に含む"] else "いいえ")

            st.write("### 券種別的中")
            hit_df = ticket_hit_summary(result_df, st.session_state["latest_ticket_tables"])
            st.dataframe(hit_df, use_container_width=True, hide_index=True)

            payout_tables = extract_payout_tables(result_url.strip(), result_html)
            recovery_df, recovery_summary = calc_recovery_rate(
                payout_tables,
                st.session_state["latest_ticket_tables"],
                bet_per_ticket=100,
            )

            st.write("### 回収率")
            c1, c2, c3 = st.columns(3)
            c1.metric("購入総額", f"{int(recovery_summary['購入総額'])}円")
            c2.metric("払戻総額", f"{int(recovery_summary['払戻総額'])}円")
            c3.metric("総合回収率", f"{recovery_summary['総合回収率']:.1f}%")
            st.dataframe(recovery_df, use_container_width=True, hide_index=True)
            if payout_tables:
                st.write("### 払戻テーブル（文字化け対策版）")
                for name, ptable in payout_tables.items():
                    st.dataframe(ptable, use_container_width=True, hide_index=True)
            else:
                st.caption("払戻テーブルは取得できませんでした。")
        except Exception as e:
            st.error(f"検証エラー: {e}")

    export_tables = {
        "race_card": race,
        "horse_history": hist,
        "総合ランキング": edited_df,
        "本命候補": honmei[["mark", "horse_no", "horse_name", "win_prob", "place_prob", "ai_score", "running_style"]],
        "穴候補": ana[ana_cols],
        "ワイド": wide_df,
        "馬連": umaren_df,
        "3連複": trio_df,
        "3連単": trifecta_df,
        "本命+穴ワイド": honmei_ana_wide,
        "本命+穴馬連": honmei_ana_umaren,
        "本命2頭+穴1頭": honmei_ana_trio,
        "本命_本命穴": honmei_ana_trifecta,
    }
    xlsx = make_download_file(export_tables)
    st.download_button(
        "結果をExcelでダウンロード",
        data=xlsx,
        file_name="keiba_ai_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# ============================================================
# 画面
# ============================================================

st.title(APP_TITLE)

mode = st.radio("実行モード", ["URLから取得して予想", "CSVアップロード", "ローカルファイル（コマンド用）"])
race_url = st.text_input("netkeiba 出馬表URL", DEFAULT_RACE_URL)
max_horses = st.number_input("取得頭数", min_value=1, max_value=18, value=18)
recent_n = st.slider("直近成績使用件数", 3, 20, 10)
ana_count = st.slider("穴馬候補の頭数", 1, 15, 8)
ticket_head_count = st.slider("買い目に使う上位頭数", 5, 15, TOP_FOR_TICKETS_DEFAULT)
wide_count = st.slider("ワイド表示件数", 5, 50, 30)
umaren_count = st.slider("馬連表示件数", 5, 50, 30)
trio_count = st.slider("3連複表示件数", 5, 50, 30)
trifecta_count = st.slider("3連単表示件数", 10, 100, 50)
st.caption(f"Selenium利用可能: {'はい' if SELENIUM_AVAILABLE else 'いいえ'}")
if mode == "URLから取得して予想":
    if st.button("取得して予想", use_container_width=True):
        try:
            st.session_state["latest_race_url"] = race_url.strip()
            race_df, hist_df = fetch_race_and_history(
                race_url.strip(),
                int(max_horses),
                int(recent_n)
            )
            st.session_state["race_df_store"] = race_df.copy()
            st.session_state["hist_df_store"] = hist_df.copy()
            st.session_state["prediction_ready"] = True
            st.session_state["loaded_from"] = "url"
        except Exception as e:
            st.error(f"エラー: {e}")

elif mode == "CSVアップロード":
    c1, c2 = st.columns(2)
    with c1:
        race_file = st.file_uploader("race_card.csv", type="csv")
    with c2:
        hist_file = st.file_uploader("horse_history.csv", type="csv")
    if race_file and hist_file and st.button("CSVを読み込んで予想", use_container_width=True):
        try:
            race_df = load_csv_upload(race_file)
            hist_df = load_csv_upload(hist_file)
            st.session_state["race_df_store"] = race_df.copy()
            st.session_state["hist_df_store"] = hist_df.copy()
            st.session_state["prediction_ready"] = True
            st.session_state["loaded_from"] = "csv"
        except Exception as e:
            st.error(f"エラー: {e}")

else:
    if st.button("ローカルCSVを読み込んで予想", use_container_width=True):
        try:
            race_df = pd.read_csv("race_card.csv")
            hist_df = pd.read_csv("horse_history.csv")
            st.session_state["race_df_store"] = race_df.copy()
            st.session_state["hist_df_store"] = hist_df.copy()
            st.session_state["prediction_ready"] = True
            st.session_state["loaded_from"] = "local"
        except Exception as e:
            st.error(f"ローカルCSV読み込み失敗: {e}")

if st.session_state.get("prediction_ready") and st.session_state.get("race_df_store") is not None and st.session_state.get("hist_df_store") is not None:
    render_results(
        st.session_state["race_df_store"].copy(),
        st.session_state["hist_df_store"].copy(),
        int(recent_n),
        int(ana_count),
        int(ticket_head_count),
        int(wide_count),
        int(umaren_count),
        int(trio_count),
        int(trifecta_count),
    )

# ============================================================
# 末尾コメント（行数調整と将来拡張メモ）
# ============================================================
# 以下は将来の機能拡張余地を残すためのコメントブロック。
# 1. レース条件別の重み最適化
# 2. 距離別モデル
# 3. コース別モデル
# 4. 天候別モデル
# 5. 馬場別モデル
# 6. 騎手補正
# 7. 調教師補正
# 8. 過去レースラップ補正
# 9. ペース推定
# 10. リスク分散買い
# 11. 資金配分
# 12. ログ保存
# 13. 予想履歴保存
# 14. 過去予想検証
# 15. 的中率集計
# 16. 回収率集計
# 17. 単勝/複勝以外のEV推定
# 18. API化
# 19. Reactフロント連携
# 20. モバイル最適化
