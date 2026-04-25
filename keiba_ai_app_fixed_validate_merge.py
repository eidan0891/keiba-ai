from __future__ import annotations
import html
import traceback
import json
from datetime import datetime, date, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Iterable
from pathlib import Path
from itertools import combinations, permutations
from collections import defaultdict
import os
import re
import io
import math
import time
import random
import warnings
import pandas as pd
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    joblib = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = False
except Exception:
    webdriver = None
    Options = None
    WebDriverWait = None
    SELENIUM_AVAILABLE = False

# ===== 安定化デフォルト =====
REQUEST_TIMEOUT = 10
REQUEST_SLEEP = 1.0
APP_TITLE = "にゃんこ競馬AI予想 完全安定版"
DEFAULT_RACE_URL = ""
LOG_FILE = "race_log.csv"
HONMEI_COUNT_DEFAULT = 6
ANA_COUNT_DEFAULT = 6
ANA_POP_MIN_DEFAULT = 4
ANA_ODDS_MIN_DEFAULT = 10.0
ANA_GAP_MIN_DEFAULT = 0.0
TOP_FOR_TICKETS_DEFAULT = 7
WIDE_COUNT_DEFAULT = 10
UMAREN_COUNT_DEFAULT = 10
TRIO_COUNT_DEFAULT = 10
TRIFECTA_COUNT_DEFAULT = 10
ANA_COUNT_UI_DEFAULT = 6
RECENT_N_DEFAULT = 5
MAX_HORSES_DEFAULT = 18
JP_COLUMNS = {}
JP_EDITABLE = []
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

def safe_app_warning(msg: str):
    try:
        st.warning(msg)
    except Exception:
        pass

def safe_app_error(msg: str):
    try:
        st.error(msg)
    except Exception:
        pass

import html
import traceback
import json
from datetime import datetime, date, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from itertools import combinations, permutations
import os
import re
import io
import math
import time
import random
import warnings
import pandas as pd
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    joblib = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False

try:
    from selenium import webdriver
    SELENIUM_AVAILABLE = True
except Exception:
    webdriver = None
    SELENIUM_AVAILABLE = False


# ============================================================
# 安全化ユーティリティ
# ============================================================
# 共通タイムアウト定数
REQUEST_TIMEOUT = 10


# 未定義防止の共通デフォルト
REQUEST_SLEEP = 1.0
DEFAULT_RACE_URL = ""

# ===== 全体デフォルト定義（未定義落ち防止）=====
APP_TITLE = "にゃんこ競馬AI予想 完全版"
LOG_FILE = "race_log.csv"
# =============================================


# ===== 全体デフォルト定義（未定義落ち防止・最終版）=====
TOP_FOR_TICKETS_DEFAULT = 7
WIDE_COUNT_DEFAULT = 10
UMAREN_COUNT_DEFAULT = 10
TRIO_COUNT_DEFAULT = 10
TRIFECTA_COUNT_DEFAULT = 10
ANA_COUNT_UI_DEFAULT = 6
RECENT_N_DEFAULT = 5
MAX_HORSES_DEFAULT = 18
# =======================================================


# ===== 最終安全デフォルト =====
JP_COLUMNS = {}
JP_EDITABLE = []
# =============================


def _safe_df(df: Any) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        out = df.copy()
    else:
        try:
            out = pd.DataFrame(df)
        except Exception:
            out = pd.DataFrame()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x) for x in col if str(x) and str(x) != "nan"]).strip("_")
            for col in out.columns
        ]
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")].copy()
    return out

def _safe_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    df = _safe_df(df)
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    v = df[col]
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0] if v.shape[1] else pd.Series([default] * len(df), index=df.index)
    if not isinstance(v, pd.Series):
        v = pd.Series([v] * len(df), index=df.index)
    if len(v) != len(df):
        vals = list(v)[:len(df)] + [default] * max(0, len(df) - len(v))
        v = pd.Series(vals, index=df.index)
    else:
        v = pd.Series(v.to_numpy(), index=df.index)
    return v

def _safe_num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    return pd.to_numeric(_safe_series(df, col, default), errors="coerce").fillna(default)

def _safe_text(df: pd.DataFrame, col: str, default="") -> pd.Series:
    return _safe_series(df, col, default).fillna(default).astype(str)

def safe_pick_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = _safe_df(df)
    use = [c for c in cols if c in out.columns]
    return out[use].copy() if use else out.copy()

def safe_numeric(df, col):
    return _safe_num(df, col, default=np.nan)

from dataclasses import dataclass
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Selenium利用可否フラグ（未導入環境でも落とさない）
try:
    from selenium import webdriver
    SELENIUM_AVAILABLE = True
except Exception:
    webdriver = None
    SELENIUM_AVAILABLE = False

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from itertools import combinations, permutations
import os
import re
import io
import random
import math
import time
import pandas as pd
import numpy as np

import pandas as pd

def safe_numeric(df, col):
    """
    必ず df と同じ長さの 1次元 Series を返す。
    列なし / 同名列重複 / scalar / None / 長さ不一致を吸収。
    """
    n = len(df)

    if col not in df.columns:
        return pd.Series([None] * n, index=df.index)

    val = df[col]

    if isinstance(val, pd.DataFrame):
        if val.shape[1] == 0:
            return pd.Series([None] * n, index=df.index)
        val = val.iloc[:, 0]

    if not isinstance(val, pd.Series):
        val = pd.Series([val] * n, index=df.index)

    if len(val) != n:
        vals = list(val)[:n] + [None] * max(0, n - len(val))
        val = pd.Series(vals, index=df.index)
    else:
        val = pd.Series(val.to_numpy(), index=df.index)

    try:
        return pd.to_numeric(val, errors="coerce")
    except Exception:
        return pd.Series([None] * n, index=df.index)

def init_log() -> None:
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "date", "race", "mode", "bet_type", "buy", "payout", "hit", "return_rate"
        ]).to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

def save_log(race_name: str, mode: str, bet_type: str, buy: int, payout: int, hit: bool) -> None:
    init_log()
    try:
        df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=["date", "race", "mode", "bet_type", "buy", "payout", "hit", "return_rate"])

    rr = (payout / buy) if buy > 0 else 0.0
    new = pd.DataFrame([{
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "race": race_name,
        "mode": mode,
        "bet_type": bet_type,
        "buy": int(buy),
        "payout": int(payout),
        "hit": bool(hit),
        "return_rate": float(rr),
    }])

    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

def load_log_summary():
    init_log()
    try:
        df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=["date", "race", "mode", "bet_type", "buy", "payout", "hit", "return_rate"])

    if len(df) == 0:
        return 0, 0, 0.0, df

    total_buy = int(pd.to_numeric(df["buy"], errors="coerce").fillna(0).sum())
    total_payout = int(pd.to_numeric(df["payout"], errors="coerce").fillna(0).sum())
    total_rr = (total_payout / total_buy * 100) if total_buy > 0 else 0.0
    return total_buy, total_payout, total_rr, df


def summarize_log_by_bet_type(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["券種", "件数", "的中率", "回収率", "収支"])
    work = df.copy()
    work["buy"] = pd.to_numeric(work["buy"], errors="coerce").fillna(0)
    work["payout"] = pd.to_numeric(work["payout"], errors="coerce").fillna(0)
    work["hit"] = work["hit"].astype(str).isin(["True", "true", "1", "当", "○"]) | work["hit"].astype(bool)
    rows = []
    for bet_type, g in work.groupby("bet_type", dropna=False):
        buy = float(g["buy"].sum())
        payout = float(g["payout"].sum())
        rows.append({
            "券種": str(bet_type),
            "件数": int(len(g)),
            "的中率": round(float(g["hit"].mean()) * 100, 1) if len(g) else 0.0,
            "回収率": round((payout / buy * 100) if buy > 0 else 0.0, 1),
            "収支": int(payout - buy),
        })
    return pd.DataFrame(rows).sort_values(["回収率", "件数"], ascending=[False, False]).reset_index(drop=True)


def _prep_adv_df(df: pd.DataFrame) -> pd.DataFrame:
    work = _safe_df(df)
    for col in ["win_prob", "place_prob", "gap", "ana_score", "odds_f", "pop_f", "top5_prob"]:
        work[col] = _safe_num(work, col, default=0.0).to_numpy()
    work["running_style"] = _safe_text(work, "running_style", "不明").to_numpy()
    if "horse_no" not in work.columns:
        work["horse_no"] = np.arange(1, len(work) + 1)
    else:
        nums = _safe_num(work, "horse_no", default=0)
        fixed = []
        for i, v in enumerate(nums.tolist(), start=1):
            try:
                iv = int(v)
                fixed.append(iv if iv > 0 else i)
            except Exception:
                fixed.append(i)
        work["horse_no"] = fixed
    if "horse_name" not in work.columns:
        work["horse_name"] = [f"馬{x}" for x in range(1, len(work) + 1)]
    else:
        names = _safe_text(work, "horse_name", "")
        defaults = pd.Series([f"馬{x}" for x in range(1, len(work) + 1)], index=work.index)
        work["horse_name"] = names.replace("", np.nan).fillna(defaults).to_numpy()
    return work

def _style_balance_bonus(styles: set) -> float:
    s = {str(x) for x in styles if str(x) not in ["", "不明", "nan", "None"]}
    if len(s) >= 3:
        return 0.08
    if len(s) == 2:
        return 0.04
    return -0.02


def build_trio_table_strong(top_df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    work = _prep_adv_df(top_df)
    if len(work) < 3:
        return pd.DataFrame(columns=["券種", "組み合わせ", "馬1", "馬2", "馬3", "score"])
    rows = []
    for i, j, k in combinations(range(len(work)), 3):
        a, b, c = work.iloc[i], work.iloc[j], work.iloc[k]
        styles = {a.get("running_style", "不明"), b.get("running_style", "不明"), c.get("running_style", "不明")}
        try:
            score = (
                float(a.get("place_prob", 0)) * 0.9 + float(b.get("place_prob", 0)) * 0.9 + float(c.get("place_prob", 0)) * 0.9
                + (max(float(a.get("gap", 0)), 0.0) + max(float(b.get("gap", 0)), 0.0) + max(float(c.get("gap", 0)), 0.0)) * 1.4
                + (float(a.get("win_prob", 0)) + float(b.get("win_prob", 0)) + float(c.get("win_prob", 0))) * 0.6
                + _style_balance_bonus(styles)
            )
        except Exception:
            score = 0.0
        rows.append({
            "券種": "3連複",
            "組み合わせ": f"{int(a.get('horse_no', i+1))}-{int(b.get('horse_no', j+1))}-{int(c.get('horse_no', k+1))}",
            "馬1": str(a.get("horse_name", "")),
            "馬2": str(b.get("horse_name", "")),
            "馬3": str(c.get("horse_name", "")),
            "score": score,
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(int(limit)).reset_index(drop=True)

def build_trifecta_table_strong(top_df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    work = _prep_adv_df(top_df)
    if len(work) < 3:
        return pd.DataFrame(columns=["組み合わせ", "1着", "2着", "3着", "score"])
    rows = []
    for i, j, k in permutations(range(len(work)), 3):
        a, b, c = work.iloc[i], work.iloc[j], work.iloc[k]
        styles = {a.get("running_style", "不明"), b.get("running_style", "不明"), c.get("running_style", "不明")}
        try:
            score = (
                float(a.get("win_prob", 0)) * 1.8
                + float(b.get("place_prob", 0)) * 0.95
                + float(c.get("top5_prob", 0)) * 0.75
                + max(float(c.get("gap", 0)), 0.0) * 1.5
                + _style_balance_bonus(styles)
            )
            if max(float(a.get("pop_f", 0)), float(b.get("pop_f", 0)), float(c.get("pop_f", 0))) >= 6:
                score += 0.03
        except Exception:
            score = 0.0
        rows.append({
            "組み合わせ": f"{int(a.get('horse_no', i+1))}→{int(b.get('horse_no', j+1))}→{int(c.get('horse_no', k+1))}",
            "1着": str(a.get("horse_name", "")),
            "2着": str(b.get("horse_name", "")),
            "3着": str(c.get("horse_name", "")),
            "score": score,
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(int(limit)).reset_index(drop=True)

def _normalize_training_df(df: pd.DataFrame) -> pd.DataFrame:
    work = _safe_df(df)
    ren = {}
    for c in list(work.columns):
        s = str(c).strip()
        if s in ["着順", "rank"]:
            ren[c] = "rank"
        elif s in ["人気", "pop", "pop_f", "popularity"]:
            ren[c] = "pop"
        elif s in ["オッズ", "odds", "odds_f", "単勝"]:
            ren[c] = "odds"
        elif s in ["後3F", "上がり3F", "上り3F", "last3f"]:
            ren[c] = "last3f"
        elif s in ["馬場状態", "馬場", "ground"]:
            ren[c] = "ground"
        elif s in ["天候", "weather"]:
            ren[c] = "weather"
    work = _safe_df(work.rename(columns=ren))
    for col in ["rank", "pop", "odds", "last3f"]:
        work[col] = _safe_num(work, col, default=np.nan).to_numpy()
    work["ground"] = _safe_text(work, "ground", "").to_numpy()
    work["weather"] = _safe_text(work, "weather", "").to_numpy()
    work["ground_code"] = pd.Series(work["ground"]).astype(str).map({"良": 0, "稍重": 1, "重": 2, "不良": 3}).fillna(0).to_numpy()
    work["weather_code"] = pd.Series(work["weather"]).astype(str).map({"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}).fillna(0).to_numpy()
    return work

def train_odds_linked_models(df: pd.DataFrame, model_dir: str = "models") -> Dict[str, str]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn / joblib が必要です。")
    work = _normalize_training_df(df)
    use = work.dropna(subset=["rank"]).copy()
    if len(use) < 5:
        raise ValueError("学習データが少なすぎます。最低5行以上、できれば20行以上のCSVを使ってください。")
    feature_cols = ["odds", "pop", "last3f", "ground_code", "weather_code"]
    for c in feature_cols:
        if c not in use.columns:
            use[c] = 0
    X = use[feature_cols].fillna(0)
    y_top3 = (pd.to_numeric(use["rank"], errors="coerce").fillna(999) <= 3).astype(int)
    y_rank = pd.to_numeric(use["rank"], errors="coerce").fillna(999).astype(float)
    if y_top3.nunique() < 2:
        y_top3 = pd.Series([1 if i % 2 == 0 else 0 for i in range(len(use))], index=use.index)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X, y_top3)
    reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    reg.fit(X, y_rank)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    top3_path = model_path / "top3_model.joblib"
    rank_path = model_path / "rank_model.joblib"
    meta_path = model_path / "model_meta.joblib"
    joblib.dump(clf, top3_path)
    joblib.dump(reg, rank_path)
    joblib.dump({"feature_cols": feature_cols}, meta_path)
    return {"top3_model": str(top3_path), "rank_model": str(rank_path), "meta": str(meta_path)}

def load_trained_models(model_dir: str = "models") -> Optional[Dict[str, object]]:
    if not SKLEARN_AVAILABLE:
        return None
    try:
        model_path = Path(model_dir)
        top3_path = model_path / "top3_model.joblib"
        rank_path = model_path / "rank_model.joblib"
        meta_path = model_path / "model_meta.joblib"
        if not top3_path.exists() or not rank_path.exists() or not meta_path.exists():
            return None
        meta = joblib.load(meta_path)
        return {
            "top3_model": joblib.load(top3_path),
            "rank_model": joblib.load(rank_path),
            "feature_cols": meta.get("feature_cols", ["odds", "pop", "last3f", "ground_code", "weather_code"]),
        }
    except Exception:
        return None

def apply_trained_models_to_prediction_df(df: pd.DataFrame, model_bundle: Optional[Dict[str, object]]) -> pd.DataFrame:
    out = _safe_df(df)
    out["model_top3_prob"] = _safe_num(out, "model_top3_prob", default=np.nan).to_numpy()
    out["model_rank_pred"] = _safe_num(out, "model_rank_pred", default=np.nan).to_numpy()
    out["model_value_score"] = _safe_num(out, "model_value_score", default=np.nan).to_numpy()
    if model_bundle is None or len(out) == 0:
        return out
    try:
        work = _normalize_training_df(out)
        feature_cols = model_bundle.get("feature_cols", ["odds", "pop", "last3f", "ground_code", "weather_code"])
        for c in feature_cols:
            if c not in work.columns:
                work[c] = 0
        X = work[feature_cols].fillna(0)
        top3_model = model_bundle.get("top3_model")
        rank_model = model_bundle.get("rank_model")
        if top3_model is not None:
            try:
                out["model_top3_prob"] = top3_model.predict_proba(X)[:, 1]
            except Exception:
                out["model_top3_prob"] = top3_model.predict(X)
        if rank_model is not None:
            out["model_rank_pred"] = rank_model.predict(X)
        odds_col = "odds_f" if "odds_f" in out.columns else ("odds" if "odds" in out.columns else None)
        if odds_col:
            odds_num = _safe_num(out, odds_col, default=0)
            out["model_value_score"] = pd.to_numeric(out["model_top3_prob"], errors="coerce").fillna(0).to_numpy() * odds_num.to_numpy()
        else:
            out["model_value_score"] = pd.to_numeric(out["model_top3_prob"], errors="coerce").fillna(0).to_numpy()
    except Exception:
        return out
    return out

def fetch_text_with_encoding(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return decode_response_html(resp, url)
    except Exception as e:
        safe_app_warning(f"取得失敗: {e}")
        return ""

def decode_response_html(resp: requests.Response, url: str = "") -> str:
    raw = resp.content

    # NAR / 地方競馬は Shift_JIS 系優先
    if "nar.netkeiba.com" in url:
        best = ""
        best_score = -10**9
        for enc in ["cp932", "shift_jis", "euc_jp", "utf-8"]:
            try:
                txt = raw.decode(enc, errors="replace")
            except Exception:
                continue

            score = (
                txt.count("払戻") * 40
                + txt.count("単勝") * 20
                + txt.count("複勝") * 20
                + txt.count("馬連") * 20
                + txt.count("ワイド") * 20
                + txt.count("3連複") * 20
                + txt.count("3連単") * 20
                + txt.count("着順") * 20
                + txt.count("馬名") * 20
                + sum(1 for c in txt if ("\u4e00" <= c <= "\u9fff") or ("\u3040" <= c <= "\u30ff")) * 2
                - txt.count("ｽ") * 25
                - txt.count("郢") * 25
                - txt.count(" ") * 25
            )
            if score > best_score:
                best_score = score
                best = txt
        if best:
            return best

    # 通常ページ
    best = ""
    best_score = -10**9
    for enc in [getattr(resp, "apparent_encoding", None), resp.encoding, "cp932", "shift_jis", "euc_jp", "utf-8"]:
        try:
            if not enc:
                continue
            txt = raw.decode(enc, errors="replace")
        except Exception:
            continue

        score = (
            txt.count("着順") * 20
            + txt.count("馬名") * 20
            + sum(1 for c in txt if ("\u4e00" <= c <= "\u9fff") or ("\u3040" <= c <= "\u30ff")) * 2
            - txt.count("ｽ") * 20
            - txt.count("郢") * 20
            - txt.count(" ") * 20
        )
        if score > best_score:
            best_score = score
            best = txt

    return best if best else raw.decode("utf-8", errors="replace")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}



# ---- 未定義落ち防止の保険値 ----
payout_tables = {}
tables_out = {}
bet_per_ticket = 100


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown("""
<style>
:root{
  --nk-primary:#4f7cff;
  --nk-primary-2:#7a5cff;
  --nk-border:#e5e7eb;
  --nk-soft:#f8fafc;
  --nk-text:#0f172a;
  --nk-sub:#6b7280;
}
.block-container{padding-top:0.8rem !important; padding-bottom:0.9rem !important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fbfdff 0%,#f3f6fb 100%); border-right:1px solid #e5e7eb;}
[data-testid="stSidebar"] .block-container{padding-top:1rem;}
.nk-settings-title{font-size:2.4rem;font-weight:800;color:var(--nk-text);margin:0 0 .2rem 0;line-height:1.55 !important;overflow:visible !important;padding-top:.5rem;}
.nk-settings-sub{color:var(--nk-sub);margin:0 0 1rem 0;}
.nk-top-actions{display:flex;gap:12px;justify-content:flex-end;}
.nk-card{border:1px solid var(--nk-border);background:#fff;border-radius:18px;padding:18px 18px;box-shadow:0 8px 22px rgba(15,23,42,.04);}
.nk-card-title{font-size:1.2rem;font-weight:800;color:var(--nk-text);margin-bottom:.15rem;}
.nk-card-sub{font-size:.92rem;color:var(--nk-sub);margin-bottom:.75rem;}
.nk-mini{font-size:.88rem;color:var(--nk-sub);}
.nk-side-card{border:1px solid #dbe5ff;background:#fff;border-radius:16px;padding:14px 14px;margin:8px 0 14px 0;box-shadow:0 6px 18px rgba(79,124,255,.06);}
.nk-side-label{font-weight:800;color:#111827;margin-bottom:.25rem;}
.nk-side-sub{font-size:.88rem;color:#6b7280;}
div[data-testid="stCheckbox"], div[data-testid="stSelectbox"], div[data-testid="stSlider"], div[data-testid="stRadio"], div[data-testid="stTextInput"], div[data-testid="stNumberInput"], div[data-testid="stFileUploader"]{
  background:#fff;border:1px solid var(--nk-border);border-radius:14px;padding:10px 12px;
}
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button{
  border-radius:12px;border:1px solid #d7e3ff;background:#fff;color:#1f2937;font-weight:700;
}
div[data-testid="stButton"] > button[kind="primary"]{
  background:linear-gradient(90deg,var(--nk-primary),var(--nk-primary-2)); color:#fff; border:none;
}
</style>
""", unsafe_allow_html=True)

if "race_df_store" not in st.session_state:
    st.session_state["race_df_store"] = None
if "hist_df_store" not in st.session_state:
    st.session_state["hist_df_store"] = None
if "prediction_ready" not in st.session_state:
    st.session_state["prediction_ready"] = False
if "loaded_from" not in st.session_state:
    st.session_state["loaded_from"] = ""


# ============================================================
# UI状態同期
# ============================================================
def _apply_quick_profile(profile: str) -> None:
    profile = clean_text(profile) or "カスタム"
    st.session_state["rr_quick_profile"] = profile
    if profile == "的中率":
        st.session_state["rr_predict_mode"] = "的中率重視"
        st.session_state["rr_honmei_count"] = 5
        st.session_state["rr_ana_count_logic"] = 3
        st.session_state["rr_ana_pop_min"] = 6
        st.session_state["rr_ana_odds_min"] = 10.0
        st.session_state["rr_ana_gap_min"] = 0.02
    elif profile == "バランス":
        st.session_state["rr_predict_mode"] = "バランス"
        st.session_state["rr_honmei_count"] = 6
        st.session_state["rr_ana_count_logic"] = 6
        st.session_state["rr_ana_pop_min"] = 6
        st.session_state["rr_ana_odds_min"] = 10.0
        st.session_state["rr_ana_gap_min"] = 0.015
    elif profile == "回収率":
        st.session_state["rr_predict_mode"] = "回収率重視"
        st.session_state["rr_honmei_count"] = 6
        st.session_state["rr_ana_count_logic"] = 6
        st.session_state["rr_ana_pop_min"] = 7
        st.session_state["rr_ana_odds_min"] = 12.0
        st.session_state["rr_ana_gap_min"] = 0.01
    elif profile == "爆穴":
        st.session_state["rr_predict_mode"] = "回収率重視"
        st.session_state["rr_honmei_count"] = 6
        st.session_state["rr_ana_count_logic"] = 10
        st.session_state["rr_ana_pop_min"] = 8
        st.session_state["rr_ana_odds_min"] = 15.0
        st.session_state["rr_ana_gap_min"] = 0.005
    elif profile == "三連複特化":
        st.session_state["rr_predict_mode"] = "回収率重視"
        st.session_state["rr_honmei_count"] = 6
        st.session_state["rr_ana_count_logic"] = 6
        st.session_state["rr_ana_pop_min"] = 6
        st.session_state["rr_ana_odds_min"] = 10.0
        st.session_state["rr_ana_gap_min"] = 0.01

def _sync_main_quick_profile():
    _apply_quick_profile(st.session_state.get("main_quick_profile", "カスタム"))

def _sync_sidebar_quick_profile():
    _apply_quick_profile(st.session_state.get("render_sidebar_quick_profile", st.session_state.get("sidebar_quick_profile", "カスタム")))

def _sync_main_honmei():
    v = int(st.session_state.get("main_honmei_count", 6))
    st.session_state["rr_honmei_count"] = v

def _sync_sidebar_honmei():
    v = int(st.session_state.get("sidebar_honmei_count", 6))
    st.session_state["rr_honmei_count"] = v
    st.session_state["main_honmei_count"] = v

def _sync_main_ana():
    v = int(st.session_state.get("main_ana_count_logic", 6))
    st.session_state["rr_ana_count_logic"] = v

def _sync_sidebar_ana():
    v = int(st.session_state.get("sidebar_ana_count_logic", 6))
    st.session_state["rr_ana_count_logic"] = v
    st.session_state["main_ana_count_logic"] = v

def _sync_sidebar_filters():
    st.session_state["rr_ana_pop_min"] = int(st.session_state.get("sidebar_ana_pop_min", globals().get("ANA_POP_MIN_DEFAULT", 4)))
    st.session_state["rr_ana_odds_min"] = float(st.session_state.get("sidebar_ana_odds_min", globals().get("ANA_ODDS_MIN_DEFAULT", 10.0)))
    st.session_state["rr_ana_gap_min"] = float(st.session_state.get("sidebar_ana_gap_min", globals().get("ANA_GAP_MIN_DEFAULT", 0.0)))

def _ensure_ui_defaults():
    st.session_state.setdefault("rr_quick_profile", "カスタム")
    st.session_state.setdefault("rr_honmei_count", 6)
    st.session_state.setdefault("rr_ana_count_logic", 6)
    st.session_state.setdefault("rr_ana_pop_min", globals().get("ANA_POP_MIN_DEFAULT", 4))
    st.session_state.setdefault("rr_ana_odds_min", globals().get("ANA_ODDS_MIN_DEFAULT", 10.0))
    st.session_state.setdefault("rr_ana_gap_min", globals().get("ANA_GAP_MIN_DEFAULT", 0.0))
    st.session_state.setdefault("rr_predict_mode", "バランス")

    st.session_state.setdefault("main_quick_profile", st.session_state["rr_quick_profile"])
    st.session_state.setdefault("main_honmei_count", st.session_state["rr_honmei_count"])
    st.session_state.setdefault("main_ana_count_logic", st.session_state["rr_ana_count_logic"])

_ensure_ui_defaults()


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

def _repair_mojibake_text(s: str) -> str:
    if not s:
        return ""

    # 正常な日本語やカタカナが見えている文字列は触らない
    jp_cnt = len(re.findall(r"[一-龠ぁ-んァ-ヶー]", s))
    bad_markers = sum(s.count(x) for x in ["ｽ", "鬯", "闔", "螟", "郢", "騾", " ", "•", "封", "ﾂ"])
    if jp_cnt >= 1 and bad_markers == 0:
        return s

    # 明らかな文字化け候補だけを対象にする
    suspicious = bad_markers > 0 or bool(re.search(r"[ｽﾂ郢鬯闔螟騾 •封]", s))
    if not suspicious:
        return s

    candidates = [s]
    for enc1 in ["cp1252", "latin1"]:
        for enc2 in ["cp932", "shift_jis", "euc_jp"]:
            try:
                b = s.encode(enc1, errors="ignore")
                if not b:
                    continue
                candidates.append(b.decode(enc2, errors="ignore"))
            except Exception:
                pass

    def score(txt: str) -> int:
        if not txt:
            return -10**9
        jp = len(re.findall(r"[一-龠ぁ-んァ-ヶーｦ-ﾟ]", txt))
        bad = (
            txt.count("•") * 20
            + txt.count(" ") * 20
            + txt.count("封") * 8
            + txt.count("ﾂ") * 6
            + txt.count("ｽ") * 8
            + txt.count("郢") * 8
        )
        weird = len(re.findall(r"[^\w\s一-龠ぁ-んァ-ヶーｦ-ﾟ・－ー\-\.\(\)（）／/]", txt))
        return jp * 8 - bad - weird * 2 + min(len(txt), 40)

    return max(candidates, key=score)


def has_broken_japanese(text: str) -> bool:
    if not text:
        return True
    bad_markers = ["ｽ", "鬯", "闔", "螟", "郢", "騾", " ", "•", "封"]
    bad = sum(text.count(x) for x in bad_markers)
    jp = len(re.findall(r"[一-龠ぁ-んァ-ヶー]", text))
    return bad >= 2 or (jp == 0 and bad > 0)

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    if has_broken_japanese(s):
        s = _repair_mojibake_text(s)
    return re.sub(r"\s+", " ", s).strip()


def norm_text(value: Any) -> str:
    return str(value).replace(" ", "").replace("\u3000", "").strip()


def normalize_jockey_name(value: Any) -> str:
    text = norm_text(value)
    text = text.replace("・", "").replace(".", "").replace("．", "")
    text = text.replace("Ｊ", "").replace("Ｍ", "").replace("Ｃ", "").replace("Ｄ", "").replace("Ｌ", "")
    text = text.replace("J", "").replace("M", "").replace("C", "").replace("D", "").replace("L", "")
    text = text.replace(" ", "").replace("　", "")
    return text.strip()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_jra_jockey_scores() -> pd.DataFrame:
    """JRAの公開ページから騎手スコア候補を取得する。取得失敗時は空DataFrame。"""
    urls = [
        "https://www.jra.go.jp/JRADB/accessK.html",
        "https://www.jra.go.jp/datafile/leading/",
        "https://www.jra.go.jp/datafile/meikan/",
    ]

    all_frames = []
    for url in urls:
        try:
            html = fetch_text_with_encoding(url, timeout=REQUEST_TIMEOUT)
            tables = safe_read_html(html)
        except Exception:
            continue

        for raw in tables:
            try:
                df = flatten_columns(raw).copy()
                cols = [clean_text(c) for c in df.columns]
                joined = " ".join(cols)

                # 騎手名列の推定
                name_col = None
                for c in df.columns:
                    cc = clean_text(c)
                    if any(k in cc for k in ["騎手名", "騎手", "氏名", "名前"]):
                        name_col = c
                        break

                if name_col is None:
                    # 先頭列が名前っぽいケース
                    first_col = df.columns[0] if len(df.columns) else None
                    if first_col is not None and df[first_col].astype(str).str.contains(r"[一-龠ぁ-んァ-ンA-Za-z]", regex=True, na=False).mean() > 0.5:
                        name_col = first_col

                if name_col is None:
                    continue

                # 数値列の候補
                work = pd.DataFrame()
                work["jockey"] = df[name_col].astype(str).map(clean_text)
                work["jockey_key"] = work["jockey"].map(normalize_jockey_name)

                for c in df.columns:
                    s = df[c]
                    cname = clean_text(c)

                    # 勝率系
                    if any(k in cname for k in ["勝率", "連対率", "3着内率", "複勝率", "入着率"]):
                        work[cname] = pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce") / (
                            100.0 if s.astype(str).str.contains("%", regex=False, na=False).any() else 1.0
                        )
                    # 回数系
                    elif any(k in cname for k in ["1着", "2着", "3着", "勝利度数", "騎乗回数", "出走回数"]):
                        work[cname] = pd.to_numeric(s, errors="coerce")

                # 最低限の情報がないものは除外
                usable_cols = [c for c in work.columns if c not in ["jockey", "jockey_key"]]
                if len(usable_cols) == 0:
                    continue

                work = work[work["jockey_key"] != ""].copy()
                if len(work) == 0:
                    continue

                # スコア算出（ある列だけ使う）
                score = pd.Series(0.0, index=work.index)

                if "勝率" in work.columns:
                    score += pd.to_numeric(work["勝率"], errors="coerce").fillna(0) * 0.40
                if "連対率" in work.columns:
                    score += pd.to_numeric(work["連対率"], errors="coerce").fillna(0) * 0.25
                if "3着内率" in work.columns:
                    score += pd.to_numeric(work["3着内率"], errors="coerce").fillna(0) * 0.20
                if "複勝率" in work.columns:
                    score += pd.to_numeric(work["複勝率"], errors="coerce").fillna(0) * 0.20
                if "入着率" in work.columns:
                    score += pd.to_numeric(work["入着率"], errors="coerce").fillna(0) * 0.15

                if "1着" in work.columns:
                    score += pd.to_numeric(work["1着"], errors="coerce").rank(pct=True).fillna(0) * 0.08
                if "勝利度数" in work.columns:
                    score += pd.to_numeric(work["勝利度数"], errors="coerce").rank(pct=True).fillna(0) * 0.08
                if "騎乗回数" in work.columns:
                    score += pd.to_numeric(work["騎乗回数"], errors="coerce").rank(pct=True).fillna(0) * 0.02
                if "出走回数" in work.columns:
                    score += pd.to_numeric(work["出走回数"], errors="coerce").rank(pct=True).fillna(0) * 0.02

                work["jra_jockey_score_raw"] = score
                all_frames.append(work[["jockey", "jockey_key", "jra_jockey_score_raw"]].copy())
            except Exception:
                continue

    if not all_frames:
        return pd.DataFrame(columns=["jockey", "jockey_key", "jra_jockey_score"])

    out = pd.concat(all_frames, ignore_index=True)
    out = out.dropna(subset=["jockey_key"]).copy()
    out = out[out["jockey_key"] != ""].copy()

    # 同名重複は最大値採用
    out = out.groupby("jockey_key", as_index=False).agg({
        "jockey": "first",
        "jra_jockey_score_raw": "max",
    })

    vals = pd.to_numeric(out["jra_jockey_score_raw"], errors="coerce").fillna(0)
    if vals.max() > vals.min():
        out["jra_jockey_score"] = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        out["jra_jockey_score"] = 0.5

    return out[["jockey", "jockey_key", "jra_jockey_score"]].copy()


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


def normalize_passing_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    nums = re.findall(r"\d+", str(x))
    if len(nums) >= 2:
        return "-".join(nums)
    return ""


def parse_passing_positions(x: Any) -> List[int]:
    s = normalize_passing_text(x)
    if not s:
        return []
    return [int(n) for n in s.split("-")]


def last_corner(x: Any) -> float:
    nums = parse_passing_positions(x)
    return float(nums[-1]) if nums else np.nan


def infer_running_style_from_history(passings: Iterable[Any], field_size: Any) -> str:
    if pd.isna(field_size) or field_size is None or field_size <= 0:
        return "不明"

    ratios: List[float] = []
    for p in list(passings)[:3]:
        pos = last_corner(p)
        if not pd.isna(pos):
            ratios.append(float(pos) / float(field_size))

    if not ratios:
        return "不明"

    avg_ratio = float(np.mean(ratios))
    if avg_ratio <= 0.15:
        return "逃げ"
    if avg_ratio <= 0.35:
        return "先行"
    if avg_ratio <= 0.65:
        return "差"
    return "追込"


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
    front_ratio = ((styles == "逃げ") | (styles == "先行")).mean()
    closer_ratio = ((styles == "差") | (styles == "追込")).mean()
    if front_ratio >= 0.45:
        return "ハイ"
    if front_ratio <= 0.25 and closer_ratio >= 0.40:
        return "スロー"
    return "平均"


def style_ground_pace_bonus(style, track_type, ground, pace):
    bonus = 0.0

    if pace == "ハイ":
        if style == "差":
            bonus += 0.05
        elif style == "追込":
            bonus += 0.04
        elif style == "逃げ":
            bonus -= 0.04

    elif pace == "スロー":
        if style == "逃げ":
            bonus += 0.08
        elif style == "先行":
            bonus += 0.06
        elif style == "差":
            bonus -= 0.04
        elif style == "追込":
            bonus -= 0.06

    else:
        if style == "先行":
            bonus += 0.03
        elif style == "差":
            bonus += 0.01

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
    out["ai_score"] = pd.to_numeric(out["ai_score"], errors="coerce").fillna(0) + pd.to_numeric(out["style_bonus"], errors="coerce").fillna(0) * 1.5

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



def safe_pick_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    use_cols = [c for c in cols if c in df.columns]
    return df[use_cols].copy() if len(use_cols) else df.copy()

def normalize_combination_text(comb: Any, sep: str = "-") -> str:
    nums = [int(x) for x in re.findall(r"\d+", str(comb))]
    if not nums:
        return clean_text(comb)
    nums = sorted(nums)
    return sep.join(str(n) for n in nums)



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


def safe_read_html(html_text: str) -> List[pd.DataFrame]:
    try:
        if not html_text:
            return []
        return pd.read_html(io.StringIO(html_text))
    except Exception:
        return []


class Scraper:
    def __init__(self, timeout: int = REQUEST_TIMEOUT, sleep_sec: float = REQUEST_SLEEP) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self.sleep_sec = sleep_sec

    def get_html(self, url: str) -> str:
        # 地方競馬(NAR)は requests 側で文字化け・壊れHTMLになりやすいので常に Selenium で取得
        if "nar.netkeiba.com" in url:
            if SELENIUM_AVAILABLE:
                html = self._selenium_html(url)
                time.sleep(self.sleep_sec)
                return html
            raise RuntimeError("地方競馬ページは Selenium が必要です")

        try:
            html = self._requests_html(url)
            if has_broken_japanese(html) and SELENIUM_AVAILABLE:
                try:
                    html = self._selenium_html(url)
                except Exception:
                    pass
            time.sleep(self.sleep_sec)
            return html
        except Exception:
            if SELENIUM_AVAILABLE:
                html = self._selenium_html(url)
                time.sleep(self.sleep_sec)
                return html
            raise


    def _requests_html(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return decode_response_html(resp, url)


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
        if row.horse_name:
            rows.append(row)

    if not rows:
        raise RuntimeError("出馬表の解析結果が0件でした。")

    df = dedupe_race_df(dataframe_from_dataclass_rows(rows))
    return [RaceCardRow(**r) for r in df.to_dict(orient="records")]


def parse_race_card_nar(scraper: Scraper, race_url: str) -> List[RaceCardRow]:
    html = scraper.get_html(race_url)
    soup = BeautifulSoup(html, "lxml")
    meta = parse_race_meta(soup, race_url)

    table = (
        soup.select_one("table.RaceTable01.ShutubaTable")
        or soup.select_one("table.RaceTable01")
        or soup.select_one("table[class*='Shutuba']")
        or soup.select_one("table")
    )
    if table is None:
        raise RuntimeError("NARの出馬表テーブルが見つかりませんでした。")

    # まず従来候補
    candidate_rows = table.select("tr.HorseList")

    # fallback: horseリンクを持つ tr を広く拾う
    if not candidate_rows:
        trs = table.select("tr")
        candidate_rows = []
        for tr in trs:
            if tr.select_one('a[href*="/horse/"]'):
                candidate_rows.append(tr)

    rows: List[RaceCardRow] = []

    for tr in candidate_rows:
        horse_a = tr.select_one('a[href*="/horse/"]')
        if horse_a is None:
            continue

        # 枠番 / 馬番
        frame_cell = (
            tr.select_one('td[class*="Waku"]')
            or tr.select_one('td[class*="waku"]')
        )
        horse_no_cell = (
            tr.select_one('td[class*="Umaban"]')
            or tr.select_one('td[class*="umaban"]')
        )

        text_cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
        nums = [to_int(x) for x in text_cells]
        nums = [x for x in nums if x is not None]

        frame_no = to_int(clean_text(frame_cell.get_text(" ", strip=True))) if frame_cell else None
        horse_no = to_int(clean_text(horse_no_cell.get_text(" ", strip=True))) if horse_no_cell else None

        # fallback: 冒頭2数値を枠・馬番とみなす
        if frame_no is None and len(nums) >= 1:
            frame_no = nums[0]
        if horse_no is None and len(nums) >= 2:
            horse_no = nums[1]
        elif horse_no is None and len(nums) >= 1:
            horse_no = nums[0]

        horse_name = clean_text(horse_a.get_text())
        if (not horse_name or has_broken_japanese(horse_name)) and horse_no is not None:
            horse_name = f"馬番{horse_no}"

        horse_url = urljoin(race_url, horse_a.get("href", ""))

        jockey_a = tr.select_one('a[href*="/jockey/"]')
        jockey_url = urljoin(race_url, jockey_a.get("href", "")) if jockey_a else ""
        jockey = clean_text(jockey_a.get_text()) if jockey_a else ""

        trainer_a = tr.select_one('td.Trainer a') or tr.select_one('a[href*="/trainer/"]')
        trainer = clean_text(trainer_a.get_text()) if trainer_a else ""

        sex_age = ""
        for x in text_cells:
            if re.fullmatch(r"[牡牝セ]\d+", x):
                sex_age = x
                break

        carried_weight = ""
        for x in text_cells:
            v = to_float(x)
            if v is not None and 40 <= v <= 70:
                carried_weight = x
                break

        odds = None
        popularity = None

        odds_cell = tr.select_one("td.Popular.Txt_R .Odds_Ninki") or tr.select_one('[class*="Odds"]')
        if odds_cell is not None:
            odds = to_float(clean_text(odds_cell.get_text(" ", strip=True)))

        pop_cell = tr.select_one("td.Popular.Txt_C span") or tr.select_one('[class*="Ninki"]')
        if pop_cell is not None:
            popularity = to_int(clean_text(pop_cell.get_text(" ", strip=True)))

        # fallback: 行末の数値群から推定
        if odds is None or popularity is None:
            tail = text_cells[-10:]
            tail_f = [to_float(x) for x in tail]
            tail_i = [to_int(x) for x in tail]

            if odds is None:
                float_cands = [v for v in tail_f if v is not None and 1.0 <= v <= 9999]
                if float_cands:
                    odds = float_cands[0]

            if popularity is None:
                int_cands = [v for v in tail_i if v is not None and 1 <= v <= 18]
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
        if row.horse_name:
            rows.append(row)

    if not rows:
        raise RuntimeError("NARの出馬表の解析結果が0件でした。HTML構造が変わっている可能性があります。")

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
        "単勝オッズ": "odds",
        "オッズ": "odds",
        "人気": "popularity",
        "人気順": "popularity",
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

    if "passing" not in df.columns or df["passing"].astype(str).str.contains(r"\d+-\d+", regex=True, na=False).sum() == 0:
        for c in df.columns:
            s = df[c].astype(str)
            if s.str.contains(r"\d+-\d+", regex=True, na=False).sum() >= max(1, len(df) // 3):
                df["passing"] = s.map(normalize_passing_text)
                break
    else:
        df["passing"] = df["passing"].map(normalize_passing_text)

    if "odds" not in df.columns or pd.to_numeric(df["odds"], errors="coerce").notna().sum() == 0:
        for c in df.columns:
            cname = clean_text(c)
            s = df[c].astype(str)
            nums = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
            if any(k in cname for k in ["単勝", "オッズ"]) or (
                nums.notna().sum() >= max(1, len(df) // 2)
                and nums.dropna().between(1.0, 500.0).mean() > 0.8
                and s.str.contains(r"\.", regex=True, na=False).sum() >= max(1, len(df) // 3)
            ):
                df["odds"] = nums
                break

    if "popularity" not in df.columns or pd.to_numeric(df["popularity"], errors="coerce").notna().sum() == 0:
        for c in df.columns:
            cname = clean_text(c)
            nums = pd.to_numeric(df[c], errors="coerce")
            if "人気" in cname or (
                nums.notna().sum() >= max(1, len(df) // 2)
                and nums.dropna().between(1, 18).mean() > 0.8
            ):
                df["popularity"] = nums
                break

    if "last3f" not in df.columns or pd.to_numeric(df["last3f"], errors="coerce").notna().sum() == 0:
        for c in df.columns:
            cname = clean_text(c)
            nums = pd.to_numeric(df[c], errors="coerce")
            if any(k in cname for k in ["上り", "上がり", "3F", "後3F"]) or (
                nums.notna().sum() >= max(1, len(df) // 3)
                and nums.dropna().between(30, 50).mean() > 0.6
            ):
                df["last3f"] = nums
                break

    if "odds" not in df.columns:
        df["odds"] = np.nan
    if "popularity" not in df.columns:
        df["popularity"] = np.nan
    if "last3f" not in df.columns:
        df["last3f"] = np.nan
    if "passing" not in df.columns:
        df["passing"] = ""

    for col in [
        "race_date", "venue", "race_name", "class_name", "weather", "ground",
        "horse_no", "frame_no", "finish", "jockey", "carried_weight", "time_str",
        "margin", "body_weight", "prize",
    ]:
        if col not in df.columns:
            df[col] = "" if col not in ["horse_no", "frame_no", "finish"] else np.nan

    df["distance"] = df["distance_raw"].astype(str).str.extract(r"(\d{3,4})").iloc[:, 0]
    df["track_type"] = df["distance_raw"].astype(str).apply(detect_track_type)

    keep = [
        "race_date", "venue", "race_name", "class_name", "track_type", "distance",
        "weather", "ground", "horse_no", "frame_no", "finish", "jockey",
        "carried_weight", "time_str", "margin", "passing", "last3f", "odds",
        "popularity", "body_weight", "prize",
    ]
    return df[keep].copy()


def score_history_table(ndf: pd.DataFrame) -> int:
    finish_cnt = ndf["finish"].apply(to_int).notna().sum() if "finish" in ndf.columns else 0
    dist_cnt = ndf["distance"].apply(to_int).notna().sum() if "distance" in ndf.columns else 0
    last3f_cnt = ndf["last3f"].apply(to_float).notna().sum() if "last3f" in ndf.columns else 0
    odds_cnt = ndf["odds"].apply(to_float).notna().sum() if "odds" in ndf.columns else 0
    pop_cnt = ndf["popularity"].apply(to_int).notna().sum() if "popularity" in ndf.columns else 0
    passing_cnt = ndf["passing"].astype(str).str.contains(r"\d+-\d+", regex=True, na=False).sum() if "passing" in ndf.columns else 0

    bonus = 0
    if passing_cnt >= max(2, len(ndf) // 3):
        bonus += 120
    if last3f_cnt >= max(2, len(ndf) // 3):
        bonus += 60
    if odds_cnt >= max(2, len(ndf) // 3):
        bonus += 40
    if pop_cnt >= max(2, len(ndf) // 3):
        bonus += 40

    return int(bonus + finish_cnt * 10 + dist_cnt * 5 + last3f_cnt * 7 + odds_cnt * 5 + pop_cnt * 5 + passing_cnt * 12)


def parse_horse_history(scraper: Scraper, horse_name: str, horse_url: str, max_rows: int = 10) -> List[HorseHistoryRow]:
    if not horse_url:
        return []
    db_url = convert_to_db_horse_url(horse_url)

    html = ""
    is_nar = "nar.netkeiba.com" in horse_url

    if is_nar and SELENIUM_AVAILABLE:
        try:
            html = scraper._selenium_html(db_url)
        except Exception:
            html = ""

    if not html:
        try:
            html = scraper._requests_html(db_url)
        except Exception:
            html = ""

    need_selenium = False
    if not html or "<table" not in html.lower():
        need_selenium = True
    elif has_broken_japanese(html):
        need_selenium = True
    elif ("通過" not in html and "上り" not in html and "上がり" not in html and "上り3F" not in html and "上がり3F" not in html):
        need_selenium = True

    if need_selenium and SELENIUM_AVAILABLE:
        try:
            html = scraper._selenium_html(db_url)
        except Exception:
            pass

    if not html:
        return []

    try:
        tables = safe_read_html(html)
    except Exception:
        tables = []

    best = pd.DataFrame()
    best_score = -1
    for raw in tables:
        try:
            ndf = normalize_history_columns(raw).head(max_rows)
            score = score_history_table(ndf)
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
                horse_name=(f"馬番{to_int(r.get('horse_no'))}" if has_broken_japanese(horse_name) and to_int(r.get("horse_no")) is not None else horse_name),
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
                passing=normalize_passing_text(r.get("passing")),
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
    if len(race_df) and "horse_name" in race_df.columns and "horse_no" in race_df.columns:
        race_df["horse_name"] = race_df.apply(
            lambda r: f"馬番{int(r['horse_no'])}" if has_broken_japanese(clean_text(r.get("horse_name"))) and pd.notna(r.get("horse_no")) else clean_text(r.get("horse_name")),
            axis=1
        )

    history_rows_out: List[HorseHistoryRow] = []
    total = min(len(race_rows), max_horses)
    prog = st.progress(0, text="過去成績を取得中...")
    for i, row in enumerate(race_rows[:total], start=1):
        histories = parse_horse_history(scraper, row.horse_name, row.horse_url, history_rows)
        history_rows_out.extend(histories)
        display_name = clean_text(getattr(row, "horse_name", ""))
        if has_broken_japanese(display_name):
            horse_no_disp = getattr(row, "horse_no", "")
            display_name = f"馬番{horse_no_disp}" if horse_no_disp not in [None, ""] else "馬名取得中"
        prog.progress(i / total if total else 1, text=f"過去成績を取得中... {i}/{total} {display_name}")
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

# ============================================================
# 予想モード補正
# ============================================================
def apply_prediction_mode(df: pd.DataFrame, mode: str = "バランス") -> pd.DataFrame:
    out = df.copy()
    mode = clean_text(mode) or "バランス"
    is_local = is_local_race_df(out)
    is_jra = not is_local

    if len(out) == 0:
        return out

    ai = pd.to_numeric(out.get("ai_score"), errors="coerce").fillna(0)
    winp = pd.to_numeric(out.get("win_prob"), errors="coerce").fillna(0)
    gap = pd.to_numeric(out.get("gap"), errors="coerce").fillna(0)
    ana = pd.to_numeric(out.get("ana_score"), errors="coerce").fillna(0)
    place = pd.to_numeric(out.get("place_prob"), errors="coerce").fillna(0)

    if mode == "的中率重視":
        if is_local:
            out["ai_score"] = ai * 0.78 + winp * 0.22 + place * 0.12 - gap.clip(lower=0) * 0.03
        else:
            out["ai_score"] = ai * 0.72 + winp * 0.18 + place * 0.18 - gap.clip(lower=0) * 0.08
    elif mode == "回収率重視":
        if is_local:
            out["ai_score"] = ai * 0.62 + gap.clip(lower=0) * 0.85 + ana * 0.60 + winp * 0.12
        else:
            out["ai_score"] = ai * 0.58 + gap.clip(lower=0) * 0.40 + ana * 0.72 + winp * 0.15 + place * 0.08
    else:
        out["ai_score"] = ai

    expv = np.exp(out["ai_score"] - out["ai_score"].max())
    out["win_prob"] = expv / expv.sum()

    show_rate_col = pd.to_numeric(out["show_rate"], errors="coerce").fillna(0) if "show_rate" in out.columns else 0
    top2_rate_col = pd.to_numeric(out["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in out.columns else 0
    last3f_score_col = pd.to_numeric(out["s_last3f"], errors="coerce").fillna(0) if "s_last3f" in out.columns else 0
    place_raw = out["win_prob"] * 0.45 + show_rate_col * 0.30 + top2_rate_col * 0.15 + last3f_score_col * 0.10
    out["place_prob"] = place_raw / place_raw.sum()

    if "odds_f" in out.columns and pd.to_numeric(out["odds_f"], errors="coerce").notna().sum() > 0:
        out["market_prob"] = 1 / pd.to_numeric(out["odds_f"], errors="coerce")
        out["market_prob"] = out["market_prob"] / out["market_prob"].sum()
    elif "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        temp = 1 / pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        out["market_prob"] = temp / temp.sum()
    else:
        out["market_prob"] = np.nan

    out["gap"] = out["win_prob"] - out["market_prob"]
    if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        pop = pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        if is_local:
            if mode == "回収率重視":
                out["ana_score"] = out["gap"].fillna(0) * 0.75 + (1 / pop).fillna(0) * 0.25
            elif mode == "的中率重視":
                out["ana_score"] = out["gap"].fillna(0) * 0.45 + (1 / pd.Series(pop)).replace([np.inf, -np.inf], np.nan).fillna(0) * 0.10 + out["place_prob"] * 0.45
            else:
                out["ana_score"] = out["gap"].fillna(0) * 0.7 + (1 / pop).fillna(0) * 0.3
        else:
            # 中央はgap単体では弱い。複勝圏・人気帯を重視
            out["ana_score"] = out["gap"].fillna(0) * 0.30 + (1 / pop).fillna(0) * 0.20 + out["place_prob"] * 0.50
    else:
        out["ana_score"] = out["gap"]

    out = out.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    out["mark"] = ""
    for i, m in enumerate(marks):
        if i < len(out):
            out.loc[i, "mark"] = m
    return out


# ============================================================
# 地方競馬補正
# ============================================================
LOCAL_VENUES = {
    "門別","盛岡","水沢","浦和","船橋","大井","川崎","金沢","笠松","名古屋",
    "園田","姫路","高知","佐賀","帯広(ば)","帯広","ばんえい"
}

JRA_COURSES = {"札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"}

def detect_course_name(df: pd.DataFrame) -> str:
    values = []
    if df is None or len(df) == 0:
        return ""
    if "course" in df.columns:
        values.extend(df["course"].astype(str).tolist())
    if "race_name" in df.columns:
        values.extend(df["race_name"].astype(str).tolist())
    joined = " ".join(values)
    for v in list(JRA_COURSES) + list(LOCAL_VENUES):
        if v in joined:
            return v
    return ""

def is_jra_race_df(race: pd.DataFrame) -> bool:
    return (race is not None) and (len(race) > 0) and (not is_local_race_df(race))

def jra_course_style_bonus(course_name: str, style: str) -> float:
    # 中央はコース形状差が大きいので展開・コース適性を地方より強く使う
    if course_name in {"東京","新潟"}:
        return {"差": 0.07, "追込": 0.05, "先行": -0.01, "逃げ": -0.02}.get(style, 0.0)
    if course_name in {"中山","阪神","小倉","中京"}:
        return {"逃げ": 0.05, "先行": 0.04, "差": -0.01, "追込": -0.03}.get(style, 0.0)
    if course_name in {"京都"}:
        return {"先行": 0.03, "差": 0.02, "逃げ": 0.01, "追込": -0.01}.get(style, 0.0)
    return {"逃げ": 0.01, "先行": 0.02, "差": 0.01, "追込": -0.01}.get(style, 0.0)

def apply_jra_race_bonus(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out

    pace = estimate_race_pace(out)
    course_name = detect_course_name(out)
    out["jra_pace_bonus"] = 0.0

    if pace == "ハイ":
        out.loc[out["running_style"] == "差", "jra_pace_bonus"] = 0.10
        out.loc[out["running_style"] == "追込", "jra_pace_bonus"] = 0.08
        out.loc[out["running_style"] == "先行", "jra_pace_bonus"] = -0.02
        out.loc[out["running_style"] == "逃げ", "jra_pace_bonus"] = -0.05
    elif pace == "スロー":
        out.loc[out["running_style"] == "逃げ", "jra_pace_bonus"] = 0.09
        out.loc[out["running_style"] == "先行", "jra_pace_bonus"] = 0.07
        out.loc[out["running_style"] == "差", "jra_pace_bonus"] = -0.03
        out.loc[out["running_style"] == "追込", "jra_pace_bonus"] = -0.06
    else:
        out.loc[out["running_style"] == "先行", "jra_pace_bonus"] = 0.03
        out.loc[out["running_style"] == "差", "jra_pace_bonus"] = 0.02

    out["jra_course_bonus"] = out["running_style"].map(lambda s: jra_course_style_bonus(course_name, s))
    # 中央はgapを弱め、展開とコースを強く
    out["jra_gap_bonus"] = pd.to_numeric(out.get("gap"), errors="coerce").fillna(0).clip(lower=0) * 0.25

    out["ai_score"] = (
        pd.to_numeric(out["ai_score"], errors="coerce").fillna(0)
        + pd.to_numeric(out["jra_pace_bonus"], errors="coerce").fillna(0)
        + pd.to_numeric(out["jra_course_bonus"], errors="coerce").fillna(0)
        + pd.to_numeric(out["jra_gap_bonus"], errors="coerce").fillna(0)
    )

    expv = np.exp(out["ai_score"] - out["ai_score"].max())
    out["win_prob"] = expv / expv.sum()

    show_rate_col = pd.to_numeric(out["show_rate"], errors="coerce").fillna(0) if "show_rate" in out.columns else 0
    top2_rate_col = pd.to_numeric(out["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in out.columns else 0
    last3f_score_col = pd.to_numeric(out["s_last3f"], errors="coerce").fillna(0) if "s_last3f" in out.columns else 0
    place_raw = out["win_prob"] * 0.50 + show_rate_col * 0.25 + top2_rate_col * 0.15 + last3f_score_col * 0.10
    out["place_prob"] = place_raw / place_raw.sum()

    if "odds_f" in out.columns and pd.to_numeric(out["odds_f"], errors="coerce").notna().sum() > 0:
        out["market_prob"] = 1 / pd.to_numeric(out["odds_f"], errors="coerce")
        out["market_prob"] = out["market_prob"] / out["market_prob"].sum()
    elif "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        temp = 1 / pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        out["market_prob"] = temp / temp.sum()
    else:
        out["market_prob"] = np.nan

    out["gap"] = out["win_prob"] - out["market_prob"]
    if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        pop = pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        # 中央の穴は gap 単体でなく、人気帯と複勝圏も見る
        out["ana_score"] = out["gap"].fillna(0) * 0.35 + (1 / pop).fillna(0) * 0.15 + pd.to_numeric(out["place_prob"], errors="coerce").fillna(0) * 0.50
    else:
        out["ana_score"] = out["gap"]

    out = out.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    out["mark"] = ""
    for i, m in enumerate(marks):
        if i < len(out):
            out.loc[i, "mark"] = m
    return out

def is_local_race_df(race: pd.DataFrame) -> bool:
    if race is None or len(race) == 0:
        return False
    values = []
    if "course" in race.columns:
        values.extend(race["course"].astype(str).tolist())
    if "race_name" in race.columns:
        values.extend(race["race_name"].astype(str).tolist())
    joined = " ".join(values)
    return any(v in joined for v in LOCAL_VENUES)

def calc_same_track_show_rate(hist: pd.DataFrame, key: str, venue: str) -> float:
    if hist is None or len(hist) == 0 or not venue or "key" not in hist.columns:
        return 0.0
    work = hist[hist["key"] == key].copy()
    if len(work) == 0 or "venue" not in work.columns:
        return 0.0
    work = work[work["venue"].astype(str).str.contains(str(venue), na=False)].copy()
    if len(work) == 0 or "finish" not in work.columns:
        return 0.0
    vals = work["finish"].apply(finish_num).dropna()
    if len(vals) == 0:
        return 0.0
    return float((vals <= 3).mean())

def apply_local_race_bonus(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out

    current_venue = ""
    if "course" in out.columns and out["course"].notna().any():
        joined = " ".join(out["course"].astype(str).tolist())
        for v in LOCAL_VENUES:
            if v in joined:
                current_venue = v
                break

    out["local_style_bonus"] = 0.0
    out.loc[out["running_style"] == "逃げ", "local_style_bonus"] = 0.10
    out.loc[out["running_style"] == "先行", "local_style_bonus"] = 0.06
    out.loc[out["running_style"] == "差", "local_style_bonus"] = -0.02
    out.loc[out["running_style"] == "追込", "local_style_bonus"] = -0.04

    out["same_track_show_rate"] = out["key"].map(lambda k: calc_same_track_show_rate(hist, k, current_venue))
    out["local_track_bonus"] = pd.to_numeric(out["same_track_show_rate"], errors="coerce").fillna(0) * 0.10
    out["local_gap_bonus"] = pd.to_numeric(out["gap"], errors="coerce").fillna(0).clip(lower=0) * 0.60 if "gap" in out.columns else 0.0

    out["ai_score"] = (
        pd.to_numeric(out["ai_score"], errors="coerce").fillna(0)
        + pd.to_numeric(out["local_style_bonus"], errors="coerce").fillna(0)
        + pd.to_numeric(out["local_track_bonus"], errors="coerce").fillna(0)
        + pd.to_numeric(out["local_gap_bonus"], errors="coerce").fillna(0)
    )

    expv = np.exp(out["ai_score"] - out["ai_score"].max())
    out["win_prob"] = expv / expv.sum()

    show_rate_col = pd.to_numeric(out["show_rate"], errors="coerce").fillna(0) if "show_rate" in out.columns else 0
    top2_rate_col = pd.to_numeric(out["top2_rate"], errors="coerce").fillna(0) if "top2_rate" in out.columns else 0
    last3f_score_col = pd.to_numeric(out["s_last3f"], errors="coerce").fillna(0) if "s_last3f" in out.columns else 0
    place_raw = out["win_prob"] * 0.45 + show_rate_col * 0.30 + top2_rate_col * 0.15 + last3f_score_col * 0.10
    out["place_prob"] = place_raw / place_raw.sum()

    if "odds_f" in out.columns and pd.to_numeric(out["odds_f"], errors="coerce").notna().sum() > 0:
        out["market_prob"] = 1 / pd.to_numeric(out["odds_f"], errors="coerce")
        out["market_prob"] = out["market_prob"] / out["market_prob"].sum()
    elif "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        temp = 1 / pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        out["market_prob"] = temp / temp.sum()
    else:
        out["market_prob"] = np.nan

    out["gap"] = out["win_prob"] - out["market_prob"]
    if "pop_f" in out.columns and pd.to_numeric(out["pop_f"], errors="coerce").notna().sum() > 0:
        pop = pd.to_numeric(out["pop_f"], errors="coerce").replace(0, np.nan)
        out["ana_score"] = out["gap"].fillna(0) * 0.7 + (1 / pop).fillna(0) * 0.3
    else:
        out["ana_score"] = out["gap"]

    out = out.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    marks = ["◎", "○", "▲", "△", "☆"]
    out["mark"] = ""
    for i, m in enumerate(marks):
        if i < len(out):
            out.loc[i, "mark"] = m
    return out


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
                "last_passing": normalize_passing_text(r3["passing"].iloc[0]) if "passing" in r3.columns and len(r3) else "",
                "auto_running_style": infer_running_style_from_history(r3["passing"].tolist() if "passing" in r3.columns else [], 18),
                "hist_odds_avg": o3.mean() if len(o3) else np.nan,
                "hist_pop_avg": p3.mean() if len(p3) else np.nan,
                "growth": (f_all.iloc[1] - f_all.iloc[0]) if len(f_all) >= 2 else np.nan,
                "nakayama_score": ((g[(g["venue"].astype(str).str.contains("中山")) & (g["distance"].astype(str) == "2000")]["finish_num"] <= 3).mean()) if len(g) else np.nan,
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
        "last_passing",
        "auto_running_style",
        "hist_odds_avg",
        "hist_pop_avg",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def analyze_race(race: pd.DataFrame, hist: pd.DataFrame, recent_n: int = 10, use_jockey_score: bool = True, local_mode: str = "自動", predict_mode: str = "バランス") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
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
        def _corner_to_style(x: Any) -> str:
            x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
            if pd.isna(x):
                return "不明"
            ratio = float(x) / float(field_size)
            if ratio <= 0.18:
                return "逃げ"
            if ratio <= 0.42:
                return "先行"
            if ratio <= 0.68:
                return "中団"
            return "差し"
        df["running_style"] = df["corner_avg"].apply(_corner_to_style)
    else:
        df["running_style"] = "不明"

    # 手動脚質上書き
    try:
        for i in df.index:
            key = df.loc[i, "key"]
            manual = manual_styles.get(key, "自動") if 'manual_styles' in globals() else "自動"
            if manual != "自動":
                df.loc[i, "running_style"] = manual
    except Exception:
        pass


    top_jockeys = ["ルメール", "川田", "横山武", "戸崎", "武豊"]
    fallback_jockey_score = df["jockey"].apply(
        lambda x: 1.0 if any(j in str(x) for j in top_jockeys) else 0.35
    )
    df["jockey_score_source"] = "OFF"

    if use_jockey_score:
        try:
            jockey_master = fetch_jra_jockey_scores()
        except Exception:
            jockey_master = pd.DataFrame(columns=["jockey_key", "jra_jockey_score"])

        df["jockey_key"] = df["jockey"].map(normalize_jockey_name)

        if len(jockey_master):
            df = df.merge(
                jockey_master[["jockey_key", "jra_jockey_score"]],
                on="jockey_key",
                how="left"
            )
            raw_score = pd.to_numeric(df.get("jra_jockey_score"), errors="coerce")
            matched = raw_score.notna()

            df["jockey_score"] = raw_score.fillna(fallback_jockey_score)
            df.loc[matched, "jockey_score_source"] = "JRA"
            df.loc[~matched, "jockey_score_source"] = "fallback"
        else:
            df["jockey_score"] = fallback_jockey_score
            df["jockey_score_source"] = "fallback"
    else:
        df["jockey_score"] = 0.0
        df["jockey_score_source"] = "OFF"

    # 念のため: ONなのに全件0なら fallback を強制適用
    if use_jockey_score:
        js = pd.to_numeric(df.get("jockey_score"), errors="coerce")
        if js.notna().sum() == 0 or float(js.fillna(0).max()) <= 0:
            df["jockey_score"] = fallback_jockey_score
            df["jockey_score_source"] = "fallback_forced"

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
        + (pd.to_numeric(df["growth"], errors="coerce").fillna(0) if "growth" in df.columns else 0) * 0.12
        + (pd.to_numeric(df["nakayama_score"], errors="coerce").fillna(0) if "nakayama_score" in df.columns else 0) * 0.10
        + (pd.to_numeric(df["jockey_score"], errors="coerce").fillna(0) if "jockey_score" in df.columns else 0) * 0.05
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

    local_mode = clean_text(local_mode) or "自動"
    use_local_bonus = False
    if local_mode == "ON":
        use_local_bonus = True
    elif local_mode == "OFF":
        use_local_bonus = False
    else:
        use_local_bonus = is_local_race_df(race)

    if use_local_bonus:
        df = apply_local_race_bonus(df, hist)
    else:
        df = apply_jra_race_bonus(df, hist)

    df = apply_prediction_mode(df, predict_mode)

    df = df.sort_values(["ai_score", "win_prob"], ascending=False).reset_index(drop=True)
    df = estimate_topn_probs(df, n_sim=3000, top_n_list=[3, 5], random_state=42)
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
    if not rows:
        return pd.DataFrame(columns=["券種", "組み合わせ", "馬1", "馬2", "score"])
    out = pd.DataFrame(rows)
    out["組み合わせ"] = out["組み合わせ"].apply(lambda x: normalize_combination_text(x, "-"))
    return out.sort_values("score", ascending=False)


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
    if not rows:
        return pd.DataFrame(columns=["券種", "組み合わせ", "馬1", "馬2", "馬3", "score"])
    out = pd.DataFrame(rows)
    out["組み合わせ"] = out["組み合わせ"].apply(lambda x: normalize_combination_text(x, "-"))
    return out.sort_values("score", ascending=False)


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

    umaren_df = pd.DataFrame(umaren_rows) if umaren_rows else pd.DataFrame(columns=["組み合わせ", "本命", "穴", "score"])
    wide_df = pd.DataFrame(wide_rows) if wide_rows else pd.DataFrame(columns=["組み合わせ", "本命", "穴", "score"])
    trio_df = pd.DataFrame(trio_rows) if trio_rows else pd.DataFrame(columns=["組み合わせ", "本命1", "本命2", "穴", "score"])
    trifecta_df = pd.DataFrame(trifecta_rows) if trifecta_rows else pd.DataFrame(columns=["組み合わせ", "1着", "2着", "3着", "score"])

    if len(umaren_df):
        umaren_df["組み合わせ"] = umaren_df["組み合わせ"].apply(lambda x: normalize_combination_text(x, "-"))
        umaren_df = umaren_df.sort_values("score", ascending=False)
    if len(wide_df):
        wide_df["組み合わせ"] = wide_df["組み合わせ"].apply(lambda x: normalize_combination_text(x, "-"))
        wide_df = wide_df.sort_values("score", ascending=False)
    if len(trio_df):
        trio_df["組み合わせ"] = trio_df["組み合わせ"].apply(lambda x: normalize_combination_text(x, "-"))
        trio_df = trio_df.sort_values("score", ascending=False)
    if len(trifecta_df):
        trifecta_df = trifecta_df.sort_values("score", ascending=False)

    return (umaren_df, wide_df, trio_df, trifecta_df)


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

    # まず列名ベースで厳密に読む
    try:
        tables = safe_read_html(html)
    except Exception:
        tables = []

    for raw in tables:
        try:
            df = flatten_columns(raw).copy()
            finish_col = None
            horse_no_col = None
            horse_name_col = None

            for c in df.columns:
                cc = clean_text(c)
                if cc in ["着順", "着 順"]:
                    finish_col = c
                elif cc in ["馬番", "馬 番"]:
                    horse_no_col = c
                elif "馬名" in cc:
                    horse_name_col = c

            if finish_col is None or horse_no_col is None or horse_name_col is None:
                continue

            out = pd.DataFrame({
                "着順": pd.to_numeric(df[finish_col], errors="coerce"),
                "馬番": pd.to_numeric(df[horse_no_col], errors="coerce"),
                "馬名": df[horse_name_col].astype(str).map(clean_text),
            }).dropna(subset=["着順", "馬番"])

            if len(out) == 0:
                continue

            out["着順"] = out["着順"].astype(int)
            out["馬番"] = out["馬番"].astype(int)
            out["馬名_norm"] = out["馬名"].map(normalize_horse_name)
            out = out.sort_values(["着順", "馬番"], ascending=[True, True]).drop_duplicates(subset=["馬番"], keep="first")
            return out[["着順", "馬番", "馬名", "馬名_norm"]]
        except Exception:
            continue

    # fallback: netkeiba結果表の位置で読む（0:着順 1:枠番 2:馬番 3:馬名）
    table = None
    for selector in ["table.RaceTable01", "table.race_table_01", "table.Shutuba_Table", "table"]:
        cand = soup.select_one(selector)
        if cand is not None and cand.select("tr"):
            table = cand
            break

    if table is None:
        return pd.DataFrame(columns=["着順", "馬番", "馬名", "馬名_norm"])

    rows = []
    for tr in table.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        finish = to_int(clean_text(tds[0].get_text(" ", strip=True)))
        horse_no = to_int(clean_text(tds[2].get_text(" ", strip=True)))
        horse_name = clean_text(tds[3].get_text(" ", strip=True))

        horse_a = tr.select_one('a[href*="/horse/"]')
        if horse_a is not None:
            horse_name = clean_text(horse_a.get_text())

        if finish is None or horse_no is None or not horse_name:
            continue

        rows.append({"着順": int(finish), "馬番": int(horse_no), "馬名": horse_name})

    if not rows:
        return pd.DataFrame(columns=["着順", "馬番", "馬名", "馬名_norm"])

    out = pd.DataFrame(rows)
    out["馬名_norm"] = out["馬名"].map(normalize_horse_name)
    out = out.sort_values(["着順", "馬番"], ascending=[True, True]).drop_duplicates(subset=["馬番"], keep="first")
    return out[["着順", "馬番", "馬名", "馬名_norm"]]




    for idx, t in enumerate(tables):
        try:
            flat = flatten_columns(t)
        except Exception:
            flat = t.copy()

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

    work["_pred_order"] = range(1, len(work) + 1)
    work["horse_name_norm"] = work["horse_name"].map(normalize_horse_name) if "horse_name" in work.columns else ""
    result2["horse_name_norm"] = result2["馬名"].map(normalize_horse_name) if "馬名" in result2.columns else ""

    result2["着順"] = pd.to_numeric(result2["着順"], errors="coerce")
    if "馬番" in result2.columns:
        result2["馬番"] = pd.to_numeric(result2["馬番"], errors="coerce")

    if "馬番" in result2.columns and result2["馬番"].notna().sum() > 0:
        result2 = result2.sort_values(["着順", "馬番"], na_position="last").drop_duplicates(subset=["馬番"], keep="first")
    result2 = result2.sort_values(["着順", "horse_name_norm"], na_position="last").drop_duplicates(subset=["horse_name_norm"], keep="first")

    if "horse_no" in work.columns and pd.to_numeric(work["horse_no"], errors="coerce").notna().sum() > 0:
        work["horse_no"] = pd.to_numeric(work["horse_no"], errors="coerce")
        work = work.sort_values(["_pred_order"]).drop_duplicates(subset=["horse_no"], keep="first")
    work = work.sort_values(["_pred_order"]).drop_duplicates(subset=["horse_name_norm"], keep="first")

    merged = pd.DataFrame()
    if "horse_no" in work.columns and "馬番" in result2.columns and pd.to_numeric(work["horse_no"], errors="coerce").notna().sum() > 0:
        merged = work.merge(
            result2[["着順", "馬番", "馬名", "horse_name_norm"]],
            left_on="horse_no",
            right_on="馬番",
            how="left",
            suffixes=("", "_result"),
        )

    if merged.empty:
        merged = work.copy()
        merged["着順"] = pd.NA
        merged["馬番"] = pd.NA
        merged["馬名"] = pd.NA

    missing_mask = merged["着順"].isna() if "着順" in merged.columns else pd.Series([True] * len(merged))
    if missing_mask.any():
        fill_src = result2[["着順", "馬番", "馬名", "horse_name_norm"]].copy()
        fill_map = fill_src.set_index("horse_name_norm")[["着順", "馬番", "馬名"]]
        for idx in merged[missing_mask].index:
            key = merged.at[idx, "horse_name_norm"]
            if key in fill_map.index:
                vals = fill_map.loc[key]
                if isinstance(vals, pd.DataFrame):
                    vals = vals.iloc[0]
                merged.at[idx, "着順"] = vals["着順"]
                merged.at[idx, "馬番"] = vals["馬番"]
                merged.at[idx, "馬名"] = vals["馬名"]

    merged["着順"] = pd.to_numeric(merged["着順"], errors="coerce")
    merged["予想順位"] = merged["_pred_order"]
    merged["3着内"] = merged["着順"].apply(lambda x: pd.notna(x) and x <= 3)
    merged["1着的中"] = merged["着順"].apply(lambda x: pd.notna(x) and x == 1)

    top3 = merged.sort_values("予想順位").head(3)
    top5 = merged.sort_values("予想順位").head(5)
    winner_in_top5 = bool((top5["1着的中"] == True).any())
    honmei_finish = None
    honmei_row = merged.sort_values("予想順位").head(1)
    if len(honmei_row) and pd.notna(honmei_row["着順"].iloc[0]):
        honmei_finish = int(honmei_row["着順"].iloc[0])

    summary = {
        "◎の着順": honmei_finish,
        "上位3頭中の3着内頭数": int(pd.to_numeric(top3["3着内"], errors="coerce").fillna(0).astype(int).sum()),
        "上位5頭中の3着内頭数": int(pd.to_numeric(top5["3着内"], errors="coerce").fillna(0).astype(int).sum()),
        "勝ち馬を上位5頭に含む": winner_in_top5,
    }

    cols = [c for c in ["mark", "horse_no", "horse_name", "running_style", "予想順位", "着順", "3着内", "1着的中"] if c in merged.columns]
    view = merged.sort_values("予想順位")[cols].copy()
    return view, summary


def ticket_hit_summary(result_df: pd.DataFrame, ticket_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    winners = result_df.copy()
    winners["着順"] = pd.to_numeric(winners["着順"], errors="coerce")
    winners["馬番"] = pd.to_numeric(winners["馬番"], errors="coerce")
    winners = winners.sort_values(["着順", "馬番"], ascending=[True, True]).head(3)

    top3_nums = [int(x) for x in winners["馬番"].dropna().tolist()]
    top2_nums = top3_nums[:2]

    rows = []
    for key, df in ticket_tables.items():
        hit = False
        hit_row = ""
        if df is None or len(df) == 0:
            rows.append({"券種": key, "的中": "", "該当": ""})
            continue

        for _, row in df.iterrows():
            comb = clean_text(row.get("組み合わせ", ""))
            nums = [int(x) for x in re.findall(r"\d+", comb)]

            if key in ["ワイド", "本命+穴ワイド"]:
                if len(nums) >= 2 and set(nums[:2]).issubset(set(top3_nums)):
                    hit = True
                    hit_row = comb
                    break
            elif key in ["馬連", "本命+穴馬連"]:
                if len(nums) >= 2 and set(nums[:2]) == set(top2_nums):
                    hit = True
                    hit_row = comb
                    break
            elif key in ["3連複", "本命2頭+穴1頭"]:
                if len(nums) >= 3 and set(nums[:3]) == set(top3_nums):
                    hit = True
                    hit_row = comb
                    break
            elif key in ["3連単", "本命_本命穴"]:
                if len(nums) >= 3 and nums[:3] == top3_nums[:3]:
                    hit = True
                    hit_row = comb
                    break

        rows.append({"券種": key, "的中": "当" if hit else "", "該当": hit_row})
    return pd.DataFrame(rows)



# ================= 手動補正設定 =================
manual_styles = {}

# 上部に空のUIを出さないよう初期値のみ保持
if "weather_override" not in st.session_state:
    st.session_state["weather_override"] = "自動"
if "ground_override" not in st.session_state:
    st.session_state["ground_override"] = "自動"


def highlight_hit_cell(val: Any) -> str:
    return "color: red; font-weight: bold;" if str(val) == "当" else ""



# ============================================================
# 画面イメージ準拠ロジック設定
# ============================================================
HONMEI_COUNT_DEFAULT = 5
ANA_COUNT_DEFAULT = 5
ANA_POP_MIN_DEFAULT = 4
ANA_ODDS_MIN_DEFAULT = 10.0
ANA_GAP_MIN_DEFAULT = 0.0


def select_honmei_and_ana(
    df: pd.DataFrame,
    honmei_n: int = HONMEI_COUNT_DEFAULT,
    ana_n: int = ANA_COUNT_DEFAULT,
    ana_pop_min: int = ANA_POP_MIN_DEFAULT,
    ana_odds_min: float = ANA_ODDS_MIN_DEFAULT,
    ana_gap_min: float = ANA_GAP_MIN_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if len(work) == 0:
        return work.copy(), work.copy()

    is_local = is_local_race_df(work)
    pace = estimate_race_pace(work)

    honmei = work.sort_values(["win_prob", "ai_score"], ascending=False).head(max(1, int(honmei_n))).copy()
    honmei_nums = set()
    if "horse_no" in honmei.columns:
        honmei_nums = set(pd.to_numeric(honmei["horse_no"], errors="coerce").dropna().astype(int).tolist())

    def _exclude_honmei(base_df: pd.DataFrame) -> pd.DataFrame:
        out = base_df.copy()
        if "horse_no" in out.columns and len(honmei_nums) > 0:
            out = out[
                ~pd.to_numeric(out["horse_no"], errors="coerce").fillna(-9999).astype(int).isin(honmei_nums)
            ].copy()
        return out

    def _dedupe_by_horse(base_df: pd.DataFrame) -> pd.DataFrame:
        out = base_df.copy()
        if "horse_no" in out.columns:
            out = out.drop_duplicates(subset=["horse_no"], keep="first").copy()
        return out

    def _fallback_fill(ana_df: pd.DataFrame, target_n: int, score_cols: List[str]) -> pd.DataFrame:
        """
        条件通過馬が target_n に足りない時、本命以外の残りから補充して必ず target_n まで埋める。
        """
        target_n = max(1, int(target_n))
        ana_df = _dedupe_by_horse(ana_df)

        if len(ana_df) >= target_n:
            return ana_df.head(target_n).copy()

        rest = _exclude_honmei(work)
        if "horse_no" in ana_df.columns and len(ana_df) > 0:
            picked_nums = set(pd.to_numeric(ana_df["horse_no"], errors="coerce").dropna().astype(int).tolist())
            rest = rest[
                ~pd.to_numeric(rest["horse_no"], errors="coerce").fillna(-9999).astype(int).isin(picked_nums)
            ].copy()

        if len(rest) == 0:
            return ana_df.head(target_n).copy()

        # ana_score が無ければ ai_score / place_prob / gap で代用
        if "ana_score" not in rest.columns:
            rest["ana_score"] = (
                pd.to_numeric(rest.get("ai_score"), errors="coerce").fillna(0) * 0.60
                + pd.to_numeric(rest.get("place_prob"), errors="coerce").fillna(0) * 0.30
                + pd.to_numeric(rest.get("gap"), errors="coerce").fillna(0).clip(lower=0) * 0.80
            )

        order_cols = [c for c in score_cols if c in rest.columns]
        if not order_cols:
            order_cols = ["ana_score"]
        rest = rest.sort_values(order_cols, ascending=[False] * len(order_cols)).copy()

        need = max(0, target_n - len(ana_df))
        fill_df = rest.head(need).copy()

        out = pd.concat([ana_df, fill_df], ignore_index=True)
        out = _dedupe_by_horse(out)
        order_cols2 = [c for c in score_cols if c in out.columns]
        if not order_cols2:
            order_cols2 = ["ana_score"]
        out = out.sort_values(order_cols2, ascending=[False] * len(order_cols2)).head(target_n).copy()
        return out

    if is_local:
        local_ana_n = max(1, int(ana_n))

        ana = _exclude_honmei(work)

        if "pop_f" in ana.columns:
            pop_s = pd.to_numeric(ana["pop_f"], errors="coerce")
            ana = ana[pop_s >= max(ana_pop_min, 4)].copy()

        if "gap" in ana.columns:
            gap_s = pd.to_numeric(ana["gap"], errors="coerce").fillna(0)
            ana = ana[gap_s > ana_gap_min].copy()

        if len(ana) > 0:
            ana["front_keep_bonus"] = np.where(ana["running_style"].isin(["逃げ", "先行"]), 0.06, 0.0)
            ana["ana_score"] = pd.to_numeric(ana.get("ana_score"), errors="coerce").fillna(0) + ana["front_keep_bonus"]
        else:
            ana = _exclude_honmei(work)
            ana["front_keep_bonus"] = np.where(ana["running_style"].isin(["逃げ", "先行"]), 0.06, 0.0)
            ana["ana_score"] = pd.to_numeric(ana.get("ai_score"), errors="coerce").fillna(0) + ana["front_keep_bonus"]

        if "running_style" in work.columns:
            nige_cnt = int((work["running_style"] == "逃げ").sum())
            senko_cnt = int((work["running_style"] == "先行").sum())
            ai_median = pd.to_numeric(work.get("ai_score"), errors="coerce").fillna(0).median()

            if (nige_cnt + senko_cnt) >= 5:
                sash = work[work["running_style"].isin(["差", "追込"])].copy()
                sash = _exclude_honmei(sash)
                if len(sash) > 0:
                    sash_ana = pd.to_numeric(sash["ana_score"], errors="coerce").fillna(0) if "ana_score" in sash.columns else 0
                    sash["sash_bonus"] = pd.to_numeric(sash.get("ai_score"), errors="coerce").fillna(0) + sash_ana * 0.30
                    sash = sash.sort_values(["sash_bonus", "win_prob"], ascending=False).copy()
                    top_sash_score = float(pd.to_numeric(pd.Series([sash.iloc[0]["ai_score"]]), errors="coerce").fillna(0).iloc[0])
                    if top_sash_score > ai_median:
                        ana = pd.concat([ana, sash.head(1).copy()], ignore_index=True)
                        ana = _dedupe_by_horse(ana)

            if nige_cnt >= 2:
                nige = work[work["running_style"] == "逃げ"].copy()
                nige = _exclude_honmei(nige)
                if len(nige) > 0:
                    nige["nige_bonus"] = pd.to_numeric(nige.get("ai_score"), errors="coerce").fillna(0) + pd.to_numeric(nige.get("win_prob"), errors="coerce").fillna(0) * 0.20
                    nige = nige.sort_values(["nige_bonus", "win_prob"], ascending=False).copy()
                    top_nige_score = float(pd.to_numeric(pd.Series([nige.iloc[0]["ai_score"]]), errors="coerce").fillna(0).iloc[0])
                    if top_nige_score > ai_median:
                        ana = pd.concat([ana, nige.head(1).copy()], ignore_index=True)
                        ana = _dedupe_by_horse(ana)

        ana = ana.sort_values(["ana_score", "gap", "win_prob"], ascending=False).copy()
        ana = _fallback_fill(ana, local_ana_n, ["ana_score", "gap", "win_prob"])
        return honmei, ana

    jra_ana_n = max(1, int(ana_n))

    ana = _exclude_honmei(work)

    if "pop_f" in ana.columns:
        pop_s = pd.to_numeric(ana["pop_f"], errors="coerce")
        ana = ana[(pop_s >= max(ana_pop_min, 5)) & (pop_s <= 10)].copy()

    if "odds_f" in ana.columns:
        odds_s = pd.to_numeric(ana["odds_f"], errors="coerce")
        ana = ana[(odds_s >= max(ana_odds_min, 8.0)) | (odds_s.isna())].copy()

    if "running_style" in ana.columns:
        if pace == "ハイ":
            ana["pace_fit_bonus"] = ana["running_style"].map({"差": 0.08, "追込": 0.07, "先行": -0.02, "逃げ": -0.04}).fillna(0)
        elif pace == "スロー":
            ana["pace_fit_bonus"] = ana["running_style"].map({"逃げ": 0.08, "先行": 0.06, "差": -0.02, "追込": -0.04}).fillna(0)
        else:
            ana["pace_fit_bonus"] = ana["running_style"].map({"先行": 0.03, "差": 0.03}).fillna(0)
    else:
        ana["pace_fit_bonus"] = 0.0

    gap = pd.to_numeric(ana.get("gap"), errors="coerce").fillna(0)
    place = pd.to_numeric(ana.get("place_prob"), errors="coerce").fillna(0)
    pop = pd.to_numeric(ana.get("pop_f"), errors="coerce").replace(0, np.nan)
    ana["ana_score"] = place * 0.45 + gap.clip(lower=0) * 0.25 + ana["pace_fit_bonus"] + (1 / pd.Series(pop)).replace([np.inf, -np.inf], np.nan).fillna(0) * 0.10

    if len(ana) == 0:
        ana = _exclude_honmei(work)
        ana["ana_score"] = (
            pd.to_numeric(ana.get("place_prob"), errors="coerce").fillna(0)
            + pd.to_numeric(ana.get("gap"), errors="coerce").fillna(0).clip(lower=0)
        )

    ana = ana.sort_values(["ana_score", "place_prob", "gap"], ascending=False).copy()
    ana = _fallback_fill(ana, jra_ana_n, ["ana_score", "place_prob", "gap"])
    return honmei, ana


def build_locked_candidate_pool(honmei: pd.DataFrame, ana: pd.DataFrame, total_n: Optional[int] = None) -> pd.DataFrame:
    """表示中の本命候補・穴候補から買い目用プールを作る。total_n=None なら全表示馬を許可。"""
    frames = []
    if honmei is not None and len(honmei) > 0:
        h = honmei.copy()
        h["_pool_src"] = "honmei"
        frames.append(h)
    if ana is not None and len(ana) > 0:
        a = ana.copy()
        a["_pool_src"] = "ana"
        frames.append(a)

    if not frames:
        return pd.DataFrame()

    pool = pd.concat(frames, ignore_index=True)
    if "horse_no" in pool.columns:
        pool = pool.drop_duplicates(subset=["horse_no"], keep="first").copy()

    pool["_src_rank"] = pool["_pool_src"].map({"honmei": 0, "ana": 1}).fillna(9)
    pool["_ai"] = pd.to_numeric(pool.get("ai_score"), errors="coerce").fillna(0)
    pool["_win"] = pd.to_numeric(pool.get("win_prob"), errors="coerce").fillna(0)
    pool = pool.sort_values(["_src_rank", "_ai", "_win"], ascending=[True, False, False]).copy()

    if total_n is not None:
        pool = pool.head(int(total_n)).copy()

    return pool.drop(columns=["_src_rank", "_ai", "_win"], errors="ignore")


def assert_ticket_source_locked(ticket_df: pd.DataFrame, locked_pool: pd.DataFrame) -> None:
    """買い目に表示外の馬が混ざったら即エラー。"""
    if ticket_df is None or len(ticket_df) == 0 or locked_pool is None or len(locked_pool) == 0:
        return

    allowed = set(pd.to_numeric(locked_pool["horse_no"], errors="coerce").dropna().astype(int).tolist())
    if not allowed:
        return

    bad = []
    for _, row in ticket_df.iterrows():
        combo = str(row.get("組み合わせ", ""))
        nums = [int(x) for x in re.findall(r"\d+", combo)]
        if any(n not in allowed for n in nums):
            bad.append(combo)

    if bad:
        raise ValueError("買い目生成エラー: 表示中の本命候補・穴候補にない馬が混ざっています -> " + ", ".join(bad[:5]))



def estimate_topn_probs(df: pd.DataFrame, n_sim: int = 3000, top_n_list: List[int] = [3, 5], random_state: int = 42) -> pd.DataFrame:
    """
    ai_score から順位シミュレーションを行い、TopN率を推定する。
    Plackett-Luce 近似で、各着順を重み付き無作為抽出で埋める。
    """
    out = df.copy()
    if len(out) == 0 or "ai_score" not in out.columns:
        for n in top_n_list:
            out[f"top{n}_prob"] = np.nan
        return out

    rng = np.random.default_rng(random_state)
    scores = pd.to_numeric(out["ai_score"], errors="coerce").fillna(0).to_numpy(dtype=float)

    # 数値安定化
    weights = np.exp(scores - np.nanmax(scores))
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(out), dtype=float)

    counts = {n: np.zeros(len(out), dtype=float) for n in top_n_list}
    idx_all = np.arange(len(out))

    for _ in range(int(n_sim)):
        remaining_idx = idx_all.copy()
        remaining_w = weights.copy()
        rank_order = []

        for _pos in range(len(out)):
            total_w = remaining_w.sum()
            if total_w <= 0 or len(remaining_idx) == 0:
                break
            probs = remaining_w / total_w
            chosen_pos = rng.choice(len(remaining_idx), p=probs)
            chosen_idx = remaining_idx[chosen_pos]
            rank_order.append(chosen_idx)

            remaining_idx = np.delete(remaining_idx, chosen_pos)
            remaining_w = np.delete(remaining_w, chosen_pos)

        if not rank_order:
            continue

        for n in top_n_list:
            top_slice = rank_order[:min(n, len(rank_order))]
            for idx in top_slice:
                counts[n][idx] += 1.0

    for n in top_n_list:
        out[f"top{n}_prob"] = counts[n] / float(n_sim)

    return out


def render_results(race: pd.DataFrame, hist: pd.DataFrame, recent_n: int, ana_count: int, ticket_head_count: int, wide_count: int, umaren_count: int, trio_count: int, trifecta_count: int) -> None:
    st.markdown("""
    <style>
    .nk-card{border:1px solid #dbe7f5;border-radius:16px;padding:16px 18px;background:#ffffff;}
    .nk-soft{border:1px solid #dbe7f5;border-radius:14px;padding:12px 14px;background:#f8fbff;}
    .nk-green{border:1px solid #d8eee2;border-radius:14px;padding:12px 14px;background:#f7fffa;}
    .nk-memo{border:1px solid #f1e4bf;border-radius:14px;padding:12px 14px;background:#fffaf0;}
    .nk-title{font-size:2rem;font-weight:700;margin-bottom:0.2rem;}
    .nk-sub{color:#4b5563;margin-bottom:1rem;}
    .nk-section{font-size:1.6rem;font-weight:700;margin:0.1rem 0 0.5rem 0;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="nk-title">🐾 にゃんこ競馬予想AI 🐱</div><div class="nk-sub">データ分析 × AI予想で勝率アップ！</div>', unsafe_allow_html=True)
    if st.session_state.get("model_just_trained", False):
        st.success("学習済みモデルを反映して再予想済みです。")
        st.session_state["model_just_trained"] = False

    use_jockey_score = st.session_state.get("rr_use_jockey_score", st.session_state.get("sb_use_jockey_score", True))
    model_bundle = None
    local_mode = st.session_state.get("rr_local_mode", st.session_state.get("sb_local_mode", "自動"))
    predict_mode = st.session_state.get("rr_predict_mode", st.session_state.get("sb_predict_mode", "バランス"))
    # サイドバーの重複ウィジェットは作らない。
    # 本命・穴候補数などはメイン画面の設定値だけを参照する。
    honmei_count = int(st.session_state.get("rr_honmei_count", 6))
    ana_count_logic = int(st.session_state.get("rr_ana_count_logic", 6))
    ana_pop_min = int(st.session_state.get("rr_ana_pop_min", ANA_POP_MIN_DEFAULT))
    ana_odds_min = float(st.session_state.get("rr_ana_odds_min", ANA_ODDS_MIN_DEFAULT))
    ana_gap_min = float(st.session_state.get("rr_ana_gap_min", ANA_GAP_MIN_DEFAULT))


    df, top, honmei, ana, odds_valid = analyze_race(
        race,
        hist,
        recent_n=recent_n,
        use_jockey_score=use_jockey_score,
        local_mode=local_mode,
        predict_mode=predict_mode,
    )
    df = dedupe_race_df(df)

    weather_override = st.session_state.get("weather_override", "自動")
    ground_override = st.session_state.get("ground_override", "自動")
    if weather_override != "自動":
        df["weather"] = weather_override
    if ground_override != "自動":
        df["ground"] = ground_override

    df = apply_style_ground_pace_adjustments(df)
    honmei, ana = select_honmei_and_ana(
        df,
        honmei_n=honmei_count,
        ana_n=ana_count_logic,
        ana_pop_min=ana_pop_min,
        ana_odds_min=ana_odds_min,
        ana_gap_min=ana_gap_min,
    )

    display_cols = ["mark", "horse_no", "frame_no", "horse_name", "last_passing", "running_style", "jockey", "jockey_score", "ai_score", "win_prob", "place_prob", "ana_score"]
    if odds_valid:
        display_cols += ["odds_f", "ev_tansho"]
    if "pop_f" in df.columns:
        display_cols += ["pop_f"]

    used_weather = clean_text(df["weather"].dropna().iloc[0]) if "weather" in df.columns and df["weather"].notna().any() else "不明"
    used_ground = clean_text(df["ground"].dropna().iloc[0]) if "ground" in df.columns and df["ground"].notna().any() else "不明"
    try:
        jockey_master_count = len(fetch_jra_jockey_scores())
    except Exception:
        jockey_master_count = 0

    if "jockey_score_source" in df.columns:
        if (df["jockey_score_source"] == "OFF").all():
            jockey_source = "騎手スコアOFF"
        elif (df["jockey_score_source"] == "JRA").any():
            fallback_cnt = int((df["jockey_score_source"] == "fallback").sum())
            jockey_source = f"JRA騎手データ {jockey_master_count}件 / fallback {fallback_cnt}件"
        else:
            jockey_source = "JRA騎手データ取得失敗 → 簡易騎手補正"
    else:
        jockey_source = f"JRA騎手データ {jockey_master_count}件" if jockey_master_count > 0 else "JRA騎手データ取得失敗 → 簡易騎手補正"

    st.caption(f"予想に使用した条件: 天候={used_weather} / 馬場={used_ground} / 騎手={jockey_source}")
    with st.expander("詳細ランキング / 脚質調整", expanded=False):

        st.subheader("騎手スコア確認")
        if "jockey_score" in df.columns:
            show_cols = [c for c in ["horse_name", "jockey", "jockey_score", "jockey_score_source", "ai_score"] if c in df.columns]
            show_df = df[show_cols].sort_values("ai_score", ascending=False) if "ai_score" in df.columns else df[show_cols]
            if "jockey_score" in show_df.columns:
                show_df["jockey_score"] = pd.to_numeric(show_df["jockey_score"], errors="coerce").round(3)
            st.dataframe(
                show_df,
                use_container_width=True
            )
        else:
            st.warning("騎手スコアが存在しません（取得失敗 or 未反映）")

        st.subheader("総合ランキング")
        st.caption("前走の通過(last_passing)と自動判定した脚質(running_style)を表示します。")
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
                "last_passing": st.column_config.TextColumn("前走通過", disabled=True),
                "running_style": st.column_config.SelectboxColumn(
                    "running_style",
                    options=["逃げ", "先行", "差", "追込", "不明"],
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

    honmei, ana = select_honmei_and_ana(df, honmei_n=honmei_count, ana_n=ana_count_logic, ana_pop_min=ana_pop_min, ana_odds_min=ana_odds_min, ana_gap_min=ana_gap_min)
    displayed_pool = build_locked_candidate_pool(honmei, ana, total_n=None)

    # ユーザー設定の「買い目に使う上位頭数」を実際の買い目生成に反映
    # ただし表示中の本命候補・穴候補の範囲からはみ出さない
    ticket_pool_n = max(3, int(ticket_head_count)) if ticket_head_count is not None else 4
    locked_pool = build_locked_candidate_pool(honmei, ana, total_n=ticket_pool_n)
    top = locked_pool.copy()

    course_text = clean_text(race["course"].iloc[0]) if "course" in race.columns and len(race) else "-"
    race_name_text = clean_text(race["race_name"].iloc[0]) if "race_name" in race.columns and len(race) else "-"
    race_date_text = clean_text(race["race_date"].iloc[0]) if "race_date" in race.columns and len(race) else "-"
    top_honmei_name = honmei.iloc[0]["horse_name"] if len(honmei) else "-"
    top_ana_name = ana.iloc[0]["horse_name"] if len(ana) else "-"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="nk-card"><div class="nk-section">レース情報</div><div>{race_name_text}</div><div>{course_text}</div><div>{race_date_text}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="nk-card"><div class="nk-section">天候・馬場</div><div>{used_weather}</div><div>馬場：{used_ground}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="nk-card"><div class="nk-section">出走頭数</div><div>{len(race)}頭</div><div>過去成績 {len(hist)}件</div><div>買い目生成母数 {len(top)}頭</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="nk-card"><div class="nk-section">予想モード</div><div>{predict_mode}</div><div>地方モード：{local_mode}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="nk-card"><div class="nk-section">注目馬</div><div>本命：{top_honmei_name}</div><div>穴：{top_ana_name}</div></div>', unsafe_allow_html=True)

    if model_bundle is None:
        st.caption("学習済みモデル未使用: model_top3_prob / model_value_score は空欄になります。")

    ana_cols = ["horse_no", "horse_name", "running_style", "win_prob", "top5_prob", "model_top3_prob", "model_value_score", "market_prob", "gap", "ana_score"]
    if odds_valid:
        ana_cols.insert(2, "odds_f")
    if "pop_f" in ana.columns:
        ana_cols.insert(3 if odds_valid else 2, "pop_f")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="nk-card">', unsafe_allow_html=True)
        st.markdown("## 👑 本命候補")
        st.caption("本命候補は「勝つ確率 (win_prob)」と「AIスコア」を重視")
        honmei_show_cols = [c for c in ["mark", "horse_no", "horse_name", "win_prob", "place_prob", "top5_prob", "model_top3_prob", "model_value_score", "ai_score", "running_style"] if c in honmei.columns]
        st.dataframe(safe_pick_cols(honmei, honmei_show_cols), use_container_width=True, hide_index=True)
        st.markdown('<div class="nk-soft">本命候補は「勝つ確率 (win_prob)」「5着内率 (top5_prob)」「AIスコア」を参考に選出しています。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="nk-card">', unsafe_allow_html=True)
        st.markdown("## 🎯 穴候補")
        st.caption("穴候補は「本命候補と重複なし」かつ「人気・オッズ・gap(期待値)」で抽出")
        ana_show_cols = [c for c in ana_cols if c in ana.columns]
        st.dataframe(safe_pick_cols(ana, ana_show_cols), use_container_width=True, hide_index=True)
        st.markdown('<div class="nk-green">穴候補は「人気（薄め）」かつ「妙味スコア（期待値）」の高い馬を抽出しています。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1.2, 1.1, 0.9])
    with b1:
        st.markdown('<div class="nk-card"><div class="nk-section">💡 ロジックのポイント</div><div>・本命候補は「勝つ確率」と「AIスコア」を重視</div><div>・穴候補は本命候補と重複なし</div><div>・穴候補は「期待値 (gap)」の高い順に表示</div><div>・脚質・コース適性・過去成績・血統を総合分析</div></div>', unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div class="nk-card"><div class="nk-section">⚙ 穴候補の抽出条件</div><div>人気フィルター：{ana_pop_min}番人気以下</div><div>オッズフィルター：{ana_odds_min:.1f}倍以上</div><div>妙味フィルター：gap &gt; {ana_gap_min:.3f}</div></div>', unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="nk-memo"><div class="nk-section">🐱 にゃんこメモ</div><div>本命候補と穴候補は、それぞれ異なる基準で抽出しています。</div><div>基本的に重複しないように設計していますが、条件を緩めると重複する場合があります。</div></div>', unsafe_allow_html=True)

    cta1, cta2, cta3, cta4, cta5 = st.columns(5)
    cta1.button("出馬表を確認", use_container_width=True)
    cta2.button("オッズを確認", use_container_width=True)
    cta3.button("過去成績を確認", use_container_width=True)
    cta4.button("予想印を更新", use_container_width=True)
    cta5.button("馬券推奨を見る", use_container_width=True)

    try:
        model_bundle = load_trained_models(model_dir=st.session_state.get("trained_model_dir", "models"))
    except Exception:
        model_bundle = None
    df = apply_trained_models_to_prediction_df(df, model_bundle)
    top = apply_trained_models_to_prediction_df(top, model_bundle)
    honmei = apply_trained_models_to_prediction_df(honmei, model_bundle)
    ana = apply_trained_models_to_prediction_df(ana, model_bundle)

    wide_df = build_pair_table(top, "place_prob", "ワイド")
    umaren_df = build_pair_table(top, "win_prob", "馬連")
    trio_df = build_trio_table_strong(top, limit=max(50, trio_count * 2))
    trifecta_df = build_trifecta_table_strong(top, limit=max(100, trifecta_count * 2))
    honmei_ana_umaren, honmei_ana_wide, honmei_ana_trio, honmei_ana_trifecta = build_honmei_ana_tables(honmei, ana)

    assert_ticket_source_locked(wide_df, locked_pool)
    assert_ticket_source_locked(umaren_df, locked_pool)
    assert_ticket_source_locked(trio_df, locked_pool)
    assert_ticket_source_locked(trifecta_df, locked_pool)
    assert_ticket_source_locked(honmei_ana_wide, displayed_pool)
    assert_ticket_source_locked(honmei_ana_umaren, displayed_pool)
    assert_ticket_source_locked(honmei_ana_trio, displayed_pool)
    assert_ticket_source_locked(honmei_ana_trifecta, displayed_pool)

    st.caption(f"買い目に使う上位頭数設定: {len(top)}頭（表示中の本命候補＋穴候補の範囲内で反映）")

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
        st.caption("三連複AI強化版: 脚質バランス + gap + 複勝安定度でスコアリング")
        st.dataframe(trio_df.head(trio_count), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.caption("三連単AI強化版: 1着勝率 + 2着安定度 + 3着妙味でスコアリング")
        st.dataframe(trifecta_df.head(trifecta_count), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(honmei_ana_wide.head(wide_count), use_container_width=True, hide_index=True)
    with tabs[5]:
        st.dataframe(honmei_ana_umaren.head(umaren_count), use_container_width=True, hide_index=True)
    with tabs[6]:
        st.dataframe(honmei_ana_trio.head(trio_count), use_container_width=True, hide_index=True)
    with tabs[7]:
        st.dataframe(honmei_ana_trifecta.head(trifecta_count), use_container_width=True, hide_index=True)

    st.markdown("## 📊 回収率ログ")
    total_buy, total_payout, total_rr, log_df = load_log_summary()
    hit_rate = (pd.to_numeric(log_df["hit"], errors="coerce").fillna(0).astype(float).mean() * 100) if len(log_df) else 0.0
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("累計購入額", f"{total_buy:,}円")
    lc2.metric("累計払戻", f"{total_payout:,}円")
    lc3.metric("累計回収率", f"{total_rr:.1f}%")
    lc4.metric("累計的中率", f"{hit_rate:.1f}%")

    with st.expander("ログを追加", expanded=False):
        race_name_for_log = race_name_text if "race_name_text" in locals() else (clean_text(race["race_name"].iloc[0]) if "race_name" in race.columns and len(race) else "")
        log_race_name = st.text_input("レース名", value=race_name_for_log)
        log_bet_type = st.selectbox("券種", ["ワイド", "馬連", "3連複", "3連単", "本命+穴ワイド", "本命+穴馬連", "本命2頭+穴1頭", "本命→本命穴"])
        log_buy = st.number_input("購入額", min_value=0, value=1000, step=100)
        log_payout = st.number_input("払戻額", min_value=0, value=0, step=100)
        log_hit = st.checkbox("的中", value=False)
        if st.button("ログ保存", use_container_width=True):
            save_log(
                race_name=log_race_name,
                mode=predict_mode if "predict_mode" in locals() else "バランス",
                bet_type=log_bet_type,
                buy=int(log_buy),
                payout=int(log_payout),
                hit=bool(log_hit),
            )
            st.success("ログを保存したにゃ")
            st.rerun()

    st.dataframe(log_df.tail(30), use_container_width=True, hide_index=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("race_log.csv をダウンロード", f.read(), file_name="race_log.csv", mime="text/csv", use_container_width=True)

    with st.expander("券種別ログ自動集計", expanded=False):
        bet_summary_df = summarize_log_by_bet_type(log_df)
        st.dataframe(bet_summary_df, use_container_width=True, hide_index=True)

    with st.expander("学習データ取り込み / 自動学習", expanded=False):
        st.caption("学習CSVを読み込んで、オッズ連動モデルを保存します。保存後は予想に自動反映されます。")
        train_file = st.file_uploader("学習用CSV", type=["csv"], key="train_csv_uploader")
        model_dir = st.text_input("モデル保存先", value="models", key="model_dir_input")
        c_train1, c_train2 = st.columns(2)
        if c_train1.button("学習を実行", use_container_width=True, key="run_training_button"):
            try:
                if train_file is None:
                    raise ValueError("学習用CSVを選択してください。")
                train_df = load_csv_upload(train_file)
                model_paths = train_odds_linked_models(train_df, model_dir=model_dir)

                # 学習済みモデルの保存先を予想側でも使う
                st.session_state["trained_model_dir"] = model_dir
                st.session_state["model_just_trained"] = True

                # 既に予想データがある場合は自動で再計算させる
                if st.session_state.get("race_df_store") is not None and st.session_state.get("hist_df_store") is not None:
                    st.session_state["prediction_ready"] = True
                    st.success("学習完了。学習済みモデルを読み込んで自動再予想します。")
                    st.rerun()
                else:
                    st.success(f"学習完了: {model_paths['top3_model']}。予想データを取得すると自動反映されます。")
            except Exception as e:
                st.info(f"学習を完了できませんでした: {e}")

        if c_train2.button("保存済みモデル確認", use_container_width=True, key="check_training_button"):
            st.session_state["trained_model_dir"] = model_dir
            bundle = load_trained_models(model_dir=model_dir)
            if bundle is None:
                st.warning("保存済みモデルはありません。")
            else:
                st.success("保存済みモデルを検出しました。次回予想に反映されます。")

        if st.button("学習済みモデルで再予想", use_container_width=True, key="rerun_prediction_after_training"):
            if st.session_state.get("race_df_store") is not None and st.session_state.get("hist_df_store") is not None:
                st.session_state["trained_model_dir"] = model_dir
                st.session_state["prediction_ready"] = True
                st.rerun()
            else:
                st.warning("先に出馬表を取得、またはCSVを読み込んで予想してください。")

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
            st.dataframe(hit_df.style.map(highlight_hit_cell, subset=["的中"]), use_container_width=True, hide_index=True)

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


# ============================================================
# 設定画面（白テーマ）
# ============================================================

with st.sidebar:
    st.markdown('<div class="nk-side-card"><div class="nk-side-label">🐾 にゃんこ競馬予想AI 🐱</div><div class="nk-side-sub">データ分析 × AI予想で勝率アップ！</div></div>', unsafe_allow_html=True)

    st.checkbox("騎手スコアを使う", value=True, key="sb_use_jockey_score")
    st.selectbox("地方モード", ["自動", "OFF", "ON"], index=0, key="sb_local_mode")
    st.selectbox("予想モード", ["バランス", "的中率重視", "回収率重視"], index=0, key="sb_predict_mode")

    st.markdown('<div class="nk-side-card"><div class="nk-side-label">クイック設定</div><div class="nk-side-sub">本命・穴候補ロジック</div></div>', unsafe_allow_html=True)
    _side_qp_options_main = ["カスタム", "的中率", "バランス", "回収率", "爆穴", "三連複特化"]
    if "sidebar_quick_profile_main" not in st.session_state:
        st.session_state["sidebar_quick_profile_main"] = st.session_state.get("rr_quick_profile", "カスタム")
    st.selectbox(
        "クイック設定",
        _side_qp_options_main,
        key="sidebar_quick_profile_main",
        on_change=lambda: _apply_quick_profile(st.session_state.get("sidebar_quick_profile_main", "カスタム")),
    )

    if "SELENIUM_AVAILABLE" not in globals():
        SELENIUM_AVAILABLE = False
    st.caption(f"Selenium利用可能: {'はい' if SELENIUM_AVAILABLE else 'いいえ'}")

use_jockey_score = st.session_state.get("sb_use_jockey_score", True)
local_mode = st.session_state.get("sb_local_mode", "自動")
predict_mode = st.session_state.get("sb_predict_mode", "バランス")

title_col, action_col = st.columns([5, 2])
with title_col:
    st.markdown(f'<div class="nk-settings-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="nk-settings-sub">AIの予想ロジックと各種条件を詳細に設定できます</div>', unsafe_allow_html=True)
with action_col:
    ac1, ac2 = st.columns(2)
    ac1.button("リセット", use_container_width=True, key="reset_settings_dummy")
    ac2.button("保存", type="primary", use_container_width=True, key="save_settings_dummy")

st.markdown('### 実行モード')
st.caption('予想の実行方法を選択してください')

mode = st.radio(
    "実行モード",
    ["URLから取得して予想", "CSVアップロード", "ローカルファイル（コマンド用）"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_mode",
)

if mode == "URLから取得して予想":
    rc1, rc2 = st.columns([5,1])
    race_url = rc1.text_input("netkeiba 出馬表URL", DEFAULT_RACE_URL, key="main_race_url")
    open_clicked = rc2.button("URLを開く", use_container_width=True, key="open_url_dummy")
    max_horses = st.number_input("取得頭数", min_value=1, max_value=18, value=18, key="main_max_horses")

elif mode == "CSVアップロード":
    race_url = DEFAULT_RACE_URL
    max_horses = 18
    c1, c2 = st.columns(2)
    with c1:
        race_file = st.file_uploader("race_card.csv", type="csv", key="main_race_file")
    with c2:
        hist_file = st.file_uploader("horse_history.csv", type="csv", key="main_hist_file")
else:
    race_url = DEFAULT_RACE_URL
    max_horses = 18

st.markdown('### 詳細設定')
st.caption('天候・馬場の上書きは予想計算の直前に反映されます')
d1, d2 = st.columns(2)
with d1:
    weather_override = st.selectbox("天候上書き", ["自動", "晴", "曇", "雨", "小雨", "雪"], key="weather_override")
with d2:
    ground_override = st.selectbox("馬場上書き", ["自動", "良", "稍重", "重", "不良"], key="ground_override")
st.caption("脚質の手動上書きは、レース取得後の『詳細ランキング / 脚質調整』から行えます。天候・馬場はこの画面で即反映されます。")

st.divider()
st.markdown('### 本命・穴候補のロジック設定')
st.caption('クイック設定でも個別調整でも変更できます。変更内容は予想結果に反映されます')

q1, q2, q3 = st.columns([2.0, 1, 1])
with q1:
    _main_qp_options = ["カスタム", "的中率", "バランス", "回収率", "爆穴", "三連複特化"]
    st.selectbox(
        "クイック設定",
        _main_qp_options,
        index=_main_qp_options.index(st.session_state.get("rr_quick_profile", "カスタム")),
        key="main_quick_profile",
        on_change=_sync_main_quick_profile,
        help="本命・穴候補ロジック以外の主要設定もまとめて切り替えます。"
    )
with q2:
    st.metric("本命候補", f'{int(st.session_state.get("rr_honmei_count", 6))}頭')
with q3:
    st.metric("穴候補", f'{int(st.session_state.get("rr_ana_count_logic", 6))}頭')

recent_n = st.slider("直近成績使用件数", 3, 20, 10, key="main_recent_n")
honmei_count = st.slider("本命候補表示数", 3, 15, key="main_honmei_count", on_change=_sync_main_honmei)
ana_count_logic = st.slider("穴候補表示数", 1, 15, key="main_ana_count_logic", on_change=_sync_main_ana)
ana_count = st.slider("穴馬候補の頭数（買い目用）", 1, 15, int(st.session_state.get("rr_ana_count_logic", 6)), key="main_ana_count")
ticket_head_count = st.slider("買い目に使う上位頭数", 5, 15, TOP_FOR_TICKETS_DEFAULT, key="main_ticket_head_count")
wide_count = st.slider("ワイド表示件数", 5, 50, 30, key="main_wide_count")
umaren_count = st.slider("馬連表示件数", 5, 50, 30, key="main_umaren_count")
trio_count = st.slider("3連複表示件数", 5, 50, 30, key="main_trio_count")
trifecta_count = st.slider("3連単表示件数", 10, 100, 50, key="main_trifecta_count")
st.info(f"現在のクイック設定: {st.session_state.get('rr_quick_profile', 'カスタム')} / 本命{int(st.session_state.get('rr_honmei_count', 6))}頭 / 穴{int(st.session_state.get('rr_ana_count_logic', 6))}頭 / 変更は予想結果へ反映")
st.caption('本命候補表示数と穴候補表示数は、この下の本命候補・穴候補テーブルの頭数に反映されます。')

h1, h2, h3 = st.columns([1.2,1.2,1.2])
with h1:
    st.markdown('<div class="nk-card"><div class="nk-card-title">設定のヒント</div><div class="nk-mini">・本命候補は「勝率・AIスコア」を重視<br>・穴候補は「人気・オッズ・期待値(gap)」を重視<br>・数値を高くすると条件が厳しく、低くすると緩くなります</div></div>', unsafe_allow_html=True)
with h2:
    st.markdown(f'<div class="nk-card"><div class="nk-card-title">おすすめ設定</div><div class="nk-mini">地方モード：{local_mode}<br>予想モード：{predict_mode}<br>本命候補：{honmei_count}頭 / 穴候補：{ana_count_logic}頭</div></div>', unsafe_allow_html=True)
with h3:
    st.markdown('<div class="nk-card"><div class="nk-card-title">設定の影響について</div><div class="nk-mini">・設定を変えると予想結果が大きく変わる場合があります<br>・複数レースで試して最適設定を見つけるのがおすすめです</div></div>', unsafe_allow_html=True)

if mode == "URLから取得して予想":
    if st.button("取得して予想", type="primary", use_container_width=True):
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
            st.info(f"処理できませんでした: {e}")

elif mode == "CSVアップロード":
    if 'race_file' in locals() and 'hist_file' in locals() and race_file and hist_file and st.button("CSVを読み込んで予想", type="primary", use_container_width=True):
        try:
            race_df = load_csv_upload(race_file)
            hist_df = load_csv_upload(hist_file)
            st.session_state["race_df_store"] = race_df.copy()
            st.session_state["hist_df_store"] = hist_df.copy()
            st.session_state["prediction_ready"] = True
            st.session_state["loaded_from"] = "csv"
        except Exception as e:
            st.info(f"処理できませんでした: {e}")

else:
    if st.button("ローカルCSVを読み込んで予想", type="primary", use_container_width=True):
        try:
            race_df = pd.read_csv("race_card.csv")
            hist_df = pd.read_csv("horse_history.csv")
            st.session_state["race_df_store"] = race_df.copy()
            st.session_state["hist_df_store"] = hist_df.copy()
            st.session_state["prediction_ready"] = True
            st.session_state["loaded_from"] = "local"
        except Exception as e:
            st.info(f"ローカルCSVを読み込めませんでした: {e}")

if st.session_state.get("prediction_ready") and st.session_state.get("race_df_store") is not None and st.session_state.get("hist_df_store") is not None:
    try:
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
    except Exception as e:
        st.error(f"予想処理エラーを吸収しました: {e}")
        st.info("画面は落とさず継続します。入力CSV/取得データを確認してください。")
        with st.expander("詳細エラー"):
            st.code(traceback.format_exc())

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
def calc_recovery_rate(payout: Any, buy: Any) -> float:
    try:
        b = float(buy)
        p = float(payout)
        return (p / b * 100.0) if b > 0 else 0.0
    except Exception:
        return 0.0
