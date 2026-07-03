"""
結算腳本：對 results/*.xlsx 的歷史預測對答案。

資料源：TWSE / TPEX 官方日行情（免 token，按日抓全市場，快取在 scripts/.price_cache/）。
輸出：終端摘要 + results/結算報告.xlsx（總結 / 明細兩張表）。

用法：python scripts/score_results.py
"""
import glob
import os
import re
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "scripts", ".price_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TARGET_RETURN = 0.015   # 舊主模型目標：次日最高 >= 收盤 +1.5%
OC_TARGET = 0.006       # 新主模型目標：次日開盤買收盤賣 >= +0.6%（覆蓋交易成本）
LIMIT_RETURN = 0.095    # 漲停候選目標：次日最高 >= 收盤 +9.5%
TOP_N = 10              # 每日信心排序取前 N 檔的分組

HEADERS = {"User-Agent": "Mozilla/5.0"}


def _to_float(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return None


def get_json(url, params, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def fetch_day_quotes(d: date):
    """抓某日全市場 OHLC，回傳 {sid: (open, high, low, close)}；非交易日回傳 None。"""
    quotes = {}

    j = get_json(
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
        {"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"},
    )
    if j.get("stat") != "OK":
        return None
    for t in j.get("tables", []):
        if "每日收盤行情" not in (t.get("title") or ""):
            continue
        idx = {name: i for i, name in enumerate(t["fields"])}
        for row in t["data"]:
            sid = row[idx["證券代號"]].strip()
            o, h, l, c = (_to_float(row[idx[k]]) for k in ("開盤價", "最高價", "最低價", "收盤價"))
            if re.fullmatch(r"\d{4}", sid) and c is not None:
                quotes[sid] = (o, h, l, c)

    j = get_json(
        "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
        {"date": d.strftime("%Y/%m/%d"), "response": "json"},
    )
    for t in j.get("tables", []):
        if "上櫃股票行情" not in (t.get("title") or ""):
            continue
        idx = {name: i for i, name in enumerate(t["fields"])}
        for row in t["data"]:
            sid = row[idx["代號"]].strip()
            o, h, l, c = (_to_float(row[idx[k]]) for k in ("開盤", "最高", "最低", "收盤"))
            if re.fullmatch(r"\d{4}", sid) and c is not None:
                quotes[sid] = (o, h, l, c)

    return quotes or None


def load_day_quotes(d: date):
    """帶檔案快取的 fetch_day_quotes。非交易日以空檔案記錄，避免重抓。"""
    cache = os.path.join(CACHE_DIR, d.strftime("%Y%m%d") + ".csv")
    if os.path.exists(cache):
        if os.path.getsize(cache) == 0:
            return None
        df = pd.read_csv(cache, dtype={"sid": str})
        return {r.sid: (r.open, r.high, r.low, r.close) for r in df.itertuples()}

    quotes = fetch_day_quotes(d)
    time.sleep(1)
    if quotes is None:
        open(cache, "w").close()
        return None
    pd.DataFrame(
        [(sid, *v) for sid, v in quotes.items()],
        columns=["sid", "open", "high", "low", "close"],
    ).to_csv(cache, index=False)
    return quotes


# ==========================================
# 讀入所有預測
# ==========================================
files = sorted(glob.glob(os.path.join(ROOT, "results", "*.xlsx")))
files = [f for f in files if re.search(r"_(\d{8})\.xlsx$", f)]
if not files:
    sys.exit("results/ 沒有預測檔")

preds = []
for f in files:
    file_date = pd.Timestamp(re.search(r"_(\d{8})\.xlsx$", f).group(1))
    df = pd.read_excel(f, sheet_name="全部")
    df["代號"] = df["代號"].astype(str).str.zfill(4)
    # 舊檔沒有 模型資料日 / 漲停候選分數%
    if "模型資料日" in df.columns:
        df["基準日"] = pd.to_datetime(df["模型資料日"])
    else:
        df["基準日"] = file_date
    if "漲停候選分數%" not in df.columns:
        df["漲停候選分數%"] = pd.NA
    df["檔案日"] = file_date
    preds.append(df[["檔案日", "基準日", "等級", "代號", "股名",
                     "XGB信心%", "預測日收盤", "漲停候選分數%"]])

pred_df = pd.concat(preds, ignore_index=True)
print(f"📂 讀入 {len(files)} 個檔案、{len(pred_df)} 筆預測")

# ==========================================
# 抓行情（基準日隔天起到今天）
# ==========================================
start = pred_df["基準日"].min().date() + timedelta(days=1)
end = date.today()
day_quotes = {}
d = start
while d <= end:
    if d.weekday() < 5:
        q = load_day_quotes(d)
        if q:
            day_quotes[pd.Timestamp(d)] = q
    d += timedelta(days=1)
trading_days = sorted(day_quotes)
print(f"📈 取得 {len(trading_days)} 個交易日行情（{trading_days[0].date()} ~ {trading_days[-1].date()}）")


def next_quote(base_date, sid):
    for td in trading_days:
        if td > base_date:
            q = day_quotes[td].get(sid)
            return (td, *q) if q else None
    return None


# ==========================================
# 逐筆結算
# ==========================================
rows = []
for r in pred_df.itertuples():
    nq = next_quote(r.基準日, r.代號)
    if nq is None or r.預測日收盤 is None or pd.isna(r.預測日收盤):
        continue
    td, o, h, l, c = nq
    pc = float(r.預測日收盤)
    rows.append({
        "檔案日": r.檔案日, "基準日": r.基準日, "結算日": td,
        "等級": r.等級, "代號": r.代號, "股名": r.股名,
        "XGB信心%": r._6, "漲停候選分數%": r._8, "預測日收盤": pc,
        "次日開": o, "次日高": h, "次日收": c,
        "達標": int(h >= pc * (1 + TARGET_RETURN)),
        "開收達標": int(o is not None and o > 0 and (c - o) / o >= OC_TARGET),
        "漲停達標": int(h >= pc * (1 + LIMIT_RETURN)),
        "開收報酬%": round((c - o) / o * 100, 2) if o else None,
        "收收報酬%": round((c - pc) / pc * 100, 2),
    })

detail = pd.DataFrame(rows)
print(f"✅ 結算 {len(detail)} 筆（略過 {len(pred_df) - len(detail)} 筆無次日行情）")


# ==========================================
# 分組統計
# ==========================================
def summarize(df, label):
    oc = df["開收報酬%"].dropna()
    return {
        "分組": label, "筆數": len(df),
        "達標率%": round(df["達標"].mean() * 100, 1),
        "開收達標率%": round(df["開收達標"].mean() * 100, 1),
        "漲停率%": round(df["漲停達標"].mean() * 100, 2),
        "開收平均%": round(oc.mean(), 3),
        "開收中位%": round(oc.median(), 3),
        "開收勝率%": round((oc > 0).mean() * 100, 1),
        "收收平均%": round(df["收收報酬%"].mean(), 3),
    }


summary = [summarize(detail, "全部（基準）")]
for g in ["A 強訊號", "B XGB獨立", "C Tree獨立", "D 無訊號"]:
    sub = detail[detail["等級"] == g]
    if len(sub):
        summary.append(summarize(sub, g))

top_conf = (detail.sort_values("XGB信心%", ascending=False)
                  .groupby("檔案日").head(TOP_N))
summary.append(summarize(top_conf, f"每日XGB信心前{TOP_N}"))

has_limit = detail[detail["漲停候選分數%"].notna()].copy()
if len(has_limit):
    summary.append(summarize(has_limit, "有漲停分數的全部（基準）"))
    top_limit = (has_limit.sort_values("漲停候選分數%", ascending=False)
                          .groupby("檔案日").head(TOP_N))
    summary.append(summarize(top_limit, f"每日漲停分數前{TOP_N}"))

summary_df = pd.DataFrame(summary)
print("\n" + "=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)
print(f"（達標 = 次日最高 ≥ 預測日收盤 +{TARGET_RETURN:.1%}（舊目標）；"
      f"開收達標 = 次日開盤買收盤賣 ≥ +{OC_TARGET:.1%}（新目標）；開收報酬未含成本）")

out = os.path.join(ROOT, "results", "結算報告.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as w:
    summary_df.to_excel(w, sheet_name="總結", index=False)
    detail.to_excel(w, sheet_name="明細", index=False)
print(f"\n✅ 已儲存：{out}")
