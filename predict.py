"""
每日選股預測腳本 - 由 GitHub Actions 自動執行
執行完畢後將 xlsx 存入 results/ 資料夾，並 commit 回 repo
"""
import requests
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import time, re, os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

# ==========================================
# 設定
# ==========================================
API_TOKEN = os.environ.get("FINMIND_TOKEN", "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wNC0wOSAwMDoxNjoyMCIsInVzZXJfaWQiOiJkaWduaXR5a2VuIiwiZW1haWwiOiJ6MDkxMjA5NjQ3MEBnbWFpbC5jb20iLCJpcCI6IjExOC4xNjkuMTQwLjE5OCJ9.x99Wo6d4oAFk0ALHY8q2zh7Huyd9SrWyMrRRc6I216E")

URLS = {
    "上市買超": "https://fubon-ebrokerdj.fbs.com.tw/Z/ZG/ZG_F.djhtm",
    "上櫃買超": "https://fubon-ebrokerdj.fbs.com.tw/z/zg/zg_F_1_1.djhtm",
}

os.makedirs("results", exist_ok=True)

# ==========================================
# 確定預測日（前一個交易日）
# ==========================================
base_date = datetime.utcnow() + timedelta(hours=8)  # 轉台灣時間
base_date = base_date - timedelta(days=1)
while base_date.weekday() >= 5:                      # 跳過週末
    base_date -= timedelta(days=1)

print(f"📅 預測基準日：{base_date.strftime('%Y-%m-%d')}")

# ==========================================
# API
# ==========================================
def get_fm_data(dataset, stock_id="", start="", end="", retries=4):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {"dataset": dataset, "data_id": stock_id,
              "start_date": start, "end_date": end, "token": API_TOKEN}
    for attempt in range(retries):
        try:
            time.sleep(0.5)
            res = requests.get(url, params=params, timeout=10)
            rj  = res.json()
            if rj.get('status') == 200:
                return pd.DataFrame(rj.get('data', []))
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()

# ==========================================
# 爬取主力買超名單
# ==========================================
def get_consensus_universe():
    print("🌐 爬取主力買超名單...")
    seen, rank = {}, 0
    headers  = {'User-Agent': 'Mozilla/5.0'}
    patterns = [
        r"Link2Stk\('(\d{4,6})'\)[^>]*>[\d]{4,6}([^<]+)<",
        r"Link2Stk\('(\d{4,6})'\)[^>]*>\d*\s*([^\d<][^<]*)<",
        r"goLink\(['\"](\d{4,6})['\"][^)]*\)[^>]*>([^<]+)<",
        r"stockid=(\d{4,6})[^>]*>([^<]{2,10})<",
    ]
    for label, url in URLS.items():
        count_before = len(seen)
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = 'cp950'
            found = []
            for pat in patterns:
                found = re.findall(pat, res.text)
                if found:
                    break
            for sid, sname in found:
                sname = sname.strip()
                if re.match(r'^(00|01|009)', sid) or not re.match(r'^\d{4}$', sid):
                    continue
                if sid not in seen:
                    rank += 1
                    seen[sid] = {"sid": sid, "sname": sname, "count": 1, "rank": rank}
            print(f"  {label}：{len(seen) - count_before} 檔")
        except Exception as e:
            print(f"  {label}：失敗 ({e})")
    result = list(seen.values())
    print(f"✅ 合計 {len(result)} 檔")
    return result

# ==========================================
# 分析單支股票
# ==========================================
def analyze_stock(item, base_date):
    sid, sname = item['sid'], item['sname']
    start_date = (base_date - timedelta(days=1095)).strftime('%Y-%m-%d')
    end_date   = base_date.strftime('%Y-%m-%d')

    df_price = get_fm_data("TaiwanStockPrice", sid, start_date, end_date)
    if df_price.empty or len(df_price) < 60:
        return None

    df_inst = get_fm_data("TaiwanStockInstitutionalInvestorsBuySell", sid, start_date, end_date)
    df_hold = get_fm_data("TaiwanStockShareholding", sid, start_date, end_date)
    df_loan = get_fm_data("TaiwanStockLoanAndShortSell", sid, start_date, end_date)

    if df_inst.empty or df_hold.empty:
        return None

    df_price['Date'] = pd.to_datetime(df_price['date']).dt.normalize()
    f = df_inst[df_inst['name'].str.contains('外資|Foreign', case=False)].copy()
    f['Date'] = pd.to_datetime(f['date']).dt.normalize()
    f = f.groupby('Date')[['buy', 'sell']].sum().reset_index()
    df_hold['Date'] = pd.to_datetime(df_hold['date']).dt.normalize()

    df = pd.merge(
        df_price[['Date','open','max','min','close','Trading_Volume','Trading_turnover']],
        f, on='Date', how='left'
    ).fillna(0)
    df = pd.merge(
        df,
        df_hold[['Date','ForeignInvestmentShares','ForeignInvestmentRemainRatio']],
        on='Date', how='left'
    ).ffill()

    if not df_loan.empty:
        df_loan['Date'] = pd.to_datetime(df_loan['date']).dt.normalize()
        l_col = 'lending_balance' if 'lending_balance' in df_loan.columns else 'lending_remain_qty'
        s_col = 'short_sell_balance' if 'short_sell_balance' in df_loan.columns else 'short_sell_remain_qty'
        df_loan = df_loan[['Date', l_col, s_col]].copy()
        df_loan.columns = ['Date', '借券餘額', '借券賣出']
        df = pd.merge(df, df_loan, on='Date', how='left').fillna(0)
    else:
        df['借券餘額'] = 0
        df['借券賣出'] = 0

    df['價差對比']    = ((df['max'] - df['min']) / df['close'] * 100).round(2)
    df['買進張數']    = df['buy'] / 1000
    df['賣出張數']    = df['sell'] / 1000
    df['買賣超張數']  = df['買進張數'] - df['賣出張數']
    df['今日持股張數'] = df['ForeignInvestmentShares'] / 1000
    df['餘額數差異']  = df['今日持股張數'].diff()
    df['實際差異數']  = (df['餘額數差異'] - df['買賣超張數']).round(2)
    df['控一'] = (
        ((df['close'] - df['min']) - (df['max'] - df['close'])) /
        (df['max'] - df['min'] + 0.001) * (df['Trading_Volume'] / 1000)
    ).round(0)
    df['控二']        = df['控一'].rolling(5).sum().fillna(0)
    df['控三']        = (df['close'].rolling(5).mean() - df['close'].rolling(20).mean()).round(2)
    df['借券潛在']    = (df['借券餘額'] - df['借券賣出']) / 1000
    df['持股比例差異'] = df['ForeignInvestmentRemainRatio'].diff().round(3)
    df['總股本']      = df['今日持股張數'] / (df['ForeignInvestmentRemainRatio'] / 100 + 0.0001)
    df['本2%張數']    = df['總股本'] * 0.02
    df['開收比']      = ((df['open'] - df['close'].shift(1)) /
                         (df['close'].shift(1) + 0.001)).round(4)

    features = [
        '價差對比', '持股比例差異', '買賣超張數', '實際差異數',
        '控一', '控二', '控三', '借券潛在', '本2%張數', '開收比'
    ]

    base_dt  = pd.Timestamp(base_date).normalize()
    train_df = df[df['Date'] <= base_dt].copy()
    train_df['Target'] = np.where(
        (train_df['max'].shift(-1) - train_df['close']) / train_df['close'] >= 0.015, 1, 0
    )
    train_df = train_df.dropna(subset=features + ['Target'])

    if len(train_df) < 100 or len(train_df['Target'].unique()) < 2:
        return None

    X, y = train_df[features], train_df['Target']
    scale = max((y == 0).sum() / (y == 1).sum(), 1.0)

    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=scale, random_state=42,
        eval_metric='logloss', verbosity=0
    )
    xgb_model.fit(X, y)

    tree_model = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')
    tree_model.fit(X, y)

    today_row = df[df['Date'] == base_dt]
    if today_row.empty:
        return None

    latest   = today_row[features].fillna(0)
    close_px = float(today_row['close'].values[0])
    xgb_pred = int(xgb_model.predict(latest)[0])
    xgb_prob = round(float(xgb_model.predict_proba(latest)[0][1]) * 100, 1)
    tree_pred = int(tree_model.predict(latest)[0])
    tree_prob = round(float(tree_model.predict_proba(latest)[0][1]) * 100, 1)

    if   xgb_pred == 1 and tree_pred == 1: grade = "A 強訊號"
    elif xgb_pred == 1 and tree_pred == 0: grade = "B XGB獨立"
    elif xgb_pred == 0 and tree_pred == 1: grade = "C Tree獨立"
    else:                                   grade = "D 無訊號"

    return {
        "等級": grade, "代號": sid, "股名": sname,
        "XGB信心%": xgb_prob, "Tree信心%": tree_prob,
        "xgboost": xgb_pred, "s_Tree": tree_pred,
        "預測日收盤": close_px,
        "開收比": float(today_row['開收比'].values[0]),
        "實際差異數": float(today_row['實際差異數'].values[0]),
        "價差對比": float(today_row['價差對比'].values[0]),
        "控一": float(today_row['控一'].values[0]),
        "買超排名": item['rank'],
    }

# ==========================================
# 主流程
# ==========================================
universe = get_consensus_universe()
total    = len(universe)
print(f"\n🔥 開始分析 {total} 檔...\n")

final_results, failed = [], []
for i, item in enumerate(universe):
    print(f"[{i+1}/{total}] {item['sid']} {item['sname']}...", end='  ')
    try:
        res = analyze_stock(item, base_date)
        if res:
            final_results.append(res)
            print(f"✅ {res['等級']}")
        else:
            failed.append(item['sid'])
            print("⚠️ 略過")
    except Exception:
        failed.append(item['sid'])
        print("❌ 失敗")
    time.sleep(0.5)

if failed:
    print(f"\n⚠️ 失敗 {len(failed)} 檔：{', '.join(failed)}")

if not final_results:
    print("❌ 無任何結果，中止")
    exit(1)

result_df = pd.DataFrame(final_results)
grade_order = {"A 強訊號": 0, "B XGB獨立": 1, "C Tree獨立": 2, "D 無訊號": 3}
result_df['等級排序'] = result_df['等級'].map(grade_order)
result_df = result_df.sort_values(
    by=['等級排序', 'XGB信心%'], ascending=[True, False]
).drop(columns='等級排序').reset_index(drop=True)

# 摘要
print("\n" + "=" * 50)
for grade in ["A 強訊號", "B XGB獨立", "C Tree獨立"]:
    sub = result_df[result_df['等級'] == grade]
    if len(sub):
        print(f"{grade}（{len(sub)}檔）：" +
              "、".join(f"{r['代號']}{r['股名']}" for _, r in sub.iterrows()))
print("=" * 50)

# 存 Excel
filename = f"results/v4主力_{base_date.strftime('%Y%m%d')}.xlsx"
with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    result_df[result_df['等級'] == 'A 強訊號'].to_excel(writer, sheet_name='A強訊號',    index=False)
    result_df[result_df['等級'] == 'B XGB獨立'].to_excel(writer, sheet_name='B_XGB獨立', index=False)
    result_df[result_df['等級'] == 'C Tree獨立'].to_excel(writer, sheet_name='C_Tree獨立',index=False)
    result_df.to_excel(writer, sheet_name='全部', index=False)

print(f"\n✅ 已儲存：{filename}")
