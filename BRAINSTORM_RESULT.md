# 台股 AI 分析平台 - 腦力激盪結果

## 執行摘要

建立一個基於傳統 ML（非 LLM）的台股量化分析平台，採用「POC 驗證後再工程化」策略。透過 LightGBM/XGBoost 預測股票報酬率，搭配 Walk-Forward 回測驗證，並提供 Web 介面讓使用者調控指標參數、查看分析結果。技術棧以 Supabase + GitHub Actions + SvelteKit 為核心，強調輕量化與個人開發者可維護性。

**方案哲學：「務實先行，數據說話：用現有資源快速驗證信號存在性，有效才談完美。」**

---

## 最終方案

### 三大核心決策

| 決策點 | 選擇 | 理由 |
|--------|------|------|
| **資料策略** | 混合策略 | 現存股票先跑（~600-800檔），驗證有效再補歷史/下市股票資料 |
| **技術棧** | Supabase + GitHub Actions | 免費託管、自動備份、零維運負擔，適合個人開發者 |
| **成功標準** | 分層驗證 | Phase 1 寬鬆探索（Sharpe>1.1, p<0.10）→ Phase 2 嚴謹（Sharpe>1.2, p<0.05）|

### 技術棧

```yaml
資料層:
  Database: Supabase (PostgreSQL, 免費 500MB)
  Storage: Supabase Storage (模型/資料檔)

資料抓取:
  排程: GitHub Actions (每日 18:00 UTC+8)
  來源: yfinance / FinMind / 公開觀測站 XBRL

ML Pipeline:
  訓練: 本地 Python (scikit-learn, XGBoost)
  模型: Ensemble (Ridge + Random Forest + XGBoost)
  回測: 自建引擎 + Walk-Forward 驗證

Web 前端:
  Framework: SvelteKit
  Charts: Apache ECharts
  Hosting: Vercel (免費)

基礎設施:
  CI/CD: GitHub Actions
  監控: Supabase Dashboard + Sentry
```

### ML Pipeline 設計

#### 模型矩陣
| 模型 | 角色 | 目的 |
|------|------|------|
| Buy & Hold (0050) | 基準線 | 最低標準 |
| Ridge Regression | 線性對照 | 驗證非線性必要性 |
| Random Forest | 主力模型 1 | 特徵重要性分析 |
| XGBoost | 主力模型 2 | 最佳性能 |
| Ensemble (RF+XGB) | 最終模型 | 降低單模型風險 |

#### 特徵工程（~50-80 features）
- **價量技術指標 (30%)**：MA, RSI, MACD, Bollinger Bands, Volume Profile, OBV
- **財務基本面 (40%)**：P/E, P/B, ROE, EPS 成長率, 負債比, 自由現金流
- **市場結構 (20%)**：產業相對強度, 市值分類
- **另類數據 (10%)**：法人買賣超, 融資融券變化

#### 交易成本模型
```
買進成本: 0.1425% (手續費)
賣出成本: 0.1425% + 0.3% (手續費 + 證交稅)
來回成本: ~0.6% (回測基準)
滑價測試: Phase 3 加入 0.8-1.0% 敏感性分析
```

#### Walk-Forward 驗證
```
訓練窗口: 3 年 (滾動)
測試窗口: 1 季
重訓練: 每季度
時間分割: 2018-2021 訓練 / 2022 驗證 / 2023-2024 測試
```

### 偏差處理

| 偏差類型 | 處理方式 |
|----------|----------|
| **前瞻偏差** | 財報只用公告日+1天後的資料；盤後資料 14:00 後才可用 |
| **生存者偏差** | Phase 1 標註警告，Phase 2 補全歷史（TEJ 或手動） |
| **過擬合** | Walk-Forward + 線性基準比較 + 統計檢驗 |

### Web 平台功能（3 頁 MVP）

```
/dashboard    → 最新訊號、累積報酬、風險指標、權益曲線 (vs 0050)
/backtest     → 參數面板（日期/持股數/模型）+ 結果視覺化
/stocks/:id   → 技術指標、財務指標、預測分數、同產業比較
```

---

## 分階段開發計劃

| Phase | 時間 | 目標 | 成功標準 | 停損線 |
|-------|------|------|----------|--------|
| **0: 環境準備** | Week 1-2 | Supabase/GitHub 設定 | 能存取資料 | - |
| **1: 資料管道** | Week 3-7 | 抓取+特徵工程+驗證 | 50+ features, 資料完整 | 7.5週未完成→重評 |
| **2: 基準模型** | Week 8-12 | 回測引擎+模型比較 | Sharpe > 1.1 | Sharpe < 0.8→停止 |
| **3: 模型優化** | Week 13-17 | XGBoost+Ensemble+WF | Sharpe > 1.2, p<0.05 | Walk-forward Sharpe<1.0 |
| **4: Web 平台** | Week 18-25 | SvelteKit 3頁 MVP | 可操作的分析介面 | - |
| **5: 測試優化** | Week 26-30 | 端到端測試+部署 | 穩定運行 | - |

**總時程：約 6-7 個月**

---

## 專案目錄結構

```
finai/
├── data/                    # 資料管道
│   ├── collectors/          # 資料抓取 (price.py, financials.py, universe.py)
│   ├── processors/          # 特徵工程 + 清洗 (features.py, cleaning.py)
│   └── loaders/             # Supabase 介面 (supabase.py)
├── models/                  # ML 模型
│   ├── baseline.py          # Buy & Hold, Ridge
│   ├── tree_models.py       # RF, XGBoost
│   ├── ensemble.py          # 集成模型
│   └── training.py          # 訓練邏輯
├── backtest/                # 回測引擎
│   ├── engine.py            # 核心引擎
│   ├── portfolio.py         # 投資組合管理
│   ├── costs.py             # 交易成本
│   └── metrics.py           # 績效指標
├── web/                     # SvelteKit 前端
│   └── src/routes/          # Dashboard, Backtest, Stock Explorer
├── scripts/                 # 自動化腳本
│   ├── daily_update.py      # 每日資料更新
│   ├── weekly_retrain.py    # 每週重訓練
│   └── monthly_report.py    # 月度報告
├── .github/workflows/       # GitHub Actions
├── tests/                   # 測試
├── configs/                 # 設定檔 (YAML)
└── docs/                    # 文件
```

---

## 討論過程摘要

### 關鍵洞見
1. **先驗證再工程化**：最大風險不是技術選型，而是「花 6 個月做出來發現模型不賺錢」
2. **簡化是核心主題**：HMM→刪除、Mean-Variance→Equal Weight、PostgreSQL→Supabase、Great Expectations→自定義驗證
3. **資料基礎決定一切**：前瞻偏差、生存者偏差、交易成本是量化策略的三大陷阱
4. **線性基準不可少**：Ridge Regression 能證明「非線性模型的額外複雜度是否值得」
5. **分層驗證避免過早放棄**：寬鬆探索 → 嚴謹驗證，兼顧效率與科學性

### 被否決的方案
| 方案 | 否決原因 |
|------|----------|
| HMM 市場狀態識別 | 台股歷史太短、參數不穩定、無文獻支持 |
| Mean-Variance 投資組合 | 常態假設不成立、協方差不穩、極端權重 |
| PostgreSQL + Redis + Airflow | 對個人開發者維護負擔過重 |
| Great Expectations | 學習曲線陡峭，自定義驗證即可 |
| React 前端 | SvelteKit 更輕量、學習成本低 |
| Tick-level 滑價模型 | 資料取得困難，固定百分比即足夠 |

### 已知限制
1. 免費資料源不含已下市股票（Phase 1 標註警告）
2. 交易成本 0.6% 可能樂觀（Phase 3 做敏感性分析）
3. GitHub Actions 免費額度有限（~2000 分鐘/月）
4. 個人電腦訓練速度限制（大規模超參搜索受限）

---

## 討論統計
- 總輪數：3
- 最終評分：78/100（35 → 72 → 78）
- 高嚴重度問題：8 個提出 → 全部解決
- 關鍵決策：3 個明確選擇
- 方案迭代：v1.0 → v2.0 → v3.0（最終版）

---

## 下一步行動

1. **今天**：建立 Git repo + Supabase 專案
2. **本週**：完成 `data/collectors/price.py`（抓取 50 檔測試）
3. **下週**：完成特徵工程（至少 30 features）
4. **第一個月**：跑出第一個 Ridge Regression 回測結果
