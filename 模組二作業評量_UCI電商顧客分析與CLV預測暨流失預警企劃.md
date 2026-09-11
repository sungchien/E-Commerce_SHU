---
puppeteer:
  displayHeaderFooter: true
  scale: 1.15
  headerTemplate: '<div style="font-size: 11px; margin: 0 auto;">模組二期末綜合作業評量：UCI 電商交易 Log 分析、RFM K-Means 顧客分群與 CLV 預測暨流失預警企劃</div>'
  footerTemplate: '<div style="font-size: 11px; margin: 0 auto;">第 <span class="pageNumber"></span> 頁 / 共 <span class="totalPages"></span> 頁</div>'
  margin:
    top: "1.5cm"
    bottom: "1.5cm"
    left: "1.5cm"
    right: "1.5cm"
---

<style>
  /* 全域字型、字級與行距 */
  body {
    font-size: 13pt !important;
    line-height: 1.7 !important;
    font-family: "Microsoft JhengHei", "PingFang TC", "Helvetica Neue", sans-serif;
  }

  /* 階層標題微調 (作業評量篇幅有限，h2 不強制分頁) */
  h1 { font-size: 24pt !important; margin-bottom: 0.5em !important; }
  h2 { font-size: 18pt !important; }
  h3 { font-size: 15pt !important; }
  h4 { font-size: 13.5pt !important; }

  /* 表格文字放大與排版優化 */
  table, th, td {
    font-size: 12pt !important;
    line-height: 1.5 !important;
  }

  /* 程式碼區塊 */
  pre, code {
    font-size: 11.5pt !important;
    font-family: Consolas, "Courier New", monospace !important;
  }

  /* Mermaid 流程圖節點字體放大 */
  .mermaid text {
    font-size: 14px !important;
  }
</style>

---

# 模組二期末綜合作業評量：UCI 電商交易 Log 分析、RFM K-Means 顧客分群與 CLV 預測暨流失預警企劃

> 課程名稱： 電子商務服務
> 評量配分： 佔學期總成績 15%
> 組隊方式： 3 ~ 5 人為一組 自由組隊合作
> 繳交期限： 發布後 一週內 （11/29 23:59 前繳交）
> 格式規範： 最多 3 張 A4 為限 （PDF 格式，字體 12 pt，內文可附圖表或表格，免寫程式碼或可附摘要分析表）

---

## 評量情境背景（Business Context）

你是某國際跨國線上零售電商（Online Retail）聘用的「CRM 顧客數據分析與成長行銷顧問團隊」。
公司過去一年累積了約 54 萬筆交易流水帳明細（Transaction Logs），過去行銷部門長期採用傳統的「全站統一常規折扣與聯播網廣告」進行顧客維繫，導致行銷預算過度分散，對沉睡客浪費大量廣告費，卻未給予高價值 VIP 顧客足夠的尊榮維護，導致營收成長進入瓶頸期。

幸運的是，你的團隊已經運用第九週至第十一週所學的資料科學與機器學習方法，完成了三大階段的深度顧客分析：

1. W9 交易 Log 探索性分析與第一階段 Baseline CLV：
   - 成功將 54 萬筆明細聚合為 4,338 位獨立顧客維度。原始 AOV 呈現極大正偏斜（偏斜度 $+41.69$），透過 95th Percentile Winsorization 截斷修正得出穩健 AOV $315.40$。
   - 算出一流失率 38.45%、留存期 2.60 年、穩健第一階段全局 Baseline CLV 為 $3,502.80$。揭露了「全局平均迷思（Global Average Trap）」——全站公式會嚴重低估頂級 VIP、嚴重高估沉睡客！
   - 驗證帕累托法則（80/20 法則）：全站前 20% 高價值顧客貢獻了 74.59% 的總營收。
2. W10 RFM 特徵工程與第二階段分群 CLV：
   - 透過對數轉換（`np.log1p`）與 `StandardScaler` 標準化消除偏斜與量綱霸權，運用 K-Means 非監督式機器學習（$K=4$）將顧客劃分為 4 大 Persona 畫像群體：
     - Cluster 0（Champions 核心頂級 VIP）： 佔比 29.5%，專屬流失率 10.20%，留存期 9.80 年，分群 CLV 高達 $57,046.20$。
     - Cluster 1（Loyalists 主力消費客）： 佔比 24.2%，專屬流失率 22.50%，留存期 4.44 年，分群 CLV 為 $5,550.00$。
     - Cluster 2（Promising 潛力新客）： 佔比 25.8%，專屬流失率 36.80%，留存期 2.72 年，分群 CLV 為 $1,142.40$。
     - Cluster 3（Hibernating 沉睡流失客）： 佔比 20.5%，專屬流失率 78.50%，留存期 1.27 年，分群 CLV 僅剩 $393.70$。
3. W11 個體化機器學習迴歸 CLV 預測與流失預警：
   - 建立時間序列觀察窗口（P1~P3 歷史特徵 $X_i \to$ P4 個體消費金額 $y_i$），透過 95 分位數截斷與 10-Fold 交叉驗證（10-Fold CV）進行迴歸預測（$R^2 = 0.6200$，OOF MAE 顯著降低）。
   - 建構二維 CLV-流失風險決策矩陣，釐清舊客留存成本遠低於新客獲客（5~7 倍成本差異），並透過歷史同群體新客基準（Cohort Benchmark）設定 CPA 上限。

現在，電子商務執行長（CEO）與行銷長（CMO）要求你的團隊在 3 頁 A4 篇幅內，提交一份兼具「定量數據證據」與「商業自動化落地方案」的企劃報告。

---

## 作業四大必答任務（4 Core Tasks）

### 任務一：三大 CLV 演進階段分析與「全局平均迷思」診斷（配分：20%）

請根據第九週至第十一週的三大 CLV 計算演進階段，進行比較與痛點診斷：
1. 演進比較： 比較「第一階段全局 Baseline CLV（$3,502.80$）」與「第二階段分群 CLV（Champions $57,046.20$ vs. Hibernating $393.70$）」的估算差異。
2. 痛點診斷與資源浪費試算：
   - 假定公司行銷部門準備撥款行銷預算，若依據「第一階段全局公式」，行銷團隊會認為沉睡客（Hibernating）未來能帶來 $3,502.80$ 的價值，因而對 888 位沉睡客投放高額數位再行銷廣告；但事實上分群 CLV 揭露沉睡客真實價值僅剩 $393.70$。
   - 請說明為什麼全局一刀切公式會導致「VIP 資源投入不足（欠估 $53,543.40$）」與「沉睡客廣告資源嚴重浪費（高估 $3,109.10$）」？

### 任務二：RFM K-Means 特徵前處理與 Persona 畫像評析（配分：30%）

請靈活運用第十週所學的機器學習前處理與分群觀念，回答以下問題：
1. 特徵前處理效益：
   - 為什麼在將 RFM 特徵送入 K-Means 之前，必須先進行對數轉換（`np.log1p`）與 `StandardScaler` 標準化？若不做這兩項前處理，對歐式距離與分群結果會產生什麼破壞性影響？
2. 4 大 Persona 商業畫像解讀：
   - 請說明 K-Means 將顧客分為 4 大群體的商業可解釋性。為什麼 Champions 群體的專屬流失率（10.20%）會遠低於 Hibernating 群體（78.50%）？這對留存期（$\text{Lifespan}_k = \frac{1}{\text{Churn}_k}$）與 CLV 有何影響？

### 任務三：設計「自動化 CRM 事件驅動觸發矩陣與流失預警機制」（配分：30%）

請為行銷團隊設計一套系統化的 CRM 自動化觸發與流失預警機制（不用寫程式碼，用表格或條列式清晰呈現）：
1. 4 大客群自動化觸發條件矩陣：
   - 核心頂級 VIP： 說明為何自動化觸發門檻設為「單筆下單金額 $\ge \$2,000$ 或 累積消費 $\ge \$5,000$」？其數據依據為何？
   - 主力消費客： 說明為何觸發條件設為「距離上次購物滿 30 天未回購」？
   - 潛力新客： 說明為何觸發條件設為「首購完成後第 7 天」自動發送二購 Drip Campaign？
   - 沉睡流失客： 說明為何觸發條件設為「Recency $> 180$ 天」？
2. 二維 CLV-流失風險決策矩陣（2D CRM Matrix）：
   - 請結合第十一週迴歸模型預測的 P4 消費金額 $\hat{y}_i$ 與流失預警訊號（$y \to 0$），規劃「高價值高風險（即將流失 VIP）」與「高價值低風險（穩定 VIP）」的差異化干預腳本。

### 任務四：預期商業效益評估與高階主管簡報結論（配分：20%）

請為 CEO 與 CMO 評估新策略的商業回報：
1. 舊客留存 vs 新客獲客經濟效益：
   - 根據 Bain & HBR 研究，獲取新客成本（CPA）是舊客留存的 5 至 7 倍。請說明停用「沉睡客高額廣告」並轉為「VIP 尊榮維護與新客二購自動化」如何為企業帶來顯著的 ROI 提升。
2. 獲客 CPA 上限設定：
   - 請說明如何運用「歷史同群體首購新客（Historical New-Customer Cohort Benchmark）」來為數位廣告投放團隊設定客觀的 CPA 上限。
3. 高階主管 3 句話總結：
   - 請用 3 句話向 CEO / CMO 總結本企劃精髓，說明這套自動化 CRM 系統如何實現「精準營運、降低流失、提升 CLV」。

---

## 評分標準（Rubric）

| 評分維度 | 配分 | 評分細節標準 |
| :--- | :--- | :--- |
| 數據邏輯與公式精確度 | 25% | CLV 三階段演進概念清晰，AOV Winsorization、流失率與 Lifespan 計算邏輯正確。 |
| 跨章節觀念整合能力 | 35% | 能貫通 W9 EDA 偏斜度、W10 RFM K-Means 特徵前處理與 W11 時間序列迴歸預測。 |
| 商業落地與企劃可行性 | 25% | CRM 自動化觸發門檻與 2D 流失預警矩陣具備實務操作可行性，能直接落地執行。 |
| 報告排版與精煉溝通 | 15% | 嚴格遵守 3 頁 A4 限制，排版簡潔專業、圖表清晰、無冗長贅字。 |

---

## 繳交須知

- 檔案命名格式： `組別_第X組_模組二企劃書.pdf`
- 繳交方式： 請由組長於下週上課前上傳至學校 TronClass 課程平台。
- 學術誠信： 歡迎使用 AI 工具（如 ChatGPT、Claude）協助發想與潤飾文案，但所有數據計算與商業邏輯必須經由小組討論確認。
