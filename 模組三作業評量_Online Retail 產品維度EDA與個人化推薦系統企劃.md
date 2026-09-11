---
puppeteer:
  displayHeaderFooter: true
  scale: 1.15
  headerTemplate: '<div style="font-size: 11px; margin: 0 auto;">模組三期末綜合作業評量：Online Retail 產品維度 EDA 與個人化推薦系統企劃</div>'
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

# 模組三期末綜合作業評量：Online Retail 產品維度 EDA 與個人化推薦系統企劃

> 課程名稱： 電子商務服務
> 評量配分： 佔學期總成績 15%
> 組隊方式： 3 ~ 5 人為一組 自由組隊合作
> 繳交期限： 發布後 一週內 （12/20 23:59 前繳交）
> 格式規範： 最多 3 張 A4 為限 （PDF 格式，字體 12 pt，內文可附圖表或表格，免寫程式碼或可附摘要分析表）

---

## 評量情境背景（Business Context）

你是某跨境歐美零售電商平台聘用的「商業數據分析與推薦系統顧問團隊」。
平台目前擁有多達 3,800 種跨國商品與數十萬筆交易 Log，然而行銷副總與技術總監面臨三大營運瓶頸：
1. 新訪客冷啟動（Cold-Start）與流量浪費： 新進站的無歷史行為訪客跳出率高達 65%，缺乏全域熱門榜單指引。
2. 購物籃客單價（AOV）停滯： 顧客在購物車頁面缺乏高相關性的搭售建議，未能激發「買了 A 順手加購 B」的連帶銷售。
3. 顧客無法獲得個體化推薦： 舊客重複瀏覽相同熱門排行榜，無法享有「一對一個人化猜你喜歡」的尊榮體驗。

幸運的是，你的團隊已運用第十二週至第十四週所學的演算法，完成三階段分析：
- W12 產品維度 EDA： 發現符合 Pareto 80/20 法則，前 20% A 類商品貢獻 78.5% 全站營收；建立「熱門商品推薦器（Popularity-Based Recommender）」破解冷啟動。
- W13 購物籃關聯規則挖掘： 運用 FP-Growth 演算法探勘出強關聯組合，例如 $\{\text{燭台}\} \implies \{\text{粉茶杯}\}$ 具備高達 100% 之 Confidence 與 Lift = $1.25$。
- W14 協同過濾與 A/B Test 架構： 建立顧客-商品矩陣（Customer-Item Matrix），運用餘弦相似度（Cosine Similarity）實作 User-Based 與 Item-Based 協同過濾，並規劃線上 A/B Test 實驗架構（Z-Test / t-Test, $p\text{-value} < 0.05$）與多階層保底降級機制。

現在，行銷副總與技術總監要求你的團隊在 3 頁 A4 篇幅內，向高階決策層提交一份兼具「數據計算」、「演算法架構」與「商業落地可行性」的推薦系統綜合作業企劃書。

---

## 作業四大必答任務（4 Core Tasks）

### 任務一：產品維度 EDA 與 Pareto ABC 分級暨熱門商品冷啟動策略（配分：20%）

1. Pareto 80/20 數據診斷與 ABC 分級：
   - 假定平台總營收為 $1,000 萬元，共有 3,800 種商品。請說明 A 類商品（前 20% 高營收品項）、B 類商品（中間 30% 品項）與 C 類長尾商品（後 50% 品項）的營收貢獻分佈。
   - 說明為什麼長尾 C 類商品不適合直接放置於 App 首頁頂部？
2. 熱門商品推薦器設計：
   - 請為無歷史行為紀錄的新訪客（Cold-Start），設計一套首頁「流量密碼熱門排行榜」推薦邏輯。你將如何綜合考量「近 7 天銷售件數（Quantity）」與「總銷售額（Revenue）」進行加權排序？

### 任務二：購物籃關聯規則搭售與加購推播企劃（配分：30%）

請運用第十三週關聯規則之核心指標（Support, Confidence, Lift），分析以下探勘結果並設計搭售策略：

1. 指標計算與雙向非對稱性解析：
   - 假設全站 5 筆示範訂單中，$\text{Support}(\text{燭台}) = 0.60$、$\text{Support}(\text{粉茶杯}) = 0.80$，共同支持度 $\text{Support}(\text{燭台, 粉茶杯}) = 0.60$：
     - 請計算規則 1（$\text{燭台} \implies \text{粉茶杯}$）與規則 2（$\text{粉茶杯} \implies \text{燭台}$）各自的 Confidence 與 Lift。
     - 說明為什麼兩條規則的 Lift 完全相同（$1.25$），但 Confidence 卻分別為 100% 與 75%？這在購物車頁面的「單向加購推播」上有何決策啟示？
2. 購物車與結帳頁面搭售企劃：
   - 請為電商平台設計一套 3 級搭售策略（如：結帳頁 85 折組合包、購物車彈窗加購品選單）。

### 任務三：一對一個人化協同過濾 (CF) 推薦器與保底降級架構企劃（配分：30%）

1. 餘弦相似度（Cosine Similarity）幾何解析與計算：
   - 假設顧客 A 購物向量為 $[4, 3]$（買 4 件綠茶杯、3 件粉茶杯），顧客 B 為 $[8, 6]$，顧客 C 為 $[0, 5]$：
     - 請試算 $\text{Cosine}(A, B)$ 與 $\text{Cosine}(A, C)$。
     - 說明為什麼 $\text{Cosine}(A, B) = 1.00$（夾角 $\theta = 0^\circ$）？這代表什麼商業意義？
2. User-Based CF vs. Item-Based CF 選擇：
   - 請比較 Amazon 採用的 Item-Based CF 與 User-Based CF 在「計算複雜度」、「離線快取」與「商業解釋性」上的優劣。
3. 全站多階層保底推薦架構設計（Multi-Tier Fallback Cascade Architecture）：
   - 請繪製或條列說明包含「第一線個人化協同過濾 $\to$ 第二線購物籃關聯規則 $\to$ 第三線熱門排行榜保底」的完整降級邏輯。

### 任務四：離線指標評估與線上 A/B Test 實驗驗證與商業效益估算（配分：20%）

1. 離線指標試算（Precision@5 與 Recall@5）：
   - 假設顧客小明測試集實際購買了 4 個商品 $\{\text{綠茶杯}, \text{粉茶杯}, \text{燭台}, \text{抹布}\}$，推薦系統為小明推薦了 Top 5 清單 $\{\text{綠茶杯}, \text{粉茶杯}, \text{保溫瓶}, \text{馬克杯}, \text{餐巾紙}\}$。
   - 請計算 Precision@5、Recall@5 與 F1-Score@5，並說明在手機 App 彈窗推播時，為何應優先追求高的 Precision@K？
2. 線上 A/B Test 實驗設計與假設檢定：
   - 設計 50% 流量（對照組 A：熱門榜單）vs 50% 流量（實驗組 B：個人化 CF）之 A/B Test。
   - 說明如何透過雙樣本比例 Z 檢定與雙樣本獨立 t 檢定評估 CVR 與 AOV，並說明 $p\text{-value} < 0.05$ 的判讀標準。

---

## 評分標準（Rubric）

| 評分維度 | 配分 | 評分細節標準 |
| :--- | :--- | :--- |
| 數據邏輯與公式精確度 | 25% | Pareto 分級、Support/Confidence/Lift、Cosine 相似度與 Precision/Recall 計算精確無誤。 |
| 跨章節觀念整合能力 | 35% | 能貫通 Pareto EDA、Apriori/FP-Growth 關聯規則、協同過濾與 A/B Test 驗證。 |
| 商業落地與企劃可行性 | 25% | 多階層保底架構與搭售企劃具備實務操作可行性，能直接作推薦系統規劃書。 |
| 報告排版與精煉溝通 | 15% | 嚴格遵守 3 頁 A4 限制，排版簡潔專業、圖表清晰、無冗長贅字。 |

---

## 繳交須知

- 檔案命名格式： `組別_第X組_模組三企劃書.pdf`
- 繳交方式： 請由組長於 12/20 23:59 前上傳至學校 TronClass 課程平台。
- 學術誠信： 歡迎使用 AI 工具協助發想與潤飾文案，但所有數據計算與商業邏輯必須經由小組討論確認。
