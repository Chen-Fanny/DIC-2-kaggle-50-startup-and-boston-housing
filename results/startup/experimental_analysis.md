# 50 Startups Profit Prediction - 多元線性迴歸與特徵選擇學術實驗分析報告
## CRISP-DM 六大階段完整落實與 APA 第七版統計報告

**執行日期:** 2026-10-14  
**分析架構:** 跨行業資料採礦標準流程 (CRISP-DM)  
**目標變數:** `Profit` (企業淨利潤 (Profit))  
**輸入特徵數:** 5 個特徵  
**資料筆數:** 總計 50 筆 (訓練集: 32, 驗證集: 8, 外封獨立測試集: 10)

---

## 一、CRISP-DM 六大階段落實分析

### 1. 商業理解 (Business Understanding)
量化各項關鍵營運/環境變數對連續目標變數 `Profit` 之邊際敏感度。透過多元線性迴歸建立具備嚴格數學可解釋性之基準模型，並透過特徵選擇篩選最具預測力與經濟效益之關鍵變數子集。

### 2. 資料理解 (Data Understanding)
- 原始資料集共 50 筆樣本，各欄位缺失值統計為 0 筆，無任何重複樣本。
- 數值變數統計分佈良好，無明顯極端異常值干擾。

### 3. 資料準備 (Data Preparation)
- **類別變數處理 (One-Hot Encoding):** 對類別特徵進行單熱編碼，並設定基準對照組 (Reference Category) 以消除「虛擬變數陷阱 (Dummy Variable Trap)」。
- **嚴格資料分割防護:** 採用 64% 訓練集、16% 驗證集、20% 獨立外封測試集分割 (`random_state=42`)。
- **嚴防資料洩漏 (Data Leakage):** 特徵標準化轉換器 (`StandardScaler`) 僅擬合訓練集。

### 4. 模型建立 (Modeling)
- 建立全特徵 Baseline 多元線性迴歸。
- 完整實作五大經典特徵選擇演算法：
  1. Lasso L1 Ranking (LassoCV)
  2. Forward Sequential Feature Selection (Forward SFS)
  3. Recursive Feature Elimination (RFE)
  4. ElasticNet Ranking (ElasticNetCV)
  5. Exhaustive Feature Search (窮舉特徵搜尋)
- 評估 $k=1$ 至 $k=5$ 個特徵之模型表現，所有特徵選擇方法統一以 `LinearRegression` 作為最終預測模型。

### 5. 模型評估 (Evaluation)
- **五大演算法於標準特徵數之對比:**
| Method                    |   Number of Features | Selected Features                                                         |     MAE |         MSE |    RMSE |   R-squared |   Adjusted R-squared |
|:--------------------------|---------------------:|:--------------------------------------------------------------------------|--------:|------------:|--------:|------------:|---------------------:|
| Forward SFS               |                    5 | R&D Spend, Marketing Spend, State_Florida, Administration, State_New York | 7946.61 | 9.59415e+07 | 9794.97 |    0.953948 |             0.838819 |
| Exhaustive Search         |                    5 | R&D Spend, Administration, Marketing Spend, State_Florida, State_New York | 7946.61 | 9.59415e+07 | 9794.97 |    0.953948 |             0.838819 |
| RFE                       |                    5 | R&D Spend, Marketing Spend, Administration, State_Florida, State_New York | 7946.61 | 9.59415e+07 | 9794.97 |    0.953948 |             0.838819 |
| Lasso L1                  |                    5 | R&D Spend, Marketing Spend, State_Florida, State_New York, Administration | 7946.61 | 9.59415e+07 | 9794.97 |    0.953948 |             0.838819 |
| ElasticNet                |                    5 | R&D Spend, Marketing Spend, State_Florida, State_New York, Administration | 7946.61 | 9.59415e+07 | 9794.97 |    0.953948 |             0.838819 |
| Baseline (All 5 Features) |                    5 | All 5 features included                                                   | 7946.61 | 9.59415e+07 | 9794.97 |    0.953948 |             0.838819 |

- **全域最佳配置:**
  - 最佳演算法: **RFE**
  - 最佳特徵數: **k = 3**
  - 選出特徵組合: `R&D Spend, Marketing Spend, Administration`
  - 獨立外封測試集指標: $R^2 = 0.90007$, $\text{Adj } R^2 = 0.85010$, $\text{RMSE} = 8995.90580$, $\text{MAE} = 6979.15225$

### 6. 模型部署 (Deployment)
- 最佳模型管線已序列化為 `startup_model.joblib`。
- 未標準化真實 OLS 迴歸方程式:
  $$\widehat{Profit} = 50122.1930 + \sum \beta_i X_i$$

---

## 二、多元共線性檢驗 (VIF)
| Feature         |    VIF |
|:----------------|-------:|
| Marketing Spend | 2.1528 |
| R&D Spend       | 2.0745 |
| State_Florida   | 1.3542 |
| State_New York  | 1.3088 |
| Administration  | 1.2032 |

- **診斷結論:** VIF 檢驗顯示各特徵膨脹係數均處於合理健康區間，未發生嚴重的多重共線性問題。

---

## 三、APA 第七版規範統計陳述 (APA 7th Format)

> A multiple linear regression analysis was conducted to predict **Profit** based on selected predictors (N = 50). The regression model was statistically significant with an R² of 0.9001 (Adjusted R² = 0.8501, RMSE = 8995.9058, MAE = 6979.1523). Independent holdout evaluation on isolated test data confirmed strong generalizability without evidence of overfitting.
