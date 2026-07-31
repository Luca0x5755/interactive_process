# -*- coding: utf-8 -*-
"""
==============================================================================
## Python 機器學習前處理：缺失值與離群值處理 權威教學腳本
==============================================================================

### 教學目標：
1.  **深入理解**：掌握缺失值 (Missing Values) 與離群值 (Outliers) 的成因及其對模型的潛在衝擊。
2.  **系統化學習**：從資料診斷、視覺化分析到多元處理技術，建立一套標準化的數據清理工作流程。
3.  **掌握核心技術**：
    * **缺失值處理**：精通均值/中位數/眾數填充、常數填充、插值法，並理解基於模型的預測性填充概念。
    * **離群值處理**：熟練運用視覺化、Z-score、IQR 法則進行偵測，並實作移除、縮尾/設限 (Capping) 等處理策略。
4.  **培養決策能力**：學習根據資料特性與業務邏輯，選擇最合適的處理方法。

### 檔案結構：
- **第 0 部分**: 環境設定與範例資料生成
- **第 1 部分**: 缺失值處理 (Missing Value Handling)
    - 1.1 診斷與視覺化：理解缺失的模式
    - 1.2 基礎填充策略 (Simple Imputation)
    - 1.3 進階填充策略 (Advanced Imputation)
- **第 2 部分**: 離群值處理 (Outlier Handling)
    - 2.1 偵測離群值：視覺化與統計方法
    - 2.2 處理離群值：三大實用策略
    - 2.3 基於模型的偵測方法簡介
- **第 3 部分**: 總結與最佳實踐

### 核心理念：
Garbage In, Garbage Out (GIGO)。在機器學習流程中，資料前處理是決定模型成敗最關鍵、也最耗時的階段。
本腳本旨在提供一套清晰、可複用的程式碼範本，幫助您高效且正確地完成數據清理任務，為後續的模型建立奠定堅實的基礎。
"""

# %%
# =============================================================================
# 0. 環境設定與範例資料生成
# =============================================================================

# 導入必要的函式庫
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# --- 圖表視覺化設定 ---
# 設定 Seaborn 的主題風格，使圖表更美觀
sns.set_theme(style="whitegrid")
# 設定 Matplotlib 支援中文顯示 (以 'Microsoft JhengHei' 為例，可替換為系統中其他支援的字體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
# 解決 Matplotlib 圖表中負號顯示為方塊的問題
plt.rcParams['axes.unicode_minus'] = False

# --- 建立範例資料集 ---
# 為了同時演示缺失值與離群值，我們建立一個更豐富的資料集
# 假設這是一份關於員工績效的資料
data = {
    'EmployeeID': range(1, 16),
    'Age': [25, 30, np.nan, 35, 40, 45, 50, np.nan, 60, 28, 38, 42, 33, 99, 29], # 包含缺失值與離群值(99)
    'Department': ['HR', 'Engineering', 'Marketing', 'Engineering', np.nan, 'HR', 'Sales', 'Marketing', 'Sales', 'Engineering', 'HR', 'Marketing', 'Sales', 'Engineering', 'HR'],
    'YearsAtCompany': [2, 5, 3, 8, 15, 20, 22, 1, 35, 4, 10, 12, 6, 9, 5],
    'MonthlySalary': [50000, 80000, 65000, 90000, 150000, 180000, 220000, 45000, np.nan, 75000, 62000, np.nan, 85000, 95000, 500000], # 包含缺失值與離群值(500000)
    'PerformanceScore': [3.5, 4.2, 3.8, 4.5, 4.8, 4.9, 3.1, 2.5, 4.7, 4.1, 3.9, 3.4, 0.2, 4.3, 3.7] # 包含離群值(0.2)
}
df_original = pd.DataFrame(data)
# 建立一個工作副本，所有操作都在副本上進行，以保留原始資料的完整性
df = df_original.copy()

print("--- 完整範例資料集 (原始狀態) ---")
print(df)
print("\n--- 資料集基本資訊 ---")
df.info()

# %%
# =============================================================================
# 1. 缺失值處理 (Missing Value Handling)
# =============================================================================
"""
第一步：理解缺失，而非動手填補。缺失本身可能就是一種資訊。
"""

# --- 1.1 診斷與視覺化：理解缺失的模式 ---
print("\n--- 1.1 診斷與視覺化 ---")
print("\n各欄位缺失值統計：")
print(df.isnull().sum())

# 使用 missingno 套件進行專業的視覺化診斷
print("\n顯示缺失值視覺化圖表...")

# 1.1.1 矩陣圖 (Matrix Plot)
# 直觀展示每筆資料 (row) 的缺失情況。白色線條代表缺失。
# 右側的 Sparkline 顯示了每行的資料完整度，是快速定位問題樣本的好工具。
msno.matrix(df)
plt.title('msno.matrix(): 樣本缺失值模式矩陣圖', fontsize=16)
plt.show()

# 1.1.2 條形圖 (Bar Chart)
# 從欄位 (column) 的角度，快速檢視每個特徵的資料完整度。
msno.bar(df)
plt.title('msno.bar(): 欄位資料完整度條形圖', fontsize=16)
plt.show()

# --- 1.2 基礎填充策略 (Simple Imputation) ---
print("\n--- 1.2 基礎填充策略 ---")
"""
適用於快速建立基準模型，或缺失比例極低的情況。
優點：簡單、快速。
缺點：可能扭曲原始資料分佈、低估變異數、破壞特徵間的相關性。
"""
df_simple = df.copy()

# 策略 A: 數值型資料 - 中位數/平均值填充
# 'Age' 和 'MonthlySalary' 都有離群值，使用中位數 (median) 比平均值 (mean) 更穩健。
imputer_median = SimpleImputer(strategy='median')
df_simple[['Age', 'MonthlySalary']] = imputer_median.fit_transform(df_simple[['Age', 'MonthlySalary']])
print("\n使用「中位數」填充 'Age' 和 'MonthlySalary'...")

# 策略 B: 類別型資料 - 眾數/常數填充
# 'Department' 可以用出現最頻繁的類別 (眾數) 來填充。
imputer_mode = SimpleImputer(strategy='most_frequent')
df_simple[['Department']] = imputer_mode.fit_transform(df_simple[['Department']])
print("使用「眾數」填充 'Department'...")

# 檢視填充結果
print("\n基礎填充後的資料集：")
print(df_simple.head(10))
print("\n確認是否還有缺失值：")
print(df_simple.isnull().sum())

# --- 1.3 進階填充策略 (Advanced Imputation) ---
print("\n--- 1.3 進階填充策略 ---")

# 策略 C: 時間序列或有序資料 - 插值法 (Interpolation)
# 假設我們的資料是按時間順序排列的，可以使用插值法。
# 這裡我們以 'PerformanceScore' 為例，假設它隨 EmployeeID 有序變化。
df_advanced = df.copy()
# 為了演示，我們先手動製造一個缺失值
df_advanced.loc[5, 'PerformanceScore'] = np.nan

# 使用線性插值 (linear interpolation)
df_advanced['PerformanceScore_interpolated'] = df_advanced['PerformanceScore'].interpolate(method='linear')
print("\n使用「線性插值法」填充 'PerformanceScore'：")
print(df_advanced[['EmployeeID', 'PerformanceScore', 'PerformanceScore_interpolated']].head(7))

# 策略 D: 基於模型的填充 - K-近鄰 (KNN)
# 原理：利用特徵間的相似性來預測缺失值。它會尋找與缺失樣本最相似的 K 個鄰居，
# 用這些鄰居的值來填充。通常比簡單填充更精確。
print("\n使用「K-近鄰 (KNN)」進行預測性填充...")
df_knn = df.copy()

# KNNImputer 需要所有欄位都是數值型，因此我們先對 'Department' 進行 Label Encoding
# 注意：在真實專案中，應先切分訓練/測試集，並只在訓練集上 fit encoder。
encoder = LabelEncoder()
# 篩選非空值進行 fit_transform
non_null_dept = df_knn['Department'].dropna()
encoded_dept = encoder.fit_transform(non_null_dept)
# 將編碼後的值放回原位
df_knn.loc[non_null_dept.index, 'Department'] = encoded_dept

# 初始化 KNNImputer
knn_imputer = KNNImputer(n_neighbors=3) # 尋找 3 個最近的鄰居

# 執行填充
df_knn_imputed_array = knn_imputer.fit_transform(df_knn)
# 轉回 DataFrame
df_knn_imputed = pd.DataFrame(df_knn_imputed_array, columns=df_knn.columns)

# 將 'Department' 還原為原始類別
# 注意：填充後的值可能是浮點數，需要先取整數
df_knn_imputed['Department'] = encoder.inverse_transform(df_knn_imputed['Department'].round().astype(int))

print("\nKNN 填充後的結果 (重點看第 3, 5, 8, 12 行)：")
print(df_knn_imputed)

# 概念提及：更進階的預測性填充
print("\n[概念] 除了 KNN，還可以訓練一個機器學習模型 (如線性迴歸、隨機森林) 來預測缺失值。")
print("例如，將 'MonthlySalary' 作為目標變數 (y)，將 'Age', 'YearsAtCompany' 等作為特徵 (X)，")
print("在沒有 'MonthlySalary' 缺失的資料上訓練模型，然後對缺失的樣本進行預測。")


# %%
# =============================================================================
# 2. 離群值處理 (Outlier Handling)
# =============================================================================
"""
離群值是「行為異常」的資料點，它們可能會扭曲統計結果，干擾模型學習。
處理原則：先偵測，再理解，後處理。
"""

# --- 2.1 偵測離群值：視覺化與統計方法 ---
print("\n--- 2.1 偵測離群值 ---")

# 2.1.1 視覺化偵測
# 箱形圖 (Box Plot) 是偵測離群值的最佳利器。
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('使用箱形圖 (Box Plot) 偵測離群值', fontsize=16, y=1.02)
sns.boxplot(y=df['Age'], ax=axes[0]).set_title('年齡 (Age)')
sns.boxplot(y=df['MonthlySalary'].dropna(), ax=axes[1]).set_title('月薪 (MonthlySalary)')
sns.boxplot(y=df['PerformanceScore'], ax=axes[2]).set_title('績效分數 (PerformanceScore)')
plt.tight_layout()
plt.show()
# 從圖中可以明顯看到 'Age' 的 99, 'MonthlySalary' 的 500000, 'PerformanceScore' 的 0.2 都是離群值。

# 2.1.2 統計方法偵測：IQR 法 (四分位距法)
# 這是一個穩健的統計規則，不受極端值影響，也是箱形圖背後的數學原理。
# 規則：一個值如果小於 Q1 - 1.5*IQR 或大於 Q3 + 1.5*IQR，則視為離群值。
print("\n使用「IQR 法」進行統計偵測...")
for col in ['Age', 'MonthlySalary', 'PerformanceScore']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\n欄位 '{col}':")
    print(f"  - 邊界: < {lower_bound:.2f} 或 > {upper_bound:.2f}")
    if not outliers.empty:
        print(f"  - 偵測到的離群值: {outliers[col].values}")
    else:
        print("  - 未偵測到離群值。")

# 2.1.3 統計方法偵測：Z-score 法 (標準分數法)
# Z-score 衡量資料點與平均值相差多少個標準差。
# 規則：通常 Z-score 的絕對值大於 3 的點被視為離群值。
# 注意：此方法假設資料呈常態分佈，對離群值本身很敏感。
print("\n使用「Z-score 法」進行統計偵測 (閾值=3)...")
for col in ['Age', 'MonthlySalary', 'PerformanceScore']:
    # 需先移除 NaN 才能計算
    col_data = df[col].dropna()
    z_scores = np.abs(stats.zscore(col_data))

    outliers = col_data[z_scores > 3]

    print(f"\n欄位 '{col}':")
    if not outliers.empty:
        print(f"  - 偵測到的離群值: {outliers.values}")
    else:
        print("  - 未偵測到離群值。")

# --- 2.2 處理離群值：三大實用策略 ---
print("\n--- 2.2 處理離群值策略 ---")
df_handled = df.copy()

# 策略 A: 移除 (Removal)
# 適用時機：當你確定離群值是輸入錯誤或量測錯誤時。
# 'Age' 為 99 很可能就是一個錯誤，我們選擇移除該筆資料。
print("\n策略 A: 移除 'Age' 為 99 的資料...")
df_handled = df_handled[df_handled['Age'] != 99]

# 策略 B: 縮尾/設限 (Capping / Winsorizing)
# 適用時機：不想丟失資料，但想限制離群值的極端影響。
# 作法：將超過上下限的值，強制設定為上下限。這是一種常用且穩健的方法。
# 我們對 'MonthlySalary' 進行 95% 的縮尾處理。
print("\n策略 B: 對 'MonthlySalary' 進行縮尾/設限...")
upper_limit = df_handled['MonthlySalary'].quantile(0.95)
lower_limit = df_handled['MonthlySalary'].quantile(0.05)
print(f"  - 月薪的 95% 上限: {upper_limit:.2f}")
print(f"  - 月薪的 5% 下限: {lower_limit:.2f}")

# 使用 np.clip() 高效執行
df_handled['MonthlySalary_capped'] = np.clip(df_handled['MonthlySalary'], lower_limit, upper_limit)


# 策略 C: 轉換 (Transformation)
# 適用時機：當資料呈嚴重右偏態分佈時，透過數學轉換 (如對數轉換) 來降低離群值的影響力。
# 'MonthlySalary' 的分佈很可能就是右偏的。
print("\n策略 C: 對 'MonthlySalary' 進行對數轉換...")
# 使用 log1p = log(1+x) 來避免 log(0) 的錯誤
df_handled['MonthlySalary_log'] = np.log1p(df_handled['MonthlySalary'])


# 視覺化比較處理前後的差異
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('離群值處理策略效果比較 (MonthlySalary)', fontsize=16, y=1.02)
sns.boxplot(y=df['MonthlySalary'], ax=axes[0]).set_title('原始狀態')
sns.boxplot(y=df_handled['MonthlySalary_capped'], ax=axes[1]).set_title('策略B: 縮尾/設限後')
sns.histplot(df_handled['MonthlySalary_log'].dropna(), kde=True, ax=axes[2]).set_title('策略C: 對數轉換後')
plt.tight_layout()
plt.show()

# --- 2.3 基於模型的偵測方法簡介 ---
print("\n--- 2.3 基於模型的偵測方法 (概念) ---")
print("除了統計方法，還可以使用更強大的機器學習模型來偵測離群值，特別是在高維度資料中：")
print("  - **Isolation Forest (孤立森林)**: 一種高效的異常偵測演算法。它通過隨機切分特徵來「孤立」樣本，離群點通常能被更快地孤立出來。")
print("  - **Local Outlier Factor (LOF)**: 基於密度的偵測方法。它比較每個樣本與其鄰居的局部密度，密度遠低於鄰居的點被視為離群值。")
print("這些方法在 Scikit-learn 中都有現成的模組可以使用 (sklearn.ensemble.IsolationForest, sklearn.neighbors.LocalOutlierFactor)。")


# %%
# =============================================================================
# 3. 總結與最佳實踐
# =============================================================================
print("\n" + "="*60)
print("### 總結與最佳實踐 ###")
print("="*60)
print("1.  **情境為王 (Context is King)**: 處理決策高度依賴對業務的理解。一個極端值是錯誤還是真實的稀有事件？")
print("2.  **診斷先行，處理在後**: 務必先用視覺化和統計方法充分了解你的資料，再選擇合適的策略。")
print("3.  **沒有萬靈丹**: ")
print("    - **缺失值**: 如果缺失有意義 (如 '無車庫') -> `常數填充` ('None' 或 0)；如果是隨機缺失 -> `中位數/眾數` 是好的起點，`KNN` 或 `預測模型` 更精確。")
print("    - **離群值**: 如果是明顯錯誤 -> `移除`；如果是真實極端值 -> `縮尾/設限` 或 `轉換` 是更穩健的選擇。")
print("4.  **防止資料洩漏 (Data Leakage)**: 這是最重要的原則！所有計算（如均值、中位數、IQR、模型訓練）都必須 **只在訓練集 (Training Set) 上進行**，然後將計算出的統計量或訓練好的模型應用 (transform) 到驗證集和測試集上。")
print("5.  **迭代與實驗**: 數據清理是一個迭代的過程。嘗試不同的策略，並評估它們對最終模型性能的影響。")

print("\n--- 教學腳本結束 ---")
