"""
===================================================================
🐍 Python 機器學習：特徵縮放 (Feature Scaling) 完整教學 🐍
===================================================================

您好！這是一份循序漸進的程式碼教學，專為希望深入理解「特徵縮放」的學習者設計。
我們將從最基本的概念開始，逐步探索各種技術，並最終在真實資料集上進行應用。

本教學涵蓋以下核心主題：
1.  **基本概念**：為何特徵縮放至關重要？
2.  **核心技術**：詳解並實作兩種最主要的縮放方法：
    -   **標準化 (Standardization / Z-score Normalization)**
    -   **歸一化 (Normalization / Min-Max Scaling)**
3.  **異常值處理**：探討異常值對不同縮放方法的巨大影響，並介紹更穩健的 `RobustScaler`。
4.  **偏態資料轉換**：學習如何使用對數轉換 (Log Transform) 和冪轉換 (Power Transformation) 來處理偏態分佈的特徵。
5.  **黃金準則**：在訓練集和測試集上應用縮放的正確流程，以避免資料洩漏 (Data Leakage)。
6.  **綜合實戰**：在一個真實的 `insurance` 資料集上，應用所學知識，為不同特徵制定並實施最佳的預處理策略。

請依照程式碼區塊 (cell) 的順序執行，以獲得最佳的學習體驗。
"""

# %%
# ===================================================================
# 0. 匯入必要的函式庫
# ===================================================================
# 讓我們從匯入所有會用到的工具開始。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import scipy.stats as stats

# 設定視覺化風格，讓圖表更美觀
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'Arial Unicode MS'] # 優先使用微軟正黑體，其次是黑體-繁，再其次是蘋果的通用字體
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題


# %%
# ===================================================================
# 1. 為何需要特徵縮放？ (The "Why")
# ===================================================================
print("="*60)
print("1. 為何需要特徵縮放？")
print("="*60)
print("""
在許多機器學習演算法中，特別是那些基於距離計算（如 KNN）或梯度下降優化（如線性迴歸、神經網路）的模型，
特徵縮放是一個至關重要的預處理步驟。

想像一個資料集包含兩個特徵：
- 'age': 範圍 0 到 100
- 'income': 範圍 0 到 1,000,000

如果直接使用這些原始資料，模型會不成比例地被 'income' 這個數值範圍遠大於 'age' 的特徵所主導。
'income' 上一個單位的變化，其影響力遠大於 'age' 的一個單位變化，這顯然不公平。

**特徵縮放的目的，就是將所有特徵的數值放在一個公平、可比較的尺度上，
確保每個特徵都能對模型的結果做出其應有的貢獻，而不是被數值大小所綁架。**
""")

# %%
# ===================================================================
# 2. 準備與檢視資料
# ===================================================================
print("\n" + "="*60)
print("2. 準備與檢視資料")
print("="*60)

# 我們將使用 'insurance' 資料集，它包含了不同尺度的數值特徵。
try:
    # 讀取資料 (假設資料集在同一個目錄下的 data 資料夾中)
    # 如果您將 'insurance.csv' 放在與此 .py 檔相同的位置，請移除 'data/'
    df = pd.read_csv('insurance.csv')
    print("✅ 成功載入 Insurance 資料集！")

    # 為簡化起見，我們先只處理數值特徵
    df_numeric = df.select_dtypes(include=np.number)
    print("\n資料集中的數值特徵：")
    print(df_numeric.head())

    print("\n原始數值特徵的統計描述：")
    print(df_numeric.describe())

    # 視覺化原始資料的分佈
    print("\n🔍 正在繪製原始特徵分佈圖...")
    df_numeric.plot(kind='kde', subplots=True, layout=(2, 2), figsize=(12, 8), sharex=False, title="原始特徵的機率密度分佈")
    plt.suptitle("原始特徵的機率密度分佈 (KDE Plot)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

except FileNotFoundError:
    print("❌ 錯誤：找不到 'insurance.csv' 檔案。")
    print("請確認 'insurance.csv' 是否存在於正確的路徑下，或修改下方的讀取路徑。")
    df_numeric = pd.DataFrame() # 創建一個空的 DataFrame 以免後續程式碼出錯


# %%
# ===================================================================
# 3. 核心縮放技術：Standardization vs. Normalization
# ===================================================================
if not df_numeric.empty:
    print("\n" + "="*60)
    print("3. 核心縮放技術：Standardization vs. Normalization")
    print("="*60)

    # --- 3.1 標準化 (Standardization) ---
    print("\n--- 3.1 標準化 (Standardization) ---")
    print("原理：將數據轉換為 **均值為 0，標準差為 1** 的分佈。")
    print("公式：z = (x - μ) / σ")
    print("優點：適用範圍廣，對異常值相對不敏感，是大多數情況下的首選。")
    print("------------------------------------")

    scaler_std = StandardScaler()
    df_standardized = pd.DataFrame(scaler_std.fit_transform(df_numeric), columns=df_numeric.columns)

    # --- 3.2 歸一化 (Normalization) ---
    print("\n--- 3.2 歸一化 (Normalization) ---")
    print("原理：將數據重新縮放到一個固定的區間，通常是 **[0, 1]**。")
    print("公式：X_norm = (X - X_min) / (X_max - X_min)")
    print("優點：數據範圍固定、直觀。")
    print("缺點：**對異常值非常敏感！** 一個極端值就會扭曲整個特徵的縮放結果。")
    print("---------------------------------")

    scaler_minmax = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df_numeric), columns=df_numeric.columns)

    # --- 視覺化比較 ---
    print("\n🔍 正在繪製縮放前後的比較圖...")
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 6))

    ax1.set_title('原始數據 (Original)')
    sns.kdeplot(data=df_numeric[['age', 'bmi']], ax=ax1)

    ax2.set_title('標準化後 (Standardized)')
    sns.kdeplot(data=df_standardized[['age', 'bmi']], ax=ax2)

    ax3.set_title('歸一化後 (Normalized)')
    sns.kdeplot(data=df_normalized[['age', 'bmi']], ax=ax3)

    plt.suptitle("不同縮放方法的比較 (以 age 和 bmi 為例)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("\n--- 標準化後描述 (均值≈0, 標準差≈1) ---")
    print(df_standardized.describe().round(2))

    print("\n--- 歸一化後描述 (最小值=0, 最大值=1) ---")
    print(df_normalized.describe().round(2))

    print("\n**觀察**：縮放只改變特徵的 **尺度 (scale)**，而不改變其 **分佈形狀 (shape)**。")


# %%
# ===================================================================
# 4. 異常值的衝擊：一個災難性的例子
# ===================================================================
print("\n" + "="*60)
print("4. 異常值的衝擊：為何 Min-Max Scaling 很危險？")
print("="*60)
print("""
這個實驗將血淋淋地展示，為何我們一再強調「歸一化 (Min-Max Scaling) 對異常值非常敏感」。
我們將手動加入一個極端值，觀察它如何徹底摧毀 Min-Max Scaling 的結果。
""")

# 創建一個沒有異常值的原始數據
np.random.seed(42)
data_normal = np.random.normal(loc=100, scale=20, size=100)
# 加入一個巨大的異常值
data_with_outlier = np.append(data_normal, 800) # 異常值設為 800

# 將它們放入 DataFrame 中以便比較
data_to_scale = data_with_outlier.reshape(-1, 1)

# 初始化三種不同的縮放器
ss = StandardScaler()
mms = MinMaxScaler()
rs = RobustScaler() # 專門用來處理異常值的穩健型縮放器

# 應用縮放
df_scaled_outlier = pd.DataFrame({
    'Original': data_with_outlier,
    'StandardScaled': ss.fit_transform(data_to_scale).flatten(),
    'MinMaxScaled': mms.fit_transform(data_to_scale).flatten(),
    'RobustScaled': rs.fit_transform(data_to_scale).flatten() # 使用中位數和四分位距(IQR)，對異常值不敏感
})

# --- 視覺化比較縮放結果 ---
print("\n🔍 正在繪製異常值對不同縮放器的影響圖...")
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

sns.histplot(df_scaled_outlier['Original'], kde=True, ax=axes[0]).set_title("原始數據 (含異常值)")
sns.histplot(df_scaled_outlier['StandardScaled'], kde=True, ax=axes[1]).set_title("StandardScaler\n(受影響，但分佈尚存)")
sns.histplot(df_scaled_outlier['MinMaxScaled'], kde=True, ax=axes[2]).set_title("MinMaxScaler\n(災難！數據被壓縮到極小區間)")
sns.histplot(df_scaled_outlier['RobustScaled'], kde=True, ax=axes[3]).set_title("RobustScaler\n(效果最好，幾乎不受影響)")

plt.suptitle("一個異常值如何摧毀特徵縮放", fontsize=18, y=1.02)
plt.tight_layout()
plt.show()

print("""
### 結果解讀 ###
- **StandardScaler** (中左圖): 雖然均值和標準差被異常值拉偏，但我們大致還能看出原始數據的分佈形狀。
- **MinMaxScaler** (中右圖): 這是 **災難性** 的結果。巨大的異常值 `800` 被映射到 `1`，導致所有正常數據都被壓縮到 0 到 0.2 這個極小的區間內，內部差異幾乎完全消失。
- **RobustScaler** (右圖): **效果驚人地好**。由於它使用對異常值不敏感的中位數和四分位距(IQR)，它成功地縮放了數據的核心部分，完全沒有被異常值影響。

**結論：如果你的數據含有或可能含有異常值，請優先考慮 `RobustScaler` 或 `StandardScaler`，並謹慎使用 `MinMaxScaler`。**
""")


# %%
# ===================================================================
# 5. 處理偏態資料：冪轉換 (Power Transformation)
# ===================================================================
if not df_numeric.empty:
    print("\n" + "="*60)
    print("5. 處理偏態資料：冪轉換 (Power Transformation)")
    print("="*60)
    print("""
縮放只改變尺度，不改變形狀。但如果特徵分佈本身是歪斜的（例如，收入、費用），
許多模型（特別是線性模型）的性能會因此下降。
冪轉換是一系列旨在 **改變變數分佈形狀**，使其更對稱、更像常態（高斯）分佈的數學方法。
""")

    # 'charges' 是一個典型的右偏（正偏）分佈
    charges = df_numeric[['charges']]

    # --- 5.1 對數轉換 (Log Transform) ---
    # 對於右偏資料，取對數是最簡單有效的轉換方法
    # 使用 np.log1p(x) 等同於 np.log(1+x)，可以避免 x=0 的情況
    charges_log = np.log1p(charges)

    # --- 5.2 更通用的 Yeo-Johnson 轉換 ---
    # Scikit-learn 的 PowerTransformer 可以自動找到最佳的轉換參數
    # 'yeo-johnson' 方法可以處理正數、0 和負數，適用性最強
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    charges_p_transformed = power_transformer.fit_transform(charges)

    # --- 視覺化比較轉換效果 ---
    print("\n🔍 正在繪製冪轉換前後的比較圖...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(charges, kde=True, ax=axes[0]).set_title("原始 'charges' (高度右偏)")
    sns.histplot(charges_log, kde=True, ax=axes[1], color='green').set_title("對數轉換後 (更對稱)")
    sns.histplot(charges_p_transformed, kde=True, ax=axes[2], color='purple').set_title("Yeo-Johnson 轉換後 (接近常態)")
    plt.suptitle("使用冪轉換馴服偏態資料", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- 使用 Q-Q 圖評估常態性 ---
    print("\n🔍 正在繪製 Q-Q 圖以評估常態性...")
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    stats.probplot(charges['charges'], dist=stats.norm, plot=ax1)
    ax1.set_title('原始 "charges" 的 Q-Q 圖')

    ax2 = fig.add_subplot(1, 2, 2)
    stats.probplot(charges_p_transformed.flatten(), dist=stats.norm, plot=ax2)
    ax2.set_title('Yeo-Johnson 轉換後 的 Q-Q 圖')
    plt.suptitle("Q-Q 圖：點越貼近紅線，代表數據越接近常態分佈", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("**觀察**：轉換後的數據點（右圖）幾乎完美地落在紅線上，證明轉換非常成功。")


# %%
# ===================================================================
# 6. 黃金準則：Fit on Train, Transform on Train & Test
# ===================================================================
if not df_numeric.empty:
    print("\n" + "="*60)
    print("6. 黃金準則：Fit on Train, Transform on Train & Test")
    print("="*60)
    print("""
這是特徵縮放中最關鍵、最容易出錯的地方，直接關係到 **資料洩漏 (Data Leakage)**。

**第一原理**：測試集是用來模擬模型在未來從未見過的真實數據上的表現。因此，任何關於數據分佈的資訊
（如均值、標準差、最大/最小值）都**只能從訓練集中學習**，絕對不能讓模型「偷看」到測試集的資訊。

**正確流程**：
1.  將數據集劃分為訓練集和測試集。
2.  在 **訓練集** 上呼叫縮放器的 `.fit()` 或 `.fit_transform()` 方法來學習縮放參數。
3.  使用 **同一個** 學習到的縮放器，分別對 **訓練集** 和 **測試集** 呼叫 `.transform()` 方法來應用縮放。
""")

    # 1. 劃分資料
    X = df_numeric.drop('charges', axis=1)
    y = df_numeric['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"資料已劃分：訓練集 {X_train.shape[0]} 筆，測試集 {X_test.shape[0]} 筆。")

    # 2. 創建並在訓練集上 fit 縮放器
    scaler_final = StandardScaler()
    scaler_final.fit(X_train) # ✅ 只在 X_train 上學習均值和標準差

    # 3. 對訓練集和測試集進行 transform
    X_train_scaled = scaler_final.transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    # 轉換回 DataFrame 以便查看
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    print("\n--- 訓練集縮放後描述 (均值≈0, 標準差≈1) ---")
    print(X_train_scaled_df.describe().round(2))

    print("\n--- 測試集縮放後描述 ---")
    print(X_test_scaled_df.describe().round(2))

    print("""
**注意**：測試集縮放後的均值和標準差 **不會** 嚴格等於 0 和 1。
這是**完全正常且正確的**！因為我們是用訓練集的參數來轉換它的，這才是真實模擬模型在新數據上表現的方法。
""")


# %%
# ===================================================================
# 7. 綜合實戰：為 Insurance 資料集制定完整預處理策略
# ===================================================================
if not df_numeric.empty:
    print("\n" + "="*60)
    print("7. 綜合實戰：為 Insurance 資料集制定完整預處理策略")
    print("="*60)
    print("""
現在，我們將綜合運用前面學到的所有知識，為 `insurance` 資料集中的每個數值特徵
量身打造一個完整的預處理方案。

**策略制定**：
1.  **`age`**: 分佈較為均勻，沒有明顯偏態。適合使用 `StandardScaler`。
2.  **`bmi`**: 看起來非常接近常態分佈，是 `StandardScaler` 的理想對象。
3.  **`children`**: 離散的計數變數，值域小但可視為有輕微的右偏和潛在的異常行為（相對於0、1、2），使用 `RobustScaler` 更穩健。
4.  **`charges`**: 明顯的 **高度右偏** 分佈。必須 **先進行冪轉換，再進行縮放**。
""")

    # 建立工作副本
    df_processed = df_numeric.copy()
    print("\n處理前的資料預覽：")
    print(df_processed.head())

    # --- 步驟 1: 處理 charges (冪轉換 + 縮放) ---
    pt_charges = PowerTransformer(method='yeo-johnson', standardize=True) # standardize=True 會在轉換後直接進行標準化
    df_processed['charges'] = pt_charges.fit_transform(df_processed[['charges']])

    # --- 步驟 2: 處理 age 和 bmi (標準化) ---
    scaler_std_ab = StandardScaler()
    cols_to_scale_std = ['age', 'bmi']
    df_processed[cols_to_scale_std] = scaler_std_ab.fit_transform(df_processed[cols_to_scale_std])

    # --- 步驟 3: 處理 children (穩健縮放) ---
    scaler_robust_c = RobustScaler()
    df_processed['children'] = scaler_robust_c.fit_transform(df_processed[['children']])

    print("\n✅ 所有數值特徵處理完成。")
    print("\n處理後的資料預覽：")
    print(df_processed.head())

    print("\n處理後的資料統計描述：")
    print(df_processed.describe().round(2))

    # --- 視覺化比較最終處理前後的結果 ---
    print("\n🔍 正在繪製最終處理前後的對比圖...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('特徵處理前後的最終分佈對比', fontsize=20)

    # 繪製原始分佈
    for i, col in enumerate(df_numeric.columns):
        sns.histplot(df_numeric[col], kde=True, ax=axes[0, i])
        axes[0, i].set_title(f'原始: {col}')

    # 繪製處理後的分佈
    for i, col in enumerate(df_processed.columns):
        sns.histplot(df_processed[col], kde=True, ax=axes[1, i], color='green')
        axes[1, i].set_title(f'處理後: {col}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("""
**最終成果**：
- **charges (右一)**: 對比最為震撼。從高度右偏變成了漂亮的常態分佈。
- **age, bmi (左一、左二)**: 形狀不變，但尺度已被標準化。
- **children (右二)**: 已被穩健地縮放。

現在，這個 `df_processed` DataFrame 中的所有特徵都在可比較的尺度上，並且消除了偏態，
它已經準備好被投入到機器學習模型中進行訓練了！
""")


# %%
# ===================================================================
# 8. 總結與回顧
# ===================================================================
print("\n" + "="*60)
print("8. 總結與回顧")
print("="*60)
print("""
恭喜您完成了本次特徵縮放的學習之旅！讓我們回顧一下最重要的知識點：

| 縮放方法              | 原理                       | 優點                               | 缺點/風險                      | 適用場景                                           |
|-----------------------|----------------------------|------------------------------------|--------------------------------|----------------------------------------------------|
| **Standardization** | 均值=0, 標準差=1           | 保留分佈和異常值資訊，適用範圍廣。 | 數據沒有被限制在特定範圍內。   | **大多數機器學習演算法的預設首選。**              |
| **Normalization** | 縮放到 [0, 1]              | 數據範圍固定，直觀。               | **對異常值非常敏感**。         | 需要特定數據範圍的演算法（如神經網路、圖像處理）。 |
| **RobustScaler** | 使用中位數和 IQR           | **對異常值穩健**，效果好。         | 計算比前兩者稍慢。             | **當數據中存在或懷疑存在異常值時的首選。**           |
| **Power Transform** | 對數/Box-Cox/Yeo-Johnson   | 修正偏態，使數據更接近常態分佈。   | 會改變原始數據的分佈。         | 用於線性模型或假設數據為常態分佈的模型前。         |

**最重要的黃金準則：永遠記得 `fit on train, transform on train and test`，以避免資料洩漏！**

希望這份教學對您有所幫助！
""")
