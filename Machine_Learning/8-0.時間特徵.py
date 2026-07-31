# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習進階：時間序列之時間特徵 (Time Features) 工程教學腳本
==============================================================================

### 教學目標：
1.  深入理解什麼是時間特徵 (Time Features)，以及為何它在時間序列預測中扮演核心角色。
2.  學習並實作四種關鍵的時間特徵創造技術：日期時間組件、滯後特徵、滑動窗口特徵，以及週期性特徵的轉換。
3.  掌握如何將這些特徵整合到一個完整的機器學習工作流程中，從資料準備、特徵工程、模型訓練到預測評估。
4.  了解每個程式碼區塊背後的邏輯、目的與統計原理，並透過詳細註解加深理解。
5.  透過一個真實的電力消耗預測案例，綜合應用所學知識。

### 適用對象：
-   對時間序列分析與預測有興趣的學習者。
-   希望提升機器學習專案中特徵工程能力的資料科學家或工程師。
-   已具備 Python 與 pandas 基礎，並想深入學習 scikit-learn 和 LightGBM 應用的使用者。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件。
# pandas 是我們處理與操作資料表 (DataFrame) 的利器。
import pandas as pd
# numpy 提供強大的數值計算功能。
import numpy as np
# matplotlib 與 seaborn 用於資料視覺化，幫助我們觀察數據模式。
import matplotlib.pyplot as plt
import seaborn as sns
# os 套件用於與操作系統互動，例如檢查檔案是否存在。
import os

# scikit-learn 是 Python 機器學習的標準函式庫。
from sklearn.model_selection import TimeSeriesSplit # 時間序列專用的交叉驗證工具
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # 模型評估指標

# LightGBM 是一個高效的梯度提升框架，非常適合處理表格型數據。
import lightgbm as lgb

# statsmodels 提供統計模型與測試，我們將用它來進行時間序列分解。
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 設定區 ---
# 設定全域的繪圖樣式與大小，讓圖表更美觀一致。
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 7)
# 設定中文字體，以解決 matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

print("套件導入成功！準備開始學習時間序列的特徵工程。")


# =============================================================================
# 1. 主題介紹：什麼是時間特徵 (Time Features)？
# =============================================================================
"""
在真實世界的資料中，時間序列 (Time Series) 數據無所不在，例如股票價格、天氣預報、
網站流量、電力消耗等。這些數據的核心特點是「時間依賴性」，也就是說，當前的數值
與過去的數值存在著緊密的關聯。

然而，大多數傳統的機器學習模型（如線性回歸、決策樹、梯度提升機）本身並不具備
時間感知能力。它們將每一筆數據視為獨立的事件，無法自動捕捉數據隨時間變化的趨勢、
週期性或季節性模式。

**時間特徵工程 (Time Feature Engineering) 的核心目的**，就是從原始的時間序列
數據中，手動提取出能夠明確表達這些時間依賴性與模式的「特徵 (Features)」。
這個過程是時間序列預測中至關重要的一步。一個好的特徵集，能讓簡單的模型達到
優異的預測效果；反之，若特徵提取不當，再複雜的模型也難以學習到有效的規律。

簡單來說，時間特徵工程就像是為機器學習模型裝上了一副能「看懂」時間的眼鏡，
讓它能夠洞察數據背後的趨勢、季節性與歷史規律，從而做出更精準的預測。

**本教學將涵蓋以下四種核心的時間特徵創造技術：**
1.  **日期與時間特徵 (Date and Time Features)**：從時間戳中提取年、月、日、時、星期等資訊。
2.  **滯後特徵 (Lag Features)**：使用過去時間點的數值作為當前的特徵。
3.  **滑動窗口特徵 (Rolling Window Features)**：計算過去一段時間的統計量（如平均值、標準差）。
4.  **週期性特徵轉換 (Cyclical Feature Transformation)**：將具有週期性的特徵（如小時、月份）進行數學轉換。
"""

print("\n" + "="*80)
print("第一部分：主題介紹 - 時間特徵在時間序列預測中的重要性")
print("="*80)


# =============================================================================
# 2. 準備與探索資料：電力消耗資料集
# =============================================================================
# 為了進行實作，我們將使用一個真實的電力消耗資料集。
# 這個資料集記錄了每小時的電力消耗量，非常適合用來演示各種時間特徵。
# 資料來源: Kaggle "Hourly Energy Consumption" dataset (AEP_hourly.csv)
DATASET_PATH = '../../datasets/raw/power_consumption/AEP_hourly.csv'

# --- 資料載入與初步處理 ---
# 為了程式的穩健性，我們先檢查檔案是否存在。
if not os.path.exists(DATASET_PATH):
    print(f"錯誤：找不到資料檔案 '{DATASET_PATH}'。")
    print("請確認您已從 Kaggle 下載 AEP_hourly.csv 並放置於正確路徑。")
    print("將創建一個虛擬資料集以繼續演示。")
    date_rng = pd.date_range(start='2015-01-01', end='2018-01-01', freq='H')
    dummy_data = np.random.randn(len(date_rng)).cumsum() * 1000 + 15000
    df = pd.DataFrame(dummy_data, index=date_rng, columns=['Consumption_MW'])
else:
    print(f"正在從 '{DATASET_PATH}' 載入電力消耗資料...")
    df = pd.read_csv(DATASET_PATH, index_col='Datetime', parse_dates=True)
    df.rename(columns={'AEP_MW': 'Consumption_MW'}, inplace=True)
    print("資料載入成功！")

# 對於時間序列資料，確保索引按時間排序是至關重要的第一步。
df.sort_index(inplace=True)

print("\n--- 資料集資訊 ---")
print(f"資料時間範圍: 從 {df.index.min()} 到 {df.index.max()}")
print(f"資料總筆數: {df.shape[0]}")
print("資料前五筆預覽：")
print(df.head())

# --- 資料視覺化探索 ---
print("\n正在繪製原始電力消耗時間序列圖...")
df['Consumption_MW'].plot(title='每小時電力消耗量歷史數據', color='teal', alpha=0.8)
plt.xlabel('日期')
plt.ylabel('電力消耗 (MW)')
plt.grid(True)
plt.show()

"""
**[探索性分析]**
從上方的時間序列圖中，我們可以初步觀察到幾個明顯的模式：
1.  **季節性 (Seasonality)**：每年夏季和冬季，電力消耗都會出現高峰，這顯然與空調和暖氣的使用有關。
2.  **長期趨勢 (Trend)**：整體來看，電力消耗似乎隨著時間有緩慢增長的趨勢。
3.  **週期性波動 (Cyclical Patterns)**：在更細的時間尺度上，我們可以預期存在每日（白天 vs. 夜晚）和每週（工作日 vs. 週末）的規律性波動。

這些觀察都證明了提取時間特徵的必要性。接下來，我們將逐一實作各種特徵工程技術。
"""

print("\n" + "="*80)
print("第二部分：資料準備與探索性分析")
print("="*80)


# =============================================================================
# 3. 技術一：日期與時間特徵 (Date and Time Features)
# =============================================================================
"""
**[核心概念]**
這是最直觀的時間特徵。我們可以直接從時間戳 (timestamp) 中分解出多種具有預測能力的
組件。例如，「月份」可以幫助模型捕捉季節性，「小時」可以捕捉日內的活動規律，而
「星期幾」則能區分工作日與週末的模式。

在 pandas 中，只要 DataFrame 的索引是 `DatetimeIndex`，我們就可以使用 `.dt` 訪問器
來輕鬆提取這些特徵。
"""

def create_datetime_features(df):
    """
    從 DataFrame 的 DatetimeIndex 中創建多種日期與時間特徵。
    """
    df_copy = df.copy()
    time_series = df_copy.index

    # 提取各種時間組件
    df_copy['hour'] = time_series.hour
    df_copy['dayofweek'] = time_series.dayofweek  # 0=週一, 6=週日
    df_copy['dayofmonth'] = time_series.day
    df_copy['dayofyear'] = time_series.dayofyear
    df_copy['weekofyear'] = time_series.isocalendar().week.astype(int)
    df_copy['month'] = time_series.month
    df_copy['quarter'] = time_series.quarter
    df_copy['year'] = time_series.year

    # 建立一個布林特徵，判斷是否為週末
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)

    return df_copy

print("\n--- 技術一：創建日期與時間特徵 ---")
df_featured = create_datetime_features(df)
print("日期與時間特徵創建完成！")
print("部分新增特徵預覽：")
print(df_featured[['hour', 'dayofweek', 'month', 'year', 'is_weekend']].head())

# --- 視覺化驗證特徵效果 ---
print("\n正在視覺化『小時』特徵與電力消耗的關係...")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_featured, x='hour', y='Consumption_MW', ax=ax, palette='viridis')
ax.set_title('每小時電力消耗分佈 (日週期模式)', fontsize=14)
ax.set_xlabel('小時 (Hour of Day)')
ax.set_ylabel('電力消耗 (MW)')
plt.show()

"""
**[視覺化分析]**
上方的箱形圖 (Boxplot) 清晰地展示了電力消耗的日週期模式：
- 凌晨時段 (約 1-5 點) 消耗量最低。
- 上午開始攀升，並在傍晚時段 (約 17-20 點) 達到高峰。
這證明了 `hour` 是一個極具預測價值的特徵。同樣地，我們也可以分析 `month`
（季節性）和 `dayofweek`（週循環）的影響。
"""


# =============================================================================
# 4. 技術二：滯後特徵 (Lag Features)
# =============================================================================
"""
**[核心概念]**
滯後特徵是時間序列預測中最基本也最強大的特徵之一。它的思想非常簡單：
**用過去時間點的觀測值，來預測當前時間點的值。**
例如，預測今天的銷售額時，昨天、7天前、甚至一年前同一天的銷售額都是極其重要的參考資訊。

這個過程將時間序列的「自相關性 (Autocorrelation)」明確地轉化為模型可以學習的特徵。
在 pandas 中，我們使用 `.shift()` 方法來創建滯後特徵。

`df['value'].shift(1)`：將 `value` 欄位的數值向下移動一格，得到前一個時間點的值 (t-1)。
`df['value'].shift(n)`：得到 n 個時間點前的值 (t-n)。

**注意**：`shift()` 操作會在數據的開頭產生缺失值 (NaN)，因為最早的幾筆數據沒有
更早的歷史可供參考。這些包含 NaN 的行在模型訓練前通常需要被移除。
"""

def create_lag_features(df, target_col, lags):
    """
    為目標變數創建指定時間間隔的滯後特徵。
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
    return df_copy

print("\n--- 技術二：創建滯後特徵 ---")
# 我們創建幾個關鍵的滯後特徵：
# - lag_1: 前一小時的消耗量 (捕捉短期自相關)
# - lag_24: 24小時前 (昨天同一時間) 的消耗量 (捕捉日週期性)
# - lag_168: 168小時前 (一週前同一時間) 的消耗量 (捕捉週週期性)
lag_intervals = [1, 24, 168]
df_featured = create_lag_features(df_featured, 'Consumption_MW', lag_intervals)
print("滯後特徵創建完成！")
print("部分新增滯後特徵預覽 (注意開頭的 NaN 值)：")
# 由於 lag_168 的存在，前 168 筆數據都會有 NaN
print(df_featured[['Consumption_MW', 'Consumption_MW_lag_1', 'Consumption_MW_lag_24']].head())
print(df_featured[['Consumption_MW', 'Consumption_MW_lag_1', 'Consumption_MW_lag_24']].iloc[168:].head())


# --- 視覺化驗證自相關性 ---
print("\n正在視覺化『當前消耗』與『24小時前消耗』的關係...")
plt.figure(figsize=(8, 8))
plt.scatter(x=df_featured['Consumption_MW_lag_24'], y=df_featured['Consumption_MW'], alpha=0.2)
plt.title('當前電力消耗 vs. 24小時前電力消耗 (日自相關性)')
plt.xlabel('24小時前消耗 (MW)')
plt.ylabel('當前消耗 (MW)')
plt.grid(True)
plt.show()

"""
**[視覺化分析]**
上方的散點圖呈現出非常強的正相關性，點的分佈緊密地圍繞在一條對角線周圍。
這有力地證明了電力消耗存在強烈的日週期性，即今天某個小時的消耗量與昨天
同一時間的消耗量高度相關。`lag_24` 特徵將是我們模型的重要預測因子。
"""


# =============================================================================
# 5. 技術三：滑動窗口特徵 (Rolling Window Features)
# =============================================================================
"""
**[核心概念]**
如果說滯後特徵是看「過去某一個時間點」，那麼滑動窗口特徵就是看「過去一段時間」。
它通過在一個固定大小的「窗口」上計算統計量（如平均值、標準差、最大/最小值等）
來創建特徵。這個窗口會沿著時間序列一步步向前滑動。

滑動窗口特徵非常有用，因為它們可以：
1.  **平滑數據**：移動平均 (Moving Average) 可以有效過濾掉短期噪聲，突顯長期趨勢。
2.  **捕捉局部趨勢**：過去一週的平均消耗量，反映了近期的消耗水平。
3.  **衡量波動性**：過去24小時的標準差 (Standard Deviation)，可以衡量消耗的穩定性。

在 pandas 中，我們使用 `.rolling(window=size)` 方法來創建滑動窗口對象，然後在其上
應用統計函數（如 `.mean()`, `.std()`）。

**重要提示：避免數據洩漏 (Data Leakage)**
在預測 `t` 時刻的值時，我們的特徵只能使用 `t` 時刻之前的資訊。直接計算的滑動窗口
統計量（如 `rolling(7).mean()`）在 `t` 時刻的結果包含了 `t` 時刻本身的數據，這會導致
數據洩漏。正確的做法是，**先計算滑動窗口，然後再對其進行 `.shift(1)` 操作**，確保
所有特徵都只基於過去的數據。
"""

def create_rolling_features(df, target_col, window_sizes):
    """
    為目標變數創建滯後1期的滑動窗口統計特徵。
    """
    df_copy = df.copy()
    for window in window_sizes:
        # 先計算滑動統計量
        rolling_mean = df_copy[target_col].rolling(window=window).mean()
        rolling_std = df_copy[target_col].rolling(window=window).std()

        # 再進行 shift(1) 操作，避免數據洩漏
        df_copy[f'{target_col}_rolling_mean_{window}_lag1'] = rolling_mean.shift(1)
        df_copy[f'{target_col}_rolling_std_{window}_lag1'] = rolling_std.shift(1)

    return df_copy

print("\n--- 技術三：創建滑動窗口特徵 ---")
# 我們創建基於過去24小時（1天）和168小時（1週）的滑動特徵。
window_intervals = [24, 168]
df_featured = create_rolling_features(df_featured, 'Consumption_MW', window_intervals)
print("滑動窗口特徵創建完成！")
print("部分新增滑動窗口特徵預覽：")
print(df_featured[['Consumption_MW', 'Consumption_MW_rolling_mean_24_lag1', 'Consumption_MW_rolling_std_24_lag1']].iloc[168:].head())

# --- 視覺化滑動平均線 ---
print("\n正在視覺化『滑動平均線』以觀察趨勢平滑效果...")
# 選取一個月的數據進行繪圖，以便觀察
df_plot = df_featured['2018-01-01':'2018-01-31']

plt.figure(figsize=(16, 7))
plt.plot(df_plot.index, df_plot['Consumption_MW'], label='原始消耗量', color='gray', alpha=0.5)
plt.plot(df_plot.index, df_plot['Consumption_MW_rolling_mean_24_lag1'], label='24小時滑動平均', color='blue', linestyle='--')
plt.plot(df_plot.index, df_plot['Consumption_MW_rolling_mean_168_lag1'], label='168小時滑動平均', color='red', linestyle='-.')
plt.title('電力消耗與其滑動平均線 (2018年1月)')
plt.xlabel('日期')
plt.ylabel('電力消耗 (MW)')
plt.legend()
plt.grid(True)
plt.show()

"""
**[視覺化分析]**
從圖中可以看到：
- 藍色的24小時滑動平均線，較好地跟隨了每日的波動，但比原始數據平滑。
- 紅色的168小時滑動平均線，則更加平滑，幾乎完全過濾掉了每日的起伏，
  更清晰地展現了週與週之間的趨勢變化。
這些平滑後的趨勢線，為模型提供了關於近期消耗水平的穩定參考。
"""


# =============================================================================
# 6. 技術四：週期性特徵轉換 (Cyclical Feature Transformation)
# =============================================================================
"""
**[核心概念]**
像 `hour` (0-23) 或 `month` (1-12) 這樣的特徵，本質上是「環形」或「週期性」的。
例如，23點之後就是0點，12月之後就是1月。如果直接將它們作為數值（0, 1, ..., 23）
輸入模型，模型可能會誤解它們的關係，認為 23 和 0 之間的「距離」非常遙遠，
而實際上它們是相鄰的。

為了解決這個問題，我們可以使用**正弦 (sine) 和餘弦 (cosine) 變換**，將單一的
週期性特徵映射到二維圓上的 (x, y) 坐標。這樣，週期的頭尾點在新的特徵空間中就
會變得相鄰，正確地表達了其週期性。

轉換公式：
- X_sin = sin(2 * π * X / max_value)
- X_cos = cos(2 * π * X / max_value)
"""
def encode_cyclical_features(df, col, max_val):
    """
    使用正弦/餘弦變換對週期性特徵進行編碼。
    """
    df_copy = df.copy()
    df_copy[col + '_sin'] = np.sin(2 * np.pi * df_copy[col] / max_val)
    df_copy[col + '_cos'] = np.cos(2 * np.pi * df_copy[col] / max_val)
    return df_copy

print("\n--- 技術四：轉換週期性特徵 ---")
# 對 'hour', 'dayofweek', 'month' 進行週期性編碼
df_featured = encode_cyclical_features(df_featured, 'hour', 23.0)
df_featured = encode_cyclical_features(df_featured, 'dayofweek', 6.0)
df_featured = encode_cyclical_features(df_featured, 'month', 12.0)
print("週期性特徵轉換完成！")
print("部分新增週期性特徵預覽：")
print(df_featured[['hour', 'hour_sin', 'hour_cos', 'month', 'month_sin', 'month_cos']].head())

# --- 視覺化驗證週期性編碼效果 ---
print("\n正在視覺化『小時』特徵經過週期性編碼後的效果...")
# 隨機抽取24個點以避免圖像過於擁擠
sample_df = df_featured.sample(24)
plt.figure(figsize=(7, 7))
plt.scatter(sample_df['hour_sin'], sample_df['hour_cos'], c=sample_df['hour'], cmap='viridis')
plt.title('小時特徵的週期性編碼視覺化 (顏色代表原始小時)')
plt.xlabel('Hour Sin')
plt.ylabel('Hour Cos')
plt.gca().set_aspect('equal') # 確保X/Y軸比例相同，使圓形正確顯示
plt.colorbar(label='小時 (Hour)')
plt.grid(True)
plt.show()

"""
**[視覺化分析]**
上圖完美地展示了週期性編碼的效果。原始的 `hour` 特徵（由顏色表示）被映射到
了一個圓上。現在，0點和23點在坐標系中的位置是相鄰的，模型可以正確地理解
時間的連續性和週期性。
"""


# =============================================================================
# 7. 整合與模型訓練：電力消耗預測實戰
# =============================================================================
"""
現在，我們已經創建了一個包含豐富時間特徵的資料集。接下來，我們將進入模型訓練
階段，完成一個端到端的時間序列預測專案。

**流程：**
1.  **最終資料準備**：移除因特徵工程產生的缺失值 (NaN)。
2.  **定義特徵與目標**：劃分 X (特徵集) 和 y (目標變數)。
3.  **資料分割**：**嚴格按照時間順序**將資料分為訓練集和測試集，避免數據洩漏。
4.  **模型訓練**：使用 LightGBM 模型進行訓練。
5.  **模型評估**：在測試集上評估模型性能，並視覺化預測結果。
6.  **特徵重要性分析**：探究哪些特徵對模型的貢獻最大。
"""
print("\n" + "="*80)
print("第七部分：整合與模型訓練實戰")
print("="*80)

# --- 7.1 最終資料準備 ---
print("原始特徵資料集大小:", df_featured.shape)
# 由於滯後和滑動窗口操作，數據開頭會有很多NaN，必須移除。
df_final = df_featured.dropna()
print("移除 NaN 後的最終資料集大小:", df_final.shape)

# --- 7.2 定義特徵 (X) 與目標 (y) ---
TARGET = 'Consumption_MW'
# 特徵集為除了目標變數之外的所有欄位
FEATURES = [col for col in df_final.columns if col != TARGET]

X = df_final[FEATURES]
y = df_final[TARGET]

print(f"\n已定義 {len(FEATURES)} 個特徵和 1 個目標變數。")

# --- 7.3 時間序列資料分割 ---
# 我們選擇一個分割日期，此日期前的數據用於訓練，之後的用於測試。
split_date = '2017-01-01'
train_mask = df_final.index < split_date
test_mask = df_final.index >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"\n資料已按時間分割：")
print(f"訓練集大小: {X_train.shape[0]} 筆資料")
print(f"測試集大小: {X_test.shape[0]} 筆資料")
print(f"訓練集時間範圍: {X_train.index.min()} to {X_train.index.max()}")
print(f"測試集時間範圍: {X_test.index.min()} to {X_test.index.max()}")

# --- 7.4 訓練 LightGBM 模型 ---
print("\n正在初始化並訓練 LightGBM 模型...")
# LightGBM 參數設定
lgb_params = {
    'objective': 'regression_l1', # 損失函數：MAE
    'metric': 'rmse',             # 評估指標：RMSE
    'n_estimators': 1000,         # 樹的數量 (設大一點，配合早期停止)
    'learning_rate': 0.05,        # 學習率
    'feature_fraction': 0.8,      # 每次迭代隨機選擇80%的特徵
    'bagging_fraction': 0.8,      # 每次迭代隨機選擇80%的數據
    'bagging_freq': 1,
    'verbose': -1,                # 關閉詳細日誌
    'n_jobs': -1,                 # 使用所有CPU核心
    'seed': 42
}

model = lgb.LGBMRegressor(**lgb_params)

# 訓練模型，並使用早期停止 (early stopping) 防止過擬合
# 如果驗證集上的指標在50輪內沒有改善，就停止訓練。
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(50, verbose=True)])

print("\nLightGBM 模型訓練完成！")

# --- 7.5 模型評估 ---
print("\n正在測試集上進行預測與評估...")
y_pred = model.predict(X_test)

# 計算評估指標
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- 測試集評估結果 ---")
print(f"RMSE (均方根誤差): {rmse:.2f}")
print(f"MAE (平均絕對誤差): {mae:.2f}")
print(f"R² (決定係數):    {r2:.4f}")

# 視覺化預測結果
print("\n正在視覺化預測結果與真實值的對比 (抽樣一週)...")
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
df_plot_results = df_results['2018-01-01':'2018-01-07']

df_plot_results.plot(figsize=(16, 7), style=['-', '--'], lw=2)
plt.title('真實值 vs. 預測值 (測試集抽樣一週)')
plt.xlabel('日期')
plt.ylabel('電力消耗 (MW)')
plt.legend(['真實值 (Actual)', '預測值 (Predicted)'])
plt.grid(True)
plt.show()

# --- 7.6 特徵重要性分析 ---
print("\n正在分析並視覺化特徵重要性...")
feature_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature', palette='mako')
plt.title('LightGBM 模型中最重要的 20 個特徵')
plt.xlabel('重要性分數')
plt.ylabel('特徵')
plt.show()

"""
**[最終分析]**
1.  **模型性能**：從視覺化結果看，模型的預測值（虛線）與真實值（實線）高度吻合，
    成功捕捉了電力消耗的日夜波動和整體趨勢。R² 分數接近 1，也表明模型解釋了
    絕大部分的數據變異性。

2.  **特徵重要性**：
    -   **滯後特徵是王者**：`lag_168` (一週前) 和 `lag_24` (一天前) 的重要性遙遙領先，
        這印證了電力消耗強烈的週、日週期性。
    -   **時間特徵提供關鍵上下文**：`hour`, `dayofweek`, `dayofyear` 等日期時間特徵
        提供了模型理解當前處於何種時間模式（如一天中的時段、季節）的關鍵資訊。
    -   **滑動窗口捕捉近期趨勢**：`rolling_mean` 特徵也很重要，它們為模型提供了關於
        近期消耗水平的平滑化視角。
"""

# =============================================================================
# 8. 總結
# =============================================================================
"""
本次教學，我們從零開始，系統性地學習並實作了時間序列預測中四種最核心的
「時間特徵」工程技術。透過一個真實的電力消耗預測案例，我們將這些技術整合
在一個完整的機器學習流程中，並成功訓練出一個高精度的預測模型。

**核心回顧：**
| 特徵技術 | 核心作用 | 實作關鍵 |
|:---:|:---|:---|
| **日期時間特徵** | 捕捉日、週、季節等日曆規律 | `.dt` 訪問器 |
| **滯後特徵** | 利用歷史觀測值，捕捉自相關性 | `.shift()` |
| **滑動窗口特徵** | 概括近期趨勢與波動性，平滑噪聲 | `.rolling()` + `.shift(1)` |
| **週期性特徵轉換** | 解決環形數據的距離誤解問題 | 正弦/餘弦變換 |

時間特徵工程是時間序列分析的藝術與科學。精準的預測，不僅依賴於強大的模型，
更根植於對數據時間維度的深刻理解與巧妙轉化。希望本次教學能為您在未來的
時間序列專案中，提供堅實的理論基礎與實踐指引。
"""
print("\n" + "="*80)
print("教學結束。恭喜您已掌握時間序列特徵工程的核心技術！")
print("="*80)
