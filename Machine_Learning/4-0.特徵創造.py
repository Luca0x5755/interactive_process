# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習進階：特徵創造 (Feature Creation) 全方位教學腳本
==============================================================================

### 教學目標：
1.  理解特徵創造的定義、核心價值，以及為何它是提升模型效能的關鍵。
2.  學習並實作三大核心的特徵創造技術：交互特徵、分組聚合特徵與時間衍生特徵。
3.  掌握如何使用 pandas 與 scikit-learn 等套件，在真實機器學習工作流程中，靈活應用這些技術。
4.  透過真實世界的案例（紐約市計程車行程時間預測），整合所有特徵創造技巧，從原始資料中挖掘深度預測力。
5.  了解不同特徵創造技術的適用情境、優點與實作細節。

### 適用對象：
-   已具備 Python 基礎與機器學習初步概念的學習者。
-   希望深入了解資料預處理，並提升模型預測準確率的資料分析師或工程師。
-   對特徵工程 (Feature Engineering) 技術有濃厚興趣的開發者。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件
# pandas 是我們處理與操作資料表 (DataFrame) 的利器
import pandas as pd
# numpy 提供強大的數值計算功能
import numpy as np
# scikit-learn 是 Python 機器學習的標準函式庫，我們將從中引用特徵處理工具
from sklearn.preprocessing import PolynomialFeatures
# 視覺化套件，用於資料探索與結果展示
import matplotlib.pyplot as plt
import seaborn as sns

print("套件導入成功！準備開始學習特徵創造。")


# =============================================================================
# 1. 主題介紹：什麼是特徵創造 (Feature Creation)？
# =============================================================================
"""
在機器學習的領域中，數據的品質和其所包含的資訊量，往往直接決定了模型效能的上限。
原始數據 (Raw Data) 雖然真實，但其格式與內容未必最適合機器學習演算法直接使用。

**特徵創造的核心目的**，就是從現有的資料中，透過組合、轉換、或分解，建構出更具
預測能力的特徵，以提升模型的性能和解釋性。這個過程不僅僅是技術操作，更是一門
結合了領域知識 (Domain Knowledge)、數據直覺與創意思考的藝術。

簡單來說，如果特徵編碼是將非數值資料「翻譯」成數字，那麼特徵創造就是扮演「偵探」
的角色，從現有的線索（原始特徵）中，挖掘出隱藏的、更深層次的關聯性與模式，為
模型提供更強大的破案（預測）武器。

### 為什麼特徵創造至關重要？
1.  **捕捉非線性關係**: 許多真實世界的關係並非簡單的線性關係，特徵創造（如交互特徵）可以幫助模型捕捉這些複雜模式。
2.  **提升模型性能**: 提供更具資訊量的特徵，能直接且顯著地提升模型的準確性。
3.  **增強模型解釋性**: 精心設計的特徵（如「人均收入」）比分散的原始特徵（「總收入」、「人口數」）更容易被理解和解釋。
4.  **處理高基數類別特徵**: 對於唯一值極多的類別特徵，將其轉換為聚合特徵是有效的降維手段。
5.  **利用時間上下文**: 將時間戳分解為週期性特徵（如「星期幾」、「小時」），有助於模型理解時間趨勢。
"""

print("\n" + "="*60)
print("第一部分：主題介紹 - 特徵創造的重要性")
print("="*60)


# =============================================================================
# 2. 技術一：交互特徵 (Interaction Features)
# =============================================================================
"""
**定義**：透過組合兩個或多個原始特徵而創建的新特徵，能夠捕捉到原始特徵之間
無法單獨表示的複雜、非線性關係。

**常見方法**：
-   **乘法**: `特徵A * 特徵B` (例如：長 * 寬 = 面積)
-   **除法**: `特徵A / 特徵B` (例如：總收入 / 人口數 = 人均收入)
-   **多項式特徵**: 自動生成特徵的各種多項式組合，如 `A^2`, `B^2`, `A*B`。
"""
print("\n" + "="*60)
print("第二部分：技術一 - 交互特徵")
print("="*60)

# --- 2.1 準備範例資料 ---
interaction_data = {'feature1': [1, 2, 3, 4], 'feature2': [10, 20, 30, 40]}
df_interaction_raw = pd.DataFrame(interaction_data)
print("\n用於交互特徵的原始數據：")
print(df_interaction_raw)

# --- 2.2 使用 scikit-learn 自動生成多項式與交互特徵 ---
print("\n--- 2.2 scikit-learn 自動生成 (degree=2) ---")
# degree=2 表示生成最高二次的特徵 (a, b, a^2, b^2, a*b)
# include_bias=False 表示不需要常數項
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_interaction_raw)
poly_feature_names = poly.get_feature_names_out(df_interaction_raw.columns)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
print("自動生成的多項式與交互特徵：")
print(df_poly)

# --- 2.3 僅生成交互項 ---
print("\n--- 2.3 scikit-learn 僅生成交互項 (interaction_only=True) ---")
poly_interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
interaction_only_features = poly_interaction.fit_transform(df_interaction_raw)
interaction_feature_names = poly_interaction.get_feature_names_out(df_interaction_raw.columns)
df_interaction_only = pd.DataFrame(interaction_only_features, columns=interaction_feature_names)
print("僅生成的交互特徵：")
print(df_interaction_only)

# --- 2.4 手動創建交互特徵 ---
print("\n--- 2.4 基於領域知識手動創建 ---")
df_manual = df_interaction_raw.copy()
# 乘法交互
df_manual['F1_times_F2'] = df_manual['feature1'] * df_manual['feature2']
# 除法交互
df_manual['F1_div_F2'] = df_manual['feature1'] / df_manual['feature2']
print("手動創建的交互特徵：")
print(df_manual)


# =============================================================================
# 3. 技術二：分組聚合特徵 (Group Aggregation Features)
# =============================================================================
"""
**定義**：對資料中某個類別變數進行分組（如按客戶ID），然後對每個組內的其他
數值特徵應用統計函數（如均值、總和、計數等）來創建新特徵。

**核心價值**：
-   捕捉群體行為模式，從「個體」行為中提煉出「群體」洞察。
-   為模型提供更宏觀、更豐富的上下文資訊。
-   是處理高基數類別特徵的有效降維手段。
"""
print("\n" + "="*60)
print("第三部分：技術二 - 分組聚合特徵")
print("="*60)

# --- 3.1 準備範例資料 ---
agg_data = {'customer_id': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
            'product_category': ['Elec', 'Books', 'Books', 'Elec', 'Home', 'Elec', 'Books', 'Home'],
            'purchase_amount': [120, 30, 25, 150, 80, 100, 40, 90],
            'rating': [4, 5, 3, 5, 4, 4, 5, 3]}
df_agg_raw = pd.DataFrame(agg_data)
print("\n用於分組聚合的原始數據：")
print(df_agg_raw)

# --- 3.2 按單一類別變數分組與聚合 ---
print("\n--- 3.2 按 'customer_id' 分組聚合 ---")
# 按 customer_id 分組，計算 purchase_amount 的平均、總和與次數
customer_agg = df_agg_raw.groupby('customer_id')['purchase_amount'].agg([
    'mean', 'sum', 'count'
]).reset_index()
# 重新命名，使其更具描述性
customer_agg.columns = ['customer_id', 'avg_purchase', 'total_purchase', 'purchase_count']
print("按客戶ID聚合的特徵：")
print(customer_agg)

# 將聚合特徵合併回原始 DataFrame
print("\n將客戶聚合特徵合併回原始資料：")
df_merged_customer = pd.merge(df_agg_raw, customer_agg, on='customer_id', how='left')
print(df_merged_customer.head())

# --- 3.3 對多個特徵進行聚合 ---
print("\n--- 3.3 按 'product_category' 對多個特徵進行聚合 ---")
# 定義聚合的配置
agg_config = {
    'purchase_amount': ['mean', 'max', 'sum'],  # 對購買金額聚合
    'rating': ['mean', 'std', 'count']          # 對評分聚合
}
category_agg = df_agg_raw.groupby('product_category').agg(agg_config)
# 展平多級索引的列名
category_agg.columns = ['_'.join(col).strip() for col in category_agg.columns.values]
category_agg.reset_index(inplace=True)
print("按產品類別聚合的特徵：")
print(category_agg)

# 將產品類別聚合特徵也合併回去
print("\n將所有聚合特徵合併後的最終資料：")
df_merged_all = pd.merge(df_merged_customer, category_agg, on='product_category', how='left')
print(df_merged_all.head())


# =============================================================================
# 4. 技術三：時間衍生特徵 (Time-Derived Features)
# =============================================================================
"""
**定義**：將原始的時間戳（如 '2023-10-27 10:30:00'）分解、轉換為對模型
有意義的、結構化的數值或類別特徵。

**核心價值**：
-   **捕捉週期性模式**: 如每日、每週、每年的循環。
-   **識別長期趨勢**: 提取年份等特徵，幫助模型識別趨勢。
-   **利用時間上下文**: 如「一天中的時段」（早上、下午、晚上）。
"""
print("\n" + "="*60)
print("第四部分：技術三 - 時間衍生特徵")
print("="*60)

# --- 4.1 準備範例資料 ---
date_rng = pd.date_range(start='2023-01-01', end='2023-01-03 23:59:59', freq='4H')
df_time_raw = pd.DataFrame(date_rng, columns=['timestamp'])
df_time_raw['value'] = np.random.randint(low=10, high=100, size=len(df_time_raw))
print("\n用於時間衍生的原始數據：")
print(df_time_raw.head())

# --- 4.2 提取基本時間特徵 ---
print("\n--- 4.2 提取基本時間組件 ---")
df_time = df_time_raw.copy()
df_time['year'] = df_time['timestamp'].dt.year
df_time['month'] = df_time['timestamp'].dt.month
df_time['day'] = df_time['timestamp'].dt.day
df_time['hour'] = df_time['timestamp'].dt.hour
df_time['dayofweek'] = df_time['timestamp'].dt.dayofweek # 星期一=0, 星期日=6
df_time['weekofyear'] = df_time['timestamp'].dt.isocalendar().week.astype(int)
df_time['quarter'] = df_time['timestamp'].dt.quarter
print("提取基本時間特徵後的結果：")
print(df_time[['timestamp', 'year', 'month', 'day', 'hour', 'dayofweek']].head())

# --- 4.3 創建布林型和自定義類別特徵 ---
print("\n--- 4.3 創建自定義時間狀態特徵 ---")
# 是否為週末
df_time['is_weekend'] = (df_time['timestamp'].dt.dayofweek >= 5).astype(int)

# 一天中的時段
def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Night'
df_time['time_of_day'] = df_time['hour'].apply(get_time_of_day)
print("創建自定義時間特徵後的結果：")
print(df_time[['timestamp', 'is_weekend', 'time_of_day']].head())

# --- 4.4 計算時間差特徵 ---
print("\n--- 4.4 計算時間差特徵 ---")
# 計算距離數據集第一個時間點過去了多少秒
time_since_start = (df_time['timestamp'] - df_time['timestamp'].min())
df_time['seconds_since_start'] = time_since_start.dt.total_seconds()
print("創建時間差特徵後的結果：")
print(df_time[['timestamp', 'seconds_since_start']].head())


# =============================================================================
# 5. 綜合案例：紐約計程車行程時間預測
# =============================================================================
"""
本節將綜合應用上述所有特徵創造技巧，處理一個真實世界的資料集，目標是
預測紐約市計程車的行程時間。

**資料集**: Kaggle "NYC Taxi Trip Duration"
**注意**: 執行此部分需要 `train.csv` 檔案。此處為求腳本可獨立執行，
將會創建一個微型模擬資料集來演示流程。真實操作時請替換為實際資料。
"""
print("\n" + "="*60)
print("第五部分：綜合案例 - NYC 計程車行程預測")
print("="*60)

# --- 5.1 創建模擬資料 ---
# 為了讓此腳本可以獨立運行，我們創建一個小型模擬資料集
# 其結構與真實 NYC Taxi 資料集相似
print("\n--- 5.1 準備模擬的 NYC 計程車資料 ---")
data_nyc = {
    'pickup_datetime': ['2016-03-14 17:24:55', '2016-06-12 00:43:35', '2016-01-19 11:35:24'],
    'dropoff_datetime': ['2016-03-14 17:32:30', '2016-06-12 00:54:38', '2016-01-19 11:52:21'],
    'pickup_longitude': [-73.982155, -73.980415, -73.979027],
    'pickup_latitude': [40.767937, 40.738564, 40.763939],
    'dropoff_longitude': [-73.964630, -73.999481, -74.005333],
    'dropoff_latitude': [40.765602, 40.731152, 40.710087],
    'trip_duration': [455, 663, 1017] # seconds
}
df_nyc = pd.DataFrame(data_nyc)
print("模擬的原始資料：")
print(df_nyc)

# --- 5.2 應用時間衍生特徵 ---
print("\n--- 5.2 應用時間衍生特徵 ---")
df_nyc['pickup_datetime'] = pd.to_datetime(df_nyc['pickup_datetime'])
df_nyc['pickup_hour'] = df_nyc['pickup_datetime'].dt.hour
df_nyc['pickup_weekday'] = df_nyc['pickup_datetime'].dt.dayofweek
df_nyc['pickup_weekend'] = (df_nyc['pickup_datetime'].dt.dayofweek >= 5).astype(int)
print("增加時間特徵後：")
print(df_nyc[['pickup_datetime', 'pickup_hour', 'pickup_weekday', 'pickup_weekend']])

# --- 5.3 應用交互特徵 (地理距離) ---
print("\n--- 5.3 應用交互特徵 (Haversine 距離) ---")
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半徑 (km)
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

df_nyc['distance_km'] = haversine_distance(
    df_nyc['pickup_latitude'], df_nyc['pickup_longitude'],
    df_nyc['dropoff_latitude'], df_nyc['dropoff_longitude']
)
print("增加地理距離特徵後：")
print(df_nyc[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'distance_km']])

# --- 5.4 目標變數轉換 ---
print("\n--- 5.4 目標變數轉換 (Log Transformation) ---")
# 真實資料中，trip_duration 常呈右偏分佈，取對數可使其更接近常態分佈
df_nyc['log_trip_duration'] = np.log1p(df_nyc['trip_duration'])

# 視覺化比較 (使用模擬數據可能不明顯，但在真實大數據上效果顯著)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_nyc['trip_duration'], kde=True, ax=axes[0])
axes[0].set_title('Original Trip Duration')
sns.histplot(df_nyc['log_trip_duration'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Log-Transformed Trip Duration')
plt.suptitle('Target Variable Transformation')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("增加對數轉換後的目標變數：")
print(df_nyc[['trip_duration', 'log_trip_duration']])


# =============================================================================
# 6. 總結
# =============================================================================
"""
恭喜您完成了本次的特徵創造教學！

特徵創造是特徵工程中極具創造性與價值的一環。它將我們的角色從單純的
數據使用者，提升為數據的「塑造者」。透過本教學，我們學習了三種強大的
特徵創造技術：

| 技術名稱             | 核心思想                                 | 關鍵工具/方法                               |
|----------------------|------------------------------------------|---------------------------------------------|
| **交互特徵** | 捕捉特徵間的非線性組合關係。             | `sklearn.PolynomialFeatures`, 手動數學運算  |
| **分組聚合特徵** | 從群體行為中提煉統計洞察。               | `pandas.groupby().agg()`                    |
| **時間衍生特徵** | 將時間戳分解為有意義的週期性與趨勢特徵。 | `pandas.to_datetime`, `.dt` 訪問器        |

在實際專案中，這些技術往往是結合使用的。成功的特徵創造，需要不斷地進行
實驗、探索，並結合對業務問題的深刻理解。記住，好的特徵能夠讓簡單的模型
發揮出強大的威力，這正是特徵工程的魅力所在。

希望本教學能為您在機器學習的道路上，提供堅實的助力！
"""
print("\n" + "="*60)
print("教學結束。您已掌握特徵創造的核心技術！")
print("="*60)
