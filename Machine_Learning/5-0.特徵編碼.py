# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習入門：特徵編碼 (Feature Encoding) 教學腳本
==============================================================================

### 教學目標：
1.  理解什麼是特徵編碼，以及為何它在機器學習中至關重要。
2.  學習並實作五種核心的特徵編碼技術：標籤編碼、獨熱編碼、目標編碼、頻率編碼與特徵雜湊。
3.  了解不同編碼技術的適用情境、優點與潛在風險。
4.  掌握使用 pandas 與 scikit-learn 套件進行資料預處理的實務技巧。

### 適用對象：
- 對機器學習有興趣的 Python 初學者。
- 希望了解資料預處理中，如何處理類別型資料的學習者。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件
# pandas 是我們處理與操作資料表(DataFrame)的利器
import pandas as pd
# numpy 提供強大的數值計算功能
import numpy as np
# scikit-learn 是 Python 機器學習的標準函式庫，我們將從中引用編碼器
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.feature_extraction import FeatureHasher


print("套件導入成功！準備開始學習特徵編碼。")


# =============================================================================
# 1. 主題介紹：什麼是特徵編碼 (Feature Encoding)？
# =============================================================================
"""
在真實世界的資料中，我們常常會遇到非數值型的數據，例如「性別」(男/女)、
「城市」(台北/東京/紐約) 或「教育程度」(高中/大學/碩士)。這些資料被稱為
「類別型特徵」(Categorical Features)。

然而，絕大多數的機器學習演算法，其底層都是基於數學運算，它們只能理解
數字，無法直接處理文字。

**特徵編碼的核心目的**，就是將這些類別型特徵轉換為機器學習模型能夠理解
和處理的數值格式。這個過程是資料預處理中非常關鍵的一步，處理得當能
大幅提升模型效能；反之，若處理不當，可能會引入錯誤的資訊，誤導模型
的學習。

簡單來說，特徵編碼就是扮演「翻譯官」的角色，將現實世界的類別語言，
翻譯成機器學習的數字語言。
"""

print("\n" + "="*60)
print("第一部分：主題介紹 - 特徵編碼的重要性")
print("="*60)


# =============================================================================
# 2. 核心概念與程式碼範例
# =============================================================================
"""
接下來，我們將深入探討五種常見的特徵編碼技術。為了方便說明，我們
先建立一個簡單的模擬資料集。
"""

# --- 建立模擬資料集 ---
df_sample = pd.DataFrame({
    'City': ['Taipei', 'Tokyo', 'New York', 'Tokyo', 'Taipei', 'Paris', 'Tokyo'],
    'Education': ['Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'Master', 'PhD'],
    'Purchased': [1, 0, 1, 1, 0, 1, 1]  # 假設的目標變數 (1: 購買, 0: 未購買)
})

print("\n我們的模擬資料集：")
print(df_sample)
print("\n接下來，我們將對 'City' 和 'Education' 這兩個欄位進行編碼。")


# --- 2.1 標籤編碼 (Label Encoding) ---
print("\n--- 2.1 標籤編碼 (Label Encoding) ---")
"""
**原理**：
標籤編碼會為每一個不重複的類別指定一個從 0 開始的整數。
例如：['Taipei', 'Tokyo', 'New York'] -> [1, 2, 0] (順序依字母而定)

**適用時機**：
- **順序型特徵 (Ordinal Features)**：當類別本身具有明確的順序關係時，
  例如「學歷」('Bachelor' < 'Master' < 'PhD')，標籤編碼可以保留這層順序意義。
- **樹狀模型**：決策樹、隨機森林等模型對數值的絕對大小不敏感，比較不容易
  被標籤編碼引入的數值順序誤導。

**風險**：
- **誤導線性模型**：若將標籤編碼應用於無順序關係的「名目型特徵」(Nominal Features)
  如 'City'，再餵給線性模型，模型可能會錯誤地解讀出 'Tokyo'(2) > 'Taipei'(1)
  這樣的順序關係，進而影響判斷。
"""

# 建立一個資料副本，避免影響原始資料
df_label = df_sample.copy()

# **範例：對「無順序」的 'City' 進行標籤編碼 (錯誤示範)**
# scikit-learn 的 LabelEncoder 會依字母順序編碼
le = LabelEncoder()
df_label['City_LabelEncoded'] = le.fit_transform(df_label['City'])
print("\n使用 LabelEncoder 對 'City' 編碼 (注意：這會引入錯誤的順序關係):")
print(df_label[['City', 'City_LabelEncoded']])
print(f"City 的對應規則: {list(le.classes_)} -> {le.transform(le.classes_)}")

# **範例：對「有順序」的 'Education' 進行標籤編碼 (正確示範)**
# 處理順序型特徵時，最好手動定義映射關係，確保順序的正確性。
education_map = {'Bachelor': 0, 'Master': 1, 'PhD': 2}
df_label['Education_LabelEncoded'] = df_label['Education'].map(education_map)
print("\n手動定義映射，對 'Education' 進行正確的順序編碼:")
print(df_label[['Education', 'Education_LabelEncoded']])


# --- 2.2 獨熱編碼 (One-Hot Encoding) ---
print("\n--- 2.2 獨熱編碼 (One-Hot Encoding) ---")
"""
**原理**：
獨熱編碼會為每一個不重複的類別創建一個新的欄位 (特徵)，並用 0 或 1 來表示
該樣本是否屬於這個類別。如果屬於，該欄位為 1，其他新欄位為 0。

**適用時機**：
- **名目型特徵 (Nominal Features)**：這是處理無順序關係類別 (如 'City')
  最安全、最標準的方法。它能確保模型不會被錯誤的順序資訊干擾。
- **線性模型或基於距離的模型**：對於邏輯斯迴歸、SVM、KNN 等對數值大小
  敏感的模型，獨熱編碼是必要的。

**風險**：
- **維度災難**：如果一個特徵的類別數量非常多 (例如: 郵遞區號)，獨熱編碼
  會產生大量的新欄位，導致資料維度暴增，增加計算成本且可能讓模型過擬合。
"""

# 建立一個資料副本
df_onehot = df_sample.copy()

# **範例：對 'City' 進行獨熱編碼**
# 使用 pandas 的 `get_dummies` 函數是最方便快速的方法
city_onehot = pd.get_dummies(df_onehot['City'], prefix='City')

# 將編碼後的新欄位與原始資料表合併
df_onehot = pd.concat([df_onehot, city_onehot], axis=1)

print("\n使用 pd.get_dummies 對 'City' 進行獨熱編碼：")
# 為了清晰顯示，我們只看相關欄位
print(df_onehot[['City', 'City_New York', 'City_Paris', 'City_Taipei', 'City_Tokyo']])


# --- 2.3 目標編碼 (Target Encoding) ---
print("\n--- 2.3 目標編碼 (Target Encoding) ---")
"""
**原理**：
目標編碼是一種監督式(Supervised)的編碼方法。它利用「目標變數」(我們想預測的對象，
如此處的 'Purchased') 的資訊來進行編碼。具體來說，它會計算該類別對應的目標變數
的平均值，並用此平均值作為該類別的編碼結果。

**適用時機**：
- **高基數特徵**：當類別數量很多，不適合獨熱編碼時，目標編碼是一個強大的選項。
- **追求極致效能**：在機器學習競賽中，好的目標編碼策略常是提升模型預測能力的關鍵。

**風險**：
- **資料洩漏 (Data Leakage) 與過擬合 (Overfitting)**：這是目標編碼最大的風險！
  如果在計算某樣本的編碼值時，使用了該樣本自身的目標值，就會造成「資料洩漏」。
  模型會學到一個過於完美的關係，導致在訓練集上表現極佳，但在未知的測試集上
  表現很差。因此，**絕對不能用天真的方式直接計算全局平均值來編碼**。
  必須採用如「交叉驗證」的策略來穩健地實作。
"""

# **範例：使用 K-Fold 交叉驗證進行穩健的目標編碼**
# 我們將展示如何避免資料洩漏
df_target = df_sample.copy()

# 初始化一個用來存放編碼結果的欄位
df_target['City_TargetEncoded'] = np.nan

# 設定 K-Fold 交叉驗證，將資料切成數份 (fold)
# 我們用一份做驗證，其他份做訓練，輪流進行
kf = KFold(n_splits=3, shuffle=False)

# 遍歷每一個 fold
for train_index, val_index in kf.split(df_target):
    # 1. 在「訓練 fold」上計算每個 City 的目標平均值
    #    這確保了計算平均值時，沒有用到「驗證 fold」的資訊
    train_fold = df_target.iloc[train_index]
    mean_map = train_fold.groupby('City')['Purchased'].mean()

    # 2. 將計算出的平均值，應用到「驗證 fold」上
    val_fold_cities = df_target.iloc[val_index]['City']
    df_target.loc[val_index, 'City_TargetEncoded'] = val_fold_cities.map(mean_map)

# 處理可能因資料切分而產生的空值 (NaN)，用全域平均值填充
global_mean = df_target['Purchased'].mean()
df_target['City_TargetEncoded'].fillna(global_mean, inplace=True)


print("\n使用 K-Fold 進行穩健的目標編碼結果：")
print(df_target[['City', 'Purchased', 'City_TargetEncoded']])
print("\n觀察：編碼值是基於其他資料點的目標平均值計算而來，避免了直接的資料洩漏。")


# --- 2.4 頻率/計數編碼 (Frequency/Count Encoding) ---
print("\n--- 2.4 頻率/計數編碼 (Frequency/Count Encoding) ---")
"""
**原理**：
這種方法利用類別的普遍性或稀有性來進行編碼。它將每個類別替換為其在資料集中
出現的總次數（計數編碼）或總頻率（頻率編碼）。

**適用時機**：
- 當類別的出現頻率本身被認為是一個有用的特徵時。例如，在反欺詐檢測中，
  出現次數極少或極多的用戶行為可能都值得關注。
- 作為一種簡單、快速的編碼方法，特別適用於樹模型。

**風險**：
- **資訊損失**：如果兩個不同的類別出現了相同的次數或頻率，它們會被賦予
  相同的編碼值，模型將無法區分它們。
"""

# 建立一個資料副本
df_freq = df_sample.copy()

# **範例：計數編碼 (Count Encoding)**
# 計算每個 City 出現的次數
count_map = df_freq['City'].value_counts().to_dict()
df_freq['City_CountEncoded'] = df_freq['City'].map(count_map)
print("\n使用計數編碼 (Count Encoding) 對 'City' 編碼：")
print(df_freq[['City', 'City_CountEncoded']])


# **範例：頻率編碼 (Frequency Encoding)**
# 計算每個 City 出現的頻率 (比例)
freq_map = df_freq['City'].value_counts(normalize=True).to_dict()
df_freq['City_FreqEncoded'] = df_freq['City'].map(freq_map)
print("\n使用頻率編碼 (Frequency Encoding) 對 'City' 編碼：")
print(df_freq[['City', 'City_FreqEncoded']])


# --- 2.5 特徵雜湊 (Feature Hashing) ---
print("\n--- 2.5 特徵雜湊 (Feature Hashing) ---")
"""
**原理**：
特徵雜湊，也稱為「雜湊技巧」(The Hashing Trick)，是一種快速且內存效率極高的
向量化技術。它不建立任何映射字典，而是使用一個「雜湊函數」(Hash Function)
將類別名稱（字串）直接轉換為一個固定長度向量中的索引位置。

**適用時機**：
- **超高基數特徵**：當類別數量達到成千上萬甚至更多時，特徵雜湊是少數幾個
  可行的方案之一，因為它不依賴於字典，內存佔用是固定的。
- **流式資料或線上學習**：當無法預先看到所有類別時，雜湊技巧可以即時處理
  新出現的類別。

**風險**：
- **雜湊碰撞 (Hash Collision)**：由於輸出的向量維度是固定的，不同的類別
  有可能被雜湊到同一個位置，這被稱為「碰撞」。碰撞會導致模型無法區分
  原始的兩個不同類別。
- **可解釋性差**：生成的特徵是抽象的，無法反推回原始的類別意義。
"""
# 建立一個資料副本
df_hash = df_sample.copy()

# **範例：對 'City' 進行特徵雜湊**
# n_features 決定了輸出向量的維度（即雜湊空間的大小）
# 我們刻意設置為 3 (小於總類別數 4)，以凸顯其維度固定的特性
hasher = FeatureHasher(n_features=3, input_type='string')

# FeatureHasher 需要一個 list of lists 作為輸入
hashed_features = hasher.fit_transform(df_hash['City'].apply(lambda x: [x]))

# 將稀疏矩陣結果轉換為 DataFrame 以便查看
hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'CityHash_{i}' for i in range(3)])

# 合併結果
df_hash = pd.concat([df_hash.reset_index(drop=True), hashed_df], axis=1)

print("\n使用特徵雜湊 (Feature Hashing) 對 'City' 編碼 (輸出維度=3)：")
print(df_hash[['City', 'CityHash_0', 'CityHash_1', 'CityHash_2']])


# =============================================================================
# 3. 總結與應用建議
# =============================================================================
print("\n" + "="*80)
print("第三部分：總結與應用建議")
print("="*80)
"""
選擇哪種編碼方法，並沒有絕對的標準答案，需要根據你的「特徵類型」、「基數高低」和
「模型選擇」來綜合判斷。

| 編碼方法       | 核心思想                                   | 優點                               | 缺點與風險                         | 適用建議                                       |
|----------------|--------------------------------------------|------------------------------------|------------------------------------|------------------------------------------------|
| **標籤編碼** | 將類別轉為 0, 1, 2... 等連續整數。         | 實現簡單、不增加資料維度。         | 易引入錯誤的順序關係，誤導線性模型。 | **有順序的特徵** (如學歷) 或用在**樹狀模型**上。 |
| **獨熱編碼** | 為每個類別建立新的 0/1 欄位。              | 完美解決順序問題，適用性廣。       | 類別多時會導致維度災難。           | **無順序的名目特徵** (如城市)，特別是配合**線性模型**時。 |
| **目標編碼** | 使用目標變數的平均值來編碼。               | 預測能力強，能處理高基數特徵。     | **極易過擬合與資料洩漏**，需謹慎實作。 | 追求極致模型效能時，且必須搭配**交叉驗證**等穩健策略。 |
| **頻率/計數編碼**| 用類別出現的頻率或次數來編碼。         | 簡單快速，能捕捉類別普遍性。       | 不同類別可能有相同頻率，造成資訊損失。 | 想快速建立基準模型，或認為**類別頻率本身是重要信號**時。 |
| **特徵雜湊** | 使用雜湊函數將類別映射到固定維度。       | **內存效率極高**，速度快，適合流式資料。 | **雜湊碰撞**會損失資訊，且**可解釋性差**。 | **超高基數特徵** (如用戶ID) 或記憶體受限的場景。 |

**進一步學習建議**：
精通特徵工程是成為優秀資料科學家的必經之路。建議你在真實的資料集上
多加練習，嘗試不同的編碼方法，並觀察它們對模型結果的影響。祝你學習愉快！
"""
print("\n教學腳本執行完畢。")
