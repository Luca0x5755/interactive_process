# -*- coding: utf-8 -*-
"""
==============================================================================
## Python 機器學習入門：從問題定義到資料探索 (教學腳本)
==============================================================================

### 教學目標：
1.  學習如何明確定義一個機器學習問題。
2.  理解不同機器學習問題的類型（監督式 vs. 非監督式）。
3.  掌握使用 Pandas 載入資料並進行系統化的初步檢視。
4.  學會透過視覺化進行單變數、雙變數及多變數分析。
5.  練習從資料分析中生成初步假設並總結發現。

### 核心概念：
Garbage In, Garbage Out (GIGO) - 這是資料科學的黃金定律。
一個模型的表現好壞，其上限取決於資料的品質。因此，在專案的初始階段，
我們必須投入足夠的時間來「認識」我們的資料，這個過程稱為「探索性資料分析 (Exploratory Data Analysis, EDA)」。
"""

# %%
# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 資料處理核心
import pandas as pd
# 數值計算
import numpy as np
# 專業視覺化套件
import seaborn as sns
import matplotlib.pyplot as plt

# 設定 Matplotlib 和 Seaborn 的視覺風格，讓圖表更美觀
sns.set_style("whitegrid")
# 設置中文字體，以防圖表標題或標籤出現亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
print("所有必要套件已成功導入！")


# %%
# =============================================================================
# 1. 問題定義 (Problem Definition)
# =============================================================================
"""
所有機器學習專案的成功，都源於一個清晰的起點。
我們必須先將一個模糊的商業問題，轉化為一個可以被數據和模型解決的具體問題。

**本教學範例的核心問題是：**
「在鐵達尼號沉船事件中，哪些特徵是影響乘客生還與否的關鍵因素？」

這個問題引導我們去探討「生還狀況 (Survived)」與其他乘客特徵，
例如：「艙等 (Pclass)」、「性別 (Sex)」、「年齡 (Age)」、「票價 (Fare)」等之間的關係。
這是一個典型的、適合入門的二元分類問題的探索階段。
"""
print("--- 1. 問題定義 ---")
print("核心問題：探討鐵達尼號乘客『生還狀況』與哪些特徵（如 Pclass, Sex, Age, Fare等）有關。")


# %%
# =============================================================================
# 2. 問題類型定義 (Problem Type Definition)
# =============================================================================
"""
在定義了問題後，我們需要將其歸類到特定的機器學習任務類型。
這有助於我們選擇正確的演算法和評估指標。

--- A. 監督式學習 (Supervised Learning) ---
特點：資料集包含「特徵 (Features)」和「標準答案 (Labels/Target)」。模型從中學習一個映射函數。

* **分類 (Classification)**：預測目標是一個離散的類別。
    * **二元分類 (Binary Classification)**：預測只有兩種可能的結果。
        > 範例：我們的鐵達尼號問題，預測乘客是「生還(1)」還是「死亡(0)」。
    * **多元分類 (Multi-class Classification)**：預測有多個互斥的可能結果。
        > 範例：根據手寫數字圖片，辨識出是 0 到 9 的哪一個數字。
    * **多標籤分類 (Multi-label Classification)**：為每個樣本預測一組不互斥的標籤。
        > 範例：根據電影簡介，為其標上「動作」、「喜劇」、「愛情」等多種類型標籤。

* **迴歸 (Regression)**：預測目標是一個連續的數值。
    > 範例：根據房屋的坪數、地點、屋齡等特徵，預測其「銷售價格」。

* **排序 (Ranking)**：對一組項目進行排序。
    > 範例：搜尋引擎根據查詢關鍵字，對網頁結果進行相關性排序。


--- B. 非監督式學習 (Unsupervised Learning) ---
特點：資料集只有「特徵」，沒有「標準答案」。模型需要自己從資料中找出結構和模式。

* **分群 (Clustering)**：將相似的資料點分到同一個群組中。
    > 範例：根據顧客的消費行為，將他們分為「高價值客群」、「潛力客群」、「流失客群」等。

* **降維 (Dimensionality Reduction)**：在保留大部分資訊的前提下，減少特徵的數量。
    > 範例：將一個有數百個特徵的資料集壓縮到只有兩個特徵，以便進行二維視覺化。

* **關聯規則學習 (Association Rule Learning)**：找出資料項之間的有趣關聯。
    > 範例：在超市交易紀錄中發現「購買尿布的顧客，也很常一起購買啤酒」的規則。

* **異常偵測 (Anomaly Detection)**：識別出與大多數資料顯著不同的資料點。
    > 範例：在信用卡交易紀錄中，偵測出可能是盜刷的異常交易模式。
"""
print("\n--- 2. 問題類型定義 ---")
print("我們的問題屬於『監督式學習』中的『二元分類』問題。")


# %%
# =============================================================================
# 3. 載入與初步檢視資料 (Load and Initial Data Review)
# =============================================================================
"""
這個階段的目標是進行基本的資料品質掃描，了解資料的概況。
"""
# 為了教學方便，我們直接使用 seaborn 內建的 "titanic" 資料集
df = sns.load_dataset('titanic')
print("\n--- 3. 成功載入 Titanic 資料集 ---")
print(f"資料集維度 (行, 列): {df.shape}")

# 步驟 3.1: 檢視資料概況 (.info())
# .info() 提供每個欄位的非空值數量和資料類型，是發現缺失值的最快方法。
print("\n--- 3.1: 資料集技術摘要 (.info()) ---")
df.info() #
# 初步發現: 'age', 'deck', 'embarked', 'embark_town' 存在缺失值。 'deck' 尤其嚴重。

# 步驟 3.2: 描述性統計 (.describe())
# 對數值型欄位，.describe() 提供分佈、集中趨勢和離散程度的概覽。
print("\n--- 3.2: 數值型欄位統計摘要 ---")
print(df.describe()) #
# 初步發現: 'fare' 的最大值 (512) 遠大於 75% 的值 (31)，暗示可能存在極端值。

# 對類別型欄位，可以觀察類別的數量(unique)、最常見的類別(top)及其頻率(freq)。
print("\n--- 3.3: 類別型欄位統計摘要 ---")
print(df.describe(include=['object', 'category'])) #

# 步驟 3.3: 處理缺失值與檢查資料類型
# 雖然完整的處理在特徵工程階段，但初步量化是必要的。
print("\n--- 3.4: 各欄位缺失值統計 ---")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({'缺失數量': missing_values, '缺失比例 (%)': missing_percentage})
print(missing_info[missing_info['缺失數量'] > 0].sort_values(by='缺失比例 (%)', ascending=False)) #
# 資料類型在 .info() 中已檢查，符合預期。


# %%
# =============================================================================
# 4. 單變數分析 (Univariate Analysis)
# =============================================================================
"""
我們逐一檢視每個獨立特徵的分佈、趨勢與異常值。
"""
print("\n--- 4. 開始單變數分析 ---")

# 4.1 目標變數: Survived
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df) #
plt.title('生還人數分佈 (0 = 死亡, 1 = 生還)')
plt.xticks([0, 1], ['死亡', '生還'])
plt.ylabel('人數')
plt.show()
print("生還率約為 38.4%。") #

# 4.2 類別型特徵: Pclass, Sex, Embarked
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x='pclass', data=df, ax=axes[0]).set_title('乘客艙等分佈')
sns.countplot(x='sex', data=df, ax=axes[1]).set_title('乘客性別分佈')
sns.countplot(x='embarked', data=df, ax=axes[2]).set_title('登船港口分佈')
plt.tight_layout()
plt.show()
# 發現: 乘客以三等艙居多，男性遠多於女性，多數人從 S 港口登船。

# 4.3 數值特徵: Age, Fare (使用直方圖和箱形圖)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['age'].dropna(), kde=True, ax=axes[0]).set_title('年齡分佈') #
sns.histplot(df[df['fare'] < 200]['fare'], kde=True, ax=axes[1]).set_title('票價分佈 (Fare < 200)') # 過濾極端值以利觀察
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(y='age', data=df, ax=axes[0]).set_title('年齡箱形圖') #
sns.boxplot(y='fare', data=df, ax=axes[1]).set_title('票價箱形圖') #
plt.show()
# 發現: 年齡主要集中在青壯年。票價是高度右偏的，且箱形圖顯示有大量離群值。


# %%
# =============================================================================
# 5. 雙變數與多變數分析 (Bivariate & Multivariate Analysis)
# =============================================================================
"""
探索特徵之間的關係，特別是目標變數 Survived 與其他特徵的關聯。
"""
print("\n--- 5. 開始雙變數與多變數分析 ---")

# 5.1 性別 vs. 生還狀況
sex_survival = df.groupby('sex')['survived'].mean()
print(f"女性生還率: {sex_survival['female']:.2%}")
print(f"男性生還率: {sex_survival['male']:.2%}")
plt.figure(figsize=(6, 4))
sns.barplot(x=sex_survival.index, y=sex_survival.values) #
plt.title('不同性別生還率')
plt.ylabel('生還率')
plt.show()
# 發現: 女性的生還率 (約 74%) 遠高於男性 (約 19%)。

# 5.2 艙等 vs. 生還狀況
pclass_survival = df.groupby('pclass')['survived'].mean()
print(f"一等艙生還率: {pclass_survival[1]:.2%}")
print(f"二等艙生還率: {pclass_survival[2]:.2%}")
print(f"三等艙生還率: {pclass_survival[3]:.2%}")
plt.figure(figsize=(8, 5))
sns.barplot(x=pclass_survival.index, y=pclass_survival.values) #
plt.title('各艙等生還率')
plt.ylabel('生還率')
plt.xlabel('乘客艙等 (Pclass)')
plt.show()
# 發現: 艙等越高，生還率越高。

# 5.3 年齡 vs. 生還狀況
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='survived', kde=True, multiple="stack") #
plt.title('年齡與生還情況的關係')
plt.legend(title='是否生還', labels=['生還', '死亡'])
plt.show()
# 發現: 兒童 (特別是 10 歲以下) 的生還率較高。

# 5.4 多變數分析: 艙等、性別與生還狀況的綜合影響
plt.figure(figsize=(10, 6))
sns.pointplot(x='pclass', y='survived', hue='sex', data=df) #
plt.title('不同艙等和性別的生還率')
plt.ylabel('生還率')
plt.xlabel('乘客艙等 (Pclass)')
plt.show()
# 強化發現: 在所有艙等中，女性的生還率都遠高於男性。

# 5.5 數值特徵相關性矩陣
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f") #
plt.title('數值特徵相關性熱力圖')
plt.show()
# 發現: Survived 與 Pclass 呈負相關(-0.34)，與 Fare 呈正相關(0.26)，驗證了先前的觀察。


# %%
# =============================================================================
# 6. 生成假設與總結 (Generate Hypotheses & Summarize)
# =============================================================================
"""
根據前述資料分析結果，我們提出數個初步假設，並對整體發現進行簡潔總結。
"""
print("\n--- 6. 初步總結與假設 ---")
summary = """
經過系統化的 EDA，我們獲得了以下關鍵洞見：

初步假設：
1.  **性別是影響生還率的最關鍵因素**：女性的生還優先級遠高於男性，這可能與「婦女和兒童優先」的原則有關。
2.  **社會階級 (由 Pclass 反映) 是次要的關鍵因素**：艙等越高，生還率越高，這可能關乎船艙位置與救援順序。
3.  **年齡是另一個重要因素**：兒童的生還率相對較高，而青壯年乘客的死亡比例較高。
4.  **票價 (Fare) 可能是一個代理變數**：它與艙等高度相關，同樣反映了社會經濟地位。

整體總結：
- **資料品質**: `age` 和 `embarked` 存在少量缺失值，`deck` 缺失嚴重，需要後續處理。`fare` 欄位有明顯的離群值。
- **預測方向**: `Sex`, `Pclass`, `Age` 和 `Fare` 是預測乘客能否生還的強力候選特徵。
- **後續步驟**: 接下來的特徵工程階段，應著重於處理缺失值、對數值特徵進行標準化或轉換，並將類別特徵進行編碼，為建立預測模型做準備。

這次 EDA 為我們接下來的資料清理和模型建立指明了清晰的方向。
"""
print(summary)
print("\n--- 資料檢視與探索階段完成 ---")
