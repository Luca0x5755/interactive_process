# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習實戰：監督式學習排序 (Supervised Ranking) 教學腳本
==============================================================================

### 教學目標：
1.  理解「排序問題」(Ranking Problem) 在監督式學習中的定義、挑戰與應用場景。
2.  學習Pointwise、Pairwise、Listwise三種主流的監督式排序方法論。
3.  掌握如何使用 `lightgbm.LGBMRanker` 這一強大的Listwise模型，進行排序模型的建立、訓練與預測。
4.  學習計算與解讀排序問題的核心評估指標，特別是 NDCG (Normalized Discounted Cumulative Gain)。
5.  了解如何將一個完整的排序模型，從資料準備到最終預測，整合到標準的機器學習工作流程中。

### 適用對象：
- 已具備基礎機器學習概念（如分類、迴歸）並希望深入特定領域的學習者。
- 需要處理搜尋結果排序、推薦系統、問答系統等相關問題的資料科學家或工程師。
- **注意**：本教學假設「特徵工程」階段已完成，將完全專注於排序模型的選擇、建構與評估。

### 使用套件：
- **pandas & numpy**: 資料處理與數值計算的基礎。
- **scikit-learn**: 用於資料切分。
- **lightgbm**: 本次教學的核心，用於實現梯度提升排序模型 (LambdaMART)。
  (若您尚未安裝，請執行: `pip install lightgbm`)
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

print("套件導入成功！準備開始學習監督式排序模型。")


# =============================================================================
# 1. 主題介紹：什麼是排序問題 (Learning to Rank)？
# =============================================================================
"""
在機器學習領域，我們熟悉的「分類」問題是預測一個離散的類別（例如，這是貓還是狗？），
而「迴歸」問題是預測一個連續的數值（例如，明天的氣溫是幾度？）。

**排序問題 (Ranking)** 則是一個更特殊的挑戰。它的目標不是預測一個絕對值或
類別，而是 **預測一個項目列表 (list of items) 的相對順序**。

舉個最經典的例子：**搜尋引擎**。
當你在Google輸入一個查詢 (Query) 時，搜尋引擎需要從數十億個網頁中，找出與你
查詢最相關的網頁，並將它們以「最有用」到「最沒用」的順序呈現給你。這個決定
「順序」的過程，就是一個典型的排序問題。

其他應用場景包含：
- **電商推薦**：在商品頁面推薦「你可能也喜歡」的商品列表。
- **廣告投放**：決定在網頁的哪個位置，依序顯示哪些廣告。
- **問答系統**：針對一個問題，對所有可能的答案進行排序。

**挑戰**：排序的挑戰在於，模型評估的標準不是單一預測的準確性，而是整個
列表的「品質」。將一個高度相關的結果排在第二位，遠比將它排在第十位要好。
因此，模型需要理解項目之間的相對重要性，而不僅僅是它們的個別分數。
"""

print("\n" + "="*60)
print("第一部分：主題介紹 - 排序問題的重要性")
print("="*60)


# =============================================================================
# 2. 監督式排序模型的三大方法論
# =============================================================================
"""
監督式排序模型主要可以分為三大家族，它們的核心區別在於如何構建損失函數
(Loss Function) 來學習「順序」。

**A. Pointwise (點對點法)**
   - **原理**：這是最簡單的方法。它將排序問題完全轉化為迴歸或分類問題。
     模型一次只看一個「(查詢, 文件)」組合，並預測該文件的「絕對相關性分數」
     (例如，從0到5分)。在預測階段，再根據這個分數對所有文件進行排序。
   - **優點**：簡單直觀，可以直接套用現有的迴歸/分類模型。
   - **缺點**：完全忽略了同一查詢下，文件之間的相對關係。它不知道將一個5分
     文件排在4分文件前面，比將3分文件排在2分文件前面更重要。

**B. Pairwise (配對法)**
   - **原理**：此方法更進一步，它學習的是成對文件的相對順序。模型一次會看
     一對「(文件A, 文件B)」，並預測「文件A是否比文件B更相關」。損失函數的
     目標是最小化「預測錯誤的配對」數量。
   - **代表模型**：RankSVM, RankNet。
   - **優點**：比Pointwise更接近排序的本質，開始考慮相對順序。
   - **缺點**：它只考慮了兩個文件間的相對順序，仍然沒有從整個列表的全局
     視角去優化。此外，訓練所需的「文件配對」數量龐大，計算成本高。

**C. Listwise (列表法)**
   - **原理**：這是目前最先進且效果最好的方法。它直接將整個文件列表作為
     一個訓練樣本。模型學習如何對這個列表進行排列，其損失函數直接與我們
     最終關心的排序評估指標（如NDCG）掛鉤或近似。
   - **代表模型**：LambdaMART, AdaRank, ListNet。
   - **優點**：直接優化排序列表的整體品質，最符合排序問題的目標，通常能
     達到最佳的效能。
   - **缺點**：模型和理論相對複雜。

**結論**：在本教學中，我們將重點實作 **Listwise** 方法，因為它在學術界和
工業界都被證明是解決排序問題最有效的方法之一。我們將使用 LightGBM 套件中
的 `LGBMRanker`，它是 LambdaMART 演算法的一個高效能實現。
"""

print("\n" + "="*60)
print("第二部分：監督式排序模型的三大方法論")
print("="*60)


# =============================================================================
# 3. 準備教學資料：模擬搜尋引擎的點擊日誌
# =============================================================================
"""
為了讓教學腳本可以獨立執行，我們將手動生成一個模擬資料集。
這個資料集模擬了使用者在一個搜尋引擎上的行為。

資料欄位說明：
- `query_id`: 查詢的唯一標識符。排序是在同一個 `query_id` 內部進行的。
- `doc_id`: 文件的唯一標識符。
- `relevance`: **目標變數 (Ground Truth)**。代表文件與查詢的真實相關性等級。
  數字越大越相關 (例如，2: 非常相關, 1: 有點相關, 0: 不相關)。
- `feature_1`, `feature_2`: 文件的特徵。在真實世界中，這可能包含
  TF-IDF分數、PageRank、文件長度、點擊率等等數百個特徵。
"""

# 生成模擬資料
data = {
    'query_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    'doc_id':   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'relevance':[2, 1, 0, 1, 0, 2, 1, 1, 2, 0, 0, 1],
    'feature_1':[0.8, 0.6, 0.1, 0.5, 0.2, 0.9, 0.7, 0.6, 0.95, 0.1, 0.2, 0.5],
    'feature_2':[0.9, 0.5, 0.2, 0.4, 0.3, 0.8, 0.5, 0.7, 0.98, 0.3, 0.1, 0.4]
}
df = pd.DataFrame(data)

# 觀察資料結構
print("\n" + "="*60)
print("第三部分：準備教學資料")
print("="*60)
print("生成的模擬資料集：")
print(df)

# 定義特徵欄位和目標欄位
features = ['feature_1', 'feature_2']
target = 'relevance'
query_id_col = 'query_id'


# =============================================================================
# 4. 模型選擇與實作：使用 LightGBM 的 LGBMRanker
# =============================================================================
"""
現在進入核心部分：建立、訓練和評估我們的 Listwise 排序模型。

**關鍵步驟 1：準備 `group` 資訊**
Listwise 模型（如 LGBMRanker）與一般模型最關鍵的不同點在於，它需要知道
哪些樣本屬於同一個「列表」（即同一個查詢）。我們需要提供一個 `group` 陣列，
其中每個數字代表對應 `query_id` 的樣本數量。

例如，我們的資料中：
- query_id 1 有 4 個文件
- query_id 2 有 4 個文件
- query_id 3 有 4 個文件
所以我們的 `group` 陣列會是 `[4, 4, 4]`。

**關鍵步驟 2：切分資料**
我們需要將資料切分為訓練集和測試集。**非常重要**的一點是，切分時必須
確保同一個 `query_id` 的所有文件，要麼全在訓練集，要麼全在測試集。
絕對不能將它們分開，否則會破壞列表的完整性，導致資料洩漏。
"""

print("\n" + "="*60)
print("第四部分：模型選擇與實作 - LGBMRanker")
print("="*60)

# --- 步驟 4.1: 準備特徵 (X) 和目標 (y) ---
X = df[features]
y = df[target]

# --- 步驟 4.2: 準備 group 資訊 ---
# 首先，我們計算每個 query_id 有多少個文件
group_counts = df.groupby(query_id_col).size().to_numpy()
print(f"\nGroup 資訊 (每個查詢的文件數): {group_counts}")

# --- 步驟 4.3: 以 `query_id` 為單位切分資料 ---
# 獲取所有不重複的 query_id
unique_query_ids = np.unique(df[query_id_col])

# 切分 query_id
train_qids, test_qids = train_test_split(unique_query_ids, test_size=0.33, random_state=42)

# 根據切分好的 query_id 來篩選出對應的資料
train_indices = df[query_id_col].isin(train_qids)
test_indices = df[query_id_col].isin(test_qids)

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# 重新計算訓練集和測試集的 group 資訊
train_group = df[train_indices].groupby(query_id_col).size().to_numpy()
test_group = df[test_indices].groupby(query_id_col).size().to_numpy()

print(f"\n訓練資料維度: {X_train.shape}, 訓練 Group: {train_group}")
print(f"測試資料維度: {X_test.shape}, 測試 Group: {test_group}")

# --- 步驟 4.4: 建立與訓練 LGBMRanker 模型 ---
# LGBMRanker 是 LightGBM 中專為排序任務設計的模型
# objective='lambdarank': 指定使用 LambdaRank/LambdaMART 演算法
# metric='ndcg': 告訴模型在訓練過程中，監控並優化 NDCG 分數
ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# 訓練模型！注意，這裡要傳入 `group` 參數
print("\n開始訓練 LGBMRanker...")
ranker.fit(
    X_train,
    y_train,
    group=train_group,
    eval_set=[(X_test, y_test)],
    eval_group=[test_group],
    eval_at=[5],  # 計算 NDCG@5
    callbacks=[lgb.early_stopping(10, verbose=False)]
)
print("模型訓練完成！")

# --- 步驟 4.5: 進行預測 ---
# 預測出來的值是每個文件的「排序分數」，分數越高代表模型認為越相關
test_predictions = ranker.predict(X_test)

# 將預測結果整理成 DataFrame，方便觀察
results_df = X_test.copy()
results_df['true_relevance'] = y_test
results_df['predicted_score'] = test_predictions
results_df['query_id'] = df[test_indices][query_id_col].values

print("\n模型在測試集上的預測結果（前幾筆）：")
print(results_df.head())


# =============================================================================
# 5. 模型評估與調優
# =============================================================================
"""
對於排序問題，我們不能使用準確率(Accuracy)或均方誤差(MSE)來評估。
我們需要專門的排序指標。

**NDCG (Normalized Discounted Cumulative Gain)**
NDCG 是最常用、也最重要的排序評估指標。它的核心思想是：
1.  **C (Cumulative Gain)**: 相關的文件越多，分數越高。
2.  **D (Discounted)**: 高度相關的文件排得越靠前，分數越高。排在後面的文件
    其重要性會被打折扣 (discount)。
3.  **N (Normalized)**: 將分數標準化到 0 到 1 之間，使得不同查詢之間的
    分數可以互相比較。1 代表完美排序，0 代表最差排序。

**計算方式**：
`sklearn.metrics.ndcg_score` 可以幫助我們計算。它需要 `y_true` (真實相關性
分數) 和 `y_score` (模型預測分數)。由於它一次只能處理一個列表，我們需要
對每個測試集中的 `query_id` 進行迴圈計算，最後取平均。
"""
print("\n" + "="*60)
print("第五部分：模型評估 - 計算 NDCG")
print("="*60)

# 由於 ndcg_score 一次只能處理一個查詢的結果，我們需要遍歷測試集中的每個查詢
ndcg_scores = []
for qid in test_qids:
    # 篩選出當前查詢的真實相關性和預測分數
    current_query_mask = results_df['query_id'] == qid
    true_relevance = results_df[current_query_mask]['true_relevance'].to_numpy().reshape(1, -1)
    predicted_scores = results_df[current_query_mask]['predicted_score'].to_numpy().reshape(1, -1)

    # 計算該查詢的 NDCG 分數
    ndcg = ndcg_score(true_relevance, predicted_scores)
    ndcg_scores.append(ndcg)
    print(f"Query ID {qid} 的 NDCG 分數: {ndcg:.4f}")

# 計算平均 NDCG
average_ndcg = np.mean(ndcg_scores)
print(f"\n測試集上的平均 NDCG 分數: {average_ndcg:.4f}")

"""
**模型調優**：
與其他機器學習模型類似，`LGBMRanker` 也有許多超參數可以調整，例如：
- `n_estimators`: 樹的數量。
- `learning_rate`: 學習率。
- `num_leaves`: 每棵樹的葉子節點數量。
- `max_depth`: 樹的最大深度。

可以使用 `GridSearchCV` 或 `RandomizedSearchCV` 等工具，結合對 NDCG 指標
的監控，來尋找最佳的超參數組合。
"""


# =============================================================================
# 6. 整合至工作流程：建立一個完整的預測函數
# =============================================================================
"""
在真實世界的應用中，我們需要將訓練好的模型封裝起來，以便在新的查詢到來時
能夠快速返回排序好的結果列表。
"""
print("\n" + "="*60)
print("第六部分：整合至工作流程")
print("="*60)

def get_ranked_list(query_features_df, model):
    """
    接收一個包含某個查詢下所有待排序文件特徵的 DataFrame，
    返回一個根據模型預測分數排序後的 DataFrame。

    Args:
        query_features_df (pd.DataFrame): 包含特徵欄位的 DataFrame。
        model (lgb.LGBMRanker): 訓練好的排序模型。

    Returns:
        pd.DataFrame: 增加了 'predicted_score' 欄位並已排序的 DataFrame。
    """
    # 提取特徵
    features_to_predict = query_features_df[features]

    # 使用模型預測排序分數
    scores = model.predict(features_to_predict)

    # 將分數加回 DataFrame
    ranked_df = query_features_df.copy()
    ranked_df['predicted_score'] = scores

    # 根據分數降序排列
    ranked_df = ranked_df.sort_values(by='predicted_score', ascending=False)

    return ranked_df

# --- 模擬應用場景 ---
# 假設有一個新的查詢 (query_id = 4)，它有3個候選文件
new_query_data = {
    'doc_id': [20, 21, 22],
    'feature_1': [0.85, 0.15, 0.65],
    'feature_2': [0.92, 0.25, 0.55]
}
new_query_df = pd.DataFrame(new_query_data)

print("\n模擬新查詢的候選文件：")
print(new_query_df)

# 使用封裝好的函數獲取排序結果
ranked_results = get_ranked_list(new_query_df, ranker)

print("\n經過排序模型預測後的結果列表：")
print(ranked_results)


# =============================================================================
# 7. 總結
# =============================================================================
"""
恭喜你完成了監督式排序的學習！

回顧本次教學的重點：
1.  我們理解了排序問題的核心是預測「相對順序」，而不僅僅是絕對值。
2.  我們探討了 Pointwise, Pairwise, Listwise 三種方法，並選擇了最強大的
    Listwise 方法進行實作。
3.  我們學會了如何準備 `group` 資訊，這是訓練 Listwise 模型的關鍵步驟。
4.  我們使用 `lightgbm.LGBMRanker` 成功訓練了一個排序模型，並用它來預測
    新查詢的結果排序。
5.  我們掌握了使用核心指標 NDCG 來評估排序模型的好壞。

排序技術是現代資訊檢索和推薦系統的基石。希望這份教學能為您打開一扇
通往更專業領域的大門！
"""
print("\n" + "="*60)
print("教學腳本執行完畢！")
print("="*60)
