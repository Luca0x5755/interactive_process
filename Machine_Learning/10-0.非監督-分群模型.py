# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習實戰：非監督式學習之分群 (Clustering) 教學腳本
==============================================================================

### 教學目標：
1.  理解什麼是非監督式學習中的「分群」，以及其在資料分析中的重要性。
2.  學習並實作三種主流的分群演算法：K-Means、DBSCAN 與階層式分群。
3.  掌握如何評估分群模型的效能，例如使用輪廓係數 (Silhouette Score) 與 Davies-Bouldin 指數。
4.  學會使用 Matplotlib 與 Seaborn 將分群結果視覺化，以利於分析與解讀。
5.  了解不同分群演算法的適用情境、核心參數與優缺點。

### 適用對象：
- 已具備 Python 基礎，並對機器學習有初步認識的學習者。
- 希望深入了解非監督式學習，特別是分群技術的資料分析師或工程師。
- 假設使用者已完成特徵工程與選擇，專注於分群演算法本身。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件。
# numpy 提供強大的數值計算功能，是科學計算的基礎。
import numpy as np
# pandas 是我們處理與操作資料表(DataFrame)的利器。
import pandas as pd
# matplotlib 與 seaborn 是 Python 中最主流的資料視覺化函式庫。
import matplotlib.pyplot as plt
import seaborn as sns
# scikit-learn 是 Python 機器學習的標準函式庫，我們將從中引用演算法與評估工具。
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
# scipy 用於階層式分群中的樹狀圖 (Dendrogram) 繪製。
from scipy.cluster.hierarchy import dendrogram, linkage

print("✅ 套件導入成功！準備開始學習非監督式分群。")


# =============================================================================
# 1. 主題介紹：什麼是分群 (Clustering)？
# =============================================================================
"""
分群是一種「非監督式學習」(Unsupervised Learning) 技術。

與「監督式學習」(Supervised Learning) 不同，非監督式學習的資料集
沒有預先定義好的標籤 (Label) 或目標 (Target)。演算法的任務是自行在
資料中找出隱藏的結構、模式或關係。

**分群的核心目的**，就是根據資料點之間的「相似性」，將它們自動分組。
理想情況下，同一群組內的資料點彼此非常相似，而不同群組之間的資料點
則差異很大。

**為何分群很重要？**
- **顧客分群 (Customer Segmentation):** 根據顧客的購買行為、人口統計資料等，
  將他們分為不同客群（如高價值客戶、潛力新客、沉睡客戶），以便制定精準的
  行銷策略。
- **異常檢測 (Anomaly Detection):** 無法歸入任何群組的孤立點，可能就是異常
  資料，例如信用卡盜刷、網路入侵或系統故障。
- **圖像分割 (Image Segmentation):** 在電腦視覺中，可將圖片中顏色或紋理
  相似的像素分在同一群，以識別物體。
- **生物資訊 (Bioinformatics):** 根據基因表現，將具有相似功能的基因分群。

簡單來說，分群幫助我們在沒有標準答案的數據海洋中，探索並定義出有意義的群體。
"""

print("\n" + "="*70)
print("第一部分：主題介紹 - 什麼是分群 (Clustering)？")
print("="*70)


# =============================================================================
# 2. 資料準備：生成模擬資料集
# =============================================================================
"""
為了專注於演算法本身，我們不使用真實世界的複雜資料，而是使用 scikit-learn
的 `make_blobs` 函式來生成一個理想的模擬資料集。

`make_blobs` 可以讓我們輕鬆控制樣本數、特徵數、群心數量以及群體的離散程度，
非常適合用來展示分群演算法的效果。
"""
# --- 資料生成 ---
# n_samples: 樣本總數
# n_features: 特徵數量 (維度)
# centers: 要生成的群心數量
# cluster_std: 群體的標準差，數值越大，群體越分散
# random_state: 亂數種子，確保每次生成的資料都一樣，方便重現結果
X, y_true = make_blobs(n_samples=300,
                       n_features=2,
                       centers=4,
                       cluster_std=1.0,
                       random_state=42)

print("\n" + "="*70)
print("第二部分：資料準備")
print("="*70)
print(f"✅ 已成功生成模擬資料，資料維度: {X.shape}")
print("真實的群組標籤 (y_true) 我們僅用於最後比較，演算法本身不會使用它。")

# --- 原始資料視覺化 ---
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=50, alpha=0.7)
plt.title('原始模擬資料分佈 (未分群)', fontsize=16)
plt.xlabel('特徵 1', fontsize=12)
plt.ylabel('特徵 2', fontsize=12)
plt.grid(True)
plt.show()


# =============================================================================
# 3. 演算法實作一：K-Means (K-均值分群)
# =============================================================================
"""
K-Means 是最經典、最廣為人知的 centroid-based (基於中心點) 分群演算法。

**核心思想：**
1.  **初始化：** 隨機選擇 K 個資料點作為初始的群心 (Centroids)。
2.  **分配 (Assignment)：** 將每個資料點分配給離它最近的那個群心。
3.  **更新 (Update)：** 重新計算每個群組的中心點（即該群組所有資料點的平均值），
    作為新的群心。
4.  **迭代：** 重複步驟 2 和 3，直到群心不再有明顯變動，或達到設定的迭代次數為止。

**重要參數：**
- `n_clusters` (即 K 值): 需要事先指定的群組數量。這是 K-Means 的最大特點，
  也是其限制之一。

**優點：**
- 演算法簡單、快速，計算效率高，適合處理大規模資料。
- 在群組為凸形 (convex) 且大小相似時，效果很好。

**缺點：**
- 需要預先設定 K 值。K 值的選擇對結果影響巨大。
- 對初始群心的選擇很敏感，可能導致局部最佳解。
- 對於非球形、大小不一或密度不均的群組，效果較差。
"""
print("\n" + "="*70)
print("第三部分：演算法實作 - K-Means")
print("="*70)

# --- 模型訓練 ---
# 我們知道真實群數是 4，所以先用 K=4 來實驗
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

print(f"✅ K-Means 模型訓練完成 (K={k})")

# --- 結果評估 ---
# 輪廓係數 (Silhouette Score):
# - 衡量一個點與其所屬群組的緊密程度，以及與其他群組的分離程度。
# - 範圍在 [-1, 1] 之間。越接近 1，表示分群效果越好。
# - 接近 0 表示群組重疊，負值表示可能分到了錯誤的群組。
silhouette_avg = silhouette_score(X, y_kmeans)

# Davies-Bouldin 指數 (Davies-Bouldin Index):
# - 計算群內離散度與群間距離的比值。
# - 指數越小，表示群內越緊密，群間越分離，分群效果越好。最小值為 0。
db_index = davies_bouldin_score(X, y_kmeans)

print(f"輪廓係數 (Silhouette Score): {silhouette_avg:.4f}")
print(f"Davies-Bouldin 指數: {db_index:.4f}")

# --- 結果視覺化 ---
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette='viridis', s=50, alpha=0.7, legend='full')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*', label='Centroids (群心)')
plt.title(f'K-Means 分群結果 (K={k})', fontsize=16)
plt.xlabel('特徵 1', fontsize=12)
plt.ylabel('特徵 2', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# --- 如何選擇最佳 K 值：手肘法 (Elbow Method) ---
"""
在真實場景中，我們通常不知道最佳的 K 值是多少。手肘法是常見的輔助工具。
原理是計算不同 K 值下的「群內誤差平方和 (WCSS, Within-Cluster Sum of Squares)」，
並觀察其變化。當 WCSS 的下降趨勢由陡峭轉為平緩時，那個轉折點就像手肘一樣，
通常是個不錯的 K 值選擇。
"""
wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans_elbow = KMeans(n_clusters=i, random_state=42, n_init='auto')
    kmeans_elbow.fit(X)
    wcss.append(kmeans_elbow.inertia_) # inertia_ 屬性就是 WCSS

plt.figure(figsize=(10, 7))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('K-Means 手肘法 (Elbow Method)', fontsize=16)
plt.xlabel('群組數量 (K)', fontsize=12)
plt.ylabel('群內誤差平方和 (WCSS)', fontsize=12)
plt.xticks(k_range)
plt.grid(True)
plt.show()
print("📈 手肘法圖表顯示，K=4 是一個明顯的轉折點，符合我們的預期。")


# =============================================================================
# 4. 演算法實作二：DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# =============================================================================
"""
DBSCAN 是一種 density-based (基於密度) 的分群演算法。

**核心思想：**
它將群組定義為「高密度區域」，並將被低密度區域分隔開的點視為不同群組。
它不需要預先指定群組數量，還能有效地找出「雜訊點 (Noise)」。

**重要參數：**
- `eps` (epsilon): 鄰域半徑。定義了一個點的「周圍」範圍有多大。
- `min_samples`: 核心點的最小鄰居數。在一個點的 `eps` 半徑內，至少要有多少個
  其他點，才能將該點視為「核心點 (Core Point)」。

**優點：**
- 不需要設定群組數量。
- 能夠發現任意形狀的分群 (例如環形、月牙形)。
- 對雜訊點不敏感，能明確標示出離群值。

**缺點：**
- 對 `eps` 和 `min_samples` 參數的選擇非常敏感。
- 當資料密度不均勻時，很難找到一組合適的參數對所有群組都有效。
"""
print("\n" + "="*70)
print("第四部分：演算法實作 - DBSCAN")
print("="*70)

# --- 模型訓練 ---
# DBSCAN 參數的選擇通常需要一些經驗或領域知識，這裡我們先嘗試一組參數
dbscan = DBSCAN(eps=0.9, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# 取得分群數量 (排除雜訊點 -1)
n_clusters_dbscan = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

print(f"✅ DBSCAN 模型訓練完成")
print(f"找到的群組數量: {n_clusters_dbscan}")
print(f"辨識出的雜訊點數量: {n_noise}")

# --- 結果評估 ---
# 注意：如果沒有找到任何群組或只有一個群組，則無法計算評估指標
if n_clusters_dbscan > 1:
    silhouette_dbscan = silhouette_score(X, y_dbscan)
    db_index_dbscan = davies_bouldin_score(X, y_dbscan)
    print(f"輪廓係數 (Silhouette Score): {silhouette_dbscan:.4f}")
    print(f"Davies-Bouldin 指數: {db_index_dbscan:.4f}")
else:
    print("由於找到的群組數不足，無法計算評估指標。")

# --- 結果視覺化 ---
plt.figure(figsize=(10, 7))
# 使用 set(y_dbscan) 來確保圖例的標籤是唯一的
unique_labels = set(y_dbscan)
# 為每個標籤設定顏色
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 將雜訊點設為灰色
        col = 'gray'
        label = 'Noise (雜訊)'
    else:
        label = f'Cluster {k}'

    class_member_mask = (y_dbscan == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], edgecolor='k', alpha=0.7, label=label)

plt.title(f'DBSCAN 分群結果 (eps=0.9, min_samples=5)', fontsize=16)
plt.xlabel('特徵 1', fontsize=12)
plt.ylabel('特徵 2', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
print("💡 提示：嘗試調整 `eps` 與 `min_samples` 的值，觀察分群結果的變化。")


# =============================================================================
# 5. 演算法實作三：階層式分群 (Hierarchical Clustering)
# =============================================================================
"""
階層式分群會建立一個樹狀的群組結構，稱為「樹狀圖 (Dendrogram)」。

**核心思想 (Agglomerative/凝聚式):**
1.  **開始：** 將每個資料點都視為一個獨立的群組。
2.  **合併：** 找到最相似（距離最近）的兩個群組，並將它們合併成一個新的群組。
3.  **迭代：** 重複步驟 2，直到所有資料點都被合併到同一個大群組中為止。

**重要參數：**
- `n_clusters`: 可以像 K-Means 一樣預設群組數。若設為 `None`，則會計算完整的樹狀結構。
- `linkage`: 合併策略，即如何計算群組之間的距離。
    - `ward`: 最小化合併後群組內的變異數總和 (常用且效果好)。
    - `complete`: 計算兩群組中，所有點對距離的最大值 (Max Linkage)。
    - `average`: 計算兩群組中，所有點對距離的平均值 (Average Linkage)。

**優點：**
- 不需要預設群組數量，可以透過樹狀圖來決定。
- 樹狀圖提供了豐富的視覺化資訊，讓我們能理解資料的階層結構。

**缺點：**
- 計算複雜度較高 (通常是 O(n^2) 或更高)，不適合非常大的資料集。
- 一旦合併完成，就無法撤銷，可能導致次佳的結果。
"""
print("\n" + "="*70)
print("第五部分：演算法實作 - 階層式分群")
print("="*70)

# --- 步驟一：繪製樹狀圖 (Dendrogram) 來決定群數 ---
# 使用 scipy 的 linkage 函式來計算階層結構
# 'ward' 是一種常用的 linkage 方法，旨在最小化群內的變異
linked = linkage(X, method='ward')

plt.figure(figsize=(14, 8))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('階層式分群 樹狀圖 (Dendrogram)', fontsize=16)
plt.xlabel('資料點索引', fontsize=12)
plt.ylabel('距離 (歐式距離)', fontsize=12)
# 畫一條水平線來輔助判斷群數
plt.axhline(y=25, color='r', linestyle='--')
plt.show()
print("🌳 樹狀圖分析：在 y=25 處用水平線切割，會得到 4 個群組。")

# --- 步驟二：根據樹狀圖分析，進行模型訓練 ---
# 根據上面的分析，我們設定 n_clusters=4
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_agg = agg_cluster.fit_predict(X)

print(f"✅ 階層式分群模型訓練完成 (n_clusters=4)")

# --- 結果評估 ---
silhouette_agg = silhouette_score(X, y_agg)
db_index_agg = davies_bouldin_score(X, y_agg)
print(f"輪廓係數 (Silhouette Score): {silhouette_agg:.4f}")
print(f"Davies-Bouldin 指數: {db_index_agg:.4f}")

# --- 結果視覺化 ---
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_agg, palette='plasma', s=50, alpha=0.7)
plt.title('階層式分群結果 (n_clusters=4)', fontsize=16)
plt.xlabel('特徵 1', fontsize=12)
plt.ylabel('特徵 2', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# 6. 結論與總結
# =============================================================================
"""
本次教學我們實作了三種主流的分群演算法，並對它們進行了評估與視覺化。

**演算法比較總結：**

| 特性         | K-Means                               | DBSCAN                                  | 階層式分群                               |
|--------------|---------------------------------------|-----------------------------------------|------------------------------------------|
| **核心思想** | 基於中心點 (Centroid-based)           | 基於密度 (Density-based)                | 基於階層/合併 (Hierarchical)             |
| **群組形狀** | 傾向於球形、大小相近                  | 可發現任意形狀                          | 彈性，取決於 linkage method              |
| **主要參數** | `n_clusters` (K值)                    | `eps`, `min_samples`                    | `n_clusters`, `linkage`                  |
| **需要群數?**| **是**，必須預先指定                  | **否**，自動決定                        | **否**，可由樹狀圖決定                   |
| **雜訊處理** | 無法直接處理，所有點都會被分群        | **是**，能有效識別雜訊點                | 無法直接處理，所有點都會被分群           |
| **計算效率** | 高，適合大規模資料                    | 中等，資料量大時較慢                    | 低 (O(n^2))，不適合大規模資料            |
| **適用情境** | 群組結構較簡單、資料量大              | 群組形狀不規則、需要找出異常值          | 需要探索資料階層結構、資料量不大         |

**給學習者的建議：**
- **沒有最好的演算法，只有最適合的演算法。** 在面對真實問題時，應該根據
  資料特性與分析目的來選擇。
- **參數調校是關鍵。** 本次教學中的參數是為了展示效果，在實務上，您需要
  透過交叉驗證、評估指標或領域知識來找到最佳參數組合。
- **動手實驗！** 嘗試修改 `make_blobs` 的參數 (如 `cluster_std` 或 `centers`)，
  或者更換 `sklearn.datasets` 中的其他資料集 (如 `make_moons`, `make_circles`)，
  觀察不同演算法的表現，這是加深理解的最好方式。
"""
print("\n" + "="*70)
print("第六部分：結論與總結")
print("="*70)
print("🎉恭喜您！您已完成本次非監督式分群的教學。")
print("希望這份腳本能幫助您建立對分群演算法的扎實理解。")
