# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習實戰：監督式學習 - 迴歸模型 (Regression) 教學腳本
==============================================================================

### 教學目標：
1.  理解監督式學習中「迴歸問題」的核心概念與應用場景。
2.  學習並實作四種主流的迴歸模型：線性迴歸、決策樹迴歸、隨機森林迴歸與梯度提升迴歸。
3.  掌握模型訓練、評估與驗證的完整流程，從資料分割到效能指標解讀。
4.  學會使用網格搜索 (Grid Search) 與交叉驗證 (Cross-Validation) 進行模型超參數調優。
5.  精通 scikit-learn 套件在迴歸任務中的實務應用。

### 適用對象：
- 已完成資料預處理與特徵選擇，準備進入模型建立階段的學習者。
- 希望系統性學習 Python 迴歸模型實作的資料科學初學者。
- 欲鞏固 scikit-learn 核心功能應用的開發者。

### 前置知識：
- 具備 Python 基礎語法能力。
- 了解 pandas 與 numpy 套件的基本操作。
- 已對特徵工程有初步認識。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始建立模型之前，讓我們先導入所有需要的套件。
# pandas 用於資料處理，numpy 用於數值運算。
import pandas as pd
import numpy as np

# scikit-learn 是我們實作機器學習演算法的核心
# 資料分割工具
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# 迴歸模型
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# 效能評估指標
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("所有必要的套件已成功導入！")


# =============================================================================
# 1. 迴歸模型定義與選擇
# =============================================================================
"""
### 1.1 什麼是監督式學習中的迴歸 (Regression)？

在監督式學習中，我們使用一組已經標記好的資料（包含輸入特徵與正確的輸出答案）來訓練模型。
其目標是學習一個從輸入到輸出的映射函數。

**迴歸**是監督式學習的一種類型，其特殊之處在於它的「輸出答案」是**連續性**的數值。
例如：
- 預測房價（例如：300萬, 520.5萬）
- 預測氣溫（例如：25.5度, -3度）
- 預測股票價格（例如：120.5元, 78.9元）
- 預測銷售量（例如：1000件, 2500件）

簡單來說，當你想要模型預測一個「數值」時，你面對的就是一個迴歸問題。

### 1.2 常見的迴歸模型

scikit-learn 提供了多種強大的迴歸模型，以下介紹四種最主流的模型：

1.  **線性迴歸 (Linear Regression)**
    -   **核心思想**：試圖找到一條直線（或超平面）來最佳地擬合資料點。
    -   **適用情境**：當特徵與目標變數之間存在線性關係時。
    -   **優勢**：模型簡單、計算速度快、易於解釋。是理解迴歸問題的絕佳起點。

2.  **決策樹迴歸 (Decision Tree Regressor)**
    -   **核心思想**：透過一系列的「是/否」問題（規則）來分割資料，最終在每個葉節點上給出一個預測值。
    -   **適用情境**：當資料中存在非線性關係與特徵互動時。
    -   **優勢**：能夠捕捉複雜的非線性模式，模型結果直觀（可視覺化）。
    -   **風險**：容易過擬合 (Overfitting)，對資料中的小變動很敏感。

3.  **隨機森林迴歸 (Random Forest Regressor)**
    -   **核心思想**：建立多棵決策樹，並將它們的預測結果進行平均。這是一種「集成學習」(Ensemble Learning) 的方法。
    -   **適用情境**：幾乎適用於所有迴歸問題，是目前最泛用且強大的模型之一。
    -   **優勢**：顯著降低單一決策樹的過擬合風險，模型穩定性與準確性高。
    -   **風險**：計算成本較高，模型解釋性比單一決策樹差。

4.  **梯度提升迴歸 (Gradient Boosting Regressor)**
    -   **核心思想**：也是一種集成學習。它循序漸進地建立多棵樹，每一棵新樹都致力於修正前面所有樹的預測錯誤。
    -   **適用情境**：當追求極致的預測精度時。
    -   **優勢**：通常能達到非常高的準確性，是許多資料科學競賽中的首選。
    -   **風險**：對超參數敏感，訓練時間可能較長，且有過擬合的風險。
"""
print("\n" + "="*60)
print("第一部分：迴歸模型定義與選擇 - 理論基礎")
print("="*60)


# =============================================================================
# 2. 準備範例資料與資料分割
# =============================================================================
# 為了讓腳本可以獨立執行，我們創建一個虛構的房價預測資料集。
# 在真實專案中，您會從 CSV 或資料庫載入資料。
# 特徵 (X): '坪數', '屋齡', '鄰近捷運站' (1:是, 0:否)
# 目標 (y): '房價' (萬)
data = {
    '坪數': [25, 30, 15, 50, 40, 22, 35, 45, 18, 28],
    '屋齡': [5, 2, 10, 1, 3, 8, 4, 6, 12, 7],
    '鄰近捷運站': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    '房價': [1500, 1800, 900, 3000, 2200, 1300, 1900, 2500, 1000, 1600]
}
df = pd.DataFrame(data)

# 分離特徵 (X) 與目標 (y)
X = df[['坪數', '屋齡', '鄰近捷運站']]
y = df['房價']

print("\n範例資料已建立：")
print(df)

# --- 資料分割 (Train-Test Split) ---
# 這是模型建立中最關鍵的一步。我們將資料集切分為「訓練集」和「測試集」。
# 訓練集 (Training Set): 用來訓練模型，讓模型學習資料中的規律。
# 測試集 (Test Set): 用來評估模型的表現。模型在訓練過程中「從未見過」測試集，
#                     因此能客觀地評估其「泛化能力」。

# test_size=0.2 表示我們將 20% 的資料作為測試集，剩下的 80% 作為訓練集。
# random_state 是一個隨機種子，設定一個固定的值可以確保每次分割的結果都一樣，方便重現實驗。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "="*60)
print("第二部分：資料準備與分割")
print(f"原始資料總筆數: {len(df)}")
print(f"訓練集筆數: {len(X_train)}")
print(f"測試集筆數: {len(X_test)}")
print("="*60)


# =============================================================================
# 3. 模型訓練與實作
# =============================================================================
# scikit-learn 的模型訓練遵循一個非常一致的模式：
# 1. 初始化 (Instantiate) 模型。
# 2. 使用 .fit() 方法在訓練資料上進行訓練。
# 3. 使用 .predict() 方法在測試資料上進行預測。

# --- 3.1 線性迴歸 ---
print("\n--- 訓練線性迴歸模型 ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("線性迴歸模型訓練完成。")

# --- 3.2 決策樹迴歸 ---
print("\n--- 訓練決策樹迴歸模型 ---")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("決策樹迴歸模型訓練完成。")

# --- 3.3 隨機森林迴歸 ---
print("\n--- 訓練隨機森林迴歸模型 ---")
# n_estimators 是森林中樹的數量，是重要的超參數之一。
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("隨機森林迴歸模型訓練完成。")

# --- 3.4 梯度提升迴歸 ---
print("\n--- 訓練梯度提升迴歸模型 ---")
# n_estimators, learning_rate 是重要的超參數。
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr_model.fit(X_train, y_train)
gbr_predictions = gbr_model.predict(X_test)
print("梯度提升迴歸模型訓練完成。")

print("\n" + "="*60)
print("第三部分：四種核心迴歸模型已完成訓練")
print("="*60)


# =============================================================================
# 4. 模型評估 (Model Evaluation)
# =============================================================================
"""
模型訓練完後，我們需要客觀的指標來衡量模型的好壞。
對於迴歸問題，常用的評估指標有：

1.  **平均絕對誤差 (Mean Absolute Error, MAE)**
    -   **計算**：所有「預測值」與「真實值」之間差的絕對值的平均。
    -   **解讀**：代表模型預測的平均誤差大小。值越小越好。單位與目標變數相同。
    -   **公式**：(1/n) * Σ|y_true - y_pred|

2.  **均方誤差 (Mean Squared Error, MSE)**
    -   **計算**：所有「預測值」與「真實值」之間差的平方的平均。
    -   **解讀**：懲罰較大的誤差。值越小越好。因為是平方，單位與目標變數不同。
    -   **公式**：(1/n) * Σ(y_true - y_pred)²

3.  **均方根誤差 (Root Mean Squared Error, RMSE)**
    -   **計算**：MSE 的平方根。
    -   **解讀**：與 MAE 類似，但對大誤差更敏感。值越小越好。單位與目標變數相同，更易於解釋。
    -   **公式**：sqrt(MSE)

4.  **決定係數 (R-squared, R²)**
    -   **計算**：衡量模型解釋力的指標。
    -   **解讀**：R² 的值介於 0 和 1 之間。越接近 1，表示模型能解釋越多目標變數的變異性，擬合效果越好。
             R² 為 0 表示模型跟猜平均值一樣差。R² 可能為負，表示模型比猜平均值還差。
"""

def evaluate_regression(y_true, y_pred, model_name):
    """一個用於計算並印出所有迴歸指標的輔助函式"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {model_name} 效能評估 ---")
    print(f"平均絕對誤差 (MAE): {mae:.2f}")
    print(f"均方誤差 (MSE): {mse:.2f}")
    print(f"均方根誤差 (RMSE): {rmse:.2f}")
    print(f"決定係數 (R²): {r2:.4f}")
    print("-" * (len(model_name) + 15))

print("\n" + "="*60)
print("第四部分：模型效能評估")
print("="*60)

# 評估各個模型
evaluate_regression(y_test, lr_predictions, "線性迴歸")
evaluate_regression(y_test, dt_predictions, "決策樹迴歸")
evaluate_regression(y_test, rf_predictions, "隨機森林迴歸")
evaluate_regression(y_test, gbr_predictions, "梯度提升迴歸")

# 比較預測結果與真實值
results_df = pd.DataFrame({'真實房價': y_test,
                           '隨機森林預測': rf_predictions,
                           '梯度提升預測': gbr_predictions})
print("\n比較隨機森林與梯度提升的預測結果：")
print(results_df)


# =============================================================================
# 5. 模型調優與驗證 (Tuning & Validation)
# =============================================================================
"""
### 5.1 為什麼需要調優與驗證？

- **超參數 (Hyperparameters)**：我們在初始化模型時設定的參數（例如 RandomForest 的 `n_estimators`），
  它們不是模型從資料中學習到的，而是我們手動設定的。不同的超參數組合會產生不同效能的模型。
- **模型調優**：就是系統性地尋找最佳超參數組合的過程。
- **交叉驗證 (Cross-Validation)**：一種更穩健的模型評估方法。它將訓練集分成好幾折 (folds)，
  輪流用其中一折當作驗證集，其他折當作訓練集。這樣可以避免單次 train-test split 的偶然性，
  得到更可靠的效能評估。

### 5.2 網格搜索 (Grid Search)

網格搜索是最常見的調優方法。它會窮盡你所指定的所有超參數組合，並透過交叉驗證來評估每一種組合的效能，
最終找出最佳的那一組。

我們以「隨機森林迴歸」為例，來尋找最佳的 `n_estimators` (樹的數量) 和 `max_depth` (樹的最大深度)。
"""
print("\n" + "="*60)
print("第五部分：模型調優與驗證")
print("="*60)

# --- 使用 GridSearchCV 進行超參數搜索 ---
# 1. 定義要搜索的超參數網格
#    這是一個字典，鍵是超參數名稱，值是你想嘗試的數值列表。
param_grid = {
    'n_estimators': [50, 100, 150],       # 嘗試 3 種樹的數量
    'max_depth': [None, 10, 20],          # 嘗試 3 種樹的最大深度 (None 表示不限制)
    'min_samples_split': [2, 5]           # 嘗試 2 種內部節點再分割所需的最小樣本數
}

# 2. 初始化 GridSearchCV
#    - estimator: 你要調優的模型 (一個新的、未訓練過的模型實例)
#    - param_grid: 超參數網格
#    - cv: 交叉驗證的折數 (通常設為 5 或 10)
#    - scoring: 評估效能的指標，對於迴歸，'neg_mean_squared_error' 是常用選項 (因為 GridSearch 會最大化分數，所以用負的MSE)
#    - n_jobs=-1: 使用所有可用的 CPU 核心來加速計算
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1,
                           verbose=1) # verbose=1 會顯示搜索過程

# 3. 在「完整訓練資料」上執行搜索
#    注意：這裡我們用 X_train 和 y_train，而不是整個 X, y。
#    GridSearchCV 內部會自動處理交叉驗證的分割。
print("\n開始進行網格搜索 (Grid Search)...")
grid_search.fit(X_train, y_train)
print("網格搜索完成！")

# 4. 檢視最佳結果
print(f"\n找到的最佳超參數組合: {grid_search.best_params_}")
print(f"在交叉驗證中得到的最佳負MSE分數: {grid_search.best_score_:.2f}")

# 5. 使用最佳模型進行預測
best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

# 6. 評估調優後的模型
print("\n--- 調優後的隨機森林模型效能評估 ---")
evaluate_regression(y_test, best_rf_predictions, "調優後隨機森林")


# =============================================================================
# 6. 結論與後續步驟
# =============================================================================
"""
恭喜你！你已經完成了從模型選擇、訓練、評估到調優的完整迴歸分析流程。

### 本次教學總結：
1.  **理解問題**：我們首先定義了迴歸問題，並了解了四種核心模型的理論基礎。
2.  **準備資料**：學習了為何以及如何將資料分割為訓練集與測試集。
3.  **建立模型**：使用 scikit-learn 實作了四種迴歸模型。
4.  **評估效能**：掌握了 MAE, MSE, RMSE, R² 等關鍵指標的計算與解讀。
5.  **模型優化**：透過網格搜索與交叉驗證，找到了模型的最佳超參數，並提升了其預測能力。

### 後續步驟：
- **特徵重要性 (Feature Importance)**：對於決策樹、隨機森林、梯度提升等模型，可以分析哪些特徵對預測最有貢獻。
- **模型儲存與部署**：使用 `joblib` 或 `pickle` 套件將訓練好的最佳模型儲存起來，以便未來在其他應用中直接使用。
- **嘗試更多模型**：scikit-learn 還提供了如 SVR, Lasso, Ridge 等更多迴歸模型，可以嘗試它們的效果。
- **進階調優**：除了網格搜索，還可以探索隨機搜索 (RandomizedSearchCV) 或貝葉斯優化等更高效的調優技術。
"""
print("\n" + "="*60)
print("教學結束！您已掌握監督式迴歸的核心實作流程。")
print("="*60)
