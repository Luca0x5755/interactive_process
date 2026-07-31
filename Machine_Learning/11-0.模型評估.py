# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習實戰：模型評估 (Model Evaluation) 教學腳本
==============================================================================

### 教學目標：
1.  理解模型評估的定義，以及為何它在機器學習工作流程中至關重要。
2.  學習並實作分類與迴歸任務中，最常見的模型評估指標。
3.  掌握訓練/測試集劃分、交叉驗證、學習曲線與驗證曲線等核心評估技術。
4.  了解如何解讀各項指標與圖表，以診斷模型問題（如：過擬合、欠擬合）並進行優化。
5.  掌握使用 scikit-learn 套件進行模型評估的實務技巧。

### 適用對象：
- 已了解機器學習基本概念，希望深入學習如何評估與選擇模型的學習者。
- 希望建立一套完整、嚴謹的模型評估流程的資料科學家或工程師。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn 套件：模型、資料集、預處理與評估工具
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

print("套件導入成功！準備開始學習模型評估。")


# =============================================================================
# 1. 主題介紹：什麼是模型評估 (Model Evaluation)？
# =============================================================================
"""
在建立機器學習模型後，我們最關心的問題是：「這個模型到底好不好？」
模型評估就是用來回答這個問題的系統性方法。

**模型評估的核心目的**，是使用量化的指標來衡量模型在「從未見過的資料」
上的表現能力。如果一個模型在訓練資料上表現完美，但在新的測試資料上
表現很差，我們稱之為「過擬合」(Overfitting)，這樣的模型是沒有實用價值的。

**為何模型評估如此重要？**
1.  **量化模型表現**：提供客觀的數字來比較不同模型或不同參數設定的優劣。
2.  **避免過擬合與欠擬合**：幫助我們診斷模型是學習得太多還是太少。
3.  **建立對模型的信心**：確保模型在部署到真實世界後，能有預期中的可靠表現。
4.  **指導模型優化**：評估結果可以告訴我們模型在哪方面做得不好，從而指導我們
    下一步的優化方向（例如：收集更多資料、調整特徵、更換演算法等）。

簡單來說，沒有經過嚴謹評估的模型，就像一輛沒有經過安全測試的汽車，我們
無法信任它能安全、可靠地完成任務。
"""

print("\n" + "="*60)
print("第一部分：主題介紹 - 模型評估的重要性")
print("="*60)


# =============================================================================
# 2. 評估技術 (1)：訓練/測試集劃分 (Train-Test Split)
# =============================================================================
"""
這是最基本也最重要的評估技術。我們將資料集切分為兩部分：
- **訓練集 (Training Set)**：用來訓練模型的資料。
- **測試集 (Test Set)**：用來評估模型表現的「未知」資料。模型在整個訓練
  過程中，完全看不到這部分資料。

這樣做可以模擬模型在真實世界中遇到新資料時的表現。
"""
print("\n" + "="*60)
print("第二部分：訓練/測試集劃分")
print("="*60)

# 2.1 準備資料：使用 scikit-learn 生成一個模擬的分類資料集
# n_samples: 樣本數
# n_features: 特徵數
# n_classes: 類別數
# flip_y: 噪點比例，增加任務難度
# random_state: 確保每次生成的資料都一樣，方便重現
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    flip_y=0.1,
    random_state=42
)

print(f"生成資料完成，資料維度 X: {X.shape}, y: {y.shape}")

# 2.2 執行劃分
# test_size=0.3: 將 30% 的資料劃為測試集，70% 為訓練集
# stratify=y: 確保劃分後，訓練集與測試集中的類別比例與原始資料相同，這在處理不平衡資料時尤其重要
# random_state=42: 確保每次劃分的結果都一樣
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"訓練集維度 X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"測試集維度 X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"原始資料類別 '1' 的比例: {np.mean(y):.2f}")
print(f"訓練集類別 '1' 的比例: {np.mean(y_train):.2f}")
print(f"測試集類別 '1' 的比例: {np.mean(y_test):.2f}")


# =============================================================================
# 3. 分類任務評估指標 (Classification Metrics)
# =============================================================================
"""
對於分類問題（如：判斷郵件是否為垃圾郵件、客戶是否會流失），我們有多種
評估指標，每種指標關注模型的不同方面。
"""
print("\n" + "="*60)
print("第三部分：分類任務評估指標")
print("="*60)

# 3.1 訓練一個簡單的分類模型
# 這裡使用邏輯迴歸作為範例
model_clf = LogisticRegression(random_state=42, max_iter=1000)
model_clf.fit(X_train, y_train)

# 3.2 在測試集上進行預測
y_pred = model_clf.predict(X_test)
y_pred_proba = model_clf.predict_proba(X_test)[:, 1] # 用於 ROC 曲線

print("模型訓練與預測完成。")

# 3.3 指標計算與解說

# --- 3.3.1 準確度 (Accuracy) ---
# 定義：(預測正確的樣本數) / (總樣本數)
# 適用情境：最直觀的指標，但在「類別不平衡」的資料中具有誤導性。
# 例如：99% 的郵件都不是垃圾郵件，一個模型即使把所有郵件都預測為「正常」，
# 準確度也能高達 99%，但它完全沒有識別垃圾郵件的能力。
accuracy = accuracy_score(y_test, y_pred)
print(f"\n[指標] 準確度 (Accuracy): {accuracy:.4f}")

# --- 3.3.2 混淆矩陣 (Confusion Matrix) ---
# 混淆矩陣是一個表格，用來視覺化模型的預測結果。
#       | 預測為 0 | 預測為 1
# ------|----------|----------
# 實際為 0 |    TN    |    FP
# 實際為 1 |    FN    |    TP
#
# TN (True Negative):  實際為 0，預測也為 0 (正確)
# FP (False Positive): 實際為 0，預測為 1 (錯誤，類型 I 錯誤)
# FN (False Negative): 實際為 1，預測為 0 (錯誤，類型 II 錯誤)
# TP (True Positive):  實際為 1，預測也為 1 (正確)
cm = confusion_matrix(y_test, y_pred)
print("\n[指標] 混淆矩陣 (Confusion Matrix):")
print(cm)

# 使用 seaborn 將混淆矩陣視覺化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['預測為 0', '預測為 1'],
            yticklabels=['實際為 0', '實際為 1'])
plt.title('混淆矩陣 (Confusion Matrix)')
plt.xlabel('預測標籤')
plt.ylabel('真實標籤')
plt.show()


# --- 3.3.3 精確度 (Precision)、召回率 (Recall) 與 F1 分數 (F1-Score) ---
# **精確度 (Precision)** = TP / (TP + FP)
#   - 在所有被模型預測為「正例」(Positive, 也就是 1) 的樣本中，有多少是真的正例。
#   - 關注的是「預測的準確性」，越高代表模型預測為正例的結果越可信。
#   - 適用情境：當我們不希望誤判時（低 FP 很重要），例如：垃圾郵件過濾，我們不希望把重要郵件誤判為垃圾郵件。

# **召回率 (Recall)** = TP / (TP + FN)
#   - 在所有真實為「正例」的樣本中，有多少被模型成功地找出來了。
#   - 關注的是「找得全不全」，越高代表模型越能找出所有正例。
#   - 適用情境：當我們不希望漏判時（低 FN 很重要），例如：癌症診斷，我們希望找出所有潛在的病患。

# **F1 分數 (F1-Score)** = 2 * (Precision * Recall) / (Precision + Recall)
#   - 精確度與召回率的調和平均數，是兩者的綜合性指標。
#   - 當 Precision 和 Recall 都很高時，F1-Score 也會高。
#   - 適用情境：當我們希望在 Precision 和 Recall 之間取得平衡時。

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n[指標] 精確度 (Precision): {precision:.4f}")
print(f"[指標] 召回率 (Recall): {recall:.4f}")
print(f"[指標] F1 分數 (F1-Score): {f1:.4f}")

# 我們也可以使用 classification_report 一次性輸出所有指標
print("\n[報告] 分類報告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['類別 0', '類別 1']))


# --- 3.3.4 ROC 曲線 (Receiver Operating Characteristic Curve) 與 AUC 值 ---
# **ROC 曲線**：
#   - 橫軸是「假正例率 (FPR)」= FP / (FP + TN)
#   - 縱軸是「真正例率 (TPR)」，也就是召回率 = TP / (TP + FN)
#   - 曲線描繪了在不同「分類閾值」下，TPR 和 FPR 的關係。
#   - 一個好的模型，其 ROC 曲線會盡量往左上角靠近，代表在維持低 FPR 的同時，能有高 TPR。
#   - 對角線（y=x）代表一個隨機猜測的模型。

# **AUC (Area Under the Curve)**：
#   - ROC 曲線下的面積，數值介於 0 到 1 之間。
#   - AUC 值越大，代表模型區分正負樣本的能力越強。
#   - AUC = 0.5: 模型沒有區分能力（同隨機猜測）。
#   - AUC = 1.0: 模型完美區分所有樣本。
#   - 優點：不受類別不平衡的影響，能全面評估模型的排序能力。

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\n[指標] AUC (Area Under Curve): {roc_auc:.4f}")

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲線 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='隨機猜測')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (False Positive Rate)')
plt.ylabel('真正例率 (True Positive Rate)')
plt.title('接收者操作特徵曲線 (ROC Curve)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# =============================================================================
# 4. 迴歸任務評估指標 (Regression Metrics)
# =============================================================================
"""
對於迴歸問題（如：預測房價、預測銷量），我們評估的是模型預測值與真實值
之間的「差距」。
"""
print("\n" + "="*60)
print("第四部分：迴歸任務評估指標")
print("="*60)

# 4.1 準備資料：使用 scikit-learn 生成一個模擬的迴歸資料集
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    noise=25,
    random_state=42
)

# 4.2 劃分資料並訓練模型
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)

print("迴歸模型訓練與預測完成。")

# 4.3 指標計算與解說

# --- 4.3.1 平均絕對誤差 (Mean Absolute Error, MAE) ---
# 定義：所有「預測值與真實值之差的絕對值」的平均。
# MAE = (1/n) * Σ|y_true - y_pred|
# 優點：單位與目標變數相同，易於理解。例如，如果預測房價，MAE=10代表平均預測誤差為10萬元。
# 缺點：沒有考慮誤差的方向。
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"\n[指標] 平均絕對誤差 (MAE): {mae:.4f}")

# --- 4.3.2 均方誤差 (Mean Squared Error, MSE) ---
# 定義：所有「預測值與真實值之差的平方」的平均。
# MSE = (1/n) * Σ(y_true - y_pred)²
# 優點：對較大的誤差給予更高的懲罰權重。
# 缺點：單位是目標變數的平方，不易直觀解釋。
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"[指標] 均方誤差 (MSE): {mse:.4f}")

# --- 4.3.3 均方根誤差 (Root Mean Squared Error, RMSE) ---
# 定義：MSE 的平方根。
# RMSE = sqrt(MSE)
# 優點：單位與目標變數相同，且保留了對大誤差懲罰的特性，是迴歸任務中最常用的指標之一。
rmse = np.sqrt(mse)
print(f"[指標] 均方根誤差 (RMSE): {rmse:.4f}")

# --- 4.3.4 R 平方 (R-squared, Coefficient of Determination) ---
# 定義：衡量模型能夠解釋目標變數變異性的比例。
# R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
# 解讀：
#   - R² 接近 1：模型解釋了大部分的變異，擬合效果好。
#   - R² 接近 0：模型解釋能力差，跟用平均值來預測差不多。
#   - R² 為負：模型表現比平均值還差（通常代表模型有嚴重問題）。
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"[指標] R 平方 (R-squared): {r2:.4f}")


# =============================================================================
# 5. 評估技術 (2)：交叉驗證 (Cross-Validation)
# =============================================================================
"""
單次的 train-test split 結果可能帶有偶然性，如果我們剛好分到「比較簡單」
的測試集，模型的評估分數可能會過高。

**交叉驗證** 是一種更穩健、更可靠的評估方法。其中最常用的是 **K-摺交叉驗證
(K-Fold Cross-Validation)**：
1.  將整個資料集分成 K 個大小相等的部分（稱為 "摺", Fold）。
2.  進行 K 次迴圈：
    - 在第 i 次迴圈中，將第 i 摺作為「驗證集」，其餘 K-1 摺作為「訓練集」。
    - 訓練模型並在驗證集上進行評估。
3.  最終的評估結果是這 K 次評估分數的平均值。

**優點**：
- 每個樣本都有機會被當作驗證資料，評估結果更全面、更穩定。
- 減少了因資料劃分偶然性帶來的偏差。
"""
print("\n" + "="*60)
print("第五部分：交叉驗證")
print("="*60)

# 這裡我們使用完整的分類資料集 X, y
# 建立一個 SVC 模型實例
model_svc = SVC(kernel='linear', random_state=42)

# 使用 cross_val_score 進行交叉驗證
# cv=5: 進行 5-摺交叉驗證
# scoring='accuracy': 指定評估指標為準確度
# n_jobs=-1: 使用所有可用的 CPU 核心來並行計算，加快速度
scores = cross_val_score(model_svc, X, y, cv=5, scoring='accuracy', n_jobs=-1)

print(f"5-摺交叉驗證的每次準確度分數: {scores}")
print(f"平均準確度: {scores.mean():.4f}")
print(f"準確度標準差: {scores.std():.4f}")
print("\n解讀：模型的平均準確度約為 85.1%，標準差為 1.2%，代表模型表現在不同資料子集上相對穩定。")


# =============================================================================
# 6. 評估技術 (3)：學習曲線與驗證曲線
# =============================================================================
"""
學習曲線和驗證曲線是強大的視覺化工具，用於診斷模型的「偏差」(Bias) 與
「方差」(Variance)，從而判斷模型是「欠擬合」還是「過擬合」。

- **偏差 (Bias)**：模型的預測值與真實值之間的系統性差異。高偏差（欠擬合）
  代表模型過於簡單，無法捕捉資料的複雜規律。
- **方差 (Variance)**：模型在不同訓練集上的表現穩定性。高方差（過擬合）
  代表模型過於複雜，對訓練資料的噪點過度敏感，導致泛化能力差。
"""
print("\n" + "="*60)
print("第六部分：學習曲線與驗證曲線")
print("="*60)

# --- 6.1 學習曲線 (Learning Curves) ---
# 學習曲線展示了模型的效能隨著「訓練樣本數量」增加而變化的情況。
# 我們會同時繪製「訓練分數」和「交叉驗證分數」。
#
# **如何解讀學習曲線？**
# 1.  **高偏差 (欠擬合)**：
#     - 訓練分數和驗證分數都很低。
#     - 兩條曲線很早就收斂在一起。
#     - **解法**：增加模型複雜度（如：使用更高次的特徵）、減少正則化、獲取更多相關特徵。增加更多訓練樣本通常無效。
# 2.  **高方差 (過擬合)**：
#     - 訓練分數很高，但驗證分數很低。
#     - 兩條曲線之間有很大的差距 (Gap)。
#     - **解法**：增加訓練樣本、降低模型複雜度、增加正則化、進行特徵選擇。

# 為了演示，我們先對資料進行標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 learning_curve 函數
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=SVC(kernel='linear', random_state=42),
    X=X_scaled,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10), # 從 10% 到 100% 的訓練樣本
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 計算平均值和標準差
train_scores_mean = np.mean(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="訓練分數 (Training score)")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="交叉驗證分數 (Cross-validation score)")
plt.title("學習曲線 (Learning Curve)")
plt.xlabel("訓練樣本數")
plt.ylabel("準確度分數")
plt.legend(loc="best")
plt.grid(True)
plt.show()

print("\n學習曲線繪製完成。請觀察圖表：如果兩條線收斂且分數不高，可能為欠擬合；如果兩條線差距很大，可能為過擬合。")


# --- 6.2 驗證曲線 (Validation Curves) ---
# 驗證曲線展示了模型的效能隨著「單一超參數」值的變化而變化的情況。
# 這有助於我們為模型找到最佳的超參數設定。
#
# **如何解讀驗證曲線？**
# - 觀察交叉驗證分數的曲線，找到其達到峰值的點，該點對應的超參數值通常是最佳選擇。
# - 如果訓練分數很高，但驗證分數很低，表示在該超參數設定下模型可能過擬合。

# 這裡我們以 SVC 的 `gamma` 參數為例 (需使用 'rbf' 核心)
# `gamma` 控制了單一樣本的影響範圍，值越大越容易過擬合
param_range = np.logspace(-6, -1, 6) # 產生 10^-6 到 10^-1 的 6 個數值

train_scores_vc, validation_scores_vc = validation_curve(
    estimator=SVC(kernel='rbf', random_state=42),
    X=X_scaled,
    y=y,
    param_name='gamma', # 要調整的超參數名稱
    param_range=param_range, # 超參數的範圍
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 計算平均值和標準差
train_scores_vc_mean = np.mean(train_scores_vc, axis=1)
validation_scores_vc_mean = np.mean(validation_scores, axis=1)

# 繪製驗證曲線
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_scores_vc_mean, 'o-', color="r", label="訓練分數 (Training score)")
plt.semilogx(param_range, validation_scores_vc_mean, 'o-', color="g", label="交叉驗證分數 (Cross-validation score)")
plt.title("驗證曲線 (Validation Curve for SVC 'gamma')")
plt.xlabel("Gamma 參數")
plt.ylabel("準確度分數")
plt.legend(loc="best")
plt.grid(True)
plt.show()

print("\n驗證曲線繪製完成。請觀察圖表：交叉驗證分數最高的點，其對應的 gamma 值是較好的選擇。")


# =============================================================================
# 7. 總結
# =============================================================================
"""
恭喜您完成了本次模型評估的教學！

### 本次教學核心回顧：
1.  **評估的重要性**：模型評估是確保模型可靠性與實用性的基石。
2.  **基礎評估技術**：Train-Test Split 是最基本的步驟，用以模擬模型在未知資料上的表現。
3.  **分類指標**：
    - **Accuracy**：直觀但要小心類別不平衡陷阱。
    - **Confusion Matrix**：提供模型預測行為的完整視圖 (TP, FP, FN, TN)。
    - **Precision, Recall, F1-Score**：在不同業務場景下，權衡「誤判」與「漏判」的代價。
    - **ROC/AUC**：評估模型整體排序與區分能力的強大工具，不受類別不平衡影響。
4.  **迴歸指標**：
    - **MAE, MSE, RMSE**：衡量預測值與真實值差距的常用指標。
    - **R-squared**：衡量模型對資料變異性的解釋能力。
5.  **進階評估技術**：
    - **Cross-Validation**：提供更穩定、更可靠的評估結果。
    - **Learning & Validation Curves**：診斷模型「欠擬合」與「過擬合」問題，並指導超參數調優的利器。

### 後續學習建議：
- 嘗試將這些評估技術應用到您自己的專案中。
- 針對不同的問題，思考哪種評估指標最為合適。
- 學習更多進階的交叉驗證策略，如 StratifiedKFold、GroupKFold 等。
- 探索更多針對不平衡資料的評估指標，如 PR-AUC (Precision-Recall AUC)。

模型評估是一個反覆迭代的過程，透過不斷的評估、診斷與優化，才能打造出
真正強大且可靠的機器學習模型。
"""
print("\n" + "="*60)
print("教學結束！希望這份腳本對您有幫助。")
print("="*60)
