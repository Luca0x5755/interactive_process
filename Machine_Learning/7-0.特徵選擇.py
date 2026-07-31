# -*- coding: utf-8 -*-
"""
==============================================================================
# Python 機器學習實戰：特徵選擇 (Feature Selection) 全方位教學腳本
==============================================================================

### 教學目標：
1.  **理解核心概念**：深入理解特徵選擇的定義、目的，以及它為何是建構高效、可解釋機器學習模型的關鍵環節。
2.  **掌握三大技術**：學習並實作三種主流的特徵選擇技術：過濾法 (Filter Methods)、包裹法 (Wrapper Methods) 與嵌入法 (Embedded Methods)。
3.  **實作導向學習**：掌握如何在 Python 中利用 Scikit-learn 等主流函式庫，將特徵選擇技術無縫整合到完整的機器學習模型建構與評估流程中。
4.  **策略比較與應用**：透過綜合案例比較不同方法的優劣，學習如何在真實專案中，根據問題需求與資源限制，選擇最合適的特徵選擇策略。

### 適用對象：
- 對機器學習與資料科學有基礎認識，希望深化特徵工程技能的學習者。
- 尋求提升模型效能、降低複雜度與增強模型可解釋性的資料科學家與工程師。
- 準備機器學習相關面試，需要系統性整理特徵選擇知識的求職者。
"""

# =============================================================================
# 0. 導入必要套件
# =============================================================================
# 在開始之前，讓我們先導入本次教學會用到的核心套件
# 基礎資料處理與數值計算
import pandas as pd
import numpy as np

# 資料視覺化
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn 資料預處理與模型評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Scikit-learn 特徵選擇工具
# --- 過濾法 (Filter Methods) ---
from sklearn.feature_selection import SelectKBest, f_classif, chi2
# --- 包裹法 (Wrapper Methods) ---
from sklearn.feature_selection import RFE, RFECV
# --- 嵌入法 (Embedded Methods) ---
from sklearn.feature_selection import SelectFromModel

# Scikit-learn 機器學習模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 關閉未來版本警告，保持輸出簡潔
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 設定視覺化風格與中文字體
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("套件導入成功！準備開始學習特徵選擇的藝術。")


# =============================================================================
# 1. 主題介紹：為什麼特徵選擇至關重要？
# =============================================================================
"""
在真實世界的機器學習專案中，我們面對的資料集往往包含大量特徵。然而，並非所有
特徵都對我們的預測目標有同等的貢獻。有些特徵可能是**冗餘的 (Redundant)**，
例如「房屋面積(平方公尺)」和「房屋面積(平方英尺)」；有些則可能是**無關的
(Irrelevant)**，例如在房價預測中，「屋主的星座」可能就與房價無關。

**特徵選擇 (Feature Selection)** 的核心目的，就是從原始特徵集中，自動化地
篩選出一個最具預測能力的特徵子集。這個過程是「降維 (Dimensionality Reduction)」
的一種形式，旨在應對所謂的「維度災難 (Curse of Dimensionality)」。
在高維度空間中，資料點會變得非常稀疏，這不僅大幅增加了模型的計算成本，也容易
導致模型學習到資料中的噪音，產生**過擬合 (Overfitting)**。

一個精心設計的特徵選擇流程能帶來三大核心優勢：
1.  **提升模型泛化能力**：移除無關和冗餘的特徵能降低模型複雜度，減少噪音干擾，
    從而降低過擬合風險，使模型在未見過的資料上表現更佳。
2.  **提高計算效率**：更少的特徵意味著模型需要更短的訓練時間和更少的記憶體，
    這在處理大規模資料集時尤為重要。
3.  **增強模型可解釋性**：一個只包含最關鍵特徵的簡潔模型，更容易被人類理解、
    解釋和信任，這在金融、醫療等高風險領域至關重要。

本教學將系統性地介紹三種主流的特徵選擇策略，帶您一步步掌握如何為您的模型
挑選出「黃金特徵」。
"""
print("\n" + "="*70)
print("第一部分：主題介紹 - 為何特徵選擇是模型成功的基石")
print("="*70)


# =============================================================================
# 2. 資料準備：鐵達尼號生存預測
# =============================================================================
"""
為了貫穿所有範例，我們將使用經典的「鐵達尼號」資料集。這個資料集的目標是根據
乘客的個人資訊（如年齡、性別、艙等）來預測其是否生還。它包含了數值與類別
特徵，是演示特徵選擇各種技術的絕佳範例。
"""
# 載入資料集
try:
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    print("鐵達尼號資料集下載成功！")
except Exception as e:
    print(f"資料下載失敗: {e}")
    print("請檢查您的網路連線。")
    # 創建一個空的 DataFrame 以避免後續錯誤
    df = pd.DataFrame()

if not df.empty:
    # 進行必要的預處理，為特徵選擇做準備
    # 1. 選擇我們感興趣的欄位
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    # 2. 填補缺失值
    # 年齡(Age)使用中位數填補
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # 登船港口(Embarked)使用眾數填補
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # 3. 編碼類別變數
    # 性別(Sex): female -> 1, male -> 0
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    # 登船港口(Embarked)進行 One-Hot Encoding
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=True, dtype=int)

    # 4. 定義特徵 (X) 與目標 (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 5. 資料標準化 (對正規化、PCA等方法很重要)
    # 建立一個標準化器物件
    scaler = StandardScaler()
    # 在特徵資料上擬合並轉換
    X_scaled = scaler.fit_transform(X)
    # 轉換回 DataFrame 以保留欄位名稱
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 6. 分割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    print("\n--- 資料準備完成 ---")
    print("預處理後的特徵 (X) 預覽:")
    print(X.head())
    print("\n標準化後的特徵 (X_scaled) 預覽:")
    print(X_scaled.head())


# =============================================================================
# 3. 過濾法 (Filter Methods)
# =============================================================================
"""
過濾法完全獨立於任何機器學習模型，它僅根據特徵本身的統計特性來評估其與目標
變數之間的關聯性，從而進行篩選。

- **優點**：計算速度極快，因為不涉及模型訓練；選出的特徵不偏好任何特定模型，
  通用性強。
- **缺點**：忽略了特徵之間的交互作用。例如，兩個特徵單獨看可能不重要，但組合
  起來卻可能非常有預測力。

常用方法：
- **ANOVA F-檢定 (f_classif)**: 用於評估「數值特徵」與「類別目標」之間的關聯性。
- **卡方檢定 (Chi-square test)**: 用於評估「類別特徵」與「類別目標」之間的關聯性。
"""
print("\n" + "="*70)
print("第三部分：過濾法 (Filter Methods) - 快速篩選特徵")
print("="*70)

if not df.empty:
    # 我們使用 SelectKBest 來選擇分數最高的 K 個特徵
    K = 5

    # 3.1 使用 ANOVA F-檢定 (f_classif)
    # 適用於數值特徵，但這裡我們對所有特徵進行計算以作比較
    print("\n--- 3.1 ANOVA F-test (f_classif) ---")
    selector_f = SelectKBest(score_func=f_classif, k=K)
    # 在訓練集上擬合選擇器
    X_train_f_selected = selector_f.fit_transform(X_train_scaled, y_train)

    # 獲取被選擇的特徵名稱與分數
    f_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'F-Score': selector_f.scores_,
        'P-Value': selector_f.pvalues_
    }).sort_values(by='F-Score', ascending=False)

    selected_features_f = X_train.columns[selector_f.get_support()]

    print(f"使用 ANOVA F-test 選擇的前 {K} 個特徵是: {selected_features_f.tolist()}")
    print("各特徵的 F-分數 (越高越好) 與 P-值 (越低越好):")
    print(f_scores.round(4))

    # 3.2 使用卡方檢定 (Chi-square)
    # 要求特徵值為非負數，我們的原始資料(One-Hot編碼後)符合此要求。
    # 卡方檢定更適用於類別特徵。
    print("\n--- 3.2 卡方檢定 (Chi-square) ---")
    # 注意：卡方檢定應在非負資料上進行，此處使用未標準化的 X_train
    selector_chi2 = SelectKBest(score_func=chi2, k=K)
    X_train_chi2_selected = selector_chi2.fit_transform(X_train, y_train)

    # 獲取被選擇的特徵名稱與分數
    chi2_scores_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Chi2-Score': selector_chi2.scores_
    }).sort_values(by='Chi2-Score', ascending=False)

    selected_features_chi2 = X_train.columns[selector_chi2.get_support()]

    print(f"使用 Chi-Square 選擇的前 {K} 個特徵是: {selected_features_chi2.tolist()}")
    print("各特徵的 Chi-Square 分數 (越高越相關):")
    print(chi2_scores_df.round(2))

# =============================================================================
# 4. 包裹法 (Wrapper Methods)
# =============================================================================
"""
包裹法將特徵選擇視為一個搜尋問題，它使用一個特定的機器學習模型來評估不同
特徵子集的效能，從而「包裹」住模型進行選擇。

- **優點**：直接針對特定模型的效能進行優化，能夠考慮特徵之間的交互作用，
  通常能找到比過濾法更好的特徵子集。
- **缺點**：計算成本非常高，因為需要反覆訓練模型。當特徵數量龐大時，會非常
  耗時，且有過擬合的風險。

常用方法：
- **遞歸特徵消除 (Recursive Feature Elimination, RFE)**:
  從所有特徵開始，反覆訓練模型、移除最不重要的特徵，直到剩下指定的特徵數量。
  `RFECV` 版本更可以透過交叉驗證自動找到最佳的特徵數量。
"""
print("\n" + "="*70)
print("第四部分：包裹法 (Wrapper Methods) - 追求極致效能")
print("="*70)

if not df.empty:
    # 4.1 使用 RFECV 自動選擇最佳特徵
    # 我們使用一個簡單的邏輯斯迴歸模型作為評估器
    print("\n--- 4.1 遞歸特徵消除 (RFECV) ---")
    print("正在執行 RFECV，這將自動尋找最佳特徵數量，可能需要一些時間...")

    model_lr = LogisticRegression(solver='liblinear', random_state=42)
    # RFECV 會透過交叉驗證來找到最佳的特徵數
    rfecv = RFECV(
        estimator=model_lr,
        step=1,                 # 每次迭代移除一個特徵
        cv=5,                   # 5折交叉驗證
        scoring='accuracy',     # 以準確率為評估指標
        n_jobs=-1               # 使用所有CPU核心
    )

    # 在標準化後的訓練集上擬合
    rfecv.fit(X_train_scaled, y_train)

    print("RFECV 完成！")
    print(f"RFECV 找到的最佳特徵數量: {rfecv.n_features_}")

    # 獲取選擇的特徵
    selected_features_rfecv = X_train.columns[rfecv.support_]
    print(f"RFECV 選擇的最佳特徵: {selected_features_rfecv.tolist()}")

    # 視覺化交叉驗證分數
    plt.figure(figsize=(10, 6))
    plt.xlabel("選擇的特徵數量")
    plt.ylabel("交叉驗證分數 (Accuracy)")
    # `rfecv.cv_results_['mean_test_score']` 在 scikit-learn > 1.2 中可用
    # 為了兼容性，使用 grid_scores_
    scores = rfecv.cv_results_['mean_test_score'] if hasattr(rfecv, 'cv_results_') else rfecv.grid_scores_
    plt.plot(range(1, len(scores) + 1), scores, marker='o')
    plt.title("遞歸特徵消除與交叉驗證 (RFECV)")
    plt.grid(True)
    plt.show()

# =============================================================================
# 5. 嵌入法 (Embedded Methods)
# =============================================================================
"""
嵌入法將特徵選擇過程「嵌入」到模型訓練的過程中。模型在學習的同時，會自動
為特徵賦予權重或重要性，並將不重要的特徵篩選掉。

- **優點**：效率比包裹法高，因為只訓練一次模型；同時也考慮了特徵的交互作用，
  是效果與效率的良好平衡。
- **缺點**：選出的特徵與所使用的模型強烈相關。更換模型可能需要重新進行特徵選擇。

常用方法：
- **L1 正規化 (Lasso)**: 透過懲罰項將不重要特徵的係數壓縮至零，實現自動特徵選擇。
- **基於樹的模型 (如隨機森林)**: 在訓練後可提供每個特徵的重要性分數，據此進行篩選。
"""
print("\n" + "="*70)
print("第五部分：嵌入法 (Embedded Methods) - 效率與效果的平衡")
print("="*70)

if not df.empty:
    # 5.1 使用 L1 正規化 (Lasso) 進行特徵選擇
    print("\n--- 5.1 L1 正規化 (Lasso) ---")
    # 我們使用帶有 L1 懲罰的邏輯斯迴歸
    # C 是正規化強度的倒數，C越小，懲罰越強，留下的特徵越少。
    l1_lr = LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=42)
    l1_lr.fit(X_train_scaled, y_train)

    # 使用 SelectFromModel 根據係數來選擇特徵
    # '1e-5' 是一個閾值，係數絕對值小於此值的特徵將被移除
    sfm_l1 = SelectFromModel(l1_lr, prefit=True, threshold=1e-5)

    selected_features_l1 = X_train.columns[sfm_l1.get_support()]
    print(f"L1 正規化選擇了 {len(selected_features_l1)} 個特徵: {selected_features_l1.tolist()}")

    # 視覺化特徵係數
    coefficients = pd.Series(l1_lr.coef_[0], index=X_train.columns)
    plt.figure(figsize=(10, 6))
    coefficients.abs().sort_values().plot(kind='barh')
    plt.title("L1 正規化邏輯斯迴歸的特徵係數 (絕對值)")
    plt.xlabel("係數絕對值 (非零者被選中)")
    plt.show()


    # 5.2 使用隨機森林的特徵重要性
    print("\n--- 5.2 隨機森林特徵重要性 ---")
    # 訓練一個隨機森林分類器
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # 樹模型對尺度不敏感，可直接用原始資料訓練
    model_rf.fit(X_train, y_train)

    # 使用 SelectFromModel 根據特徵重要性來選擇
    # threshold='median' 表示選擇重要性高於中位數的特徵
    sfm_rf = SelectFromModel(model_rf, prefit=True, threshold='median')

    selected_features_rf = X_train.columns[sfm_rf.get_support()]
    print(f"隨機森林選擇了 {len(selected_features_rf)} 個特徵: {selected_features_rf.tolist()}")

    # 視覺化特徵重要性
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances)
    plt.title('隨機森林計算的特徵重要性')
    plt.xlabel('重要性分數')
    plt.ylabel('特徵')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. 綜合案例：比較不同特徵選擇方法對模型效能的影響
# =============================================================================
"""
理論學完了，現在是見真章的時候！我們將比較使用不同方法選擇出的特徵子集，
在同一個模型（邏輯斯迴歸）上的表現如何。這能幫助我們直觀地理解不同策略的
實際效果。
"""
print("\n" + "="*70)
print("第六部分：綜合案例 - 誰是最佳特徵選擇策略？")
print("="*70)

if not df.empty:
    results = {}

    # 基準模型：使用所有特徵
    lr_baseline = LogisticRegression(solver='liblinear', random_state=42)
    lr_baseline.fit(X_train_scaled, y_train)
    y_pred_base = lr_baseline.predict(X_test_scaled)
    acc_base = accuracy_score(y_test, y_pred_base)
    results[f'基準模型 ({X_train_scaled.shape[1]} 特徵)'] = acc_base

    # 1. 過濾法 (ANOVA F-test) 選出的特徵
    X_train_f_sel = selector_f.transform(X_train_scaled)
    X_test_f_sel = selector_f.transform(X_test_scaled)
    lr_filter = LogisticRegression(solver='liblinear', random_state=42)
    lr_filter.fit(X_train_f_sel, y_train)
    y_pred_filter = lr_filter.predict(X_test_f_sel)
    acc_filter = accuracy_score(y_test, y_pred_filter)
    results[f'過濾法 (ANOVA, {X_train_f_sel.shape[1]} 特徵)'] = acc_filter

    # 2. 包裹法 (RFECV) 選出的特徵
    X_train_wrapper = rfecv.transform(X_train_scaled)
    X_test_wrapper = rfecv.transform(X_test_scaled)
    lr_wrapper = LogisticRegression(solver='liblinear', random_state=42)
    lr_wrapper.fit(X_train_wrapper, y_train)
    y_pred_wrapper = lr_wrapper.predict(X_test_wrapper)
    acc_wrapper = accuracy_score(y_test, y_pred_wrapper)
    results[f'包裹法 (RFECV, {X_train_wrapper.shape[1]} 特徵)'] = acc_wrapper

    # 3. 嵌入法 (L1) 選出的特徵
    X_train_embedded_l1 = sfm_l1.transform(X_train_scaled)
    X_test_embedded_l1 = sfm_l1.transform(X_test_scaled)
    lr_embedded_l1 = LogisticRegression(solver='liblinear', random_state=42)
    lr_embedded_l1.fit(X_train_embedded_l1, y_train)
    y_pred_embedded_l1 = lr_embedded_l1.predict(X_test_embedded_l1)
    acc_embedded_l1 = accuracy_score(y_test, y_pred_embedded_l1)
    results[f'嵌入法 (L1, {X_train_embedded_l1.shape[1]} 特徵)'] = acc_embedded_l1

    # 4. 嵌入法 (Random Forest) 選出的特徵
    # 注意：這裡要轉換標準化後的資料
    X_train_rf_sel = X_train_scaled[selected_features_rf]
    X_test_rf_sel = X_test_scaled[selected_features_rf]
    lr_embedded_rf = LogisticRegression(solver='liblinear', random_state=42)
    lr_embedded_rf.fit(X_train_rf_sel, y_train)
    y_pred_embedded_rf = lr_embedded_rf.predict(X_test_rf_sel)
    acc_embedded_rf = accuracy_score(y_test, y_pred_embedded_rf)
    results[f'嵌入法 (RF, {X_train_rf_sel.shape[1]} 特徵)'] = acc_embedded_rf

    # 視覺化比較結果
    results_df = pd.DataFrame(pd.Series(results, name='Accuracy')).sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette='viridis')
    plt.ylabel("模型準確率 (Accuracy Score)", fontsize=12)
    plt.xlabel("特徵選擇方法", fontsize=12)
    plt.title("不同特徵選擇方法對模型準確率的影響", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0.7, 0.85)

    for index, row in results_df.iterrows():
        plt.text(x=index, y=row['Accuracy'] + 0.002, s=f"{row['Accuracy']:.4f}",
                 ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

    print("\n各方法準確率排名：")
    print(results_df)

# =============================================================================
# 7. 結論：迭代的藝術與 MLOps 實踐
# =============================================================================
"""
從綜合比較中我們可以看到，並非特徵越多模型效果就越好。經過特徵選擇後，
我們可以用更少的特徵，達到與基準模型相當甚至更好的性能。

**最終建議：如何選擇特徵工程策略？**

特徵選擇並非一個線性的單向過程，而是一個高度迭代的循環。在真實專案中，
最佳策略往往是迭代地嘗試多種方法，並根據業務目標（如準確率、模型大小、
解釋性等）來權衡。

- **追求最高模型效能與精簡性**：當計算資源允許時，**包裹法 (如 RFECV)** 和
  **嵌入法 (如 L1、基於樹的方法)** 通常是最佳選擇。它們能找到與特定模型最匹配、
  最具預測力的特徵子集。

- **計算資源有限或需要快速原型驗證**：**過濾法** 提供了一個快速且有效的起點，
  能夠迅速篩掉明顯不相關的特徵，特別適用於高維資料的初步篩選。

- **處理高度共線性或希望保留大部分資訊**：除了特徵選擇，**降維技術 (如 PCA)**
  也是一個優秀的選項。它能將原始特徵組合為新的、數量更少的特徵，有效處理
  共線性問題，但代價是新特徵會失去原始的直觀可解釋性。

這個從資料準備、方法應用到效能比較的完整流程，正是現代機器學習運維
(MLOps) 精神的體現：系統化、可重複、且以數據驅動決策。

"""

print("\n--- 特徵選擇教學腳本執行完畢 ---")
