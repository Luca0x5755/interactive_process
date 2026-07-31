import pickle
import argparse
import pandas as pd
import numpy as np

# 這些是我們在訓練時最終選擇的特徵順序，必須保持一致
FINAL_MODEL_COLUMNS = [
    'age', 'bmi', 'children', 'is_obese', 'age^2', 'age bmi', 'bmi^2',
    'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest'
]

# 這些是在縮放時使用的特徵，也必須保持一致
FEATURES_TO_SCALE = ['age', 'bmi', 'children', 'age^2', 'age bmi', 'bmi^2']

def prepare_features(input_data):
    """
    將原始輸入轉換為模型所需的特徵格式。
    這一步驟必須完全複製訓練時的特徵工程流程。
    """
    # 1. 創建一個 DataFrame
    df = pd.DataFrame([input_data])

    # 2. 特徵創造 (來自 4-0 章節)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['age^2'] = df['age'] ** 2
    df['age bmi'] = df['age'] * df['bmi']
    df['bmi^2'] = df['bmi'] ** 2

    # 3. 特徵編碼 (來自 5-0 章節)
    df['sex_male'] = (df['sex'] == 'male').astype(int)
    df['smoker_yes'] = (df['smoker'] == 'yes').astype(int)

    # 處理 region (獨熱編碼)
    df['region_northwest'] = (df['region'] == 'northwest').astype(int)
    df['region_southeast'] = (df['region'] == 'southeast').astype(int)
    df['region_southwest'] = (df['region'] == 'southwest').astype(int)

    # 4. 確保欄位順序與訓練時完全一致
    df = df[FINAL_MODEL_COLUMNS]

    return df

def predict(input_data):
    """
    載入模型和 Scaler，並執行完整的預測流程。
    """
    # 載入模型和 Scaler
    try:
        with open('regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("❌ 錯誤：找不到 'regression_model.pkl' 或 'scaler.pkl'。請先執行保存資產的步驟。")
        return None

    # 1. 準備特徵
    features_df = prepare_features(input_data)

    # 2. 特徵縮放 (來自 6-0 章節)
    # 使用 *已保存* 的 scaler 來轉換新資料
    features_df[FEATURES_TO_SCALE] = scaler.transform(features_df[FEATURES_TO_SCALE])

    # 3. 執行預測
    prediction = model.predict(features_df)

    return prediction[0]


if __name__ == '__main__':
    # 使用 argparse 建立命令列參數解析器
    parser = argparse.ArgumentParser(description='根據個人資料預測醫療費用')

    # 定義所有必要的輸入參數
    parser.add_argument('--age', type=int, required=True, help='年齡 (例如: 45)')
    parser.add_argument('--sex', type=str, choices=['male', 'female'], required=True, help='性別 (male 或 female)')
    parser.add_argument('--bmi', type=float, required=True, help='身體質量指數 (例如: 28.5)')
    parser.add_argument('--children', type=int, required=True, help='子女人數 (例如: 2)')
    parser.add_argument('--smoker', type=str, choices=['yes', 'no'], required=True, help='是否抽菸 (yes 或 no)')
    parser.add_argument('--region', type=str, choices=['southwest', 'southeast', 'northwest', 'northeast'], required=True, help='居住地區')

    # 解析參數
    args = parser.parse_args()

    # 將參數轉換為字典格式
    input_data = vars(args)

    # 執行預測
    estimated_charge = predict(input_data)

    # 印出結果
    if estimated_charge is not None:
        print("\n--- 醫療費用預測結果 ---")
        print(f"預估費用為: ${estimated_charge:,.2f}")
