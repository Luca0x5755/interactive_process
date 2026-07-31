import pickle

# --- 保存訓練好的模型 ---
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ 模型已成功保存至 regression_model.pkl")

# --- 保存用於特徵縮放的 Scaler ---
# 注意：scaler 物件是在 6-0 章節中創建和擬合的
if 'scaler' in locals():
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler 物件已成功保存至 scaler.pkl")
else:
    print("❌ 錯誤：找不到名為 'scaler' 的物件。請確認您已執行 6-0 章節的程式碼。")
