
"""
=================================================================================================

 Python 機器學習資料前處理完整教學

 您好！這是一份專為學習者設計的 Python 資料前處理教學腳本。
 作為一位資深機器學習工程師，我將引導您掌握資料前處理中最核心的四大技能。
 這份腳本整合了理論、實作與註解，旨在讓您能循序漸進地理解並應用這些關鍵技術。

 四大核心環節：
 1. 分塊處理 (Chunking): 學習如何高效處理無法一次載入記憶體的大型資料集。
 2. 重複值處理 (Duplicate Value Handling): 學習識別並管理資料中的重複記錄。
 3. 資料型態轉換 (Data Type Conversion): 學習將資料轉換為正確的格式以利分析與模型訓練。
 4. 文字欄位清理 (Text Field Cleaning): 學習對非結構化的文字資料進行標準化。

=================================================================================================
"""

import pandas as pd
import numpy as np
import string
import re
import os

# --- 準備工作：創建一個模擬的資料夾與大型資料檔案 ---
# 為了讓所有範例都能順利運行，我們先創建一個名為 'temp_data' 的資料夾，
# 並在其中生成一個模擬的 CSV 檔案。
if not os.path.exists('temp_data'):
    os.makedirs('temp_data')

# 創建一個模擬的大型 CSV 檔案
mock_csv_path = 'temp_data/mock_large_dataset.csv'
mock_data_size = 1000  # 模擬 1000 筆資料
mock_df = pd.DataFrame({
    'PassengerId': np.arange(1, mock_data_size + 1),
    'Age': np.random.randint(1, 80, size=mock_data_size),
    'Fare': np.random.uniform(5, 500, size=mock_data_size),
    'Survived': np.random.randint(0, 2, size=mock_data_size)
})
mock_df.to_csv(mock_csv_path, index=False)


# =================================================================================================
#  環節一：分塊處理 (Chunking)
# =================================================================================================

# ---
#  1.1 理論說明：為何需要分塊處理？
# ---
# 在真實世界的資料分析場景中，我們時常會遇到比電腦 RAM 還要大的資料集（例如，數十 GB 的 CSV 檔案）。
# 若嘗試一次性將整個檔案讀入一個 DataFrame，會導致 `MemoryError`，使分析無法進行。
#
# Pandas 提供了 `chunksize` 這個強大的參數，讓我們可以將大型檔案像處理串流一樣，
# 一次只讀取一小部分（一個 "chunk"）到記憶體中，對其進行處理後，再讀取下一個部分。
# 這種方法是處理大數據時不可或缺的基礎技能。

print("================= 環節一：分塊處理 (Chunking) =================\n")

# ---
#  1.2 程式碼實作：分塊讀取與計算
# ---
# 假設我們想計算 `mock_large_dataset.csv` 中乘客的平均票價 (Fare)，但檔案太大無法一次讀取。
# 我們可以分塊讀取，計算每個 chunk 的票價總和與人數，最後再將它們合併計算總平均值。

try:
    # 設定 chunksize，例如每次讀取 200 筆資料
    chunk_size = 200
    # 創建一個迭代器 (Iterator)
    chunk_iterator = pd.read_csv(mock_csv_path, chunksize=chunk_size)

    # 初始化變數來儲存累計值
    total_fare = 0
    total_passengers = 0

    print(f"開始分塊處理檔案: {mock_csv_path}, 每塊 {chunk_size} 筆資料。\n")

    # 遍歷每個 chunk
    for i, chunk in enumerate(chunk_iterator):
        print(f"--- 正在處理 Chunk {i+1} ---")
        print(f"Chunk 的維度: {chunk.shape}")

        # 累加票價總和與乘客數量
        total_fare += chunk['Fare'].sum()
        total_passengers += len(chunk)

    # 計算總平均票價
    average_fare = total_fare / total_passengers if total_passengers > 0 else 0
    print("\n分塊計算完成！")
    print(f"處理的總乘客數: {total_passengers}")
    print(f"分塊計算得到的平均票價: {average_fare:.2f}")

    # 為了驗證，我們一次性讀取並計算真實平均值
    full_df_check = pd.read_csv(mock_csv_path)
    true_average_fare = full_df_check['Fare'].mean()
    print(f"一次性讀取計算的真實平均票價: {true_average_fare:.2f}\n")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 {mock_csv_path}")


# =================================================================================================
#  環節二：重複值處理 (Duplicate Value Handling)
# =================================================================================================

# ---
#  2.1 理論說明：為何要處理重複值？
# ---
# 重複的資料記錄是資料清理中常見的問題。它們可能源於資料收集過程的錯誤、系統 bug 或是資料合併不當。
# 如果不加以處理，重複值會：
# - 扭曲統計分析結果：例如，重複的銷售記錄會誇大總銷售額。
# - 引入模型偏見：模型可能會過度學習這些重複的樣本，導致泛化能力下降。
# - 造成資料洩漏：如果重複的資料不慎同時出現在訓練集和測試集中，會導致模型評估結果過於樂觀。
# 因此，識別並恰當地處理重複值是確保資料品質的重要一步。

print("================= 環節二：重複值處理 =================\n")

# ---
#  2.2 程式碼實作：識別與移除重複值
# ---

# 創建一個帶有重複值的範例 DataFrame
data_duplicates = {
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie', 'Yum Yum'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack', 'cup'],
    'rating': [4, 4, 3.5, 1, 5, 4]
}
df_dup = pd.DataFrame(data_duplicates)
print("原始 DataFrame (含重複值):")
print(df_dup)
print("\n")

# 1. 識別重複值 .duplicated()
# .duplicated() 返回一個布林 Series，標示每一行是否為重複行。
# 預設情況下，除了第一次出現的記錄外，其餘相同的記錄都會被標記為 True。
duplicates_mask = df_dup.duplicated()
print("使用 .duplicated() 識別重複行 (True 表示重複):")
print(duplicates_mask)
print("\n顯示所有被標記為重複的行:")
print(df_dup[duplicates_mask])
print("\n")

# 2. 移除重複值 .drop_duplicates()
# 預設 keep='first'，保留第一次出現的記錄。
df_no_dup_first = df_dup.drop_duplicates(keep='first')
print("移除重複項後 (保留第一筆):")
print(df_no_dup_first)
print("\n")

# 使用 keep='last'，保留最後一次出現的記錄。
df_no_dup_last = df_dup.drop_duplicates(keep='last')
print("移除重複項後 (保留最後一筆):")
print(df_no_dup_last)
print("\n")

# 3. 基於特定欄位判斷重複
# 使用 `subset` 參數來指定用於判斷重複的欄位子集。
# 例如，我們認為 'brand' 和 'style' 的組合是唯一的。
df_subset_dup = df_dup.drop_duplicates(subset=['brand', 'style'], keep='first')
print("基於 'brand' 和 'style' 移除重複項後:")
print(df_subset_dup)
print("\n")

# =================================================================================================
#  環節三：資料型態轉換 (Data Type Conversion)
# =================================================================================================

# ---
#  3.1 理論說明：為何資料型態很重要？
# ---
# 不正確的資料型態會導致多種問題：
# - 計算錯誤：將數字儲存為字串 (`object`) 會導致無法進行數學運算。
# - 記憶體浪費：使用通用 `object` 型態來儲存純數字或類別資料，會佔用遠超必要的記憶體空間。
# - 模型不相容：大多數機器學習模型無法直接處理字串型態的資料。
# - 分析功能受限：例如，如果日期被存為字串，就無法進行基於時間的篩選或特徵提取。
# 確保每個欄位都擁有最適合其內容的資料型態，是資料清理流程中的一個核心任務。

print("================= 環節三：資料型態轉換 =================\n")

# ---
#  3.2 程式碼實作：轉換資料型態
# ---

# 創建一個資料型態混雜的範例 DataFrame
data_types = {
    'OrderID': ['101', '102', '103', '104'],
    'OrderDate': ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
    'Price': ['19.99', '25.00', '15.50', '30.00'],
    'Quantity': ['2', '1', '3', 'invalid'], # 包含一個無效值
    'Category': ['A', 'B', 'A', 'C']
}
df_types = pd.DataFrame(data_types)
print("原始 DataFrame 資訊:")
df_types.info()
print("\n")

# 1. 基本轉換 .astype()
# 將 'OrderID' 轉為整數，'Price' 轉為浮點數
df_types['OrderID'] = df_types['OrderID'].astype(int)
df_types['Price'] = df_types['Price'].astype(float)

# 2. 轉換為 Category 型態 (優化記憶體)
# 對於唯一值數量有限的欄位，轉換為 'category' 型態是很好的實踐。
original_mem = df_types['Category'].memory_usage(deep=True)
df_types['Category'] = df_types['Category'].astype('category')
categorical_mem = df_types['Category'].memory_usage(deep=True)
print(f"'Category' 欄位記憶體使用：從 {original_mem} bytes 降至 {categorical_mem} bytes。\n")

# 3. 使用 pd.to_numeric() 處理錯誤
# 對含有非數值資料的欄位，直接用 .astype(int) 會報錯。
# `pd.to_numeric` 提供了 `errors='coerce'` 參數，可將無效值轉為 NaN (Not a Number)。
df_types['Quantity'] = pd.to_numeric(df_types['Quantity'], errors='coerce')
print("使用 pd.to_numeric(errors='coerce') 轉換 'Quantity' 後的值:")
print(df_types['Quantity'])
print("\n")

# 4. 使用 pd.to_datetime() 處理日期
# 這是將字串轉換為標準日期時間格式的專用函數。
df_types['OrderDate'] = pd.to_datetime(df_types['OrderDate'])
print("轉換後的 DataFrame 資訊:")
df_types.info()
print("\n")

# 轉換為 datetime 型態後，就可以輕鬆地進行日期相關操作。
df_types['Year'] = df_types['OrderDate'].dt.year
df_types['DayOfWeek'] = df_types['OrderDate'].dt.day_name()
print("從日期中提取的新特徵:")
print(df_types[['OrderDate', 'Year', 'DayOfWeek']].head())
print("\n")


# =================================================================================================
#  環節四：文字欄位清理 (Text Field Cleaning)
# =================================================================================================

# ---
#  4.1 理論說明：為何需要清理文字？
# ---
# 文字資料是非結構化的，充滿了各種「噪音」，例如：大小寫不一致、多餘的空白、標點符號等。
# 這些噪音會嚴重干擾基於文字的分析，例如計算詞頻、特徵提取或情緒分析。
# 因此，在進行自然語言處理（NLP）之前，對文字進行標準化和清理是至關重要的第一步。

print("================= 環節四：文字欄位清理 =================\n")

# ---
#  4.2 程式碼實作：清理文字資料
# ---

# 創建一個包含需要清理的文字資料的 DataFrame
data_text = {
    'ReviewID': [1, 2, 3, 4],
    'ReviewText': [
        '  This is a GREAT product! I love it. ',
        'terrible, would not recommend.',
        'Just... OK. Not bad, not good.',
        'AWESOME!!! 10/10'
    ]
}
df_text = pd.DataFrame(data_text)
print("原始 DataFrame (含髒亂文字):")
print(df_text)
print("\n")

# Pandas Series 提供 .str 存取器，可對整個 Series 進行向量化字串操作。
# 我們可以將多個清理步驟鏈接 (chain) 起來，讓程式碼更簡潔。

# 建立一個正則表達式，匹配所有標點符號
punct_regex = f"[{re.escape(string.punctuation)}]"

# 鏈式操作：1. 轉小寫 -> 2. 移除頭尾空白 -> 3. 移除標點符號
df_text['CleanedText'] = df_text['ReviewText'].str.lower().str.strip().str.replace(punct_regex, '', regex=True)

print("一次性鏈式操作清理結果:")
print(df_text[['ReviewText', 'CleanedText']])
print("\n")


# =================================================================================================
#  總結
# =================================================================================================

print("================= 學習總結 =================\n")
print("恭喜您完成了這份資料前處理教學！\n")
print("在本腳本中，我們系統性地學習了資料科學專案中最基礎也最重要的四個環節：")
print("1. 分塊處理 (Chunking): 使用 `pd.read_csv` 的 `chunksize` 參數，讓我們有能力處理超越記憶體極限的大型資料集。")
print("2. 重複值處理 (Duplicate Handling): 運用 `.duplicated()` 和 `.drop_duplicates()`，確保了資料的唯一性與分析的準確性。")
print("3. 資料型態轉換 (Data Type Conversion): 透過 `.astype()`, `pd.to_numeric()`, `pd.to_datetime()`，我們將資料轉換為正確的格式，不僅優化了記憶體，也為後續的特徵工程與模型建立打下堅實基礎。")
print("4. 文字欄位清理 (Text Field Cleaning): 利用 `.str` 存取器和鏈式操作，我們學會了如何快速地將雜亂的文字資料標準化，這是所有自然語言處理任務的起點。\n")
print("掌握這些前處理技術，是成為一位優秀資料科學家或機器學習工程師的必經之路。")
print("乾淨、高品質的資料，是產出可靠分析與強大模型的基石。\n")

# --- 清理臨時檔案 ---
try:
    os.remove(mock_csv_path)
    os.rmdir('temp_data')
    print("已清理臨時生成的資料檔案與資料夾。")
except OSError as e:
    print(f"清理臨時檔案時出錯: {e}")
