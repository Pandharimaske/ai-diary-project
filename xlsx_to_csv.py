import pandas as pd


df = pd.read_excel("/Users/pandhari/ai-diary-project/Data/Renew_data.xlsx")


df.to_csv("/Users/pandhari/ai-diary-project/Data/diary_dataset.csv", index=False)

print("Conversion completed! CSV file saved successfully.")