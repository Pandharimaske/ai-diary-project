import pandas as pd

# Load the Excel file
df = pd.read_excel("C:/Users/anura/ai-diary-project/data/diary_dataset.xlsx")

# Save it as a CSV file
df.to_csv("C:/Users/anura/ai-diary-project/Data/diary_dataset.csv", index=False)

print("Conversion completed! CSV file saved successfully.")