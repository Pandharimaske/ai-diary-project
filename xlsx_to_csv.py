import pandas as pd

# Load the Excel file
df = pd.read_excel("/Users/pandhari/ai-diary-project/Data/Renew_data.xlsx")

# Save it as a CSV file
df.to_csv("/Users/pandhari/ai-diary-project/Data/diary_dataset.csv", index=False)

print("Conversion completed! CSV file saved successfully.")