import csv

with open("/Users/pandhari/ai-diary-project/Data/diary_dataset.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

with open("diary.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Date", "Entry"])

    for line in lines:
        if "," in line:
            date, entry = line.strip().split(",", 1)
            writer.writerow([date, entry])
