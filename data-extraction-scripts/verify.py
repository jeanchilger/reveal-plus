import csv
topics = []
count = 0
with open("topics.txt") as file:
    for line in file:
        tline = line.rstrip("\n")
        if tline not in topics:
            topics.append(tline)
        else:
            count = count + 1
            print(tline)
print(count)

csv_file = open("./html/metadata.csv", "r")
reader = csv.reader(csv_file)
count2 = 0
for row in reader:
    if row[0] in topics:
        topics.remove(row[0])
        count2 = count2 + 1
print(topics)
print(count2)
print(len(topics))