from sklearn.model_selection import train_test_split

lines = []

with open("para_score_big.jsonl") as file:
    for line in file:
        lines.append(line)

train, test = train_test_split(lines, train_size=0.9, random_state=42, shuffle=True)

with open("para_test_big.jsonl", "w") as file:
    for line in test:
        file.write(line)

with open("para_train_big.jsonl", "w") as file:
    for line in train:
        file.write(line)
