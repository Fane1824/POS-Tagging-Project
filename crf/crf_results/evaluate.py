from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def confusion_matrix(actual, predicted):
    tags= list(set(actual))
    tags.sort()
    c= pd.DataFrame(np.zeros((len(tags), len(tags))), index=tags, columns=tags)
    for i in range(len(actual)):
        if actual[i] in c.index and predicted[i] in c.columns:
            c.loc[actual[i], predicted[i]]+= 1
    return c

actual_tags = []
model_tags = []

valid_languages = ['english', 'hindi', 'french']
while True:
    lang = input("Which language data do you want to evaluate? (english, hindi, french): ").lower()
    if lang in valid_languages:
        lang = lang[0:2]
        break
    else:
        print("Invalid language. Please choose from english, hindi, or french.")

valid_proficiencies = ['low', 'intermediate', 'high']
while True:
    proficiency = input("What level of proficiency do you want to evaluate? (low, intermediate, high): ").lower()
    if proficiency in valid_proficiencies:
        break
    else:
        print("Invalid proficiency level. Please choose from low, intermediate, or high.")

f = open(lang+"_"+proficiency+".txt", "r")
for line in f:
    line = line.strip()
    if not line: 
        continue
    parts = line.split()
    if len(parts) == 3:  
        actual_tags.append(parts[1])
        model_tags.append(parts[2])

print("Precision score: "+str(precision_score(actual_tags, model_tags, average="weighted", zero_division=0) * 100)+"%")
print("Recall score: "+str(recall_score(actual_tags, model_tags, average="weighted", zero_division=0) * 100)+"%")
print("F1 Score: "+str(f1_score(actual_tags, model_tags, average="weighted", zero_division=0) * 100)+"%")

print(confusion_matrix(actual_tags, model_tags))