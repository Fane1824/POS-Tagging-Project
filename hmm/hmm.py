import re
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def reader(filename):
    file = open(filename, "r")
    contents = file.read()
    file.close()
    contents = contents.split("\n")
    sentence= []
    returndata = []
    for line in contents:
        if line == "":
            sentence.insert(0, ["<s>", "<s>"])
            sentence.append(["</s>", "</s>"])
            returndata.append(sentence)
            sentence= []
            continue
        sepword = line.split("\t")
        sentence.append(sepword)
    return returndata

def emission_probabilities_table(parsedsample):
    emission_probabilities = {}
    for sentence in parsedsample:
        for word, tag in sentence:
            if tag not in emission_probabilities:
                emission_probabilities[tag] = {}
            if word not in emission_probabilities[tag]:
                emission_probabilities[tag][word] = 1
            else:
                emission_probabilities[tag][word] += 1
    for tag, word_counts in emission_probabilities.items():
        total = sum(word_counts.values())
        for word in word_counts:
            word_counts[word] /= total

    return emission_probabilities

def transition_probabilities_table(parsedsample):
    transition_probabilities = {}
    for sentence in parsedsample:
        for i in range(len(sentence) - 1):
            tag, next_tag = sentence[i][1], sentence[i + 1][1]
            if tag not in transition_probabilities:
                transition_probabilities[tag] = {}
            if next_tag not in transition_probabilities[tag]:
                transition_probabilities[tag][next_tag] = 1
            else:
                transition_probabilities[tag][next_tag] += 1
    for tag, next_tags in transition_probabilities.items():
        total = sum(next_tags.values())
        for next_tag in next_tags:
            next_tags[next_tag] /= total
    all_tags = set()
    for sentence in parsedsample:
        for _, tag in sentence:
            all_tags.add(tag)
    all_tags.add("</s>")
    for tag in all_tags:
        if tag not in transition_probabilities:
            transition_probabilities[tag] = {}
            for next_tag in all_tags:
                transition_probabilities[tag][next_tag] = 0

    return transition_probabilities

def viterbi_algorithm(sentence, emission_matrix, transition_matrix):
    words = sentence.split()
    states = list(emission_matrix.keys())
    viterbi = {state: [0] * len(words) for state in states}
    backpointer = {state: [None] * len(words) for state in states}
    for state in states:
        emission_prob = emission_matrix[state].get(words[0], 1e-10)
        transition_prob = transition_matrix['<s>'].get(state, 0)
        viterbi[state][0] = transition_prob * emission_prob
    for t in range(1, len(words)):
        for state in states:
            max_prob, prev_state = max(
                (viterbi[prev_state][t - 1] * transition_matrix[prev_state].get(state, 0), prev_state)
                for prev_state in states
            )
            emission_prob = emission_matrix[state].get(words[t], 1e-10)
            viterbi[state][t] = max_prob * emission_prob
            backpointer[state][t] = prev_state
    best_last_state = max(viterbi.keys(), key=lambda state: viterbi[state][-1])
    best_path = [best_last_state]
    for t in range(len(words) - 1, 0, -1):
        best_path.insert(0, backpointer[best_path[0]][t])
    return list(zip(words, best_path))

def confusion_matrix(actual, predicted):
    tags= list(set(actual))
    tags.sort()
    c= pd.DataFrame(np.zeros((len(tags), len(tags))), index=tags, columns=tags)
    for i in range(len(actual)):
        if actual[i] in c.index and predicted[i] in c.columns:
            c.loc[actual[i], predicted[i]]+= 1
    return c

valid_languages = ['english', 'hindi', 'french']
while True:
    lang = input("Which language data do you want to evaluate? (english, hindi, french): ").lower()
    if lang in valid_languages:
        lang = lang[0:2]
        sampledata = "training_data/" + lang + "_training.txt"
        parsedsample = reader(sampledata)
        if lang == "hi":
            lang = lang + "_bis"
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

emission_matrix = emission_probabilities_table(parsedsample)

transition_matrix = transition_probabilities_table(parsedsample)

annotated_data = "../testing_data/" + lang + "/" + proficiency + ".txt"
parsed_adata = reader(annotated_data)

actual_tags = []
model_tags = []
for sentence in parsed_adata:
    sentence = sentence[1:-1]
    s = ""
    for word in sentence:
        s += word[0] + " "
    s = s.strip()
    tags = []
    for word in sentence:
        tags.append(word[1])
    model_output = viterbi_algorithm(s, emission_matrix, transition_matrix)
    for i in range(len(tags)):
      actual_tags.append(tags[i])
      model_tags.append(model_output[i][1])
print("Precision score: "+str(precision_score(actual_tags, model_tags, average="weighted", zero_division=0) * 100)+"%")
print("Recall score: "+str(recall_score(actual_tags, model_tags, average="weighted", zero_division=0) * 100)+"%")
print("F1 Score: "+str(f1_score(actual_tags, model_tags, average="weighted", zero_division=0) * 100)+"%")
print(confusion_matrix(actual_tags, model_tags))