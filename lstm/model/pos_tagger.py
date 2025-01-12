import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
torch.manual_seed(0)
# load idx2tag and word2idx
with open('./Encoding_Dictionaries/idx2tag.pkl', 'rb') as f:
    idx2tag = pickle.load(f)

with open('./Encoding_Dictionaries/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim  # the number of features in the hidden state h
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim)  # embedding layer

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # LSTM layer

        # the linear layer that maps from hidden state space to tag space-
        self.hidden2tag = nn.Linear(
            hidden_dim, tagset_size)  # fully connected layer
        self.hidden = self.init_hidden()  # initialize the hidden state

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))  # (h0, c0)

    def forward(self, sentence):
        # get the embedding of the words
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)  # pass the embedding to the LSTM layer
        # pass the output of the LSTM layer to the fully connected layer
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # get the softmax of the output of the fully connected layer
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# load the pytorch model
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(idx2tag))
model.load_state_dict(torch.load('pos_tagger_pretrained_model.pt', map_location=torch.device('cpu')))
model.to(device)
def     prepare_sequence(seq, to_idx):
    idxs = [to_idx['<unk>'] if w not in to_idx else to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)

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

def writer(filename, data):
    file = open(filename, "w")
    for i in data:
        for j in i:
            if j[0] == "<s>":
                continue
            if j[0] == "</s>":
                file.write("\n")
                continue
            file.write("\t".join(j))
            file.write("\n")
    
    file.close()

valid_languages = ['english', 'hindi', 'french']
while True:
    lang = input("Which language data do you want to evaluate? (english, hindi, french): ").lower()
    if lang in valid_languages:
        lang = lang[0:2]
        if lang == "hi":
            lang = lang + "_ud"
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

fname = "../../testing_data/" + lang + "/" + proficiency + ".txt"
data = reader(fname)
sentences = []
for i in data:
    lst = []
    for j in i:
        if j[0] == "<s>" or j[0] == "</s>":
            continue
        lst.append(j[0])
    sentence = " ".join(lst)
    sentences.append(sentence)

newdata = []

for i in range(len(sentences)):
    sentence_copy = sentences[i].split()
    sentence = sentences[i].lower().split()
    predicted =  [idx2tag[i] for i in model(
    prepare_sequence(sentence, word2idx)).argmax(1).tolist()]
    for j in range(len(sentence)):
        data[i][j+1].append(predicted[j])
        # print(sentence_copy[j],"\t", predicted[j]) 

ansfname = "../lstm_results/" + lang + "_" + proficiency + ".txt"
writer(ansfname, data)   
