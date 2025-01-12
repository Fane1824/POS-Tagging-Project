
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle
import numpy as np
from torchmetrics import F1Score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

print("Using device:", device)

valid_languages = ['english', 'hindi', 'french']
while True:
    lang = input("Which language data do you want to evaluate? (english, hindi, french): ").lower()
    if lang in valid_languages:
        if (lang == "hindi"):
            with open('./UD_Hindi-PUD-master/hi_pud-ud-test.conllu', 'r') as f:
                train_data = f.read()
        elif (lang == "english"):
            with open('./UD_English-PUD/en_pud-ud-test.conllu', 'r') as f:
                train_data = f.read()
        else:
            with open('./UD_French-FQB/fr_fqb-ud-test.conllu', 'r') as f:
                train_data = f.read()
        break
    else:
        print("Invalid language. Please choose from english, hindi, or french.")

def get_sentences(data):
    """ Function to get sentences from the dataset """

    sentences = []
    for line in data.split('\n'):
        if line.startswith('# text = '):
            sentences.append(line[9:])

    return sentences

train_sentences = get_sentences(train_data)

def get_labels(data):
    labels = []
    words = []
    output_data = []

    for line in data.split('\n'):
        if (line):
            if line.startswith('# text = ') or line.startswith('# sent_id = '):
                continue
            temp = line.split('\t')[0:4]
            try:
                temp = temp[0:2]+[temp[3]]  
                if temp[0] == '1':
                    words = []
                    labels = []
                    output_data.append((words, labels))
                words.append(temp[1])
                labels.append(temp[2])
            except:
                pass

    return output_data


training_data = get_labels(train_data)
word2idx = {}
tags2idx = {}
idx2tag = {}
i=0
for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    
    if(i%30==0):
        rand = np.random.randint(0, len(sent))
        training_data[i][0][rand] = '<unk>'  

    for tag in tags:
        if tag not in tags2idx:
            tags2idx[tag] = len(tags2idx)
            idx2tag[len(tags2idx)-1] = tag
            
    i+=1
    

word2idx['<unk>'] = len(word2idx)
tags2idx['<unk>'] = len(tags2idx)
idx2tag[len(tags2idx)-1] = '<unk>'

# save the dictionaries
with open('./Encoding_Dictionaries/word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)


with open('./Encoding_Dictionaries/idx2tag.pkl', 'wb') as f:
    pickle.dump(idx2tag, f)


EMBEDDING_DIM = 128
HIDDEN_DIM = 128
# the learning rate which is used to update the weights
LEARNING_RATE = 0.2
NUM_EPOCHS = 10  # the number of times the model is trained on the entire dataset

print("Embedding size:", EMBEDDING_DIM)
print("Output size:", HIDDEN_DIM)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", NUM_EPOCHS)



class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim  # the number of features in thhidden state h
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim)  # embedding layer

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # LSTM layer

        # the linear layer that maps from hidden state space to tag spacembed_sizee-
        self.hidden2tag = nn.Linear(
            hidden_dim, tagset_size)  # fully connected layer
        self.hidden = self.init_hidden()  # initialize the hidden state

    def init_hidden(self):
        # initialize empty hidden state.
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

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(
    word2idx), len(tags2idx)).to(device)  # initialize the model

loss_function = nn.NLLLoss()  # define the loss function

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

def prepare_sequence(seq, to_idx):
    idxs = [to_idx['<unk>'] if w not in to_idx else to_idx[w]
            for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)

# Training the model and evaluating it on the test set

def train_model(model, data, num_epoch):

    for epoch in range(num_epoch):
        for sentence, tags in data:
            model.zero_grad()  # clear the gradients of all optimized variables
            model.hidden = model.init_hidden()  # initialize the hidden state

            # convert the sentence to a tensor
            sentence_in = prepare_sequence(sentence, word2idx)
            # convert the tags to a tensor
            targets = prepare_sequence(tags, tags2idx)

            tag_scores = model(sentence_in)  # forward pass
            # calculate the loss
            loss = loss_function(tag_scores, targets)
            loss.backward()  # backward pass
            optimizer.step()  # update the parameters

        # print("Epoch:", epoch, "Loss:", loss.item())

train_model(model, training_data, NUM_EPOCHS)

# save the model

model_path = 'pos_tagger_pretrained_model.pt'

torch.save(model.state_dict(), model_path)



