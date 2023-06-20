import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import gc

import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer





device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask = None):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

def train(model, dataloader, path):

    #Training Parameters
    model.train()
    epochs = 500
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

    #Saving path
    current_directory = os.getcwd()
    new_directory_path = os.path.join(current_directory, path)
    os.makedirs(new_directory_path, exist_ok=True)

    for epoch in range(epochs):
        total_memory, used_memory, free_memory = map(
                    int, os.popen('free -t -m').readlines()[-1].split()[1:])
        prctg = (used_memory/total_memory) * 100
        print(f"Memory used in {epoch} : {prctg}.")

        for batch in dataloader:
            batch = batch.to(device)
            optim.zero_grad()
            input = batch.clone().int()
            # src_mask = model.generate_square_subsequent_mask(batch.shape[1])
            rand_value = torch.rand(batch.shape)
            rand_mask = (rand_value < 0.15) * (input != 0)
            mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 29
            input = input.view(batch.shape)

            out = model(input.to(device))
            loss = criterion(out.view(-1, ntokens), batch.view(-1).to(device).long())
            total_loss += loss
            loss.backward()
            optim.step()
            del batch, out, loss
            torch.cuda.empty_cache()
            gc.collect()
        if (epoch)%20==0:
            print("Epoch: {} -> loss: {}".format(epoch+1, total_loss/(len(dataloader)*epoch+1)))
            namestr = "TransModel" + f"epoch{epoch}" + ".pth"
            torch.save(model.state_dict(),os.path.join(new_directory_path,namestr))
            print('Model saved in {}'.format(namestr))

def predict(model, input):
    model.eval()
    src_mask = model.generate_square_subsequent_mask(input.size(1))
    out = model(input.to(device), src_mask.to(device))
    out = out.topk(1).indices.view(-1)
    return out



import torch
import numpy as np
from torch.utils.data import  DataLoader

from datasetter import tokenizer

def data_collate_fn(dataset_samples_list):
    arr = [torch.unsqueeze(tokenizer(x),0) for x in dataset_samples_list]
    inputs = torch.cat(arr)
    return inputs



ntokens = 30 # the size of vocabulary
emsize = 300 # embedding dimension
nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 6 # the number of heads in the multiheadattention models
dropout = 0.25 # the dropout value

if __name__ == "__main__":
    
    from datasetter import RushDatasets

    dataset = RushDatasets(num = 500000 ,new = True)
    path = sys.argv[1]
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=data_collate_fn)

    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    total_params = sum(
        param.numel() for param in model.parameters()
    )

    train(model, dataloader, path)


# Predict

# print("Input: {}".format(text[0]))
# pred_inp = tokenizer("Don't speak ill of [MASK].")
# out = predict(model, pred_inp['input_ids'])
# print("Output: {}\n".format(tokenizer.decode(out)))


# TODO: 

# - Add path to save as arg.

# - Représentation channel : blocs horizontaux de 2/3   , verticaux de 2/3
# - Find new <Représentation> less sparse to make normal networks work ?
# - DenseNet Encoder Decoder
# - Parameters to change : Embed Dim/ LR / Positional Encoding Args
# - 7x7


# --------- Extras ---------
# - Check why single input is giving wrong out shape
