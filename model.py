import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer





device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def train(model, dataloader):
    model.train()
    epochs = 500
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch in dataloader:
            optim.zero_grad()
            input = batch.clone().int()
            src_mask = model.generate_square_subsequent_mask(batch.shape[1])
            rand_value = torch.rand(batch.shape)
            rand_mask = (rand_value < 0.15) * (input != 0)
            mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 29
            input = input.view(batch.shape)

            out = model(input.to(device), src_mask.to(device))
            loss = criterion(out.view(-1, ntokens), batch.view(-1).to(device).long())
            total_loss += loss
            loss.backward()
            optim.step()
    
        if (epoch+1)%40==0 or epoch==0:
            print("Epoch: {} -> loss: {}".format(epoch+1, total_loss/(len(dataloader)*epoch+1)))
            namestr = "TransModel" + f"epoch{epoch}" + ".pth"
            torch.save(model.state_dict(),namestr)
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


from datasetter import RushDatasets

dataset = RushDatasets(num = 300000 ,new = True)

dataloader = DataLoader(dataset, batch_size=50, collate_fn=data_collate_fn)
ntokens = 30 # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 6 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

total_params = sum(
	param.numel() for param in model.parameters()
)

train(model, dataloader)


# Predict

# print("Input: {}".format(text[0]))
# pred_inp = tokenizer("Don't speak ill of [MASK].")
# out = predict(model, pred_inp['input_ids'])
# print("Output: {}\n".format(tokenizer.decode(out)))


# TODO: 
# - GPT Architecture 
# - VAE/with transformers 
# - 7x7