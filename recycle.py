
class EncoderLayer(nn.Module):
   def __init__(self):
       super(EncoderLayer, self).__init__()
       self.enc_self_attn = MultiHeadAttention()
       self.pos_ffn = PoswiseFeedForwardNet()

   def forward(self, enc_inputs, enc_self_attn_mask):
       enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
       enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
       return enc_outputs, attn
   
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)


        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]
    

###################################################################################################


class TransformerMLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(hidden_size, num_heads, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        tokens, mask = inputs

        # Embed tokens and add positional encoding
        x = self.token_embedding(tokens) + self.positional_embedding[:, :tokens.shape[1], :]

        # Apply dropout
        x = self.dropout(x)

        # Masked self-attention
        attn_mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1])) == 1
        attn_mask = attn_mask.to(tokens.device)
        attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(1), False)
        x = self.transformer(x, x, tgt_mask=attn_mask)

        # Predict masked tokens
        x = self.fc(x)

        return x

# Example usage
model = TransformerMLM(vocab_size=1000, hidden_size=256, num_layers=6, num_heads=8, max_seq_len=128, dropout=0.2)
tokens = torch.randint(0, 1000, (32, 128))
mask = torch.zeros_like(tokens, dtype=torch.bool)
mask[:, 10:20] = True
predictions = model((tokens, mask))

#################################################################################################""