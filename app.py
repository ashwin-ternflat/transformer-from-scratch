import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, num_heads, dropout=0.1, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_layers=2, num_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        mask = torch.tril(torch.ones(T, T, device=idx.device)).bool()
        attn_mask = ~mask

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


with open("r&m.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


n_embd = 128
block_size = 128
model = Transformer(vocab_size, n_embd, block_size, num_layers=2, num_heads=4)
model.load_state_dict(torch.load("two-layer-transformer_char_model.pt", map_location="cpu"))
model.eval()

#Streamlit UI
st.set_page_config(page_title="Rick & Morty Wiki Bot")
st.title(" Rick & Morty Wiki-style Bot")
st.caption("Ask anything. Get answers in Rick & Morty voice... or chaos.")

# User prompt
user_prompt = st.text_area("💬 Ask a question or say something:", height=100, max_chars=block_size)

# Generation settings
col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.slider("Temperature", 0.5, 1.5, 0.8, 0.1)
with col2:
    top_k = st.slider("Top-k", 10, 100, 50, 10)
with col3:
    max_tokens = st.slider("Max new tokens", 50, 500, 200, 50)

# Generate button
if st.button(" Generate Response"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt first.")
    else:
        context = torch.tensor(encode(user_prompt), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        generated = decode(out[0].tolist())
        st.markdown("### Response")
        st.code(generated)
