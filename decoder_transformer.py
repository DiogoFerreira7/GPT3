import torch
import torch.nn as nn
from torch.nn import functional as F

# Attention is all you need implementation

# Hyper Parameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# Each head is 64 as a standard
n_embd = 384
n_head = 6
n_layer = 6
# 15% of these intermediate layers are dropped out
dropout = 0.15

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# Mapping to and from integers
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loader that creates batches and takes data to the GPU
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Self-attention - keys queries and values all come from the same source, the nodes are self attending
# in encoder and decoder transformers - we can make keys and values come from entirely independent source 
    # cross attention can be used when there is another source of nodes that we would like to pull information from

class Head(nn.Module):
    """
    # Version 1
    # Components in nth location should only have context on n-1....0 (all locations prior of context)
    # if we get future information then ruins the whole point of trying to predict it
    # For every bathc element and for every Th elemnt - we can calcualte the average of all elements prior and including this t_th token
    # This idea summarises bag of words - averaging them (however we should be careful as this is quite lossy)

    # Version 2
    # We can use matrix multiplication on a triangular matrix that is normalised to do a batched matrix multiplication
    # The normalisation allows you to get teh average for each separate row since the triangular matrix increases by one each time
    # So then we do weighted sums of all the weights (all tokens preceeding it)

    # Version 3
    # for all elements equal to 0 we make them -inf using masked_fill
    # If we then softmax along every single row and thus dim=1
    # Softmax will normalise it and gives us the same exact matrix as version 2 - by exponentiating -inf we get 0 and for 0 we get 1 which is then divided
    # F.softmax(w, dim = 1)
    # This softmax is useful in self attention because it tells us how much each token from the past we want to focus on
    # Masked fill tells us that the prior values should not be looked at
    """

    def __init__(self, head_size):
        # The head size determines the size of our key and query linear layers
        super().__init__()
        # We define the weights that are linear layers that will define the key, query and value weights
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # These affinities will be data dependent and they will start looking at eachother - tokens will find eachother more or less interresting
        # These affinities will change - however we just prevent the past looking at teh future via clamping
        # Defining a buffer like batch norms running averages
        # "tril" is registered buffer that creates a lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Every token at each position has a query and a key
        # Query - what am i looking for
        # Key - what do i contain
        # Affinities are defined by the query and key - if they are aligned they will interact to a very high amount

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # (B,T,hs) x (B,T,hs)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute our attention scores by transposing k - a communication mechanism
        # Transpose the last 2 dimensions in k
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # Dividing by the square root is important due to normalisation
        # If we have unit gaussian inputs, if we just do the variance naivelly the variance will be scaled by the head size
        # It is really important that our attention is scaled correctly diffused and normalised by our head size
        # Softmax will converge to one hot vectors if we have sharper vectors will larger values - softmax will be far too peaky and each node will gain its information
        # from predominantly one singular node but we want an aggregation of information - this wwould be too lossy so this normalisation is important
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        
        # We can then use masked_fill that will set all the values on the upper right triangle that are 0 to -inf so that we ignore them
        # Decoder block includes the masked fill - decoders will not be allowed to talk to the nodes in the future
        # Encoder blocks allow all nodes to talk to eachother - decoder will use this triangular structure
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax will normalise the model and output the affinities
        wei = F.softmax(wei, dim=-1)
        # We can prevent some of the layers from communicating - it will randomly shut off some subset of neurons and train without them
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values based on the affinities that we have from the key and query 
        v = self.value(x)
        out = wei @ v 

        return out

class MultiHeadAttention(nn.Module):
    """
    We want multiple heads of self attention in parallel
    Create multiple heads, and the head size of each should be taken into account

    Multiple communication channels independently allows us to exchange multiple different pieces of data and information
    That can help figure out what we want and better embeddings
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Projection is a linear transformation into the linear pathway - we project the multihead attention back into the main pathway
        # of the residual connection
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """
    Linear layer followed by a non linearity
    Applies to the number of embeddings
    Self attention is the communication and then the FeedForward allows the network to think and apply different methods before continuing
    to the next attention layer
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Channel sizes are 4x if they are the inner layers of the feed forward layer
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # We can dropout here - trains an ensemble of sub networks, and these during inference are merged into one large one
            # it is a regularisation technique that proves quite useful especially when we are about to scale up our model
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Intersperse communicaiton and calculations
    we can do multi head attention (just does not have cross attention yet)
    Then we apply a feed forward layer on the token level to each separate value

    Deep neural networks however will suffer from optimisation issues
    There are two main optimisations that will help with these

    Skip connections / Residual connections - these residual blocks allows us distribute gradients equally to both of the branches
    The gradients from the loss can continue to be propagated backwards without being lost (technically creates a gradient motorway)
    During initialisation gradients are unimpeded
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # The number of embeddings divided by the number of heads that we have should divide evenly
        head_size = n_embd // n_head
        # So for example if we have 32 embeddings we can use 4 heads with a size of 8 each
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)

        # Layer norms now are applied prior to any self attention layers or any feed forward layers
        # Layer norm has gamma and beta parameters that will also be trained
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Skip connections and these residual connections allow us to fork off and do some computation which here is the feed forward
        # and multihead attention
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    This is a decoder only tranformer

    Batch normalisation - across batches individual neurons have unit gaussian distribtuions with 0 mean and 1 std outputs

    Layer norm instead will do the rows instead of the columns unlike batch normalisation
    For every individual example the vectors are normalised
    Because our data is not spanned across batches we dont need any running averages anymore and thus we dont need to keep track of buffers

    Layer norm used to be applied after the transformations
    Now the layer norm is applied pre transformation by convention
    """

    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each position from 0 to block_size - 1 will get its own embedding vector
        # Gives the notion of position within the sentence
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Normalises our rows instead of our columns - and we do this right at the end of our transformer
        self.ln_f = nn.LayerNorm(n_embd)
        # Language modelling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # Positional embedding that will be integers from 0 to T-1
        # Because of the size we can broadcast it
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

        # Combination of the token embeddings with the positional embeddings - we can broadcast the positional embeddings to the token embeddings
        # (B,T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Predictions
            logits, loss = self(idx_cond)
            # Focus only on the last step prediction since that is what we care about
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            # Append new to the prior that will stack as context
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# This block only has self attention and feed forward - the cross attention is not implemented
# This is a decoder only as we are just generating text that is unconditioned
# The triangular mask has an autoregressive property that allows us to sample from our context

# If we are doing translation we want to condition on additional information e.g another language and this would require an encoder
# that would read the conditioned text - create tokens without triangular masking that can communicate through the whole content
# This context will be used for cross attention that would be provided to the outputs of the decoded
# Keys and Values are being created by the nodes from the encoder and fed into the decoder

model = GPTLanguageModel()
m = model.to(device)

# Model parameters
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# AdamW as the optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Evaluate loss on Training and Validation so we can see when we are overfitting
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"{iter}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # Optimise
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
