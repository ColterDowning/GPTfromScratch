import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters. These are the parameters that we can tune to optimize our model. They are created by the developer (me) to control the behavior of the model.
batch_size = 64 # number of independent sequences to process in parallel
block_size = 256 # number of tokens in a sequence. If this is 256, the nn will generate the 257th token
max_iters = 5000 # number of iterations to train for
eval_interval = 500 # interval at which we evaluate the loss function
learning_rate = 3e-4 # How quickly we descend the gradient.
eval_iters = 200 # Number of iterations to run the estimate_loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu' # If a gpu is available, use it.
n_embd = 384 # embedding dimension. This is the dimensionality of the vectors that will be used to represent the tokens.
n_head = 6 # number of attention heads. The more heads we have, the more information we can capture in the self-attention mechanism.
n_layer = 6 # number of layers in the transformer model.
dropout = 0.2 # probability of dropping out a neuron. This helps prevent overfitting.
#-------------

torch.manual_seed(1337) # Set the random seed for reproducibility
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that appear in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) # The number of unique tokens in the vocabulary

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # enumerate returns an iterator for each character in chars. It returns a tuple of (index, character). Ex (0, 'a'), (1, 'b'), etc. The code wrapped around that creates a dictionary.
itos = { i:ch for i,ch in enumerate(chars) } # Same thing here, but we are going from integers back to characters.
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the training dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n] # First 90% of the data
val_data = data[n:] # Last 10% of the data


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # If we are training, we use the training data, otherwise we use the validation data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Randomly sample a batch of indices from the data. The minus block_size is because we need to have enough context to generate the next token.
    x = torch.stack([data[i:i+block_size] for i in ix]) # Create the input sequences. We do this by taking the data, and slicing it into blocks of size block_size. We then stack these slices together to form a single tensor.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Create the target sequences. Same logic as above.
    return x, y

@torch.no_grad() # Decorator to tell PyTorch that we are not going to call the autograd function. This means that no intermediate tensors will store computation graphs. Efficiency =)
def estimate_loss(): # Estimates the loss of the model on the training and validation data.
    out = {} # Initialize an empty dictionary to store the losses.
    model.eval() # Set the model to evaluation mode. This is necessary because some layers, such as dropout, behave differently during training and evaluation.
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split) # Get a batch of data
            logits, loss = model(X, Y) # Get the model's predictions and the loss
            losses[k] = loss.item() # Store the loss
        out[split] = losses.mean() # Store the mean loss for the split
    model.train() # Set the model back to training mode
    return out

class Head(nn.Module):
    #One head of self-attention

    def __init__(self, head_size): # head_size is the dimensionality of the keys, queries, and values.
        super().__init__() # Initialize the parent class
        self.key = nn.Linear(n_embd, head_size, bias=False) # Create a linear layer for the keys
        self.query = nn.Linear(n_embd, head_size, bias=False) # Create a linear layer for the queries
        self.value = nn.Linear(n_embd, head_size, bias=False) # Create a linear layer for the values
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Mask the upper triangle so we don't consider future tokens
        self.dropout = nn.Dropout(dropout) # Randomly zeroes out elements to prevent overfitting

    # B: Batch size, the number of sequences processed in parallel. In deep learning, processing
    # multiple sequences in parallel (batch processing) is more efficient than processing them one by one. 
    # The batch size determines how many sequences are processed simultaneously.
    # If B = 4, it means that 4 sequences are being processed in parallel.

    # T: Sequence length, the number of tokens in the input sequence.In language models, sequences are often 
    # broken down into smaller chunks or windows. T represents the number of tokens in each chunk.
    # If T = 8, it means that each sequence consists of 8 tokens.

    # C: Channel size, the dimensionality of the input space.
    # Each token in the sequence is represented as a one-hot vector of size C, 
    # where C is the number of unique tokens in the vocabulary.
    # If C = 65, it means that there are 65 unique tokens in the vocabulary.


    def forward(self, x): # x is the input to the layer
        B,T,C = x.shape # B is the batch size, T is the sequence length, C is the number of channels
        k = self.key(x) # (B,T,C) dimensionality
        q = self.query(x) # (B,T,C) dim
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # Randomly zeroes out elements to prevent overfitting
        # perform the weighted aggregation of the values
        v = self.value(x) 
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    

class MultiHeadAttention(nn.Module):
    # Multiple heads of self attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    # a simple linear layer followed by a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

        
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) which is batch, time, channel 
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    

    # The job of this function is to generate the next token given the existing context. If I gave you the letters "Th", this function might generate the letter "e" based on the probabilities.
    # We do this by taking the current idx (tensor that represents a series of letters, or tokens), which is a (Batch, Time) dimensioned array of indices. We perform our operations
    # on it, and at the end we sample a new token from our probability distribution. That new token then gets appended to the end of the idx array so that it becomes (Batch, Time + 1) dim.
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val dataset
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
