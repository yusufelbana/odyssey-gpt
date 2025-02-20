#!/usr/bin/python3
import torch 
import torch.nn as nn
import torch.nn.functional as F

#hyper parameters
batch_size = 64
block_size = 256
embed_size = 384
n_heads,n_layers = 6,6
learning_rate = 1e-3
# n_epochs = 5: overkill training data is only ~10k lines
max_iters = 15000
device = "cuda" if torch.cuda.is_available() else "cpu"


with open("input.txt", "r") as f:
    #had this extra BOM on local, caused issues on training in google collab
    txt = f.read().lstrip('\ufeff')
    unique_chars = sorted(list(set(txt)))

encoder = {c:i for i,c in enumerate(unique_chars)}
encode = lambda string: [encoder[char] for char in string]
decoder = {i:c for i,c in enumerate(unique_chars)}
decode = lambda int_list: [decoder[i] for i in int_list]

vocab_size = len(unique_chars)
tokenized_data = encode(txt)
#print(unique_chars)

n = int(0.9 * len(tokenized_data))
train_data  = tokenized_data[:n]
test_data = tokenized_data[n:]

#lucky number :3
torch.manual_seed(19)


def get_batch(data="train"):
    data = train_data if data  == "train" else test_data
    random_t = torch.randint(0,len(data) - block_size,(batch_size,))
    # x(input) target will align with y 
    #ex: x:[1,2,3,4] y:[2,3,4,5]
    x = torch.stack([torch.tensor(data[i:i+block_size] ) for i in random_t])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1])  for i in random_t])
    x,y = x.to(device),y.to(device)
    return x,y

class Head(nn.Module):
    '''
    x is of B,T,HEAD_SIZE
    see pdf in repo, math kinda messed me up at the beginning. not an in depth explanation but just a work thru of a small example.
    https://github.com/yussz/odyssey-gpt/single_head.pdf
    '''
    def __init__(self,head_size):
        super().__init__()
        #linear pass with the last dimension going from embed => head size in order to spread out how much each head can learn. will concat result at the end.
        #Query, Key, Value 
        self.query = nn.Linear(embed_size,head_size,bias=False)
        self.key = nn.Linear(embed_size,head_size,bias=False)
        self.value = nn.Linear(embed_size,head_size,bias=False)
    def forward(self,x):
        q = self.query(x)
        k = self.query(x)
        v = self.query(x)
        #B,T,HEAD_SIZE @ B,HEAD_SIZE,T => B,T,T
        tmp = (q @ k.transpose(-2,-1)) * (k.shape[-1] ** -0.5) 
        tmp = torch.tril(tmp) # remove the upper triangular values to remove each token's awareness of the future
        tmp = tmp.masked_fill(tmp == 0, float("-inf"))
        tmp = F.softmax(tmp,dim=-1) #prob dist 
        out = tmp @v        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.lin = nn.Linear(head_size * n_heads,embed_size)
    def forward(self,x):
        #each head will have B,T,HEAD_SIZE so we concatenate on last dimension to get back to embed size since head_size * n_heads = embed_size
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        #add linear pass
        out = self.lin(out)
        return out

class FeedForward(nn.Module):
    '''
    x is of B,T,EMBED_SIZE
    '''
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embed_size,embed_size),
            nn.ReLU(),
            nn.Linear(embed_size,embed_size)
        )
    def forward(self,x):
        out = self.seq(x)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        #each head will potentially be able to discover more about the relationship between the tokens
        head_size = embed_size // n_heads
        self.sa_heads =  MultiHeadAttention(head_size)
        self.feed_forward= FeedForward()
        #two layer norms
        self.ln2 = nn.LayerNorm(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
    def forward(self,x):
        #add residuals
        x = self.ln1(x + self.sa_heads(x))
        out = self.ln2(x + self.feed_forward(x))
        return out



class OdysseyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        #vec and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size,embed_size)
        self.position_embedding = nn.Embedding(block_size,embed_size)
        #final layer norm
        self.lnf = nn.LayerNorm(embed_size)
        #sequential pass of singular attention head
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])

        #final linear pass
        self.lin_f = nn.Linear(embed_size,vocab_size)

    def forward(self,x,targets=None):
        #batch and block size(time)
        B,T = x.shape
        #add positional and vec embeddings
        x = self.token_embedding(x)
        x = x + self.position_embedding(torch.arange(T,device=device))
        #create a block for each layer
        x = self.blocks(x)
        #normalize
        x = self.lnf(x)
        #embed => vocab_size
        logits = self.lin_f(x)

        if targets == None:
            loss = None
        else:
            #x is of dim B,T,C in order to align with target we must reshape so we have a tensor(32,32), after softmax it'll line up with the targets.
            B,T,C = logits.shape
            #reshape so softmax(tensor) "matches" with a y(target)
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

m = OdysseyGPT()
m = m.to(device)
#print total params
print(f"n_params: {sum(p.numel() for p in m.parameters())}")
optimizer = torch.optim.AdamW(m.parameters(),lr = learning_rate)

        
def train():
    m.train()
    for iter in range(max_iters):
        x,y = get_batch("train")
        _,loss = m(x,y)
        #calc gradients and step 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iter % 1000 == 0:
            print(f"loss: {loss.item()}")

def generate(x, max_new_tokens):
    for _ in range(max_new_tokens):
        #get last blocksize(ex:4) tokens [1,2,3,4,5,6,7,8] => [5,6,7,8]
        x_cond = x[:,-block_size:]
        logits, loss = m.forward(x_cond)
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1) 
        #get the highest prediction and keep concatening the values onto x
        x_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x,x_next), dim=1)
    return x

# train()
# torch.save(m.state_dict(),"./model.pth")
m.load_state_dict(torch.load("./model.pth",weights_only=True))
context = torch.tensor(encode("Ulysses"), dtype=torch.long,device=device).unsqueeze(0)
#context = torch.ones((1, 1), dtype=torch.long, device=device)
answer = decode(generate(context, max_new_tokens=1000)[0].tolist())
answer = "".join(answer)
with open("output.txt", "w") as f:
    f.write(answer)




