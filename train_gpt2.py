import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn import functional as F
import math
from tqdm import tqdm
from transformers import GPT2Tokenizer
import wandb
import argparse
import os

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dims = embed_size // heads
    self.values = nn.Linear(embed_size, embed_size, bias=False)
    self.keys = nn.Linear(embed_size, embed_size, bias=False)
    self.queries = nn.Linear(embed_size, embed_size, bias=False)
    self.fc_out = nn.Linear(embed_size, embed_size)

  def forward(self, x):
    nB,nS,nE = x.size() # batch, seq len, embed size
    nH = self.heads
    v = self.values(x).view(nB,nS,nE//nH,nH).transpose(1, 2) # (nB,nE//nH,nS,nH)
    k = self.keys(x).view(nB,nS,nE//nH,nH).transpose(1, 2)
    q = self.queries(x).view(nB,nS,nE//nH,nH).transpose(1, 2)

    attention = q@(k.transpose(-2,-1)) / math.sqrt(k.size(-1))  # (nB,nE//nH,nS,nS)
    mask = torch.triu(torch.ones(nS, nS), diagonal=1).to(x.device)
    attention = attention.masked_fill(mask==1, float('-inf'))
    attention = F.softmax(attention, dim=-1)
    y = attention @ v # (nB,nE//nH,nS,nH)
    y = y.transpose(1,2) # (nB,nS,nE//nH,nH)
    y = y.contiguous().view(nB,nS,nE)
    return self.fc_out(y)

class FeedForward(nn.Module):
  def __init__(self, embed_size, expansion_factor=4):
    super(FeedForward, self).__init__()
    self.fc1 = nn.Linear(embed_size, expansion_factor * embed_size)
    self.fc2 = nn.Linear(expansion_factor * embed_size, embed_size)
  def forward(self, x):
    return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout=0.1):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.feed_forward = FeedForward(embed_size)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x):
    x = self.dropout1(self.attention(x)) + x
    x = self.norm1(x)
    x = self.dropout2(self.feed_forward(x)) + x
    x = self.norm2(x)
    return x

class GPT2(nn.Module):
  def __init__(self, vocab_size, embed_size, num_layers, heads):
    super(GPT2,self).__init__()
    self.token_emb = nn.Embedding(vocab_size, embed_size)
    self.position_emb = nn.Embedding(1024, embed_size) # learnable position emb
    self.transformer_blocks = nn.ModuleList(
        [TransformerBlock(embed_size, heads) for _ in range(num_layers)]
    )
    self.fc_out = nn.Linear(embed_size, vocab_size)

  def forward(self, x):
    seq_len = x.size(1)
    x = self.token_emb(x) + self.position_emb(torch.arange(seq_len).unsqueeze(0).to(x.device))
    for transformer in self.transformer_blocks:
      x = transformer(x)
    return self.fc_out(x)

  @torch.no_grad()
  def generate(self, prompt_token, max_len=128, temperature=1.0, top_k=None):
    for _ in tqdm(range(max_len)):
      logits = self(prompt_token)
      last_logits = logits[:,-1,:] / temperature
      if top_k != None:
        top_k_logits, _ = torch.topk(last_logits, top_k, dim=-1)
        last_logits[last_logits < top_k_logits[:,[-1]]] = -float('inf')
      prob = F.softmax(last_logits, dim=-1)
      next_token = torch.multinomial(prob, num_samples=1)
      prompt_token = torch.cat((prompt_token,next_token),dim=-1)
    return prompt_token

def batch_sampler(data, block_size, batch_size, n_token):
  idx = torch.randint(0, n_token-block_size-1, (batch_size,))
  batch_x = torch.stack([data[i:i+block_size] for i in idx])
  batch_y = torch.stack([data[i+1:i+1+block_size] for i in idx])
  return batch_x, batch_y

def train(args):
  dataset_name = args.dataset
  n_epoch = args.n_epoch
  batch_size = args.batch_size
  wandb.login()
  wandb.init(project="gpt2_c4") 

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token
  dataset = load_dataset(dataset_name) # 'wikitext','wikitext-2-raw-v1'
  train_data = dataset['train']['text']
  print(f'dataset name: {dataset_name}')

  # concatenate data
  print('concatenating the data')
  train_data_concat = ''
  for data_sample in tqdm(train_data):
    train_data_concat += data_sample

  # tokenize data
  print('tokenizing the data')
  tokenized_data_concat = tokenizer.encode(train_data_concat, return_tensors="pt")
  n_token = len(tokenized_data_concat[0])
  print(f'number of tokens: {n_token}')

  device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
  model = GPT2(vocab_size=tokenizer.vocab_size, embed_size=768, num_layers=12, heads=12)
  model.to(device)
  print(f'model modules:\n{model}')

  optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
  loss_func = nn.CrossEntropyLoss()
  block_size = 1024
  # batch_size = 4
  n_iter = n_token // block_size
  # n_epoch = 1

  print('start training')
  model.train()
  for ep in range(n_epoch):
    for i in tqdm(range(n_iter)):
      optimizer.zero_grad()
      batch_x, batch_y = batch_sampler(tokenized_data_concat[0], block_size=block_size, batch_size=batch_size, n_token=n_token)
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      batch_pred_logits = model(batch_x)
      output_size = batch_pred_logits.size(-1)
      loss = loss_func(batch_pred_logits.view(-1,output_size), batch_y.view(-1))
      loss.backward()
      optimizer.step()
      wandb.log({"loss": loss.item()})
    torch.save(model.state_dict(), 'gpt2_ckpt.pth')

def generate_response(prompt, temperature, top_k):
  device = 'cuda'
  ckpt_path = 'gpt2_ckpt.pth'
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2(vocab_size=tokenizer.vocab_size, embed_size=768, num_layers=12, heads=12).to(device)
  if os.path.isfile(ckpt_path):
      model.load_state_dict(torch.load(ckpt_path))
  prompt_token = tokenizer.encode(prompt, return_tensors='pt').to(device)
  output_token = model.generate(prompt_token, max_len=128, temperature=temperature, top_k=top_k)
  output_token = output_token.tolist()[0] 
  response = tokenizer.decode(output_token)
  print(response)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='gpt2')
  # args for training
  parser.add_argument('--mode', type=str, default='train', help='train or generate')
  parser.add_argument('--dataset', type=str, default='datablations/c4-filter-small', help='prompt')
  parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
  parser.add_argument('--batch_size', type=int, default=4, help='number of epochs')


  # args for decoding
  parser.add_argument('--prompt', type=str, default='do you know wikipedia?', help='prompt')
  parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
  parser.add_argument('--top_k', type=int, default=200, help='top k sampling')
  args = parser.parse_args()
  if args.mode == 'train':
    train(args)
  elif args.mode == 'generate':
    generate_response(args.prompt,temperature=args.temperature,top_k=args.top_k)
  else:
    print('please select a correct mode.')

