import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super(Word2Vec,self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size

        initrange = 0.5/self.emb_size
        self.in_embed = nn.Embedding(vocab_size,emb_size)
        self.in_embed.weight.data.uniform_(-initrange,initrange)

        self.out_embed = nn.Embedding(vocab_size,emb_size)
        self.out_embed.weight.data.uniform_(-initrange,initrange)

    def forward(self,center_words,pos_words,neg_words):
        batch_size = center_words.size(0)
        input_embedding = self.in_embed(center_words) #[batch,emb]
        pos_embedding = self.out_embed(pos_words) #[batch,2c,emb]
        neg_embedding = self.out_embed(neg_words)  #[batch,2c*k,emb]

        log_pos = torch.matmul(pos_embedding,input_embedding.unsqueeze(2)).squeeze() # [batch,2c]
        log_nes = torch.matmul(neg_embedding,-input_embedding.unsqueeze(2)).squeeze() #[batch,2c*k]

        log_pos_los = F.logsigmoid(log_pos).sum(1)
        log_neg_los = F.logsigmoid(log_nes).sum(1)
        loss = log_neg_los+log_pos_los

        return -loss.mean()

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()