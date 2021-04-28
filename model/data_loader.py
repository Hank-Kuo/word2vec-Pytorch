from torch.utils import data
import torch

C = 3 # context window
K = 15 # number of negative samples

class DataLoader(data.Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            idx2word: index to word mapping
            word_freqs: the frequency of each word
            word_counts: the word counts
        '''
        super(DataLoader, self).__init__()
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text] # 把單詞數字化表示。如果不在詞典中，也表示為unk
        self.text_encoded = torch.LongTensor(self.text_encoded) # nn.Embedding需要傳入LongTensor型別
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
        
    def __len__(self):
        return len(self.text_encoded) # 返回所有單詞的總數，即item的總數
    
    def __getitem__(self, idx):
        ''' 這個function返回以下資料用於訓練
            - 中心詞
            - 這個單詞附近的positive word
            - 隨機取樣的K個單詞作為negative word
        '''
        center_words = self.text_encoded[idx] # 取得中心詞
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1)) # 先取得中心左右各C個詞的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 為了避免索引越界，所以進行取餘處理
        pos_words = self.text_encoded[pos_indices] # tensor(list)
        
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        # torch.multinomial作用是對self.word_freqs做K * pos_words.shape[0]次取值，輸出的是self.word_freqs對應的下標
        # 取樣方式採用有放回的取樣，並且self.word_freqs數值越大，取樣概率越大
        # 每取樣一個正確的單詞(positive word)，就取樣K個錯誤的單詞(negative word)，pos_words.shape[0]是正確單詞數量
        return center_words, pos_words, neg_words
    
    @staticmethod
    def _to_idx(key: str, mappinp) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)