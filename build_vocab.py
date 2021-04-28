from collections import Counter
import numpy as np

# Hyper parameters for the vocab
UNK_WORD = 'UNK'

def create_corpus(path: str):
    with open(path) as f:
        text = f.read() 

    text = text.lower().split() # split text
    vocab_dict = dict(Counter(text).most_common()) # 得到單詞字典表，key是單詞，value是次數
    vocab_dict[UNK_WORD] = len(text) - np.sum(list(vocab_dict.values())) # 把不常用的單詞都編碼為"<UNK>"
    
    idx2word = [word for word in vocab_dict.keys()]
    word2idx = {word:i for i, word in enumerate(idx2word)}
    
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3./4.)

    return text, idx2word, word2idx, word_freqs, word_freqs
