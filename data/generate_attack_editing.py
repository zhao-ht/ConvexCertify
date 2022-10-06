from tqdm import tqdm
import random
import nltk

def editor(origin_sent, label, iter_num=10):
    sents = nltk.word_tokenize(origin_sent)
    iter_num = min(iter_num, len(sents))
    for iter in range(iter_num):
        index = random.randint(0, len(sents)-1)
        prob = random.random()
        if prob > 0.5 and len(sents)>3:
            sents.remove(sents[index])
        else:
            sents.insert(index+1, sents[index])
    return  ' '.join(sents), label, iter_num

