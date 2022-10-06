import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
# from keras.preprocessing import sequence

sys.path.append('..')
from data.dataset import split_imdb_files
from utils.tokenizer import Tokenizer

global imdb_tokenizer
imdb_tokenizer = None

def update_tokenizer(dataset, tokenizer):

    if dataset == 'imdb':
        global imdb_tokenizer
        imdb_tokenizer = tokenizer

def get_tokenizer(args):
    dataset = args.dataset_name
    texts = None
    if dataset == 'imdb':
        global imdb_tokenizer
        if imdb_tokenizer is not None:
            return imdb_tokenizer

        imdb_tokenizer_file = os.path.join(args.work_path,"temp/imdb_tokenizer.pickle")
        if os.path.exists(imdb_tokenizer_file):
            f=open(imdb_tokenizer_file,'rb')
            imdb_tokenizer=pickle.load(f)
            f.close()
        else:
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(args)

            imdb_tokenizer = Tokenizer(num_words=args.vocab_size, use_spacy=False)
            imdb_tokenizer.fit_on_texts(train_texts)
            f=open(imdb_tokenizer_file,'wb')
            pickle.dump(imdb_tokenizer, f)
            f.close()
        return imdb_tokenizer


def get_embedding_index(file_path, embd_dim):
    global embeddings_index
    embeddings_index = {}
    f = open(file_path, encoding="utf-8")
    for line in tqdm(f, desc = 'get_embedding_index'):
        values = line.split()
        word = "".join(values[:-embd_dim])
        try:
            coefs = np.asarray(values[-embd_dim:], dtype='float32')
        except:
            print(values)
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))


def get_embedding_matrix(args, dataset, num_words, embedding_dims):
    # global num_words, embedding_matrix, word_index
    global embedding_matrix, word_index
    word_index = get_tokenizer(args).word_index
    print('Preparing embedding matrix.')
    # num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dims))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def text_process_for_single(args, tokenizer, texts):
#     maxlen = args.max_len
#     seq = tokenizer.texts_to_sequences(texts)
#     seq = sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
#     return seq
    return

def text_process_for_single_bert(args, tokenizer, texts):
    maxlen = args.max_len
    if 'nli' in args.dataset_name:
        res = []
        for i in tqdm(range(len(texts['hypothesis'])), desc="tokenizing"):
            encoded = tokenizer(texts['hypothesis'][i], texts['premise'][i], padding='max_length', max_length = maxlen)
            res.append(encoded)
        return res
    else:
        res=[]
        for text in tqdm(texts, desc="tokenizing"):
            encoded = tokenizer.encode_plus(
                    text, 
                    None,
                    add_special_tokens = True,
                    max_length = maxlen,
                    pad_to_max_length = True,
                    is_split_into_words = True
                )
            res.append(encoded)
        return res

def label_process_for_single(args, tokenizer, labels):
    maxlen = args.max_len

    out = np.array(labels)
    return out
