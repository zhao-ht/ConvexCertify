import torch
import os
import pickle
import random
from tqdm import tqdm

import sys
sys.path.append("..")
from data.dataset import split_imdb_files, split_sst2_files, split_yelp_files
#from utils import get_tokenizer, update_tokenizer, ModifiedBertTokenizer, ModifiedRobertaTokenizer
from utils import ModifiedBertTokenizer, ModifiedRobertaTokenizer
from utils import generate_synonym_list_by_dict
from utils import get_embedding_index, get_embedding_matrix
from utils import text_process_for_single, text_process_for_single_bert, label_process_for_single
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

import pandas as pd

sys.path.append("..")
from data.dataset import split_imdb_files, split_sst2_files, split_yelp_files
from data.dataset import split_yelp_sentence_attack_files, split_imdb_sentence_attack_files, split_imdb_sentence_attack_files_easy
from data.dataset import split_yelp_editing_attack_files, split_imdb_editing_attack_files
#from utils import get_tokenizer, update_tokenizer, ModifiedBertTokenizer, ModifiedRobertaTokenizer
from utils import ModifiedBertTokenizer, ModifiedRobertaTokenizer
from utils import generate_synonym_list_by_dict
from utils import get_embedding_index, get_embedding_matrix
from utils import text_process_for_single, text_process_for_single_bert, label_process_for_single
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

import pandas as pd

def make_attack_data(args):
    return bert_make_attack_data(args)

def make_synthesized_data(args):
    if args.dataset_name != 'snli':
        if args.base_model == 'bert':
            return bert_make_synthesized_data(args)
        else:
            NotImplementedError()
    else:
        return snli_bert_make_synthesized_data(args)
            # not sure if it is correct

def mapToNumber(gold_label):
    if gold_label == 'entailment':
        return 0
    if gold_label == 'neutral':
        return 1
    if gold_label == 'contradiction':
        return 2
    return -1

def openData(path):
    data = []
    i = 0
    import jsonlines
    with jsonlines.open(path) as f:

        for line in f.iter():
            data.append({
                'premise':line['sentence1'],
                'hypothesis':line['sentence2'],
                'label':mapToNumber(line['gold_label'])
            })

    df = pd.DataFrame(data)
    return df

def removeMinVal(df):
    # print(df['label'].unique())
    # Remove data with label -1 (undefined)
    new_df = df[df.label != -1]
    new_df.reset_index(drop=True, inplace=True)
    # print(new_df['label'].unique())
    return new_df


def snli_bert_make_synthesized_data(args):
    print('#### Importing Dataset ####')
    # # try:
    # df_train = torch.load(
    #     'data_set/snli_1.0/snli_1.0_train')
    # df_val = torch.load('data_set/snli_1.0/snli_1.0_dev')
    # df_test = torch.load(
    #     'data_set/snli_1.0/snli_1.0_test')
    # except:

    df_train = openData(os.path.join(args.work_path, 'data_set/snli_1.0/snli_1.0_train.jsonl'))

    df_val = openData(os.path.join(args.work_path, 'data_set/snli_1.0/snli_1.0_dev.jsonl'))
    df_test =openData(os.path.join(args.work_path, 'data_set/snli_1.0/snli_1.0_test.jsonl'))

    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    df_train = removeMinVal(df_train)
    df_val = removeMinVal(df_val)
    df_test = removeMinVal(df_test)

    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    # torch.save({'label': list(df_train.label.values)}, 'data_set/snli_1.0/snli_1.0_train')
    # torch.save({'label': list(df_val.label.values)}, 'data_set/snli_1.0/snli_1.0_dev')
    # torch.save({'label': list(df_test.label.values)}, 'data_set/snli_1.0/snli_1.0_test')

    print('#### Download Tokenizer & Tokenizing ####')

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
    args.tokenizer = tokenizer

    print('Encoding validation data')
    try:
        encode_val = torch.load(os.path.join(args.work_path, 'temp/encode_val'))
    except:
        encode_val = [tokenizer(df_val.premise.tolist(), df_val.hypothesis.tolist(),
                                 return_tensors='pt', padding='max_length', max_length=args.max_len),
                        tokenizer(df_val.premise.tolist(),
                                  return_tensors='pt', padding='max_length', max_length=args.max_len),
                        tokenizer(df_val.hypothesis.tolist(),
                                  return_tensors='pt', padding='max_length', max_length=args.max_len)
                        ]
        torch.save(encode_val, os.path.join(args.work_path, 'temp/encode_val'))

    labels_val = df_val['label']


    print('Encoding training data')
    try:
        encode_train = torch.load(os.path.join(args.work_path, 'temp/encode_train'))
    except:
        encode_train = [tokenizer(df_train.premise.tolist(), df_train.hypothesis.tolist(),
                                 return_tensors='pt', padding='max_length', max_length=args.max_len),
                        tokenizer(df_train.premise.tolist(),
                                  return_tensors='pt', padding='max_length', max_length=args.max_len),
                        tokenizer(df_train.hypothesis.tolist(),
                                  return_tensors='pt', padding='max_length', max_length=args.max_len)
                        ]
        torch.save(encode_train, os.path.join(args.work_path, 'temp/encode_train'))

    labels_train = df_train['label']



    print('Encoding test data')
    try:
        encode_test = torch.load(os.path.join(args.work_path, 'temp/encode_test'))
    except:
        encode_test = [tokenizer(df_test.premise.tolist(), df_test.hypothesis.tolist(),
                                 return_tensors='pt', padding='max_length', max_length=args.max_len),
                        tokenizer(df_test.premise.tolist(),
                                  return_tensors='pt', padding='max_length', max_length=args.max_len),
                        tokenizer(df_test.hypothesis.tolist(),
                                  return_tensors='pt', padding='max_length', max_length=args.max_len)
                        ]
        torch.save(encode_test, os.path.join(args.work_path, 'temp/encode_test'))

    labels_test = df_test['label']

    import json
    with open(os.path.join(args.work_path,args.certified_neighbors_file_path)) as f:
        syn_dict = json.load(f)
    syn_data = {}
    unk_tok = tokenizer.vocab['[UNK]']
    # Tokenize syn data
    print("Tokenize syn data.")
    for key in syn_dict:
        if len(syn_dict[key]) != 0:
            temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)[
                'input_ids']
            temp = [t for t in temp if tokenizer.ids_to_tokens[t] != '[UNK]']
            token_of_key = \
            tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]
            syn_data[token_of_key] = temp

    syn_data[unk_tok] = [unk_tok]

    print('building dataset and dataloader')
    train_data = SynthesizedData_TextLikeSyn_Bert_snli(args, encode_train, labels_train, syn_data, update=True, tokens_to_ids=None,
                                                  ids_to_tokens=None)
    dev_data = SynthesizedData_TextLikeSyn_Bert_snli(args, encode_val, labels_val, syn_data, update=False,
                                                tokens_to_ids=train_data.tokens_to_ids,
                                                ids_to_tokens=train_data.ids_to_tokens)
    test_data = SynthesizedData_TextLikeSyn_Bert_snli(args, encode_test, labels_test, syn_data, update=False,
                                                 tokens_to_ids=train_data.tokens_to_ids,
                                                 ids_to_tokens=train_data.ids_to_tokens)

    return train_data, dev_data, test_data, syn_data


def bert_make_synthesized_data(args):
    dataset = args.dataset_name
    args.label_size = 2
    if dataset == 'imdb':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(args)
    elif dataset == 'sst2':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_sst2_files(args)
    elif dataset == 'yelp':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_yelp_files(args)
    else:
        NotImplementedError()



    import transformers
    #tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = ModifiedBertTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
    args.tokenizer = tokenizer
    
    if args.synonyms_from_file:
        if dataset == 'imdb':
            filename= args.imdb_bert_synonyms_file_path
        elif dataset == 'sst2':
            filename= args.sst2_bert_synonyms_file_path
        elif dataset == 'yelp':
            filename= args.yelp_bert_synonyms_file_path
        elif dataset == 'agnews':
            filename= args.agnews_bert_synonyms_file_path
            
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        x_train=saved['x_train']
        x_test=saved['x_test']
        x_dev=saved['x_dev']
        y_train=saved['y_train']
        y_test=saved['y_test']
        y_dev=saved['y_dev']
        
    else:
        print("Preparing synonyms.")
        
        syn_dict = {}
        
        import json
        with open(args.certified_neighbors_file_path) as f:
            syn_dict = json.load(f)   
        syn_data = {}
        unk_tok = tokenizer.vocab['[UNK]']
        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            if len(syn_dict[key])!=0:
                temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']
                temp = [t for t in temp if tokenizer.ids_to_tokens[t]!= '[UNK]']
                token_of_key = tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]
                syn_data[token_of_key] = temp

        syn_data[unk_tok] = [unk_tok]

        # Tokenize the training data
        print("Tokenize training data.")
        x_train = text_process_for_single_bert(args, tokenizer, train_texts)
        y_train = label_process_for_single(args, tokenizer, train_labels)

        x_dev = text_process_for_single_bert(args, tokenizer, dev_texts)
        y_dev = label_process_for_single(args, tokenizer, dev_labels)

        x_test = text_process_for_single_bert(args, tokenizer, test_texts)
        y_test = label_process_for_single(args, tokenizer, test_labels)

        if args.dataset_name == 'imdb':
            filename= args.imdb_bert_synonyms_file_path
        elif args.dataset_name == 'sst2':
            filename= args.sst2_bert_synonyms_file_path
        elif args.dataset_name == 'yelp':
            filename = args.yelp_bert_synonyms_file_path
        elif dataset == 'agnews':
            filename = args.agnews_bert_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        saved['x_train']=x_train
        saved['x_test']=x_test
        saved['x_dev']=x_dev
        saved['y_train']=y_train
        saved['y_test']=y_test
        saved['y_dev']=y_dev
        pickle.dump(saved,f)
        f.close()

    print('building dataset and dataloader')
    train_data = SynthesizedData_TextLikeSyn_Bert(args, x_train, y_train, syn_data, update = True, tokens_to_ids = None, ids_to_tokens = None)
    dev_data = SynthesizedData_TextLikeSyn_Bert(args, x_dev, y_dev, syn_data, update = False, tokens_to_ids = train_data.tokens_to_ids, ids_to_tokens = train_data.ids_to_tokens)
    test_data = SynthesizedData_TextLikeSyn_Bert(args, x_test, y_test, syn_data, update = False, tokens_to_ids = train_data.tokens_to_ids, ids_to_tokens = train_data.ids_to_tokens)
    
    return train_data, dev_data, test_data, syn_data


def bert_make_attack_data(args):
    dataset = args.dataset_name
    args.label_size = 2
    if dataset == 'imdb':
        if args.attack_method == 'sentence':
            test_texts, test_labels = split_imdb_sentence_attack_files_easy(args)
        elif args.attack_method == 'editing':
            test_texts, test_labels = split_imdb_editing_attack_files(args)
        else:
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(args)
    elif dataset == 'yelp':
        if args.attack_method == 'sentence':
            test_texts, test_labels = split_yelp_sentence_attack_files(args)
        elif args.attack_method == 'editing':
            test_texts, test_labels = split_yelp_editing_attack_files(args)
        else:
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_yelp_files(args)
    else:
        NotImplementedError()

    import transformers
    # tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = ModifiedBertTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
    args.tokenizer = tokenizer

    ####### just to make sure input format ########
    print("Preparing synonyms.")

    syn_dict = {}

    import json
    with open(args.certified_neighbors_file_path) as f:
        syn_dict = json.load(f)
    syn_data = {}
    unk_tok = tokenizer.vocab['[UNK]']
    # Tokenize syn data
    print("Tokenize syn data.")
    for key in syn_dict:
        if len(syn_dict[key]) != 0:
            temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)[
                'input_ids']
            temp = [t for t in temp if tokenizer.ids_to_tokens[t] != '[UNK]']
            token_of_key = \
            tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]
            syn_data[token_of_key] = temp

    syn_data[unk_tok] = [unk_tok]
    ################################################
    # Tokenize the testing data
    x_test = text_process_for_single_bert(args, tokenizer, test_texts)
    y_test = label_process_for_single(args, tokenizer, test_labels)

    print('building dataset and dataloader')
    test_data = SynthesizedData_TextLikeSyn_Bert(args, x_test, y_test, syn_data)
    return test_data

# def imdb_make_synthesized_data(args):
#     dataset = args.dataset_name
#     assert (dataset == 'imdb')
#     args.label_size = 2
#     train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(
#         args)
#     tokenizer_file = os.path.join(
#         args.work_path, "temp/imdb_tokenizer_with_ceritified_syndata.pickle")
#
#     if args.synonyms_from_file:
#         filename = args.imdb_synonyms_file_path
#         f = open(filename, 'rb')
#         saved = pickle.load(f)
#         f.close()
#         syn_data = saved['syn_data']
#         args.embeddings = saved['embeddings']
#         args.vocab_size = saved['vocab_size']
#         x_train = saved['x_train']
#         x_test = saved['x_test']
#         x_dev = saved['x_dev']
#         y_train = saved['y_train']
#         y_test = saved['y_test']
#         y_dev = saved['y_dev']
#
#         tokenizer = get_tokenizer(args)
#         dictori = tokenizer.index_word
#
#         f = open(tokenizer_file, 'rb')
#         tokenizer = pickle.load(f)
#         f.close()
#         update_tokenizer(dataset, tokenizer)
#         dictnew = tokenizer.index_word
#
#         print("Check the tokenizer.")
#         for key in dictori:
#             assert (dictori[key] == dictnew[key])
#     else:
#         tokenizer = get_tokenizer(args)
#         print("len of tokenizer before updata.", len(tokenizer.index_word))
#         print("Preparing synonyms.")
#
#         syn_texts_for_tokenizer = []
#         syn_dict = {}
#
#         import json
#         with open(args.certified_neighbors_file_path) as f:
#             certified_neighbors = json.load(f)
#
#         syn_data = [[] for i in range(1 + len(tokenizer.index_word))]
#         for index in tqdm(tokenizer.index_word, desc='processing synoyms'):
#             word = tokenizer.index_word[index]
#             syn_text = " ".join(
#                 generate_synonym_list_by_dict(certified_neighbors, word))
#             syn_texts_for_tokenizer.append(syn_text)
#             syn_dict[index] = syn_text
#
#         # update tokenizer
#         print("Fit on syn texts.")
#         tokenizer.fit_on_texts(
#             syn_texts_for_tokenizer,
#             freq_count=0)  # to keep the original index order of the tokenizer
#         update_tokenizer(dataset, tokenizer)
#         f = open(tokenizer_file, 'wb')
#         pickle.dump(tokenizer, f)
#         f.close()
#
#         print("len of tokenizer after updata.", len(tokenizer.index_word))
#         assert (len(tokenizer.index_word) < args.vocab_size)
#
#         # Tokenize syn data
#         #print("Tokenize syn data.")
#         for key in tqdm(syn_dict, desc = 'Tokenize syn data'):
#             temp = tokenizer.texts_to_sequences([syn_dict[key]])
#             syn_data[key] = temp[0]
#
#         # make embd according to the updated tokenizer
#         embedding_dim = args.embedding_dim
#         pretrained = args.embedding_file_path
#         get_embedding_index(pretrained, embedding_dim)
#         embedding_matrix = get_embedding_matrix(args, dataset, args.vocab_size,
#                                                 embedding_dim)
#         args.embeddings = torch.FloatTensor(embedding_matrix)
#
#         # Tokenize the training data
#         print("Tokenize training data.")
#
#         x_train = text_process_for_single(args, tokenizer, train_texts)
#         y_train = label_process_for_single(args, tokenizer, train_labels)
#
#         x_dev = text_process_for_single(args, tokenizer, dev_texts)
#         y_dev = label_process_for_single(args, tokenizer, dev_labels)
#
#         x_test = text_process_for_single(args, tokenizer, test_texts)
#         y_test = label_process_for_single(args, tokenizer, test_labels)
#
#         filename = args.imdb_synonyms_file_path
#         f = open(filename, 'wb')
#         saved = {}
#         saved['syn_data'] = syn_data
#         saved['embeddings'] = args.embeddings
#         saved['vocab_size'] = args.vocab_size
#         saved['x_train'] = x_train
#         saved['x_test'] = x_test
#         saved['x_dev'] = x_dev
#         saved['y_train'] = y_train
#         saved['y_test'] = y_test
#         saved['y_dev'] = y_dev
#         pickle.dump(saved, f)
#         f.close()
#
#     vocab_freq = torch.zeros(args.embeddings.shape[0]).to(
#         args.embeddings.dtype)
#     for index in tokenizer.index_word:
#         word = tokenizer.index_word[index]
#         freq = tokenizer.word_counts[word]
#         vocab_freq[index] = freq
#     args.vocab_freq = vocab_freq
#
#     print('building dataset and dataloader')
#     train_data = SynthesizedData_TextLikeSyn(args, x_train, y_train, syn_data)
#     dev_data = SynthesizedData_TextLikeSyn(args, x_dev, y_dev, syn_data)
#     test_data = SynthesizedData_TextLikeSyn(args, x_test, y_test, syn_data)
#
#     return train_data, dev_data, test_data, syn_data
#
#
# class SynthesizedData_TextLikeSyn(torch.utils.data.Dataset):
#     def __init__(self, args, x, y, syn_data):
#         super(SynthesizedData_TextLikeSyn, self).__init__()
#         self.x = x.copy()
#         self.y = y.copy()
#         self.syn_data = syn_data.copy()
#         self.args = args
#
#         for x in range(len(self.syn_data)):
#             self.syn_data[x] = [
#                 syn_word for syn_word in self.syn_data[x] if syn_word != x
#             ]
#
#         self.len_voc = len(self.syn_data) + 1
#
#     def transform(self, sent, label, text_like_syn,
#                   text_like_syn_valid):
#
#         return torch.tensor(sent, dtype=torch.long), torch.tensor(label, dtype=torch.long), torch.tensor(text_like_syn, dtype=torch.long), \
#                 torch.tensor(text_like_syn_valid, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self,
#                     index,
#                     max_num_anch_per_sent=100,
#                     num_text_like_syn=20):
#
#         sent = self.x[index]
#         label = self.y[index]
#
#         text_like_syn = []
#         text_like_syn_valid = []
#
#         anch_cnt = 0
#         anch_place = []
#         for i, x in enumerate(sent):
#             text_like_syn_valid.append([])
#             if x < len(self.syn_data):
#                 text_like_syn.append(self.syn_data[x].copy())
#                 anch_cnt += 1
#                 anch_place.append(i)
#             else:
#                 text_like_syn.append([])
#
#         if anch_cnt > max_num_anch_per_sent:
#             anch_place = random.sample(anch_place, max_num_anch_per_sent)
#
#         for i, x in enumerate(sent):
#             temp = text_like_syn[i]
#             len_temp = len(temp)
#             if len_temp == 0 or i not in anch_place:
#                 text_like_syn_valid[i] = [1]
#                 text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn - 1)])
#                 text_like_syn[i] = [x]
#                 text_like_syn[i].extend([0 for times in range(num_text_like_syn - 1)])
#             elif len_temp >= num_text_like_syn - 1:
#                 temp = random.sample(temp, num_text_like_syn - 1)
#                 temp.append(x)
#                 text_like_syn[i] = temp
#                 text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
#                 assert (len(text_like_syn[i]) == num_text_like_syn)
#             else:
#                 temp.append(x)
#                 text_like_syn_valid[i] = [1 for times in range(len(temp))]
#                 while (len(temp) < num_text_like_syn):
#                     temp.append(0)
#                     text_like_syn_valid[i].append(0)
#                 text_like_syn[i] = temp
#                 assert (len(text_like_syn[i]) == num_text_like_syn)
#
#         return self.transform(sent, label, text_like_syn, text_like_syn_valid)
#
    
class SynthesizedData_TextLikeSyn_Bert(torch.utils.data.Dataset):
    def __init__(self, args, x, y, syn_data, update = True, tokens_to_ids = None, ids_to_tokens = None):
        super(SynthesizedData_TextLikeSyn_Bert, self).__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()
        self.args = args
        
        tokenizer = self.args.tokenizer
        self.bert_base_tokens_to_ids = tokenizer.vocab
        self.bert_base_ids_to_tokens = tokenizer.ids_to_tokens
        #self.len_voc = len(self.syn_data) + 1
        self.collate_item = ["sent", "mask", "token_type_ids", "label", "text_like_syn", \
                "text_like_syn_valid", "ibp_input"]
        
        self.tokens_to_ids = tokens_to_ids
        self.ids_to_tokens = ids_to_tokens
        
        if args.vocab_size > 0 :
            if update: 
                self.update_vocab()
            if self.tokens_to_ids != None:
                self.update_input()
        else:
            self.tokens_to_ids = self.bert_base_tokens_to_ids
            self.ids_to_tokens = self.bert_base_ids_to_tokens
                
    def old2new_id(self, token_id):
        token = self.bert_base_tokens_to_ids[token_id]
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        else:
            return self.tokens_to_ids['[UNK]']
    
    def update_input(self):
        
        new_syn_data = dict([])
        
        ### update synonyms token ids ###
        for w in self.syn_data:
            new_id = self.old2new_id(w)
            new_syn_data[new_id] = [self.old2new_id(t) for t in self.syn_data[w]]
        
        self.syn_data = new_syn_data
        
        ### update dataset tokens  ####
        data_size = len(self.y)
        
        for index in tqdm(range(data_size), desc = "update token ids"):
            encoded = self.x[index]
            sent = encoded["input_ids"]
            self.x[index]["input_ids"] = [self.old2new_id(t) for t in sent]
        
    def update_vocab(self):
        data_size = len(self.y)
        
        c = Counter()
        
        for index in tqdm(range(data_size), desc = "update vocabulary"):
            encoded = self.x[index]
            sent = encoded["input_ids"]
            c.update(sent)
        
        high_freq = c.most_common(self.args.vocab_size)
        vocab_set = set([w[0] for w in high_freq])
        vocab_set.add(self.bert_base_tokens_to_ids['[CLS]'])
        vocab_set = list(vocab_set)
        vocab_set.sort()
        
        self.tokens_to_ids = dict([])
        self.ids_to_tokens = dict([])
        
        for i,w in enumerate(vocab_set):
            self.ids_to_tokens[i] = self.bert_base_ids_to_tokens[w]
            self.tokens_to_ids[self.ids_to_tokens[i]] = i

    
    def transform(self, sent, mask, token_type_ids, label, text_like_syn, text_like_syn_valid):
        return torch.tensor(sent, dtype = torch.long), torch.tensor(mask, dtype = torch.long), torch.tensor(token_type_ids, dtype = torch.long),\
            torch.tensor(label, dtype = torch.long), torch.tensor(text_like_syn, dtype = torch.long), torch.tensor(text_like_syn_valid, dtype = torch.long)
   
    def __getitem__(self, index, max_num_anch_per_sent=20,
                    num_text_like_syn=20, with_CLS = False):
        max_num_anch_per_sent = self.args.perturbed_num
        encoded = self.x[index]
        label = self.y[index]
        
        sent = encoded["input_ids"]
        mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]

        if not with_CLS:
            sent = sent[1:]
            mask = mask[1:]
            token_type_ids = token_type_ids[1:]
            
        text_like_syn=[]
        text_like_syn_valid=[]
        anch_cnt = 0
        anch_place = []
        for i, x in enumerate(sent):
            text_like_syn_valid.append([])
            if x in self.syn_data:
                text_like_syn.append(self.syn_data[x].copy())
                anch_cnt += 1
                anch_place.append(i)
            else:
                text_like_syn.append([])

        if anch_cnt > max_num_anch_per_sent:
            anch_place = random.sample(anch_place, max_num_anch_per_sent)
            
        for i, x in enumerate(sent):
            temp = text_like_syn[i]
            len_temp = len(temp)
            if len_temp == 0 or i not in anch_place:
                text_like_syn_valid[i] = [1]
                text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn - 1)])
                text_like_syn[i] = [x]
                text_like_syn[i].extend([0 for times in range(num_text_like_syn - 1)])
            elif len_temp >= num_text_like_syn - 1:
                temp = random.sample(temp, num_text_like_syn - 1)
                temp.append(x)
                text_like_syn[i] = temp
                text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
                assert (len(text_like_syn[i]) == num_text_like_syn)
            else:
                temp.append(x)
                text_like_syn_valid[i] = [1 for times in range(len(temp))]
                while (len(temp) < num_text_like_syn):
                    temp.append(0)
                    text_like_syn_valid[i].append(0)
                text_like_syn[i] = temp
                assert (len(text_like_syn[i]) == num_text_like_syn)
        
        sent, mask, token_type_ids, label, text_like_syn, text_like_syn_valid = self.transform(sent, mask,\
            token_type_ids, label, text_like_syn, text_like_syn_valid)
        
        return {"sent":sent, "mask":mask, "token_type_ids":token_type_ids, \
                "label":label, "text_like_syn":text_like_syn, \
                "text_like_syn_valid":text_like_syn_valid
        }
    
    def __len__(self):
        return len(self.y)


class SynthesizedData_TextLikeSyn_Bert_lm(torch.utils.data.Dataset):
    def __init__(self, args, x, y, syn_data, update=True, tokens_to_ids=None, ids_to_tokens=None,min_log_p_diff=-5.0):
        super(SynthesizedData_TextLikeSyn_Bert_lm, self).__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()
        self.args = args
        self.min_log_p_diff=min_log_p_diff

        tokenizer = self.args.tokenizer
        self.bert_base_tokens_to_ids = tokenizer.vocab
        self.bert_base_ids_to_tokens = tokenizer.ids_to_tokens
        # self.len_voc = len(self.syn_data) + 1
        self.collate_item = ["sent", "mask", "token_type_ids", "label", "text_like_syn", \
                             "text_like_syn_valid", "ibp_input"]

        self.tokens_to_ids = tokens_to_ids
        self.ids_to_tokens = ids_to_tokens

        if args.vocab_size > 0:
            if update:
                self.update_vocab()
            if self.tokens_to_ids != None:
                self.update_input()
        else:
            self.tokens_to_ids = self.bert_base_tokens_to_ids
            self.ids_to_tokens = self.bert_base_ids_to_tokens

    def old2new_id(self, token_id):
        token = self.bert_base_tokens_to_ids[token_id]
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        else:
            return self.tokens_to_ids['[UNK]']

    def update_input(self):

        new_syn_data = dict([])

        ### update synonyms token ids ###
        for w in self.syn_data:
            new_id = self.old2new_id(w)
            new_syn_data[new_id] = [self.old2new_id(t) for t in self.syn_data[w]]

        self.syn_data = new_syn_data

        ### update dataset tokens  ####
        data_size = len(self.y)

        for index in tqdm(range(data_size), desc="update token ids"):
            encoded = self.x[index]
            sent = encoded["input_ids"]
            self.x[index]["input_ids"] = [self.old2new_id(t) for t in sent]

    def update_vocab(self):
        data_size = len(self.y)

        c = Counter()

        for index in tqdm(range(data_size), desc="update vocabulary"):
            encoded = self.x[index]
            sent = encoded["input_ids"]
            c.update(sent)

        high_freq = c.most_common(self.args.vocab_size)
        vocab_set = set([w[0] for w in high_freq])
        vocab_set.add(self.bert_base_tokens_to_ids['[CLS]'])
        vocab_set = list(vocab_set)
        vocab_set.sort()

        self.tokens_to_ids = dict([])
        self.ids_to_tokens = dict([])

        for i, w in enumerate(vocab_set):
            self.ids_to_tokens[i] = self.bert_base_ids_to_tokens[w]
            self.tokens_to_ids[self.ids_to_tokens[i]] = i

    def transform(self, sent, mask, token_type_ids, label, text_like_syn, text_like_syn_valid):
        return torch.tensor(sent, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids,
                                                                                                        dtype=torch.long), \
               torch.tensor(label, dtype=torch.long), torch.tensor(text_like_syn, dtype=torch.long), torch.tensor(
            text_like_syn_valid, dtype=torch.long)

    def __getitem__(self, index, max_num_anch_per_sent=20,
                    num_text_like_syn=20, with_CLS=False):
        max_num_anch_per_sent = self.args.perturbed_num
        encoded = self.x[index]
        label = self.y[index]

        sent = encoded["input_ids"]
        mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]

        if not with_CLS:
            sent = sent[1:]
            mask = mask[1:]
            token_type_ids = token_type_ids[1:]

        text_like_syn = []
        text_like_syn_valid = []
        anch_cnt = 0
        anch_place = []
        for i, x in enumerate(sent):
            text_like_syn_valid.append([])
            if x in self.syn_data:
                text_like_syn.append(self.syn_data[x].copy())
                anch_cnt += 1
                anch_place.append(i)
            else:
                text_like_syn.append([])

        if anch_cnt > max_num_anch_per_sent:
            anch_place = random.sample(anch_place, max_num_anch_per_sent)

        for i, x in enumerate(sent):
            temp = text_like_syn[i]
            len_temp = len(temp)
            if len_temp == 0 or i not in anch_place:
                text_like_syn_valid[i] = [1]
                text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn - 1)])
                text_like_syn[i] = [x]
                text_like_syn[i].extend([0 for times in range(num_text_like_syn - 1)])
            elif len_temp >= num_text_like_syn - 1:
                temp = random.sample(temp, num_text_like_syn - 1)
                temp.append(x)
                text_like_syn[i] = temp
                text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
                assert (len(text_like_syn[i]) == num_text_like_syn)
            else:
                temp.append(x)
                text_like_syn_valid[i] = [1 for times in range(len(temp))]
                while (len(temp) < num_text_like_syn):
                    temp.append(0)
                    text_like_syn_valid[i].append(0)
                text_like_syn[i] = temp
                assert (len(text_like_syn[i]) == num_text_like_syn)

        sent, mask, token_type_ids, label, text_like_syn, text_like_syn_valid = self.transform(sent, mask, \
                                                                                               token_type_ids, label,
                                                                                               text_like_syn,
                                                                                               text_like_syn_valid)

        return {"sent": sent, "mask": mask, "token_type_ids": token_type_ids, \
                "label": label, "text_like_syn": text_like_syn, \
                "text_like_syn_valid": text_like_syn_valid
                }

    def __len__(self):
        return len(self.y)


class SynthesizedData_TextLikeSyn_Bert_snli(torch.utils.data.Dataset):
    def __init__(self, args, x, y, syn_data, update=True, tokens_to_ids=None, ids_to_tokens=None):
        super(SynthesizedData_TextLikeSyn_Bert_snli, self).__init__()
        self.x = x[0].copy()
        self.p=x[1].copy()
        self.h=x[2].copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()
        self.args = args

        tokenizer = self.args.tokenizer
        self.bert_base_tokens_to_ids = tokenizer.vocab
        self.bert_base_ids_to_tokens = tokenizer.ids_to_tokens
        # self.len_voc = len(self.syn_data) + 1
        self.collate_item = ["sent", "premises","hypothesis","mask","mask_premises","mask_hypothesis", "token_type_ids", "label", "text_like_syn", \
                             "text_like_syn_valid", "ibp_input"]

        self.tokens_to_ids = tokens_to_ids
        self.ids_to_tokens = ids_to_tokens

        if args.vocab_size > 0:
            if update:
                self.update_vocab()
            if self.tokens_to_ids != None:
                self.update_input()
        else:
            self.tokens_to_ids = self.bert_base_tokens_to_ids
            self.ids_to_tokens = self.bert_base_ids_to_tokens

    def old2new_id(self, token_id):
        token = self.bert_base_tokens_to_ids[token_id]
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        else:
            return self.tokens_to_ids['[UNK]']

    def update_input(self):

        new_syn_data = dict([])

        ### update synonyms token ids ###
        for w in self.syn_data:
            new_id = self.old2new_id(w)
            new_syn_data[new_id] = [self.old2new_id(t) for t in self.syn_data[w]]

        self.syn_data = new_syn_data

        ### update dataset tokens  ####
        data_size = len(self.y)

        for index in tqdm(range(data_size), desc="update token ids"):
            encoded = self.x[index]
            sent = encoded["input_ids"]
            self.x[index]["input_ids"] = [self.old2new_id(t) for t in sent]

    def update_vocab(self):
        data_size = len(self.y)

        c = Counter()

        for index in tqdm(range(data_size), desc="update vocabulary"):
            encoded = self.x[index]
            sent = encoded["input_ids"]
            c.update(sent)

        high_freq = c.most_common(self.args.vocab_size)
        vocab_set = set([w[0] for w in high_freq])
        vocab_set.add(self.bert_base_tokens_to_ids['[CLS]'])
        vocab_set = list(vocab_set)
        vocab_set.sort()

        self.tokens_to_ids = dict([])
        self.ids_to_tokens = dict([])

        for i, w in enumerate(vocab_set):
            self.ids_to_tokens[i] = self.bert_base_ids_to_tokens[w]
            self.tokens_to_ids[self.ids_to_tokens[i]] = i

    def transform(self, sent,premises,hypothesis, mask, \
                                                                                               token_type_ids, label,
                                                                                               text_like_syn,
                                                                                               text_like_syn_valid,
                                                                                                                   text_like_syn_premises,
                                                                                                                   text_like_syn_valid_premises,
                                                                                                                   text_like_syn_hypothesis,
                                                                                                                   text_like_syn_valid_hypothesis):
        return torch.tensor(sent, dtype=torch.long),torch.tensor(premises, dtype=torch.long),torch.tensor(hypothesis, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids,
                                                                                                        dtype=torch.long), \
               torch.tensor(label, dtype=torch.long), torch.tensor(text_like_syn, dtype=torch.long), torch.tensor(
            text_like_syn_valid, dtype=torch.long),torch.tensor(text_like_syn_premises, dtype=torch.long),torch.tensor(text_like_syn_valid_premises, dtype=torch.long),\
               torch.tensor(text_like_syn_hypothesis, dtype=torch.long),torch.tensor(text_like_syn_valid_hypothesis, dtype=torch.long)



    def __getitem__(self, index, max_num_anch_per_sent=20,
                    num_text_like_syn=20, with_CLS=False):
        max_num_anch_per_sent = self.args.perturbed_num
        encoded = self.x
        label = self.y[index]

        sent = encoded["input_ids"][index]
        premises=self.p["input_ids"][index]
        hypothesis = self.h["input_ids"][index]
        mask = encoded["attention_mask"][index]
        mask_premises = self.p["attention_mask"][index]
        mask_hypothesis = self.h["attention_mask"][index]

        token_type_ids = encoded["token_type_ids"][index]

        if not with_CLS:
            sent = sent[1:]
            premises=premises[1:]
            hypothesis=hypothesis[1:]
            mask = mask[1:]
            mask_premises=mask_premises[1:]
            mask_hypothesis=mask_hypothesis[1:]
            token_type_ids = token_type_ids[1:]

        def get_syn(sent,per=True):

            text_like_syn = []
            text_like_syn_valid = []
            anch_cnt = 0
            anch_place = []
            for i, x in enumerate(sent):
                text_like_syn_valid.append([])
                if (int(x) in self.syn_data) and per:
                    text_like_syn.append(self.syn_data[int(x)].copy())
                    anch_cnt += 1
                    anch_place.append(i)
                else:
                    text_like_syn.append([])

            if anch_cnt > max_num_anch_per_sent:
                anch_place = random.sample(anch_place, max_num_anch_per_sent)

            for i, x in enumerate(sent):
                temp = text_like_syn[i]
                len_temp = len(temp)
                if len_temp == 0 or i not in anch_place:
                    text_like_syn_valid[i] = [1]
                    text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn - 1)])
                    text_like_syn[i] = [x]
                    text_like_syn[i].extend([0 for times in range(num_text_like_syn - 1)])
                elif len_temp >= num_text_like_syn - 1:
                    temp = random.sample(temp, num_text_like_syn - 1)
                    temp.append(x)
                    text_like_syn[i] = temp
                    text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
                    assert (len(text_like_syn[i]) == num_text_like_syn)
                else:
                    temp.append(x)
                    text_like_syn_valid[i] = [1 for times in range(len(temp))]
                    while (len(temp) < num_text_like_syn):
                        temp.append(0)
                        text_like_syn_valid[i].append(0)
                    text_like_syn[i] = temp
                    assert (len(text_like_syn[i]) == num_text_like_syn)
            return text_like_syn, text_like_syn_valid

        text_like_syn,text_like_syn_valid=get_syn(sent)
        text_like_syn_premises, text_like_syn_valid_premises = get_syn(premises,per=False)
        text_like_syn_hypothesis, text_like_syn_valid_hypothesis = get_syn(hypothesis)

        sent, premises, hypothesis, mask, token_type_ids, label, text_like_syn,   text_like_syn_valid,  text_like_syn_premises, text_like_syn_valid_premises, text_like_syn_hypothesis,text_like_syn_valid_hypothesis= self.transform(sent,premises,hypothesis, mask, token_type_ids, label,   text_like_syn,   text_like_syn_valid,   text_like_syn_premises, text_like_syn_valid_premises, text_like_syn_hypothesis,text_like_syn_valid_hypothesis)
        # if (sent - text_like_syn[:, 0]).sum() !=0:
        #     print('error in data')
        # if (premises - text_like_syn_premises[:, 0]).sum() !=0:
        #     print('error in data')
        # if (hypothesis - text_like_syn_hypothesis[:, 0]).sum() != 0:
        #     print('error in data')
        return {"sent": sent, "premises": premises,"hypothesis":hypothesis, "mask": mask,"mask_premises":mask_premises,
                "mask_hypothesis":mask_hypothesis,"token_type_ids": token_type_ids, \
                "label": label, "text_like_syn": text_like_syn, \
                "text_like_syn_valid": text_like_syn_valid,"text_like_syn_premises":text_like_syn_premises,
        "text_like_syn_valid_premises":text_like_syn_valid_premises,"text_like_syn_hypothesis":text_like_syn_hypothesis,
                "text_like_syn_valid_hypothesis":text_like_syn_valid_hypothesis
                }

    def __len__(self):
        return len(self.y)
