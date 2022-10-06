import os
import re
import sys
import csv
from tqdm import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
import os
import re
import sys
import csv
import random
from tqdm import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

from data.generate_attack_hidden_killer import Hidden_Killer
from data.generate_attack_editing import editor
import pandas as pd

def read_imdb_files(args, filetype):

    all_labels = []
    for _ in range(12500):
        all_labels.append(0)
    for _ in range(12500):
        all_labels.append(1)

    all_texts = []
    file_list = []
    path = os.path.join(args.work_path, 'data_set/aclImdb/')
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in tqdm(file_list, desc = 'processing imdb'):
        with open(file_name, 'r', encoding='utf-8') as f:
    
            from nltk import word_tokenize
            x_raw = f.readlines()[0].strip().replace('<br />', ' ')
            x_toks = word_tokenize(x_raw)
            #num_words += len(x_toks)
            all_texts.append(' '.join(x_toks))

    return all_texts, all_labels
def split_snli_files(args):
    pass
def split_imdb_files(args):
    filename = os.path.join(args.work_path, "temp/split_imdb_files")
    if os.path.exists(filename):
        print('Read processed IMDB dataset')
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_texts=saved['train_texts']
        train_labels=saved['train_labels']
        test_texts=saved['test_texts']
        test_labels=saved['test_labels']
        dev_texts=saved['dev_texts']
        dev_labels=saved['dev_labels']
    else:
        print('Processing IMDB dataset')
        train_texts, train_labels = read_imdb_files(args, 'train')
        test_texts, test_labels = read_imdb_files(args, 'test')
        dev_texts = test_texts[12500-500:12500] + test_texts[25000-500:25000]
        dev_labels = test_labels[12500-500:12500] + test_labels[25000-500:25000]

        test_texts = test_texts[:12500-500] + test_texts[12500:25000-500]
        test_labels = test_labels[:12500-500] + test_labels[12500:25000-500]

        # test_texts = test_texts[:2500]
        # test_labels = test_labels[:2500]
        
        f=open(filename,'wb')
        saved={}
        saved['train_texts']=train_texts
        saved['train_labels']=train_labels
        saved['test_texts']=test_texts
        saved['test_labels']=test_labels
        saved['dev_texts']=dev_texts
        saved['dev_labels']=dev_labels
        pickle.dump(saved,f)
        f.close()
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def split_sst2_files(args):
    print('Processing SST2 dataset')
    from datasets import load_dataset
    import numpy as np
    # dataset = load_dataset("yahoo_answers_topics")
    # dataset = load_dataset("ag_news")
    # train_data=dataset['train']['']
    dataset = load_dataset("sst", "default")

    def convert(train_texts, train_labels):
        train_texts_new = []
        train_labels_new = []
        for i in range(len(train_texts)):
            if train_labels[i] <= 0.4 or train_labels[i] > 0.6:
                train_texts_new.append(train_texts[i])
                if train_labels[i] <= 0.4:
                    train_labels_new.append(0)
                elif train_labels[i] > 0.6:
                    train_labels_new.append(1)
        return train_texts_new, train_labels_new

    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['label']

    dev_texts = dataset['validation']['sentence']
    dev_labels = dataset['validation']['label']

    test_texts = dataset['test']['sentence']
    test_labels = dataset['test']['label']

    train_texts, train_labels = convert(train_texts, train_labels)
    dev_texts, dev_labels = convert(dev_texts, dev_labels)
    test_texts, test_labels = convert(test_texts, test_labels)

    
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def read_agnews_files(filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    path = r'./PWWS/data_set/ag_news_csv/{}.csv'.format(filetype)
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        content = line[1] + ". " + line[2]
        texts.append(content)
        labels_index.append(line[0])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(2, dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    return texts, labels, labels_index

def split_agnews_files():
    print("Processing AG's News dataset")
    train_texts, train_labels, _ = read_agnews_files('train')  # 120000
    test_texts, test_labels, _ = read_agnews_files('test')  # 7600
    return train_texts, train_labels, test_texts, test_labels,test_texts, test_labels


def split_yelp_files(args):

    def load_sent(path, max_size=-1):
        data = []
        label=[]
        with open(path+'.0') as f:
            for line in f:
                if len(data) == max_size:
                    break
                data.append(line)
                label.append(0)
        with open(path + '.1') as f:
            for line in f:
                if len(data) == max_size:
                    break
                data.append(line)
                label.append(1)
        return data,label

    print('Processing Yelp dataset')
    path = os.path.join(args.work_path, 'data_set/yelp')
    train_texts,train_labels = load_sent(path+'/sentiment.train')
    dev_texts,dev_labels = load_sent(path+'/sentiment.dev')
    test_texts, test_labels = load_sent(path + '/sentiment.test')


    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels



def read_yahoo_files():
    text_data_dir = './PWWS/data_set/yahoo_10'

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    labels = to_categorical(np.asarray(labels))
    return texts, labels, labels_index

def split_yahoo_files():
    print('Processing Yahoo! Answers dataset')
    texts, labels, _ = read_yahoo_files()
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    return train_texts, train_labels, test_texts, test_labels,test_texts, test_labels


def split_yelp_sentence_attack_files(args):
    print('Processing Yelp dataset')
    path = os.path.join(args.work_path, 'data_set/yelp_pickle')

    cache_attack_text = path + '/' + 'attack' + '_texts.pkl'
    cache_attack_label = path + '/' + 'attack' + '_labels.pkl'

    test_path_text = path + '/' + 'test' + '_texts.pkl'
    test_path_label = path + '/' + 'test' + '_labels.pkl'
    test_texts = pickle.load(open(test_path_text, 'rb'))
    test_labels = pickle.load(open(test_path_label, 'rb'))

    if os.path.exists(cache_attack_text):
        attack_texts = pickle.load(open(cache_attack_text, 'rb'))
        attack_labels = pickle.load(open(cache_attack_label, 'rb'))

        # if len(attack_texts) == args.certify_sample_num:
        return attack_texts, attack_labels

    attack_model = Hidden_Killer()

    print("sample %d samples to perform empirical attack" % (args.certify_sample_num))
    max_samples = args.certify_sample_num

    if len(test_texts) > max_samples:
        sample_index = random.sample(range(len(test_texts)), max_samples)
        test_texts = [test_texts[i] for i in sample_index]
        test_labels = [test_labels[i] for i in sample_index]

    attack_texts = []
    attack_labels = []

    for (text, label) in tqdm(zip(test_texts, test_labels), desc="generating adversaries slowly :)"):
        attack_text, attack_label = attack_model.generate(text, label)
        attack_texts.append(attack_text)
        attack_labels.append(attack_label)

    pickle.dump(attack_texts, open(cache_attack_text, 'wb'))
    pickle.dump(attack_labels, open(cache_attack_label, 'wb'))

    return attack_texts, attack_labels


def split_imdb_sentence_attack_files(args):
    print('Processing IMDB dataset')

    path = os.path.join(args.work_path, 'data_set/imdb_pickle')

    cache_attack_text = path + '/' + 'attack' + '_texts.pickle'
    cache_attack_label = path + '/' + 'attack' + '_labels.pickle'

    if os.path.exists(cache_attack_text):
        attack_texts = pickle.load(open(cache_attack_text, 'rb'))
        attack_labels = pickle.load(open(cache_attack_label, 'rb'))

        # if len(attack_texts) == args.certify_sample_num:
        return attack_texts, attack_labels

    test_path_text = path + '/' + 'test' + '_texts.pickle'
    test_path_label = path + '/' + 'test' + '_labels.pickle'
    test_texts = pickle.load(open(test_path_text, 'rb'))
    test_labels = pickle.load(open(test_path_label, 'rb'))

    test_texts = test_texts[:12500 - 500] + test_texts[12500:25000 - 500]
    test_labels = test_labels[:12500 - 500] + test_labels[12500:25000 - 500]

    attack_model = Hidden_Killer()

    print("sample %d samples to perform empirical attack" % (args.certify_sample_num))
    max_samples = args.certify_sample_num

    # if len(test_texts) > max_samples:
    #
    #     sample_index = random.sample(range(len(test_texts)), max_samples)
    #     test_texts = [test_texts[i] for i in sample_index]
    #     test_labels = [test_labels[i] for i in sample_index]
    test_texts = [test_texts[0]]
    test_labels = [test_labels[0]]

    attack_texts = []
    attack_labels = []

    out_f = open('imdb_attack_new_2.txt', 'w')
    for (text, label) in tqdm(zip(test_texts, test_labels), desc="generating adversaries slowly :)"):
        attack_text, attack_label = attack_model.generate(text, label)
        attack_texts.append(attack_text)
        attack_labels.append(attack_label)
        out_f.write(attack_text + '\t' + str(attack_label) + '\n')
        out_f.flush()

    # pickle.dump(attack_texts, open(cache_attack_text, 'wb'))
    # pickle.dump(attack_labels, open(cache_attack_label, 'wb'))

    return attack_texts, attack_labels


def split_imdb_sentence_attack_files_easy(args):
    print('Processing IMDB dataset')

    path = os.path.join(args.work_path, 'data_set/aclImdb')
    # cache_attack_text = path + '/'+'attack' + '_texts.pkl'
    # cache_attack_label = path + '/'+'attack' + '_labels.pkl'
    cache_attack_text = path + '/' + 'attack_easy_new' + '_texts.pkl'
    cache_attack_label = path + '/' + 'attack_easy_new' + '_labels.pkl'

    if os.path.exists(cache_attack_text):
        attack_texts = pickle.load(open(cache_attack_text, 'rb'))
        attack_labels = pickle.load(open(cache_attack_label, 'rb'))

        # if len(attack_texts) == args.certify_sample_num:
        return attack_texts, attack_labels

    print('Processing imdb dataset')
    filename = os.path.join(args.work_path, "temp/split_imdb_files")
    if os.path.exists(filename):
        print('Read processed IMDB dataset')
        f = open(filename, 'rb')
        saved = pickle.load(f)
        f.close()
        test_texts = saved['test_texts']
        test_labels = saved['test_labels']

    attack_model = Hidden_Killer()

    print("sample %d samples to perform empirical attack" % (args.certify_sample_num))
    max_samples = args.certify_sample_num

    if len(test_texts) > max_samples:
        sample_index = random.sample(range(len(test_texts)), max_samples)
        test_texts = [test_texts[i] for i in sample_index]
        test_labels = [test_labels[i] for i in sample_index]
    # test_texts=[test_texts[0]]
    # test_labels=[test_labels[0]]

    attack_texts = []
    attack_labels = []

    out_f = open('imdb_attack_easy.txt', 'w')
    for (text, label) in tqdm(zip(test_texts, test_labels), desc="generating adversaries slowly :)"):
        attack_text, attack_label = attack_model.generate(text, label, break_sent=True, easy=True)
        attack_texts.append(attack_text)
        attack_labels.append(attack_label)
        out_f.write(attack_text + '\t' + text + '\t' + str(attack_label) + '\n')
        out_f.flush()

    pickle.dump(attack_texts, open(cache_attack_text, 'wb'))
    pickle.dump(attack_labels, open(cache_attack_label, 'wb'))

    return attack_texts, attack_labels


def split_yelp_editing_attack_files(args):
    print('Processing Yelp dataset')
    path = os.path.join(args.work_path, 'data_set/yelp_pickle')

    cache_attack_text = path + '/' + 'editing_texts'+ str(args.editing_number) + '.pkl'
    cache_attack_label = path + '/' + 'editing_labels'+ str(args.editing_number)+'.pkl'

    if os.path.exists(cache_attack_text):
        attack_texts = pickle.load(open(cache_attack_text, 'rb'))
        attack_labels = pickle.load(open(cache_attack_label, 'rb'))

        # if len(attack_texts) == args.certify_sample_num:
        return attack_texts, attack_labels

    test_path_text = path + '/' + 'test' + '_texts.pkl'
    test_path_label = path + '/' + 'test' + '_labels.pkl'
    test_texts = pickle.load(open(test_path_text, 'rb'))
    test_labels = pickle.load(open(test_path_label, 'rb'))

    # print("sample %d samples to perform empirical attack" % (args.certify_sample_num))
    # max_samples = args.certify_sample_num

    # if len(test_texts) > max_samples:
    #     sample_index = random.sample(range(len(test_texts)), max_samples)
    #     test_texts = [test_texts[i] for i in sample_index]
    #     test_labels = [test_labels[i] for i in sample_index]

    attack_texts = []
    attack_labels = []
    iter_nums = []

    for (text, label) in tqdm(zip(test_texts, test_labels), desc="generating adversaries slowly :)"):
        attack_text, attack_label, iter_num = editor(text, label)
        attack_texts.append(attack_text)
        attack_labels.append(attack_label)
        iter_nums.append(iter_num)

    f = open('/home/zhaohaiteng/Certified-Robustness-/ConvexCertify/data_set/yelp_pickle/editing_texts' + str(
        args.editing_number) + '.pkl', 'wb')
    pickle.dump(attack_texts, f)
    f.close()
    f = open('/home/zhaohaiteng/Certified-Robustness-/ConvexCertify/data_set/yelp_pickle/editing_labels' + str(
        args.editing_number) + '.pkl', 'wb')
    pickle.dump(attack_labels, f)
    f.close()

    print("average iter num is:", np.array(iter_nums).mean())
    return attack_texts, attack_labels


def split_imdb_editing_attack_files(args):
    print('Processing imdb dataset')

    path = os.path.join(args.work_path, 'data_set/aclImdb')

    cache_attack_text = path + '/' + 'editing_texts' + str(args.editing_number) + '.pkl'
    cache_attack_label = path + '/' +  'editing_labels' + str(args.editing_number) + '.pkl'

    if os.path.exists(cache_attack_text):
        attack_texts = pickle.load(open(cache_attack_text, 'rb'))
        attack_labels = pickle.load(open(cache_attack_label, 'rb'))

        # if len(attack_texts) == args.certify_sample_num:
        return attack_texts, attack_labels


    filename = os.path.join(args.work_path, "temp/split_imdb_files")
    if os.path.exists(filename):
        print('Read processed IMDB dataset')
        f = open(filename, 'rb')
        saved = pickle.load(f)
        f.close()
        test_texts = saved['test_texts']
        test_labels = saved['test_labels']

    print("sample %d samples to perform empirical attack" % (args.certify_sample_num))
    max_samples = args.certify_sample_num

    if len(test_texts) > max_samples:
        sample_index = random.sample(range(len(test_texts)), max_samples)
        test_texts = [test_texts[i] for i in sample_index]
        test_labels = [test_labels[i] for i in sample_index]

    # test_texts=[test_texts[0]]
    # test_labels=[test_labels[0]]

    attack_texts = []
    attack_labels = []
    iter_nums = []
    for (text, label) in tqdm(zip(test_texts, test_labels), desc="generating adversaries slowly :)"):
        attack_text, attack_label, iter_num = editor(text, label, iter_num=args.editing_number)
        attack_texts.append(attack_text)
        attack_labels.append(attack_label)
        iter_nums.append(iter_num)

    f = open('/home/zhaohaiteng/Certified-Robustness-/ConvexCertify/data_set/aclImdb/editing_texts' + str(
        args.editing_number) + '.pkl', 'wb')
    pickle.dump(attack_texts, f)
    f.close()
    f = open('/home/zhaohaiteng/Certified-Robustness-/ConvexCertify/data_set/aclImdb/editing_labels' + str(
        args.editing_number) + '.pkl', 'wb')
    pickle.dump(attack_labels, f)
    f.close()

    print("average iter num is:", np.array(iter_nums).mean())
    return attack_texts, attack_labels

