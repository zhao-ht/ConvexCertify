import argparse

parser = argparse.ArgumentParser(description="Network parameters")

# Main parameters
'''
 Add parameters like this :
    parser.add_argument(
    "--experiment_name", type=str, help="Name of experiment", required=True
) 
type = str, bool, int ...
'''

parser.add_argument("--experiment_name",
                    type=str,
                    help="Name of experiment",
                    default='train',
                    required=False)

parser.add_argument("--dataset_name",
                    type=str,
                    help="Name of dataset",
                    default='yelp',
                    required=False)

parser.add_argument('--work_path', type=str, default='./', help='')



parser.add_argument('--imdb_synonyms_file_path',
                    type=str,
                    default="temp/imdb.synonyms",
                    help='')

parser.add_argument('--imdb_bert_synonyms_file_path',
                    type=str,
                    default="temp/imdb_bert.synonyms",
                    help='')

parser.add_argument('--sst2_bert_synonyms_file_path',
                    type=str,
                    default="temp/sst2_bert.synonyms",
                    help='')

parser.add_argument('--yelp_bert_synonyms_file_path',
                    type=str,
                    default="temp/yelp_bert.synonyms",
                    help='')
parser.add_argument('--agnews_bert_synonyms_file_path',
                    type=str,
                    default="temp/agnews_bert.synonyms",
                    help='')



parser.add_argument('--certified_neighbors_file_path',
                    type=str,
                    default="data_set/counterfitted_neighbors.json",
                    help='')

parser.add_argument('--embedding_dim',
                    type=int,
                    default=768,
                    help='embedding_dim')

parser.add_argument('--embedding_file_path',
                    type=str,
                    default="data_set/glove/glove.840B.300d.txt",
                    help='glove or w2v')

parser.add_argument('--vocab_size',
                    type=int,
                    default=0,
                    help='size of vocabulary')













parser.add_argument('--base_model',type = str, default = 'bert', help = 'base classifier model')
parser.add_argument('--pretrained_name',type = str, default='./bert-base-uncased', help = 'bert pretrained model to load from')



parser.add_argument('--perturbed_num',type = int, default = 51, help = 'number of maximum words in a sentence that allow perturbation')
parser.add_argument('--max_len',
                    type=int,
                    default=51,
                    help='maximum length of sentence')

parser.add_argument('--use_pretrained_embed', type=bool, default=True, help = 'use pretrained embedding')


parser.add_argument('--num_classes', type=int, default=2)






#debug

parser.add_argument('--save_model', action='store_true')  #
parser.add_argument('--save_criterion', type=str,default='')  #[best, same_fit]
parser.add_argument('--same_fit_threshold', type=float, default=10)
parser.add_argument('--criterion_id', type=int,default=0)
parser.add_argument('--criterion_maximize', action='store_true')
parser.add_argument('--train_model', action='store_true') #
parser.add_argument('--adv_train_model', action='store_true') #
parser.add_argument('--train_number', type=int, default=None) #
parser.add_argument('--test_number', type=int, default=None) #
parser.add_argument('--verify_model', action='store_true') #
parser.add_argument('--test_model',  action='store_true') #
parser.add_argument('--certify_model', action='store_true') #
parser.add_argument('--empirical_certify_model', action='store_true') #
parser.add_argument('--sentence_level_certify_model', action='store_true')
parser.add_argument('--certify_on_transferred_data',action='store_true')
parser.add_argument('--verify_certification', action='store_true') #
parser.add_argument('--use_tqdm',default=True) #
parser.add_argument('--train_on_dev', action='store_true') #

parser.add_argument('--attack_method',default=None)  # ['sentence','editing']
parser.add_argument('--editing_number',type=int,default=300)
parser.add_argument('--adv_method',type=str,default='freelb')


# optimization
parser.add_argument('--epoch', type=int, default=600000, help='')
parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
parser.add_argument('--test_batch_size', type=int, default=20,
                    help='test_batch_size')
parser.add_argument('--certify_batch_size', type=int, default=1,
                    )
parser.add_argument('--certify_first_sample_size', type=int, default=50,
                    )
parser.add_argument('--certify_second_sample_size', type=int, default=300,
                    )

parser.add_argument('--use_cpu',action='store_true')
parser.add_argument('--device', nargs='+',type=int,default=[0] , help = 'device to train on') #
parser.add_argument('--save_path', type=str,default='tem', help='') #
parser.add_argument('--save_frequency',type=int,default=1)
parser.add_argument('--parallel',  action='store_true')
parser.add_argument('--local_rank',type=int, default=-1)
parser.add_argument('--synonyms_from_file', action='store_true')
parser.add_argument('--print_interval', type=int, default=500)

parser.add_argument('--learning_rate', type=float, default=1e-5, help = 'learning rate for training')

parser.add_argument('--warm_up',action='store_true')
parser.add_argument('--learning_rate_2', type=float, default=1e-2, help = 'learning rate for training')
parser.add_argument('--warm_up_step',type=int,default=100)
parser.add_argument('--warm_up_method',type=str,default='train_vae_only') # ['classify_on_hidden','train_vae_only']

# coef fashion
parser.add_argument('--loss_type', type=str, default='bound')
parser.add_argument('--radius_margin',type=float, default=1.0)
parser.add_argument('--soft_upper_bound', type = float, default=0.99)

parser.add_argument('--IBP_fashion',type=str,default='linear')
parser.add_argument('--IBP_start_step',type=float,default=1000)
parser.add_argument('--IBP_max_step',type=float,default=1000)
parser.add_argument('--IBP_max_coef',type=float,default=0.01)
parser.add_argument('--Generator_fashion',type=str,default='linear')
parser.add_argument('--Generator_max_step',type=float,default=1)
parser.add_argument('--Generator_max_coef',type=float,default=1)



#train fashion

parser.add_argument('--generator_coef',type=float,default=0.0) #
parser.add_argument('--not_train_bert', action='store_true') #
parser.add_argument('--classify_on_hidden', action='store_true') #
parser.add_argument('--only_classify_on_hidden', action='store_true')#
parser.add_argument('--radius_on_hidden_classify', action='store_true')
parser.add_argument('--direct_input_to_bert', action='store_true')#
parser.add_argument('--IBP_loss_type',type=str,default='l2')

parser.add_argument('--not_use_scheduler',action='store_true')

parser.add_argument('--init_lr',type=float,default=None)
parser.add_argument('--init_step',type=float,default=None)


#encoder parameter
parser.add_argument('--std',type=float,default=1.0)
parser.add_argument('--latent_size', type=int, default=100, help='')
parser.add_argument('--num_layer', type=int, default=2, help='')
parser.add_argument('--ibp_loss', type=bool, default=True)
parser.add_argument('--encoder_type', type=str, default='cnnibp')
parser.add_argument('--embedding_training', type=bool, default=True, help = 'finetune pretrained embedding')
parser.add_argument('--no_wordvec_layer',action='store_true')
#parameters for TextCNN Model
parser.add_argument('--dropout', type = float, default=0.4)
parser.add_argument('--n_filters', type = int, default=200)
parser.add_argument('--filter_sizes', type = int, default=[1,2,3,4,5])

parser.add_argument('--bypass_premise',action = 'store_true')

#parameter for decoder
parser.add_argument('--decoder_type', type=str, default='cnn')
parser.add_argument('--reconstruct_by_RNN',type=bool, default=False)
parser.add_argument('--rnn_size', type=int, default=500, help='')
parser.add_argument('--rnn_num_layers', type=int, default=1, help='')
parser.add_argument('--detach_reconstruct',action='store_true')
parser.add_argument('--info_loss',type=str,default=None)   # ['reconstruct']
parser.add_argument('--reconstruct_by_vocab', type=bool, default=False)
parser.add_argument('--renormlize_embedding',default=True)
parser.add_argument('--to_vocab_trainable',type=bool, default=False)
parser.add_argument('--inconsistent_recorvery_length',action='store_true')
parser.add_argument('--recovery_length',type=int,default=10)

#parameter for verify
parser.add_argument('--soft_verify',action='store_true')
parser.add_argument('--soft_beta',type=float,default=1)
parser.add_argument('--alpha',type=float,default=0.05)

#adversarial training
parser.add_argument('--adv_init_mag',type=float,default=1e-1)
parser.add_argument('--adv_steps',type=int,default=2)
parser.add_argument('--norm_type',type=str,default='l2')
parser.add_argument('--adv_lr',type=float,default=1e-1)

parser.add_argument('--adv_max_norm',type=float,default=0)



