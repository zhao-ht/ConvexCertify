# Certified Robustness Against Natural Language Attacks by Causal Intervention



## Introduction

This is the code of our work [Certified Robustness Against Natural Language Attacks by Causal Intervention](https://proceedings.mlr.press/v162/zhao22g/zhao22g.pdf) published on ICML 2022.
<div align=center>
<img src="https://github.com/zhao-ht/ConvexCertify/blob/master/pipeline.png" width="600px">
</div>


To reproduce our experiments, please first install conda environment and download datasets.


## Update

2022.12.6: We update the training algorithm for faster training.


## Environment



```
conda create -n ciss python=3.8
source activate ciss

pip install -r requirements.txt
```

## Datasets


```
mkdir backup_checkpoint
mkdir temp
mkdir data_set
cd data_set

#YELP Dataset We use the YELP with the version from https://github.com/shentianxiao/language-style-transfer. Please download the repository above and copy language-style-transfer/data/yelp to data_set.

git clone https://github.com/shentianxiao/language-style-transfer
cp -r language-style-transfer/data/yelp ./

#IMDB Dataset

wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -zxvf aclImdb_v1.tar.gz

#GloVe Vector

wget https://nlp.stanford.edu/data/glove.840B.300d.zip
zip -r glove.840B.300d.zip file

#Synonyms

cp -r ../data/counterfitted_neighbors.json ./

cd ..
```

## Train Model

The model training includes two steps: pre-train the encoder first, then train the whole classifier.

Train the classification first:



```
YELP
CUDA_VISIBLE_DEVICES=0  nohup python train.py --train_model --test_model   --seed 111  --batch_size 10 --test_batch_size 10 --dataset_name yelp  --encoder_type cnnibp --latent_size 100  --inconsistent_recorvery_length --recovery_length 10 --max_len 51 --perturbed_num 51 --dropout 0.0 --std 1.0   --IBP_start_step 1 --IBP_max_step 1 --IBP_max_coef 0.0 --radius_margin 1.0  --not_use_scheduler  --warm_up  --warm_up_method train_encoder_first  --warm_up_step 200 --classify_on_hidden --learning_rate 1e-5 --learning_rate_2 1e-2 --print_interval 200  --init_lr 1e-5 --epoch 25  --save_model --save_criterion best --criterion_id 2 --criterion_maximize   --save_path yelp_classify &

IMDB
CUDA_VISIBLE_DEVICES=0  nohup python train.py  --train_model --test_model  --seed 111  --batch_size 10 --test_batch_size 10 --dataset_name imdb  --encoder_type cnnibp --latent_size 100  --inconsistent_recorvery_length --recovery_length 10 --max_len 513 --perturbed_num 513 --dropout 0.0 --std 1.0  --IBP_start_step 1 --IBP_max_step 1 --IBP_max_coef 0.0 --radius_margin 1.0  --not_use_scheduler  --warm_up --warm_up_method train_encoder_first  --warm_up_step 500 --classify_on_hidden --learning_rate 1e-5 --learning_rate_2 1e-2 --print_interval 200  --init_lr 1e-5 --epoch 150 --save_model --save_criterion best --criterion_id 2 --criterion_maximize   --save_path imdb_classify &

```

Then run the certified robustness training:

```
YELP
CUDA_VISIBLE_DEVICES=0 nohup python train.py --train_model --test_model  --seed 111 --batch_size 10 --test_batch_size 10 --dataset_name yelp  --encoder_type cnnibp --latent_size 100  --inconsistent_recorvery_length --recovery_length 10 --max_len 51 --perturbed_num 51 --dropout 0.0 --std 1.0  --IBP_fashion linear --IBP_start_step 1 --IBP_max_step 4000000 --IBP_max_coef 4.0 --radius_margin 1.0  --not_use_scheduler  --use_loss_ratio --init_loss_ratio 1 --loss_ratio_method adver --loss_ratio_para 0.92  --learning_rate 1e-5 --learning_rate_2 1e-2 --print_interval 200 --synonyms_from_file  --init_lr 1e-5 --epoch 100 --init_step 0 --epoch 90 --save_model --save_criterion best --criterion_id 10  --load_path yelp_classify --save_path yelp_certified_training &

IMDB
CUDA_VISIBLE_DEVICES=0 nohup python train.py --train_model --test_model --seed 111    --batch_size 10 --test_batch_size 10 --dataset_name imdb  --encoder_type cnnibp --latent_size 100  --inconsistent_recorvery_length --recovery_length 10 --max_len 513 --perturbed_num 513 --dropout 0.0 --std 1.0  --IBP_fashion multistep_linear  --IBP_max_step 0 1000000 4900000 --IBP_max_coef 0.0 0.1 4.0 --radius_margin 1.0  --not_use_scheduler  --use_loss_ratio --init_loss_ratio 1 --loss_ratio_method adver --loss_ratio_para 0.83 --learning_rate 1e-5 --learning_rate_2 1e-2 --print_interval 200 --synonyms_from_file  --init_lr 1e-5 --epoch 2000 --init_step 0  --save_model --save_criterion best --criterion_id 10  --load_path imdb_classify_3 --save_path imdb_certified_training   &

```

## Test and Certify Model

Finally, run the certification testing:


```
YELP
CUDA_VISIBLE_DEVICES=0 nohup python train.py --certify_model --alpha 0.001 --certify_second_sample_size 30000 --seed 111 --batch_size 10 --test_batch_size 10 --dataset_name yelp  --encoder_type cnnibp --latent_size 100  --inconsistent_recorvery_length --recovery_length 10 --max_len 51 --perturbed_num 51 --dropout 0.0 --std 1.0  --IBP_fashion linear --IBP_start_step 1 --IBP_max_step 4000000 --IBP_max_coef 4.0 --radius_margin 1.0  --not_use_scheduler  --use_loss_ratio --init_loss_ratio 1 --loss_ratio_method adver --loss_ratio_para 0.92  --learning_rate 1e-5 --learning_rate_2 1e-2 --print_interval 200 --synonyms_from_file  --init_lr 1e-5 --epoch 1 --init_step 0  --save_model --save_criterion best --criterion_id 10  --load_path yelp_certified_training --save_path yelp_certified_testing &

IMDB
CUDA_VISIBLE_DEVICES=0 nohup python train.py --certify_model --alpha 0.001 --certify_second_sample_size 30000 --seed 111 --batch_size 10 --test_batch_size 10 --dataset_name imdb  --encoder_type cnnibp --latent_size 100  --inconsistent_recorvery_length --recovery_length 10 --max_len 513 --perturbed_num 513 --dropout 0.0 --std 1.0  --IBP_fashion multistep_linear  --IBP_max_step 0 1000000 4900000 --IBP_max_coef 0.0 0.1 4.0 --radius_margin 1.0  --not_use_scheduler  --use_loss_ratio --init_loss_ratio 1 --loss_ratio_method adver --loss_ratio_para 0.83 --learning_rate 1e-5 --learning_rate_2 1e-2 --print_interval 200 --synonyms_from_file  --init_lr 1e-5 --epoch 1 --init_step 0  --save_model --save_criterion best --criterion_id 10  --load_path imdb_certified_training --save_path imdb_certified_testing   &

```

## Imply Your Own Model In CISS Framework

To imply your own model, please imply the base classifier (BERT in our implementation) in model.base.BaseModel specified by --base_model argument, and the encoder/decoder in model.encoder/model.decoder specified by --encoder_type/--decoder_type.


## Acknowledgement

Our implementation of Interval Bound Propagation is based on [robin's Interval Bound Propagation library](https://github.com/robinjia/certified-word-sub).
