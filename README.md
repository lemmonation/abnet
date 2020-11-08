# Adapter-Bert Networks
Code for our NeurIPS 2020 [paper](https://arxiv.org/abs/2010.06138) "Incorporating BERT into Parallel Sequence Decoding with Adapters". 
Please cite our paper if you find this repository helpful in your research:
```
@article{guo2020incorporating,
  title={Incorporating BERT into Parallel Sequence Decoding with Adapters},
  author={Guo, Junliang and Zhang, Zhirui and Xu, Linli and Wei, Hao-Ran and Chen, Boxing and Chen, Enhong},
  journal={arXiv preprint arXiv:2010.06138},
  year={2020}
}
```

## Requirements

The code is based on [fairseq-0.6.2](https://github.com/pytorch/fairseq/tree/v0.6.2), PyTorch-1.2.0 and cuda-9.2. The BERT implementation is heavily inspired by [bert-nmt](https://github.com/bert-nmt/bert-nmt) and [Huggingface Transformers](https://github.com/huggingface/transformers), many thanks to the authors for making their code avaliable.

## Instructions

Below is the instruction to reproduce our results on the IWSLT14 German-English translation task with [mask-predict](https://www.aclweb.org/anthology/D19-1633.pdf) decoding.

### Data Preprocessing

We tokenize and segment each word into wordpiece tokens using the same vocabulary as pre-trained BERT models, following the implementation in [Huggingface Transformers](https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_bert.py). We provide the wordpiece tokenized IWSLT14 De-En dataset in this [link](https://drive.google.com/file/d/1cFVwAbCLUA9YcfHk9_uH3_63ELzBYeRq/view?usp=sharing).

Then preprocess data like fairseq:
```
python preprocess.py --task bert_xymasked_wp_seq2seq \
  --source-lang de --target-lang en \
  --srcdict $TEXT/count-bert-base-german-cased-vocab.txt \
  --tgtdict $TEXT/count-bert-base-uncased-vocab.txt \
  --trainpref $TEXT/train.wordpiece --validpref $TEXT/valid.wordpiece --testpref $TEXT/test.wordpiece \
  --destdir $DATA_DIR --workers 20
```

### Train an Adapter-Bert Network

We provide an example of the training script:
```
python train.py $DATA_DIR \
  --task bert_xymasked_wp_seq2seq -s de -t en \
  -a transformer_nat_ymask_bert_two_adapter_deep_small \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --lr 0.0005 --min-lr '1e-09' \
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.0 --max-tokens 2000 --update-freq 2 --max-update 200000 \
  --left-pad-source False --adapter-dimension 512 \
  --use-adapter-bert --bert-model-name bert-base-german-cased --decoder-bert-model-name bert-base-uncased
```
We conduct our experiments on a 12GB Nvidia 1080Ti GPU, and we set `--max-tokens` to `2000` and `--update-freq` to `2` due to the limited GPU memory. In a GPU with larger memory, you can set `--max-tokens` to `4096` and `--update-freq` to `1` to speedup the training.


### Generate with Mask-Predict Decoding

We report the performance of the average of last 10 checkpoints. And we provide an example of the generation script:
```
python generate.py $DATA_DIR \
  --task bert_xymasked_wp_seq2seq --bert-model-name bert-base-german-cased \
  --path checkpoint_aver.pt --decode_use_adapter \
  --mask_pred_iter 10 --left-pad-source False \
  --batch-size 32 --beam 4 --lenpen 1.1 --remove-bpe wordpiece
```

