# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
from bert import BertTokenizer
import torch

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    BertLanguagePairDataset,
    BertXYNoisyLanguagePairDataset,
    Dictionary,
)

from fairseq.data.masked_lm_dictionary import BertWordpieceDictionary
from fairseq import tokenizer

from . import FairseqTask, register_task

def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
    ratio, pred_probs, bert_model_name,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    srcbert_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
        src_datasets.append(indexed_dataset.make_dataset(prefix + src, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=src_dict))
        tgt_datasets.append(indexed_dataset.make_dataset(prefix + tgt, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=tgt_dict))
        srcbert_datasets.append(indexed_dataset.make_dataset(bertprefix + src, impl=dataset_impl,
                                                         fix_lua_indexing=True, ))

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        srcbert_datasets = srcbert_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    berttokenizer = BertTokenizer.from_pretrained(bert_model_name)
    if split == 'test':
        return BertLanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            srcbert=srcbert_datasets,
            srcbert_sizes=srcbert_datasets.sizes if srcbert_datasets is not None else None,
            berttokenizer=berttokenizer,
        )
    else:
        return BertXYNoisyLanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            shuffle=True,
            ratio=ratio,
            pred_probs=pred_probs,
            srcbert=srcbert_datasets,
            srcbert_sizes=srcbert_datasets.sizes if srcbert_datasets is not None else None,
            berttokenizer=berttokenizer,
        )

@register_task('bert_xymasked_wp_seq2seq')
class BertXYMassWordpieceTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        parser.add_argument('--word_mask', default=0.5, type=float, metavar='RATIO',
                            help='The mask ratio')
        parser.add_argument('--word_mask_keep_rand', default="0.1,0.1,0.8", type=str,
                            help='Word prediction proability')

        parser.add_argument('--bert-model-name', default='bert-base-uncased', type=str)
        parser.add_argument('--encoder-ratio', default=1., type=float)
        parser.add_argument('--bert-ratio', default=1., type=float)
        parser.add_argument('--finetune-bert', action='store_true')
        parser.add_argument('--mask-cls-sep', action='store_true')
        parser.add_argument('--warmup-from-nmt', action='store_true', )
        parser.add_argument('--warmup-nmt-file', default='checkpoint_nmt.pt', )
        parser.add_argument('--use-adapter-bert', default=False, action='store_true')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.bert_model_name = args.bert_model_name

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = BertWordpieceDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        s = args.word_mask_keep_rand.split(',')
        s = [float(x) for x in s]
        setattr(args, 'pred_probs', torch.FloatTensor([s[0], s[1], s[2]]))

        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = BertWordpieceDictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = BertWordpieceDictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))

        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertWordpieceDictionary.load(filename)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            ratio=self.args.word_mask,
            pred_probs=self.args.pred_probs,
            bert_model_name = self.bert_model_name,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def inference_step(self, generator, models, sample, prefix_tokens=None, tgt_bert_encoder=None, tgt_bert_tokenizer=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, tgt_bert_encoder=tgt_bert_encoder, tgt_bert_tokenizer=tgt_bert_tokenizer)

    def build_generator(self, args):
        from fairseq.sequence_generator_with_bert import SequenceGeneratorWithBert
        return SequenceGeneratorWithBert(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            stop_early=(not getattr(args, 'no_early_stop', False)),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            mask_pred_iter=getattr(args, 'mask_pred_iter', 10),
            decode_use_adapter=getattr(args, 'decode_use_adapter', False),
            args=args,
        )