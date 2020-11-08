# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, bert_pad_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, bert_input=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx if not bert_input else bert_pad_idx,
            eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    if samples[0]['source_bert'] is not None:
        src_bert_tokens = merge('source_bert', left_pad=left_pad_source, bert_input=True)
    else:
        src_bert_tokens = None
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    if src_bert_tokens is not None:
        src_bert_tokens = src_bert_tokens.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    origin_target = None
    if samples[0].get('target', None) is not None:
        ntokens = sum(len(s['target']) for s in samples)
        target = merge('output', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        origin_target = merge('origin_target', left_pad=left_pad_target)
        origin_target = origin_target.index_select(0, sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=False,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            # 'bert_input': src_bert_tokens,
        },
        'target': target,
        'origin_target': origin_target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class BertXYNoisyLanguagePairDataset(FairseqDataset):
    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1, x2, x3, x4, x5] [y1, _, y3, _, y5]
    source: [x1, _, x3, _, x5]
    prev_output_tokens: [y1, _, y3, _, y5]
    output: [PAD, y2, PAD, y4, PAD]
    """
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, ratio=0.50,
        pred_probs=None, mask_source_rate=0.1,
        srcbert=None, srcbert_sizes=None, berttokenizer=None,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.src_vocab = src_dict
        self.tgt_vocab = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.ratio = ratio
        self.pred_probs = pred_probs
        self.mask_source_rate = mask_source_rate

        self.srcbert = srcbert
        self.srcbert_sizes = np.array(srcbert_sizes) if srcbert_sizes is not None else None
        self.berttokenizer = berttokenizer

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]
        if self.srcbert is not None:
            src_bert_item = self.srcbert[index]
        else:
            src_bert_item = None

        tgt_list = tgt_item.tolist()
        src_list = src_item.tolist()

        origin_target = tgt_list.copy()
        target = tgt_list.copy()
        output = tgt_list.copy()
        source = src_list.copy()

        len_to_consider = len(tgt_list)

        source_len_mask = max(1, round(len(src_list) * self.mask_source_rate))
        start = self.mask_start(len(src_list)-source_len_mask)
        source_id_mask = np.arange(start, start+source_len_mask+1)

        num_mask = np.random.randint(1, len_to_consider + 1)
        id_mask = np.arange(len_to_consider)
        np.random.shuffle(id_mask)

        id_mask = sorted(id_mask[:num_mask])

        for i, w in enumerate(tgt_list):
            if i in id_mask:
                target[i] = self.mask_word(w)
            else:
                output[i] = self.tgt_vocab.pad()

        for i, w in enumerate(src_list):
            if isinstance(source_id_mask, dict):
                if i in source_id_mask.keys():
                    source[i] = self.mask_word(w, source=True, p=source_id_mask[i])
            else:
                if i in source_id_mask:
                    source[i] = self.mask_word(w, source=True)

        assert len(target) == len(output)

        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'origin_target': torch.LongTensor(origin_target),
            'source_bert': src_bert_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_vocab.pad(), eos_idx=self.src_vocab.eos(), bert_pad_idx=self.berttokenizer.pad(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        a = max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return a

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def mask_word(self, w, source=False, p=None):
        if source:
            voc = self.src_vocab
        else:
            voc = self.tgt_vocab

        p = np.random.random() if p is None else p
        if p >= 0.2:
            return voc.mask_index
        elif p >= 0.1:
            return np.random.randint(voc.nspecial, len(voc))
        else:
            return w

    def mask_start(self, end):
        p = np.random.random()
        if p >= 0.8:
            return 0
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(end)

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.srcbert is not None:
            self.srcbert.prefetch(indices)
