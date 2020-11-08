# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data import data_utils
from fairseq.data import Dictionary
import torch

class BertWordpieceDictionary(Dictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """
    def __init__(
        self,
        pad="[PAD]",
        eos="[SEP]",
        unk="[UNK]",
        bos="[CLS]",
        mask="[MASK]",
        cls="[CLS]",
        sep="[SEP]",
    ):
        self.symbols = []
        self.count = []
        self.indices = {}
        self.unk_word, self.pad_word, self.eos_word, self.bos_word = unk, pad, eos, bos
        self.cls_word = cls
        self.sep_word = sep
        self.mask_word = mask
        # self.mask_index = self.add_symbol(mask)
        # self.cls_index = self.add_symbol(cls)
        # self.sep_index = self.add_symbol(sep)
        self.nspecial = 0

    def mask(self):
        """Helper to get index of mask symbol"""
        self.mask_index = self.indices[self.mask_word]
        return self.mask_index

    def cls(self):
        """Helper to get index of cls symbol"""
        self.cls_index = self.indices[self.cls_word]
        return self.cls_index

    def sep(self):
        """Helper to get index of sep symbol"""
        self.sep_index = self.indices[self.sep_word]
        return self.sep_index

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        self.bos_index = self.indices[self.bos_word]
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        self.pad_index = self.indices[self.pad_word]
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        self.eos_index = self.indices[self.eos_word]
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        self.unk_index = self.indices[self.unk_word]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        final_tensor = []
        for i in tensor:
            if i != self.eos() and i != self.cls():
                final_tensor.append(i)
            else:
                break

        sent = ' '.join(token_string(i) for i in final_tensor if i != self.eos() and i!= self.pad())
        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls()
        lines = f.readlines()
        indices_start_line = d._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        d.post_init()
        return d

    def post_init(self):
        _ = self.mask()
        _ = self.cls()
        _ = self.sep()
        _ = self.bos()
        _ = self.eos()
        _ = self.unk()
        _ = self.pad()

        

class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
    ):
        super().__init__(pad, eos, unk)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index


class BertDictionary(MaskedLMDictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
        cls='<cls>',
        sep='<sep>'
    ):
        super().__init__(pad, eos, unk, mask)
        self.cls_word = cls
        self.sep_word = sep
        self.cls_index = self.add_symbol(cls)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        return self.cls_index

    def sep(self):
        """Helper to get index of sep symbol"""
        return self.sep_index
