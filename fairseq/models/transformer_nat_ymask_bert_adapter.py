# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import numpy as np
import copy

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from bert import BertTokenizer
from bert import BertModelWithAdapter
from bert.modeling import BertEmbeddings, BertAttention, BertIntermediate, BertOutput, BertPreTrainedModel, BertOnlyMLMHead

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_nat_ymask_bert_two_adapter')
class TransformerNatYmaskBertTwoAdapter(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder, berttokenizer, tgt_berttokenizer, args):
        super().__init__(encoder, decoder)
        self._is_generation_fast = False
        self.encoder = encoder
        self.decoder = decoder
        self.berttokenizer = berttokenizer
        self.tgt_berttokenizer = tgt_berttokenizer
        self.mask_cls_sep = args.mask_cls_sep
        self.max_source_positions = args.max_source_positions
        self.max_target_positions = args.max_target_positions

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--decoder-bert-model-name', type=str, default='bert-base-uncased')
        parser.add_argument('--top-layer-adapter', default=-1, type=int)
        parser.add_argument('--enc-top-layer-adapter', default=-1, type=int)
        parser.add_argument('--adapter-dimension', default=2048, type=int)
        parser.add_argument('--finetune-embeddings', default=False, action='store_true')
        parser.add_argument('--finetune-whole-encoder', default=False, action='store_true')
        parser.add_argument('--train-from-scratch', default=False, action='store_true')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        src_berttokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
        tgt_berttokenizer = BertTokenizer.from_pretrained(args.decoder_bert_model_name)
        assert src_berttokenizer.pad()==tgt_berttokenizer.pad()

        bertdecoder = BertAdapterDecoderFull.from_pretrained(args.decoder_bert_model_name, args, from_scratch=args.train_from_scratch)
        enc_top_layer_adapter = getattr(args, 'enc_top_layer_adapter', -1)
        adapter_dimension = getattr(args, 'adapter_dimension', 2048)
        bertencoder = BertModelWithAdapter.from_pretrained(args.bert_model_name, adapter_dimension, enc_top_layer_adapter, from_scratch=args.train_from_scratch)
        return cls(bertencoder, bertdecoder, src_berttokenizer, tgt_berttokenizer, args)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        bert_encoder_padding_mask = src_tokens.eq(self.berttokenizer.pad())

        bert_encoder_out, _, predicted_lengths = self.encoder(src_tokens, attention_mask=1-bert_encoder_padding_mask.float())
        bert_encoder_out = bert_encoder_out.permute(1,0,2).contiguous()

        encoder_out = {
            'encoder_out': bert_encoder_out,
            'encoder_padding_mask': bert_encoder_padding_mask,
            'predicted_lengths': predicted_lengths,
        }
        decoder_out, _ = self.decoder(prev_output_tokens, encoder_out=encoder_out, padding_idx=self.berttokenizer.pad())
        return decoder_out, {'predicted_lengths': predicted_lengths}

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.max_source_positions, self.max_target_positions)

class BertAdapterDecoderFull(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertAdapterDecoderFull, self).__init__(config)
        self.bert = BertAdapterDecoderAssemble(config, args)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.onnx_trace = False

    def forward(self, prev_output_tokens, src_tokens=None, encoder_out=None, padding_idx=0, **kwargs):
        sequence_output = self.bert(prev_output_tokens, encoder_out, padding_idx)
        prediction_scores = self.cls(sequence_output)

        return prediction_scores, sequence_output

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

class BertAdapterDecoderAssemble(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertAdapterDecoderAssemble, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertAdapterDecoder(config, args)
        self.apply(self.init_bert_weights)
        self.hidden_size = config.hidden_size

    def forward(self, prev_output_tokens, encoder_out=None, padding_idx=0):

        targets_padding = prev_output_tokens.eq(padding_idx)
        position_ids = torch.arange(prev_output_tokens.size(1), dtype=torch.long, device=prev_output_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(prev_output_tokens)
        positions = self.embeddings.position_embeddings(position_ids).transpose(0, 1)
        token_type_ids = torch.zeros_like(prev_output_tokens)

        extended_attention_mask = targets_padding.unsqueeze(1).unsqueeze(2).float()
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask *= -10000.0

        embedding_output = self.embeddings(prev_output_tokens, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False,
                                      encoder_out=encoder_out['encoder_out'] if encoder_out is not None else None,
                                      encoder_padding_mask=encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                                      position_embedding=positions,
                                      targets_padding=targets_padding,
                                      )
        return encoded_layers[-1]


class BertAdapterDecoder(nn.Module):
    def __init__(self, config, args):
        super(BertAdapterDecoder, self).__init__()
        self.num_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([copy.deepcopy(BertAdapterDecoderLayer(config, args, i)) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True,
                encoder_out=None, encoder_padding_mask=None, position_embedding=None, targets_padding=None):
        all_decoder_layers = []
        for i in range(self.num_layers):
            layer_module = self.layer[i]
            hidden_states = layer_module(hidden_states,
                            encoder_out=encoder_out,
                            encoder_padding_mask=encoder_padding_mask,
                            self_attn_mask=attention_mask,
                            position_embedding=position_embedding,
                            targets_padding=targets_padding,
                            layer_num=i)
            if output_all_encoded_layers:
                all_decoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_decoder_layers.append(hidden_states)
        return all_decoder_layers

class BertAdapterDecoderLayer(nn.Module):
    def __init__(self, config, args, layer_num):
        super(BertAdapterDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.top_layer_adapter = getattr(args,'top_layer_adapter', -1)

        export = getattr(args, 'char_inputs', False)

        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            kdim=getattr(args, 'encoder_embed_dim', None),
            vdim=getattr(args, 'encoder_embed_dim', None),
            dropout=args.attention_dropout, encoder_decoder_attention=True
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn_fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.encoder_attn_fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.encoder_attn_final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = False

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        self_attn_mask=None,
        position_embedding=None,
        targets_padding=None,
        layer_num=-1,
    ):
        x = self.attention(x, self_attn_mask)

        intermediate_output = self.intermediate(x)
        x = self.output(intermediate_output, x)

        x = x.transpose(0, 1)
        
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_final_layer_norm, x, before=True)
        x = self.activation_fn(self.encoder_attn_fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.encoder_attn_fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        layer_output = self.maybe_layer_norm(self.encoder_attn_final_layer_norm, x, after=True)
        layer_output = layer_output.transpose(0,1)
        return layer_output

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def without_self_mask(self, tensor):
        dim = tensor.size(0)
        eye_matrix = torch.eye(dim)
        eye_matrix[eye_matrix == 1.0] = float('-inf')
        return eye_matrix.cuda()

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

@register_model_architecture('transformer_nat_ymask_bert_two_adapter', 'transformer_nat_ymask_bert_two_adapter')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.finetune_whole_encoder = getattr(args, 'finetune_whole_encoder', False)
    args.train_from_scratch = getattr(args, 'train_from_scratch', False)

@register_model_architecture('transformer_nat_ymask_bert_two_adapter', 'transformer_nat_ymask_bert_two_adapter_deep_small')
def transformer_nat_ymask_bert_two_adapter_deep_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    base_architecture(args)

@register_model_architecture('transformer_nat_ymask_bert_two_adapter', 'transformer_nat_ymask_bert_two_adapter_iwslt_de_en')
def transformer_nat_ymask_bert_two_adapter_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)

@register_model_architecture('transformer_nat_ymask_bert_two_adapter', 'transformer_nat_ymask_bert_two_adapter_wmt_en_de_bert_base')
def transformer_nat_ymask_bert_two_adapter_wmt_en_de_bert_base(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    base_architecture(args)

