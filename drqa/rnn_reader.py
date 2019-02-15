# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.functional as F
from . import layers

# Modification:
#   - add 'pos' and 'ner' features.
#   - use gradient hook (instead of tensor copying) for gradient masking
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None):

        # 속성 : opt, embedding, padding_idx,
        # q_emb_match, doc_rnn, question_rnn, self_attn, start_attn, end_attn
        super(RnnDocReader, self).__init___()
        # 기본 옵션 가져오기
        self.opt = opt

        # 임베딩 가져오기
        if opt['prtrained_words']:
            assert embedding is not None
            # Q word fine tune은 할것임.
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            if opt['fix_embeddings']: # 임베딩 고정
                assert opt['tune_partial'] == 0
                self.embedding.weight.requires_grad = False
            elif opt['tune_partial'] > 0: # word embedding fine_tune
                assert opt['tune_partial'] + 2 < embedding.size(0)
                offset = self.opt['tune_partial'] + 2
                # fine-tune을 진행할 토큰 제외하고는 grad 0 으로 바꿔줌. 갱신되지 않는다.
                def embedding_hook(grad, offset=offset):
                    grad[offset:] = 0
                    return grad
                self.embedding.weight.register_hook(embedding_hook)
        # 가져올 임베딩 없는 경우
        else:
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)

        # Q token 정보가 투영된 C token embedding
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # context_dim(300) + att_ques_dim(300) + num_features(4) + NER_size(7) + POS_size(18) = 629
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['use_qemb']
        if opt['pos']:
            doc_input_size += opt['pos']
        if opt['ner']:
            doc_input_size += opt['ner_size']


        # RNN Context Encoder, 옵션에 정보 담겨져있다.
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding']
        )

        # RNN Question Encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding']
        )


        # Output sizes of RNN Encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['doc_layers']

        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn': #우리 모델의 케이스
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # start/end를 위한 Linear, Bmm
        # (1, 768) => (1, 768), (32, 292, 768).bmm((768, 1) => (32, 292) for start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = F.dropout(x1_emb, p=self.opt['dropout_emb'],
                               training=self.training)
            x2_emb = F.dropout(x2_emb, p=self.opt['dropout_emb'],
                               training=self.training)

        # Context input 만들기
        drnn_input_list = [x1_emb, x1_f]
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            drnn_input_list.append(x1_pos)
        if self.opt['ner']:
            drnn_input_list.append(x1_ner)
        drnn_input = torch.cat(drnn_input_list, 2)

        # Context - Bi-LSTM 3Layers
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Qeustion - Bi-LSTM 3Layers, 중요한 토큰에 가중
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_attn_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # start/end points 예측
        start_scores = self.start_attn(doc_hiddens, question_attn_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_attn_hidden, x1_mask)
        return start_scores, end_scores