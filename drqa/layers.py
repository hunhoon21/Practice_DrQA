# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# No modification is made to this file.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

class StackedBRNN(nn.Module):
    def init__(self, input_size, hidden_size, num_layers,
               dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
               concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList() # 여러개의 Layers가 여기에 쌓인다.
        for i in range(num_layers):
            # Doc : (629 to 128) * 2 => (256 to 128) * 2 => (256 to 128) * 2
            # Question : (300 to 128) * 2 => (256 to 128) * 2 => (256 to 128) * 2
            input_size = input_size if i==0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # 패딩이 필요 없었던 경우
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # 패딩을 적용, Validation 경우
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        #그 외
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # x example : (32, 292, 300)
        # x_mask : (32, 292) => [0, 0 , 0, ..., 1, 1]
        x= x.transpose(0, 1) # => (292, 32, 300)

        outputs = [x]

        # RNN layer 통과
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            #
            if self.dropout_rate > 0:
                rnn_intput = F.dropout(rnn_input,
                                       p=self.dropout_rate,
                                       training=self.training)

            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # output 모양 맞춰주기
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # 기존 모양으로
        output = output.transpose(0,1)

        # dropout
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p = self.dropout_rate,
                               training = self.training)

        return output

    def forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # example x = > (32, 31,616)
        lengths = x_mask.data.eq(0).long().sum(1).sqeeze() # (32, )
        _, idx_sort = torch.sort(lengths, dim=0, descending=True) # 큰것부터, 같은값 끼리는 random index
        _, idx_unsort = torch.sort(idx_sort, dim=0) # 위에서 sort한 순서를 idx로 기억

        lengths = list(lengths[idx_sort])
        # idx_sort = Variable(idx_sort)
        # idx_unsort = Variable(idx_unsort)
        idx_sort = torch.tensor(idx_sort)
        idx_unsort = torch.tensor(idx_unsort)

        # 길이가 큰 것부터 x 재배치 => (32, 31, 616)
        x = x.index_select(0, idx_sort)


        x = x.transpose(0, 1) # => (31, 32, 616)

        # rnn_ input = [data = (956, 616), batch_sizes = (31, )]
        # 956 means 1_token_1, 1_token_2 , ... 1_token_31, 2_token_1, ..., 32_token_26
        # sum(batch_sizes) = 956
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)


        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # dropout
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data, # => (956, 616)
                                          p = self.dropout_rate,
                                          training=self.training)
                # 길이 정보 다시 추가, rnn_ input = [data = (956, 616), batch_sizes = (31, )]
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # 기존 모양으로 돌려주기
        for i, o in enumerate(outputs[1:], 1): # i from 1
            #o = [data = (956, 256), batch_sizes = (31, )] => (31, 32, 256)
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # 3layers 합치기
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # 기존 x batch 순서 찾아가기
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # padding 사이즈 안 맞는 경우 맞춰주기
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            #output = torch.cat([output, Variable(padding)], 1)
            output = torch.cat([output, torch.tensor(padding)], 1)

        # dropout
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p = self.dropout_rate,
                               training = self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        #
        if self.linear:
            # for example,
            # x.shape = (32, 292, 300)
            # y.shape = (32, 22, 300)
            # x.view(-1, x.size(2)) => (9344, 300)
            # self.linear(9344, 300) => (9344, 300)
            # (9344, 300).view(x.size()) => (32, 292, 300)
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # (32, 292, 300).bmm((32, 300, 22)) => (32, 292, 22)
        scores = x_proj.bmm(y_proj.transpose(2,1))
        # context_token * question_token
        # 의미는 292 context token과 22 question token의 관계라고 파악

        # unsquueze(1) :  (32, 22) => (32, 1, 22)
        y_mask = y_mask.unsqueeze(1).expand(scores.size()) # => (32, 292, 22)
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        #
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=1) # => (32*292, 22)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1)) # => (32, 292, 22)

        # (32, 292, 22).bmm(32, 22, 300) => (32, 130, 300)
        matched_seq = alpha.bmm(y)

        return matched_seq

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        # x, y => doc, ques / 2(bi) * 128(hidden_size) * 3(hideen_layer)
        Wy = self.linear(y) if self.linear is not None else y # (1, 768) => (1, 768)
        # (32, 292, 768).bmm((32, 768, 1)) => (32, 292, )
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            #training 중에는 log(probability) for NLL
            alpha = F.log_softmax(xWy, dim=1)
        else:
            # 아니면 최종 출력 probability (292, )
            alpha = F.softmax(xWy, dim=1)

        return alpha

class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1)) # => (-1, 768)
        scores = self.linear(x_flat).view(x.size(0), x.size(1)) # => (-1, 1) for self attention
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------

def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    # alpha = Variable(torch.ones(x.size(0), x.size(1)))
    alpha = torch.tensor(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha

def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len"""
    return weights.unsqueeze(1).bmm(x).squeeze(1)