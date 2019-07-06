import random

import numpy as np

import torch
import torch.nn as nn

from .dot_attention import DotAttention
from .base_rnn import BaseRNN
from .simple_decoder import SimpleDecoder
import models.utils as utils

class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default: False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default: `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default: `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default: 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id, loss_type,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, 
                 sampling_type = 'sample', add_mask = True):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True, dropout=self.dropout_p)
        
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.add_mask = add_mask
        self.n_layers = n_layers

        self.init_input = None
        self.loss_type = loss_type

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = DotAttention(self.hidden_size)
        self.decoder = SimpleDecoder(self.hidden_size, self.output_size, 
                                     sampling_type = sampling_type)

    def forward_step(self, input_var, hidden, encoder_outputs):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)
        else:
            output = output.contiguous()
        return output, hidden, attn

    def forward(self, inputs=None,
                encoder_hidden=None, encoder_outputs=None,
                dataset=None, teacher_forcing_ratio=0,
                candidates = None, logit_output = None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        ori_inputs = inputs
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             teacher_forcing_ratio, candidates)
        decoder_hidden = self._init_state(encoder_hidden)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def post_decode(step_output, step_symbols, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            sequence_symbols.append(step_symbols)

            eos_batches = step_symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)

        decoder_input = inputs[:, 0].unsqueeze(1)
        mask = torch.zeros((batch_size, self.output_size), dtype = torch.float32).to(inputs.device)
       
        for di in range(max_length-1):
            context, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, symbols = self.decoder(context, mask, candidates, logit_output = logit_output)
            decoder_output = decoder_output.log()

            if teacher_forcing_ratio < 1.0:
                ran = torch.rand(symbols.shape).to(symbols.device)
                is_ss = ran.gt(teacher_forcing_ratio).float()
                if ori_inputs is not None:
                    # vanilla + SS
                    corrects = inputs[:,di+1].unsqueeze(1).unsqueeze(2)
                else:
                    # order free + SS
                    corrects = symbols
                ##sample
                ori = self.decoder.sampling_type
                self.set_sampling_type('sample')
                _, sample_symbols = self.decoder(context, mask, candidates)
                self.set_sampling_type(ori)
                step_symbols = (is_ss * sample_symbols.float() + (1-is_ss) * corrects.float()).squeeze(1).long()
            else:
                if ori_inputs is not None:
                    step_symbols = inputs[:,di+1].unsqueeze(1)
                else:
                    step_symbols = symbols.squeeze(1)
            
            if 'candidates' in self.decoder.sampling_type: 
                candidates -= utils.to_one_hot(symbols.squeeze(2).squeeze(1), self.output_size).float()
                is_eos_batch = torch.sum(candidates, dim = 1).eq(0)
                candidates[:,self.eos_id] = is_eos_batch.float()
            
            step_output = decoder_output.squeeze(1)
            post_decode(step_output, step_symbols, attn)
            decoder_input = step_symbols
            if self.add_mask:
                # mask is one if a symbol has been predicted
                # There will be error if loss is nan
                mask[range(batch_size),step_symbols.squeeze(1)] = 1
                mask[:,self.eos_id] = 0
                #mask += utils.to_one_hot(step_symbols.squeeze(1), self.output_size).float() 

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()
        
        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio, candidates = None):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0 and candidates is None:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1)  # minus the start of sequence symbol

        return inputs, batch_size, max_length
    def set_sampling_type(self,sampling_type):
        self.decoder.sampling_type = sampling_type
