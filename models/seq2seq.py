import torch.nn as nn
import torch

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decoder_fc):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder_fc = decoder_fc
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths, target_variable=None, 
                teacher_forcing_ratio=0, candidates = None, logistic_joint_decoding = False, log_output = None):
        if not isinstance(input_lengths, list):
            input_lengths = input_lengths.tolist()

        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        output = self.decoder_fc(encoder_outputs, encoder_outputs)
        
        if log_output is not None:
            # Provide logit output from other model
            output = log_output

        if logistic_joint_decoding:
            logit_output = torch.cat((output,torch.ones((output.shape[0],2), dtype=torch.float32).to(output.device) * 0.5), dim = 1)
        else:
            logit_output = None
        
        #Encoder and decoder have different layer
        n_direction = 2 if self.encoder.bidirectional else 1

        if self.encoder.n_layers >= self.decoder.n_layers:
            n = self.decoder.n_layers * n_direction
            encoder_hidden = tuple([ h[-n:,:,:] for h in encoder_hidden])
        else:
            _, batch_size, hidden_size = encoder_hidden[0].shape 
            n = self.decoder.n_layers * n_direction
            m = self.encoder.n_layers * n_direction  
            zero = torch.zeros((n-m, batch_size, hidden_size),dtype = torch.float32).to(encoder_hidden[0].device)
            encoder_hidden =  tuple([torch.cat((h,zero.clone()), dim = 0) for h in encoder_hidden])
        
        # Feed to decoder
        decoder_outputs, decoder_hidden, ret_dict  = self.decoder(inputs=target_variable,
                              candidates = candidates,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              logit_output = logit_output)
        return decoder_outputs, decoder_hidden, ret_dict, output
