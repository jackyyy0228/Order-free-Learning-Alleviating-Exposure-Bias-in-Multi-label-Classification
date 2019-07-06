import torch.nn as nn
import torch

class Seq2vec(nn.Module):

    def __init__(self,encoder, decoder_fc ):
        super(Seq2vec, self).__init__()
        self.encoder = encoder
        self.decoder_fc = decoder_fc

    def forward(self, input_variable, input_lengths):
        if not isinstance(input_lengths, list):
            input_lengths = input_lengths.tolist()

        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        output = self.decoder_fc(encoder_outputs, encoder_outputs)
        
        return output
