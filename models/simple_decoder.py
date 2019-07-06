import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleDecoder(nn.Module):
    """ Simple Decoder Model """

    def __init__(self, hidden_size, output_size, sampling_type = 'max'):
        super(SimpleDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.sampling_type = sampling_type

    def forward(self, context, mask = None, candidates = None, logit_output = None):
        #mask : batch_size * output_size
        #candidates : (batch_size * output_size) Use in order free rnn, which 
        #             contains all the labels which was not prredicted.
        batch_size, de_len = context.size(0), context.size(1)
        logits = self.linear(context.view(-1, self.hidden_size))
        softmax = torch.softmax(logits, dim = -1).view(batch_size, de_len, self.output_size)

        if logit_output is not None and de_len == 1:
            logit_output = logit_output.unsqueeze(1)
            softmax = softmax * logit_output / (1 - logit_output + 1e-8)
        
        if mask is not None:
            sample_softmax = softmax * (1 - mask.unsqueeze(1)) 
            sample_softmax = sample_softmax / torch.sum(sample_softmax, dim = -1).unsqueeze(-1)
        else:
            sample_softmax = softmax

        if self.sampling_type == 'sample_from_candidates':
            # Can be used only in step_forward
            candidates = candidates.view(batch_size, de_len, self.output_size)
            symbols = (softmax * candidates).topk(1, dim=2)[1]
        elif self.sampling_type == 'max_from_candidates':
            candidates = candidates.view(batch_size, de_len, self.output_size)
            sample_softmax = (softmax * candidates) / torch.sum(softmax*candidates, dim = -1).unsqueeze(-1)
            symbols = torch.distributions.Categorical(sample_softmax).sample()
            symbols = symbols.unsqueeze(2)
        elif self.sampling_type == 'max':
            symbols = sample_softmax.topk(1, dim=2)[1]
        elif self.sampling_type == 'sample':
            symbols = torch.distributions.Categorical(sample_softmax).sample()
            symbols = symbols.unsqueeze(2)
            
        return softmax, symbols
