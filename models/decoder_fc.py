import torch
import torch.nn as nn
from .dot_attention import DotAttention

class DecoderFC(nn.Module):
    def __init__(self, encoder_hidden_size, output_size, fc_layer_sizes, 
                 bidirectional_encoder, use_attention = True, dropout_p = 0):
        # bidirectional_encoder : True or False
        # fc_layer_sizes : A list of numbers e.g. [512, 512, 512] indicates a nn
        #                  of 3 layers with size 512
        super(DecoderFC, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        self.num_direction = 2 if bidirectional_encoder else 1
        self.hidden_size = encoder_hidden_size
        self.use_attention = use_attention
        self.dropout_p = dropout_p
        
        # first layer
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(encoder_hidden_size * self.num_direction, fc_layer_sizes[0]))
        self.module_list.append(nn.BatchNorm1d(fc_layer_sizes[0], momentum=0.5))
        self.module_list.append(nn.LeakyReLU())
        self.module_list.append(nn.Dropout(self.dropout_p))
        
        # inner layers
        for i in range(len(fc_layer_sizes) -1 ):
            self.module_list.append(nn.Linear(fc_layer_sizes[i], fc_layer_sizes[i+1]))
            self.module_list.append(nn.BatchNorm1d(fc_layer_sizes[i+1], momentum=0.5))
            self.module_list.append(nn.LeakyReLU())
            self.module_list.append(nn.Dropout(self.dropout_p))
        # convert to hidden size in order to use attention 
        self.module_list.append(nn.Linear(fc_layer_sizes[-1], self.hidden_size * self.num_direction))
        
        if use_attention :
            self.attention = DotAttention(self.hidden_size * self.num_direction)
            
        self.fc_final = nn.Linear(self.hidden_size * self.num_direction, output_size)
          
    def forward(self, x, context = None):
        # x : (batch, seq_len, hidden_size* num_direction)
        # context : (batch, en_len, hidden_size)
        out = x[:,-1,:]
        for layer in self.module_list:
            out = layer(out)
        if self.use_attention:
            out, attn = self.attention(out.unsqueeze(1), context)
        out = self.fc_final(out.squeeze(1))
        out = torch.sigmoid(out)
        return out

