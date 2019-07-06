import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.weight.data.fill_(1.)


class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return torch.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x


class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, hidden_units, drop_rate, n_layers):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate
        # TODO
        self.n_layers = n_layers

        self.conv1 = nn.Conv2d(
            in_channels=freq_bins, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv3 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn0 = nn.BatchNorm2d(freq_bins)
        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.bn2 = nn.BatchNorm2d(hidden_units)
        self.bn3 = nn.BatchNorm2d(hidden_units)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

        a0 = self.bn0(x)
        a1 = F.dropout(F.relu(self.bn1(self.conv1(a0))),
                       p=drop_rate,
                       training=self.training)

        a2 = F.dropout(F.relu(self.bn2(self.conv2(a1))),
                       p=drop_rate,
                       training=self.training)

        emb = F.dropout(F.relu(self.bn3(self.conv3(a2))),
                        p=drop_rate,
                        training=self.training)

        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return emb

        else:
            return [a0, a1, a2, emb]


class DLMAEncoder(nn.Module):
    def __init__(self, freq_bins, classes_num, hidden_units, decoder_hidden_size, encoder_n_layer):
        super(DLMAEncoder, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, 0.5, encoder_n_layer)
        self.attention = Attention(
            hidden_units,
            classes_num,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.activation = torch.nn.LeakyReLU()

        # To RNN
        self.rnn_att = RNNContext(hidden_units, decoder_hidden_size)
        self.fc_rnn = nn.Linear(hidden_units, decoder_hidden_size * 2)
        
        # To sigmoid
        self.sigmoid_head = nn.Linear(2 * classes_num, classes_num)

    def forward(self, input):
        #batch_num = input.shape[0]
        #input = input.reshape(batch_num, 10, 128)

        # (samples_num, hidden_units, time_steps, 1)
        emb_layers = self.emb(input, return_layers=True)
        contexts = (self.rnn_att(emb_layers[-1]), self.rnn_att(emb_layers[-2]))
        #(samples_num, time_steps, 2* hidden_units)
        contexts = torch.cat(contexts, dim = 1).transpose(1,2).squeeze(-1)

        # (samples_num, classes_num)
        output1 = self.attention(emb_layers[-1])
        output2 = self.attention(emb_layers[-2])

        # (samples_num, classes_num * 2)
        cat_output = torch.cat((output1, output2), dim=1)
        
        ## fc 
        logits = self.sigmoid_head(cat_output)
        sigmoid_output = torch.sigmoid(logits)
        
        #cat_output = torch.cat((cat_output,logits.detach()), dim=1)
        # To (1,batch_size,hidden_size)
        output = self.fc_rnn(torch.mean(emb_layers[-1].squeeze(3),dim = 2))
        encoder_hidden = output
        encoder_hidden = encoder_hidden.unsqueeze(0)
        midpoint = encoder_hidden.size(2) // 2

        encoder_hidden = (encoder_hidden[:,:,:midpoint].contiguous(),encoder_hidden[:,:,midpoint:].contiguous())

        return encoder_hidden, sigmoid_output, contexts

class RNNContext(nn.Module):
    def __init__(self, hidden_units, decoder_hidden):
        super(RNNContext, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=hidden_units, out_channels=decoder_hidden,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self,x):
        att = self.conv1(x)
        att = torch.sigmoid(att)
        return att
    

