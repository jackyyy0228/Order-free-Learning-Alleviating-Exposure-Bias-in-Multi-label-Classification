import torch
from torch.autograd import Variable
import models.utils as utils

CUDA = torch.cuda.is_available()
INF = 1e5

def _inflate(tensor, times, dim):
        """
        Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)

        Args:
            tensor: A :class:`Tensor` to inflate
            times: number of repetitions
            dim: axis for inflation (default=0)

        Returns:
            A :class:`Tensor`

        Examples::
            >> a = torch.LongTensor([[1, 2], [3, 4]])
            >> a
            1   2
            3   4
            [torch.LongTensor of size 2x2]
            >> b = ._inflate(a, 2, dim=1)
            >> b
            1   2   1   2
            3   4   3   4
            [torch.LongTensor of size 2x4]
            >> c = _inflate(a, 2, dim=0)
            >> c
            1   2
            3   4
            1   2
            3   4
            [torch.LongTensor of size 4x2]

        """
        repeat_dims = [1] * tensor.dim()
        repeat_dims[dim] = times
        return tensor.repeat(*repeat_dims)

class TopKDecoder(torch.nn.Module):
    r"""
    Top-K decoding with beam search.

    Args:
        decoder_rnn (DecoderRNN): An object of DecoderRNN used for decoding.
        k (int): Size of the beam.

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (seq_len, batch, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features
          in the hidden state `h` of encoder. Used as the initial hidden state of the decoder.
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
    - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*length* : list of integers
          representing lengths of output sequences, *topk_length*: list of integers representing lengths of beam search
          sequences, *sequence* : list of sequences, where each sequence is a list of predicted token IDs,
          *topk_sequence* : list of beam search sequences, each beam is a list of token IDs, *inputs* : target
          outputs if provided for decoding}.
    """

    def __init__(self, decoder_rnn, k, beam_score_type = 'sum'):
        super(TopKDecoder, self).__init__()
        self.rnn = decoder_rnn
        self.k = k
        self.hidden_size = self.rnn.hidden_size
        self.V = self.rnn.output_size
        self.SOS = self.rnn.sos_id
        self.EOS = self.rnn.eos_id
        self.beam_score_type = beam_score_type
        self.n_layers = decoder_rnn.n_layers

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                dataset=None, teacher_forcing_ratio=0, retain_output_probs=True, 
                candidates = None, logit_output = None):
        """
        Forward rnn for MAX_LENGTH steps.  Look at :func:`seq2seq.models.DecoderRNN.DecoderRNN.forward_rnn` for details.
        """

        inputs, batch_size, max_length = self.rnn._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                 teacher_forcing_ratio)

        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: b*k x h
        encoder_hidden = self.rnn._init_state(encoder_hidden)
        if encoder_hidden is None:
            hidden = None
        else:
            if isinstance(encoder_hidden, tuple):
                n_layer_bidiretion = encoder_hidden[0].size(0) # n_layer * direction
                hidden = tuple([_inflate(h, self.k, 2).view(n_layer_bidiretion,batch_size * self.k,-1) for h in encoder_hidden])
            else:
                # TODO :Should check _inflat dimension 
                hidden = _inflate(encoder_hidden, self.k, 2).view(1, batch_size * self.k)
        # ... same idea for encoder_outputs and decoder_outputs
        if self.rnn.use_attention:
            _ ,encoder_length, encoder_output_size = encoder_outputs.shape 
            inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 1).view(batch_size * self.k,encoder_length, encoder_output_size )
        else:
            inflated_encoder_outputs = None
        # logit output
        if logit_output is not None:
            label_size = logit_output.shape[-1]
            logit_output = _inflate(logit_output,self.k,1).view(batch_size * self.k,label_size)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.zeros((batch_size * self.k, 1), dtype = torch.float32)
        sequence_scores.fill_(-1000)
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)

        # Initialize the input vector
        input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))
        
        # Initialize mask
        mask = torch.zeros((batch_size * self.k , self.V), dtype = torch.float32)

        # Initialize lengths
        lengths = torch.ones((batch_size * self.k, 1), dtype = torch.float32)
        
        # Initialize eos
        eos_indices = input_var.data.eq(self.EOS)
        eos_score = sequence_scores * eos_indices.float() #bk*1
        
        # Assign all vars to CUDA if available
        if CUDA:
            self.pos_index = self.pos_index.cuda()
            input_var = input_var.cuda()
            sequence_scores = sequence_scores.cuda()
            mask = mask.cuda()
            lengths = lengths.cuda()
            eos_indices = eos_indices.cuda()
            eos_score = eos_score.cuda()

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for t in range(max_length):
            # Run the RNN one step forward
            context, hidden, attn = self.rnn.forward_step(input_var, hidden,
                                                          inflated_encoder_outputs)
            softmax_output, _ = self.rnn.decoder(context, logit_output = logit_output)
            
            log_softmax_output = softmax_output.log().squeeze(1) #bk * v
            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output) #bk * v
            
            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = _inflate(sequence_scores, self.V, 1) #bk*V
            sequence_scores += log_softmax_output + mask
            
            # Terminated sentence can only produce eos token
            eos_mask = eos_indices.squeeze().float()
            sequence_scores[:,self.EOS] = \
                sequence_scores[:,self.EOS] * (1 - eos_mask) +  eos_score.squeeze() * eos_mask  #[bk]

            # Calculate new score
            if self.beam_score_type == 'sum':
                scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k) # b* kV
                input_var = (candidates % self.V).view(batch_size * self.k, 1)
                # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
                sequence_scores = scores.view(batch_size * self.k, 1)
            elif self.beam_score_type  == 'mean':
                # Mean of scores in each time step
                scores, candidates = (sequence_scores / lengths).view(batch_size, -1).topk(self.k, dim=1) # b* kV
                input_var = (candidates % self.V).view(batch_size * self.k, 1)
                # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
                sequence_scores = scores.view(batch_size * self.k, 1) * lengths
            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1) # b*k
            
            # Update mask
            mask = mask[predecessors.squeeze(), :] - utils.to_one_hot(input_var.squeeze(), self.V).float() * INF 
            mask[:,self.EOS] = 0

            
            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())
            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            
            eos_indices = input_var.data.eq(self.EOS) # bk* 1
            eos_score = sequence_scores * eos_indices.float()
            '''
            print(sequence_scores.view(batch_size,-1)[0])
            print(input_var.view(batch_size, -1)[0])
            print(predecessors.view(batch_size, -1)[0])
            print(log_softmax_output.view(batch_size, self.k, self.V)[0,predecessors.view(batch_size, -1)[0],input_var.view(batch_size, -1)[0]])
            print('-'*100)
            '''
            # Update lengths
            if t < max_length -1:
                sequence_scores.data.masked_fill_(eos_indices, -1000)
                lengths = lengths[predecessors.squeeze(),0].view(batch_size * self.k,1) + ( 1 - eos_indices.float())

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)
        #print(sequence_scores[:20])
        # Do backtracking to return the optimal values
        t = max_length - 1
        outputs = []
        output_symbols = []
        step_scores = []
        now_indexes = torch.arange(batch_size * self.k)
        '''
        now_idx = 0
        print("start")
        sco = 0
        '''
        while t >= 0:
            t_predecessors = stored_predecessors[t].squeeze()
            
            prev_indexes = now_indexes
            now_indexes = stored_predecessors[t].squeeze()[now_indexes]
            ''' 
            prev_idx = now_idx
            now_idx = t_predecessors[now_idx].item()
            '''
            current_symbol = stored_emitted_symbols[t][prev_indexes,0].view(batch_size,self.k)
            current_output = stored_outputs[t][now_indexes].view(batch_size, self.k, self.V)
            #score[i][j][0] = output[i][j][symbol[i][j][0]]
            current_score = current_output.gather(2, current_symbol.unsqueeze(2)).view(batch_size,self.k)
            # record the back tracked results
            step_scores.append(current_score)
            outputs.append(current_output)
            output_symbols.append(current_symbol.unsqueeze(2))
            #x = current_symbol[0][0].item()
            #print(x)
            #print(current_output[0][0][x])
            #print(x)
            '''
            out_token = stored_emitted_symbols[t][prev_idx][0].item()
            print(prev_idx,now_idx, out_token, stored_outputs[t][now_idx][out_token])
            #print(x)
            #print(stored_outputs[t][now_idx][x])
            sco += stored_outputs[t][now_idx][out_token]
            '''
            t -= 1
        
        outputs.reverse() #[ b,k,V]
        output_symbols.reverse() #[b,k]
        step_scores.reverse()

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in outputs]
        decoder_hidden = None

        metadata = {}
        metadata['output'] = outputs # seq_len [batch_size * k * V]
        if self.beam_score_type  == 'sum':
            metadata['topk_score'] = (sequence_scores ).view(batch_size, self.k) # [batch_size * k]
        elif self.beam_score_type == 'mean':
            metadata['topk_score'] = (sequence_scores / lengths).view(batch_size, self.k) # [batch_size * k]
        metadata['topk_sequence'] = output_symbols # seq_len [batch_size * k,1] 
        metadata['topk_length'] = lengths.view(batch_size, self.k) # seq_len [batch_size * k] 
        metadata['step_score'] = step_scores # seq_len [batch_size * k]
        metadata['sequence'] = [seq[:,0] for seq in output_symbols] # seq_len [batch_size] 
        '''
        idx = 0
        sco = 0
        for t in range(max_length):
            x = output_symbols[t][idx][0].item()
            x_score = decoder_outputs[t][idx][x].item()
            sco += x_score
            print(x,x_score, sco)
        print(sequence_scores[batch_size * idx][0])
        exit()
        S = [x[0][0].item() for x in step_scores]
        print([x[0][0].item() for x in step_scores])
        print(torch.sum(torch.tensor(S[:5])))
        print([x[0] for x in metadata['sequence']])
        print(lengths[0][0])
        print(metadata['topk_score'][0][0]*lengths[0][0])
        exit()
        '''
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.

        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state

        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.

            score [batch, k]: A list containing the final scores for all top-k sequences

            length [batch, k]: A list specifying the length of each sequence in the top-k candidates

            p (sequence_len, batch, k): A Tensor containing predicted sequence
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            if CUDA:
                h_n = tuple([torch.zeros(state_size).cuda(), torch.zeros(state_size).cuda()])
            else:
                h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
            if CUDA:
                h_n = h_n.cuda()
        l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        '''
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for step in reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data
        '''

        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
            score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)


