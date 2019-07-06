import torch

def logistic_rescore(top_k_symbols, logistic_output):
    # top_k_symbols : seq_len [batch_size * k,1]
    # logistic_output : [batch_size, label_set_size] 
    # eos_id : label_set_size + 1
    # sos_id : label_set_size
    batch_size, k, _ = top_k_symbols[0].shape
    label_set_size = logistic_output.shape[1]
    seq_len = len(top_k_symbols)
    ##

    positive_score = torch.log(logistic_output + 1e-8) - torch.log(1-logistic_output + 1e-8)

    ## Add eos and sos 
    logistics  = torch.zeros((batch_size, label_set_size + 2), dtype=torch.float32).to(logistic_output.device)
    logistics[:,:label_set_size] = positive_score
    
    score = torch.zeros((batch_size,k), dtype=torch.float32).to(logistic_output.device)
    
    for t in range(0, seq_len):
        # score[i][j] += logitstic_output[i][top_k_symbols[t][i][j]]
        score += logistics.gather(1,top_k_symbols[t].squeeze(2))
    top1 = score.topk(1)[1]
    #symbols[i][0] = time_symbols[i][top1[i][0]]
    sequence = [time_symbols.squeeze(2).gather(1,top1).squeeze(1) for time_symbols in top_k_symbols] # seq_len [batch_size]
    return sequence, score
