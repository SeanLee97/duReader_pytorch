import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import h5py
import numpy as np

# return pack rnn inputs
def get_pack_rnn_inputs( x, x_mask):
    lengths = x_mask.data.eq(0).long().sum(1).squeeze()
    _, idx_sort = torch.sort(lengths, dim = 0, descending = True)
    _, idx_unsort = torch.sort(idx_sort, dim = 0)

    lengths = list(lengths[idx_sort])

    # Sort x
    x = x.index_select(0, Variable(idx_sort))
    x = x.transpose(0, 1).contiguous()
    rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

    return rnn_input, Variable(idx_unsort)

    
def get_pad_rnn_outputs(output, x_mask, idx_unsort):
    output = nn.utils.rnn.pad_packed_sequence(output)[0]
    print('o: ', output.size())
    
    # Transpose and unsort
    output = output.transpose(0, 1).contiguous()
    output = output.index_select(0, idx_unsort)

    # Pad up to original batch sequence length
    if output.size(1) != x_mask.size(1):
        padding = torch.zeros(output.size(0),
                              x_mask.size(1) - output.size(1),
                              output.size(2)).type(output.data.type())
        output = torch.cat([output, Variable(padding)], 1)

    return output
    
    
def test_lstm():
    # batch:3, seq_len:5, embedding:10, hidden:20
    rnn = nn.LSTM(6, 7, 1)
    
    input = Variable(torch.randn(3, 5, 6))
    mask = Variable(torch.randn(3, 5).byte())
    print('input: ', input)
    print('mask: ', mask)
    
    input, i = get_pack_rnn_inputs(input, mask)
    output, hn = rnn(input)
    output = get_pad_rnn_outputs(output, mask, i)
    print('output: ', output)
    
    '''
    input2 = input.transpose(0,1).contiguous()
    print('ini input: ', input2)
    print('==========================')
    
    print('input: ', input)
    output, hn = rnn(input)
    print('lstm out: ', output.size())
    
    print('\n==========================\n')
    lengths = [4, 3, 1]
    rnn_input = nn.utils.rnn.pack_padded_sequence(input, lengths)
    
    print('rnn input: ', rnn_input)
    output, hn = rnn(rnn_input)
    print('lstm out: ', output)
    print('\n==========================\n')
    
    outputs = nn.utils.rnn.pad_packed_sequence(output)[0]
    print('outputs pack: ', outputs)

    input2 = Variable(torch.randn(3, 3, 6))
    
    input3 = torch.cat([input,input2], 0)
    
    print('1: ', input,'\n')
    print('==========================')
    print('2: ', input2, '\n')
    print('==========================')
    print('3:', input3)
    print('input3: ', input3.size())
    output, hn = rnn(input3)
    '''
    print('end')

def get_nll_loss(inputs, target):
    loss_list = []
    print()
    for seq, t in zip(inputs, target):
        loss = 0
        loss_list.append(seq[t])
    
    loss = torch.cat(loss_list, -1)
    
    w = 1.0 / len(loss)
    weight = [w for i in range(len(loss))]
    print('w: ', weight)
    weight = torch.Tensor(weight)
    
    print('loss: ', loss)
    print('weight: ', weight)
        
    loss2 = torch.dot(loss.cuda(), weight.cuda())
    
    loss = (-1) * loss
    print('loss2: ', loss2)
    
    loss = torch.mean(torch.cat(loss_list, -1))
    return loss

def test_loss():
    inputs = Variable(torch.randn(3, 5))
    target = Variable(torch.LongTensor([1, 0, 4]))
    pred = F.log_softmax(inputs)
    print('pred: ', pred)
    
    output = F.nll_loss(pred, target)
    print('output: ', output)
    #output.backward()
    
    mine = get_nll_loss(pred, target)
    print('mine : ', mine)

def test_h5py():
    file = h5py.File('fuck.h5','w')
    data = [[1,2,3],[4,5]]
    file.create_dataset('test', data=data)
    file.close()
    

class Hyperparameters:
    nb_epoch = 1000

def test_param():


if __name__ == '__main__':
    param = Hyperparameters()
    test_lstm()













    