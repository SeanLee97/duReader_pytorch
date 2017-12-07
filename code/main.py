import torch
import os
import pickle
import time
from datetime import datetime
from baseQA import baseQA
from loader import loadTrainDataset, loadValDataset, loadTestDataset
#from util import load_vocab, load_webQA_vocab, load_webQA_embedding
from train import train, train_epoch, eval_epoch
from test import test


class Hyperparameters:
    nb_epoch = 1000
    batch_size = 128
    tagset_size = 4
    question_size = 32
    paragraph_size = 512

    qe_embedding_size = 2
    embedding_size = 64
    
    min_count = 10
    batch_storage_size = 10000
    
    learning_rate = 0.001
    model_dir = ''

    train_json_path = '../data/preprocessed/trainset/search.train.json'
    val_json_path = '../data/preprocessed/devset/search.dev.json'
    test_json_path = '../data/test.json'

    train_h5py_path = '../data/train.h5py'
    test_h5py_path = '../data/test.h5py'
    val_h5py_path = '../data/dev.h5py'
    
    vocab_path = '../data/vocab.txt'

    word2idx = {}
    idx2word = {}
    vocab_size = 0

def load_vocab(path):
    print('Loading vocabulary...')
    f = open(path, 'rb')
    input2idx = pickle.load(f)
    input_set = list(input2idx.keys())
    input_set_size = len(input_set)
    f.close()
    print('Vacabulary size:', input_set_size, '\n')
    return input2idx, input_set_size

def train_model(param):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    param.model_dir = '../model/baseQA_' + str(datetime.now()).split('.')[0].split()[0] + '/'
    if os.path.exists(param.model_dir) == False:
        os.mkdir(param.model_dir)

    train_dataset = loadTrainDataset(param.train_h5py_path)
    val_dataset = loadValDataset(param.val_h5py_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)  
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)

    model = baseQA(param)
    if torch.cuda.is_available() == True:
        model = model.cuda()
    train(model, train_loader, val_loader, param)


def test_model(param):
    test_dataset = loadTestDataset(param.test_h5py_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = False)

    model = torch.load(param.model_path)
    test(model, test_loader, param)


if __name__ == '__main__':
    param = Hyperparameters() 
    
    print('Biu ~ ~  ~ ~ ~ Give you buffs ~ \n')
    
    param.word2idx, param.vocab_size = load_vocab(param.vocab_path)
    param.idx2word = dict(zip(param.word2idx.values(), param.word2idx.keys()))

    train_model(param)  

    #test_model(param)
    
    
    
    
    
