import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import time
from datetime import datetime
from test import get_batch_scores


def save_model(model, epoch, loss, bleu, model_dir):
    model_path = model_dir + 'bleu_' + str(round(bleu, 4)) + '_loss_' + str(round(loss, 4)) + '_' + str(epoch)
    with open(model_path, 'wb') as f:
        torch.save(model, f)
            
def train(model, train_loader, valid_loader, param):
    print('Training model...')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr = param.learning_rate)  
    
    best_loss = 1000
    max_bleu_4 = 0
    for epoch in range(param.nb_epoch): 
        train_loss = train_epoch(model, epoch, train_loader, optimizer)
        if train_loss <= best_loss:
            best_loss = train_loss
            save_model(model, epoch, train_loss, 0, parm.model_dir)

        '''
        bleu_rouge = eval_epoch(model, epoch, valid_loader, param.idx2word)
        if bleu_rouge['Bleu-4'] > max_bleu_4:
            max_bleu_4 = bleu_rouge['Bleu-4']  
            save_model(model, epoch, train_loss, max_bleu_4, param.model_dir) 
        '''

    print('Train End.\n')
    

def train_epoch(model, epoch, loader, optimizer):
    print('Train epoch :', epoch)
    model.train()
                                         
    epoch_loss = 0.0
    nb_batch = 0
    for batch_idx, (question, paragraph, answer, question_length, paragraph_length) in enumerate(loader):
        nb_batch += 1

        question = Variable(question.long())
        paragraph = Variable(paragraph.long())
        answer = Variable(answer.long(), requires_grad = False)

        if torch.cuda.is_available() == True:
            question = question.cuda()
            paragraph = paragraph.cuda()
            answer = answer.cuda()

        batch_loss = model.get_loss(question, paragraph, answer, question_length, paragraph_length) 

        optimizer.zero_grad()
        batch_loss.backward()  
        #nn.utils.clip_grad_norm(model.parameters(), max_norm = 5.0)
        optimizer.step()
            
        epoch_loss += sum(batch_loss.data.cpu().numpy())
        print('-----epoch:', epoch, ' batch:',batch_idx,' train_loss:', batch_loss.data[0])
        
    epoch_loss = epoch_loss / nb_batch
    print('\nEpoch: ', epoch, ', Train Loss: ', epoch_loss, '\n')
    return epoch_loss


def eval_epoch(model, epoch, loader, idx2word):
    print('Eval epoch :', epoch)
    model.eval()
    
    nb_batch = 0
    epoch_pre, epoch_rec, epoch_f1, epoch_pred = 0, 0, 0, 0
    for batch_idx, (question, paragraph, answer, question_length, paragraph_length) in enumerate(loader):
        nb_batch += 1
        question = Variable(question.long()).cuda()
        paragraph = Variable(paragraph.long()).cuda()
        answer = Variable(answer.long(), requires_grad = False).cuda()

        pred_answer = model.get_answer(question, paragraph, question_length, paragraph_length)
        
        question = question.data.cpu().numpy()
        evidence = evidence.data.cpu().numpy()
        pre, rec, f1 , nb_pred = get_batch_scores(pred_tags, answer, question, evidence, idx2word)
        print('----epoch:', epoch, ' batch:',batch_idx,'  can_pred:', nb_pred, '   ||  pre: ', pre, '   rec: ', rec, '   f1  :', f1)
        

    print('\nEpoch: ', epoch, '  Pred: ', epoch_pred, '  || Pre:', epoch_pre, '    Rec:', epoch_rec,'    F1:', epoch_f1, '\n')
    return bleu_rouge




if __name__ == '__main__':

    print('Hey')
    
    

    
    
    
    
    
    
