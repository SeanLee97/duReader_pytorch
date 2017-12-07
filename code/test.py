import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import time
from datetime import datetime

STOP_TAG = '#OOV#'

def strict_match(preds, ans):
    for p in preds:
        if p in ans or ans in p:
            return 1
    return 0


def get_tagging_results(tokens, tags):
    chunks = set()
    start = -1
    for i, tok in enumerate(tokens):
        tag = tags[i]
        if tag == 0:  # B
            if start >= 0: chunks.add(''.join(tokens[start:i]))
            start = i
        elif tag == 1:  # I
            if start < 0: start = i
        else:
            if start < 0: continue
            chunks.add(''.join(tokens[start:i]))
            start = -1
    if start >= 0:
        chunks.add(''.join(tokens[start:]))
        
    if len(chunks) == 0:
        chunks.add('no_answer')
    return list(chunks)


def get_batch_scores(pred_tags, answer, question, evidence, idx2word):
    nb_pred = 0
    A, C, Q = 0, 0, 0
    for pred, ans , ques, evid in zip(pred_tags, answer, question, evidence):
        ques = [ idx2word[q] for q in ques if q != 0 ]
        evid = [ idx2word[e] for e in evid ]
        #pred = [ p for p in pred ]
        ans = ''.join( [ idx2word[a] for a in ans if a != 0 ] )
        
        pred_ans = get_tagging_results(evid, pred)
        
        evid = [ e for e in evid if e != STOP_TAG ]
        print('Question: ', ''.join(ques), '\n')
        print('Evidence: ', ''.join(evid), '\n')
        #print('Tags: ', pred, '\n')
        print('Predict Answers: ', pred_ans)
        print('Golden Answers: ', ans)
        print('\n ---------------------------- \n')
        
        if len(pred_ans) > 0 :
            nb_pred += 1
        
        C += strict_match(pred_ans, ans)
        A += len(pred_ans)
        Q += 1
    
    if ( A == 0):
        pre = 0
    else:
        pre = C / A
    
    if ( Q == 0):
        rec = 0
    else:
        rec = C / Q
        
    if (pre + rec == 0):
        f1 = 0
    else:
        f1 = (2 * pre * rec) / (pre + rec)
        
    return pre, rec, f1, nb_pred


def test(model, loader, idx2word):
    print('Testing model...')
    nb_batch = 0
    epoch_pre, epoch_rec, epoch_f1, epoch_pred = 0, 0, 0, 0
    for batch_idx, (question, evidence, q_mask, e_mask, qe_feat, answer) in enumerate(loader):
        nb_batch += 1
        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        qe_feat = Variable(qe_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()

        pred_tags = model.get_tags(question, evidence, q_mask, e_mask, qe_feat)
        
        question = question.data.cpu().numpy()
        evidence = evidence.data.cpu().numpy()
        
        pre, rec, f1 , nb_pred = get_batch_scores(pred_tags, answer, question, evidence, idx2word)
        print('batch:',batch_idx,'  nb_pred:', nb_pred, '   ||  pre: ', pre, '   rec: ', rec, '   f1  :', f1)
        
        epoch_pre += pre
        epoch_rec += rec
        epoch_f1 += f1
        epoch_pred += nb_pred
                                         
    epoch_pre = epoch_pre / nb_batch
    epoch_rec = epoch_rec / nb_batch
    epoch_f1 = epoch_f1 / nb_batch
    print('Pre:', epoch_pre, '    Rec:', epoch_rec,'    F1:', epoch_f1, '\n')
    return epoch_pre, epoch_rec, epoch_f1, epoch_pred



if __name__ == '__main__':
    
    print('Hey')
    
    

    
    
    
    
    
    
