import json
import jieba
import pickle
import csv, h5py
import pandas as pd
import numpy as np
from tqdm import *
import torch
from torch import Tensor
from torch.autograd import Variable
import torch.utils.data as data
from main import Hyperparameters
from collections import Counter

STOP_TAG = "#stop#"   
UNK_TAG = "#unk#"   

def filter(ret, min_count):
	count = pd.Series(ret).value_counts()
	count = count[count >= min_count]
	char_set = list(count.index)
	return char_set

def get_vocab(param):
	ret = []
	with open(param.train_json_path) as f:
		for line in tqdm(f):
			line = json.loads(line)

			if len(line['answer_docs']) == 0 or len(line['fake_answers']) == 0:
				continue 

			document = line['documents'][line['answer_docs'][0]]
			paragraph = document['paragraphs'][document['most_related_para']]

			for p in paragraph: ret.append(p)

	ret = filter(ret, param.min_count)

	ret = sorted(list(ret))
	input_set = [STOP_TAG, UNK_TAG]
	input_set.extend(list(ret))
	input_set_size = len(input_set)
	input2idx = dict(zip(input_set, range(input_set_size)))

	print('Vacabulary size:', input_set_size, '\n')
	return input2idx, input_set_size


def save_vocab(path, input2idx):
	print('Saving bocabulary...')
	f = open(path,'wb')
	pickle.dump(input2idx, f)
	f.close()


def load_vocab(path):
	print('Loading vocabulary...')
	f = open(path, 'rb')
	input2idx = pickle.load(f)
	input_set = list(input2idx.keys())
	input_set_size = len(input_set)
	f.close()
	print('Vacabulary size:', input_set_size, '\n')
	return input2idx, input_set_size


# ------------------ save h5py file --------------------------- #	
	

def load_evidence_and_feats(evidence, feats, input2idx):
	evidence_vector = []
	feats_vector = []
	for e, f in zip(evidence, feats):
		if e in input2idx:
			evidence_vector.append(input2idx[e])
			feats_vector.append(f)	  
	return evidence_vector, feats_vector, len(evidence_vector)


def pad_sequence(seq, seq_size, word2idx): 
	vector = []
	for i in range(seq_size):
		if i >= len(seq):
			vector.append(word2idx[STOP_TAG])
		elif seq[i] not in word2idx:
			vector.append(word2idx[UNK_TAG])
		else:
			vector.append(word2idx[seq[i]])

	if len(seq) < seq_size: 
		length = len(seq)
	else: 
		length = seq_size

	return vector, length


def save_data(file, param, data, shape, i):
	if i <= param.batch_storage_size:
		for key, value in data.items():
			if value == []: continue
			file.create_dataset(key, data = value, maxshape = shape[key])

	else:
		old_len = len(file['question'])
		new_len = old_len + len(data['question'])

		for key, value in data.items():
			if value == []: continue
			new_shape = [new_len]
			for s in shape[key][1:]:
				new_shape.append(s)
			file[key].resize(new_shape)

			file[key][old_len: new_len] = value

	print(i)


def get_train_data(param, line):
	document = line['documents'][line['answer_docs'][0]]
	#paragraph = document['paragraphs'][document['most_related_para']]
	segmented_paragraph = document['segmented_paragraphs'][document['most_related_para']]
	paragraph = ''.join(segmented_paragraph)
	if len(paragraph) > param.paragraph_size:
		return [], [], []
	paragraph, paragraph_length = pad_sequence(paragraph, param.paragraph_size, param.word2idx)

	answer_span = line['answer_spans'][0]
	fake_answer = line['fake_answers'][0]
	answer_start = len(''.join(segmented_paragraph[:answer_span[0]])) 
	answer_end = len(''.join(segmented_paragraph[:answer_span[1]+1]))
	answer = [answer_start, answer_end]

	return paragraph, paragraph_length, answer

def get_val_data(param, line):
	paragraphs, paragraph_lengths, answers = [], [], []
	documents = line['documents']
	question_tokens = line['segmented_question']
	for d in documents:
		para_infos = []
		for para_tokens in d['segmented_paragraphs']:				  
			common_with_question = Counter(para_tokens) & Counter(question_tokens)
			correct_preds = sum(common_with_question.values())
			if correct_preds == 0:
				recall_wrt_question = 0
			else:
				recall_wrt_question = float(correct_preds) / len(question_tokens)
			para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
		para_infos.sort(key=lambda x: (-x[1], x[2]))
		fake_paragraph = ''.join(para_infos[0][0])
		if (len(fake_paragraph)) > param.paragraph_size:
			continue
		fake_paragraph, fake_paragraph_length = pad_sequence(fake_paragraph, param.paragraph_size, param.word2idx)
		paragraphs.append(fake_paragraph)
		paragraph_lengths.append(fake_paragraph_length)
	
	answers = line['answers']

	return paragraphs, paragraph_lengths, answers

def save_h5py_file(param, old_path, new_path):
	print('Saving (', new_path, ')...')
	file = h5py.File(new_path,'w')

	data = {'question_id':[], 'question_type':[], 'question':[], 'question_length':[], 
			'paragraph':[], 'answer':[], 'paragraph_length':[], 'paragraphs':[], 'paragraph_lengths':[]}

	shape = {'question_id':(None,), 'question_type':(None,), 'question':(None, param.question_size), 'question_length':(None,), 
			'paragraph':(None, param.paragraph_size), 'answer':(None, 2), 'paragraph_length':(None,),
			'paragraphs':(None, None, param.paragraph_size), 'paragraph_lengths':(None, None,)}
	#evaluate = {}

	i = 0
	with open(old_path) as f:
		for line in tqdm(f):
			line = json.loads(line)
			documents = line['documents']

			question = line['question']
			question_id = line['question_id']
			question_type = line['question_type']
			question_tokens = line['segmented_question']
			if len(question) > param.question_size:
				continue
			
			# train
			if old_path == param.train_json_path:
				if len(line['answer_docs']) == 0 or len(line['fake_answers']) == 0:
					continue 
				paragraph, paragraph_length, answer = get_train_data(param, line)
				if paragraph == []: continue

				data['paragraph'].append(paragraph)
				data['paragraph_length'].append(paragraph_length)
				data['answer'].append(answer)
				
			# val
			elif old_path == param.val_json_path:
				paragraphs, paragraph_lengths, answers = get_val_data(param, line)
				if paragraphs == []: continue

				data['paragraphs'].append(paragraphs)
				data['paragraph_lengths'].append(paragraph_lengths)
				#data['answers'].append(answers)
				data['question_id'].append(question_id)

			question, question_length = pad_sequence(question, param.question_size, param.word2idx)
			data['question'].append(question)
			data['question_length'].append(question_length)

			# ---------------------------------
			i += 1
			if i % param.batch_storage_size == 0:
				save_data(file, param, data, shape, i)
				data = {'question_id':[], 'question_type':[], 'question':[], 'question_length':[], 
						'paragraph':[], 'answer':[], 'paragraph_length':[], 'paragraphs':[], 'paragraph_lengths':[]}

		if i % param.batch_storage_size != 0:
			save_data(file, param, data, shape, i)

	file.close()
	print('Dataset: ', i)
	

def get_answer():
	with open(param.val_json_path) as f:
		for line in tqdm(f):
			line = json.loads(line)
			question_id = line['question_id']
			answers = line['answers']

if __name__ == '__main__':

	param =  Hyperparameters()  

	# 5143
	#word2idx, word_set_size = get_vocab(param)
	#idx2word = dict(zip(word2idx.values(), word2idx.keys()))
	#print(word2idx['苏'], idx2word[520])

	#save_vocab(param.vocab_path, word2idx)
	
	param.word2idx, param.vocab_size = load_vocab(param.vocab_path)
	param.idx2word = dict(zip(param.word2idx.values(), param.word2idx.keys()))
	#print(word2idx['苏'], idx2word[520])

	
	#save_h5py_file(param, param.train_json_path, param.train_h5py_path)
	save_h5py_file(param, param.val_json_path, param.val_h5py_path)


	
	
	
	
	
   
	
