import json
import h5py
import numpy as np
from tqdm import *
from collections import Counter
import torch
from torch.autograd import Variable

# x = (batch, seq_len, hsize)
# return (batch, hidden_size)
def attention(self, x, x_mask):
	x_flat = x.view(-1, x.size(-1))
	scores = self.att_linear(x_flat).view(x.size(0), x.size(1))
	scores.data.masked_fill_(x_mask.data, -float('inf'))
	weights = F.softmax(scores)
	out = weights.unsqueeze(1).bmm(x).squeeze(1)
	return out

def uniform_weights(x, x_mask):
	"""Return uniform weights over non-masked input."""
	alpha = Variable(torch.ones(x.size(0), x.size(1)))
	if x.data.is_cuda:
		alpha = alpha.cuda()
	alpha = alpha * x_mask.eq(0).float()
	alpha = alpha / alpha.sum(1).expand(alpha.size())
	return alpha

def weighted_avg(x, weights):
	"""x = batch * len * d
	weights = batch * len
	"""
	return weights.unsqueeze(1).bmm(x).squeeze(1)

def test_lengths():
	x = Variable(torch.randn(3,5,8))
	lens = Variable(torch.randn(3))
	alpha = Variable(torch.zeros(x.size(0), x.size(1)))
	for i in range(alpha.size(0)):
		for j in range(lens[i].data):
			alpha[i][j] = 1

	print('a: ', alpha)

	alpha = alpha / alpha.sum(1).expand(alpha.size(1), alpha.size(0))
	print('alpha: ', alpha.size())


def test_weights():
	"""Return uniform weights over non-masked x (a sequence of vectors).
	Args:
		x: batch * len * hdim
		x_mask: batch * len (1 for padding, 0 for true)
	Output:
		x_avg: batch * hdim
	"""
	x = Variable(torch.randn(3,5,8))
	x_mask = Variable(torch.zeros(3,5))

	alpha = Variable(torch.ones(x.size(0), x.size(1)))
	alpha = alpha * x_mask.eq(0).float()

	print('a: ', alpha)

	alpha = alpha / alpha.sum(1).expand(alpha.size(1), alpha.size(0))
	print('alpha: ', alpha.size())

	y = alpha.unsqueeze(1).bmm(x).squeeze(1)
	print('y: ', y.size())


def test_h5py():
	file = h5py.File('test.h5', 'w')
	data = [1,2]
	file.create_dataset('look', data = data, maxshape = (None, ))
	print(file['look'][:])



def test():
	ret = []
	train_path = '../data/preprocessed/devset/search.dev.json'
	#train_path = '../data/preprocessed/testset/zhidao.test.json'
	i = 0
	with open(train_path) as f:
		for line in tqdm(f):
			line = json.loads(line)
			documents = line['documents']
			document = ''
			tmp = 0 
			j = 0 
			question = line['question']
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
				print(len(para_infos))
				fake_passage_tokens = para_infos[0][0]
				print('fake: ', ''.join(fake_passage_tokens))
				continue

				is_selected = d['is_selected']
				title = d['title']
				most_related_para = d['most_related_para']
				paragraphs = d['segmented_paragraphs']

			question = line['question']
			question_type = line['question_type']

			if len(line['answer_docs']) == 0 or len(line['fake_answers']) == 0:
				continue 

			answer_docs = line['answer_docs'][0]
			answer_span = line['answer_spans'][0]
			fake_answer = line['fake_answers'][0]


			#if len(line['answer_docs']) != 1:
			#	print(len(line['answer_docs']),' ', len(line['answer_spans']),' ', len(line['fake_answers']))
			#continue

			document = documents[answer_docs]
			#paragraph = document['paragraphs'][document['most_related_para']]
			segmented_paragraph = document['segmented_paragraphs'][document['most_related_para']]
			paragraph = ''.join(segmented_paragraph)
			'''
			if fake_answer !=  ''.join(segmented_paragraph[answer_span[0]: answer_span[1]+1]):
				print(fake_answer)
				print(''.join(segmented_paragraph[answer_span[0]: answer_span[1]+1]))
				print()
			'''
			answer_start = len(''.join(segmented_paragraph[:answer_span[0]])) 
			answer_end = len(''.join(segmented_paragraph[:answer_span[1]+1])) 

			if paragraph != ''.join(segmented_paragraph):
				print(paragraph)
				print(''.join(segmented_paragraph))
				print()

			#continue
			if fake_answer != ''.join(paragraph[answer_start: answer_end]):
				print(fake_answer)
				print(''.join(paragraph[answer_start: answer_end]))
				print()

			#print(fake_answer)
			#print()

			i = i+1
			if i == 5:
				break

if __name__ == '__main__':
	test_lengths()