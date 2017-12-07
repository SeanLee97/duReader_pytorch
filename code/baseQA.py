import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class baseQA(nn.Module):
	
	def __init__(self, param):
		super(baseQA, self).__init__()
		self.vocab_size = param.vocab_size
		self.embedding_size = param.embedding_size   
		self.question_size = param.question_size
		self.paragraph_size = param.paragraph_size

		self.question_hidden_size = 48
		self.paragraph_hidden_size = 64

		self.question_num_layers = 1
		self.paragraph_num_layers = 1
		
		self.lookup = nn.Embedding(self.vocab_size, self.embedding_size)

		'''
		if param.pre_embeds == True :
			self.lookup.weight.data.copy_(torch.from_numpy(embeds))
			for param in self.lookup.parameters():
				param.requires_grad = False
		'''

		self.paragraph_input_size = self.embedding_size + self.question_hidden_size 
		self.question_lstm = nn.LSTM(self.embedding_size, self.question_hidden_size, self.question_num_layers, dropout = 0.1)
		self.paragraph_lstm = nn.LSTM(self.paragraph_input_size, self.paragraph_hidden_size // 2, self.paragraph_num_layers, dropout = 0.2, bidirectional = True)
		#self.match_lstm = nn.LSTM(self.e_hidden_size, self.t_hidden_size, self.num_layers)
		
		self.att_linear = nn.Linear(self.question_hidden_size, 1)
		self.start_net = nn.Linear(self.paragraph_hidden_size, self.paragraph_size)
		self.end_net = nn.Linear(self.paragraph_hidden_size, self.paragraph_size)

		#self.weight = torch.FloatTensor([1.4, 1.4, 0.8, 0.8, 0.8]).cuda()
		#self.loss_func = nn.NLLLoss(weight = self.weight)
		
		self.loss_func = nn.NLLLoss()
	
	
	def init_hidden(self, num_layers, batch_size, hidden_size):
		h0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
		c0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
		if torch.cuda.is_available() == True:
			h0, c0 = h0.cuad(), c0.cuda()
		return (h0, c0)
	
	
	# x = (batch, seq_len, hsize)
	# return (batch, hidden_size)
	def attention(self, x):
		x_flat = x.view(-1, x.size(-1))
		scores = self.att_linear(x_flat).view(x.size(0), x.size(1))
		weights = F.softmax(scores)
		out = weights.unsqueeze(1).bmm(x).squeeze(1)
		return out
	
	
	# return pack rnn inputs
	def get_pack_rnn_inputs(self, x, lengths):
		_, idx_sort = torch.sort(lengths, dim = 0, descending = True)
		_, idx_unsort = torch.sort(idx_sort, dim = 0)

		lengths = list(lengths[idx_sort])

		# sort x
		x = x.index_select(0, Variable(idx_sort))
		if torch.cuda.is_available() == True:
			x = x.cuda()
		x = x.transpose(0, 1).contiguous()
		rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

		unsort = Variable(idx_unsort)
		if torch.cuda.is_available() == True:
			unsort = unsort.cuda()
		
		return rnn_input, unsort

	
	def get_pad_rnn_outputs(self, output, seq_len, idx_unsort):
		output = nn.utils.rnn.pad_packed_sequence(output)[0]
		
		# transpose and unsort
		output = output.transpose(0, 1).contiguous()
		output = output.index_select(0, idx_unsort)

		# pad up to original batch sequence length
		if output.size(1) != seq_len:
			padding = torch.zeros(output.size(0),
								  seq_len - output.size(1),
								  output.size(2)).type(output.data.type())
			output = torch.cat([output, Variable(padding)], 1)
		
		return output
	
	
	# embeds = (batch, seq_len, embedding_size)
	# return (batch, q_size)
	def get_question_lstm(self, question, question_length):
		batch_size = question.size()[0]
		embeds = self.lookup(question)
		inputs, idx_unsort = self.get_pack_rnn_inputs(embeds, question_length)
		
		init_hidden = self.init_hidden(self.question_num_layers, batch_size, self.question_hidden_size)
		lstm_out, _ = self.question_lstm(inputs, init_hidden)
		lstm_out = self.get_pad_rnn_outputs(lstm_out, self.question_size, idx_unsort)
		#print('q lstm: ', lstm_out.size())
		
		lstm_vector = self.attention(lstm_out)
		return lstm_vector
	
	
	# return (batch, paragraph_size, paragraph_hidden_size)
	def get_paragraph_lstm(self, paragraph, paragraph_length, question_vector):
		batch_size = paragraph.size()[0]
		embeds = self.lookup(paragraph)

		question_vectors = question_vector.expand(self.paragraph_size, *question_vector.size()) 
		question_vectors = question_vectors.transpose(0,1).contiguous()

		#print('embeds: ', embeds.size())
		#print('question: ', question_vectors.size())
		inputs = torch.cat([embeds, question_vectors], -1)
		inputs, idx_unsort = self.get_pack_rnn_inputs(inputs, paragraph_length)
		
		init_hidden = self.init_hidden(self.paragraph_num_layers * 2, batch_size, self.paragraph_hidden_size // 2)
		lstm_out, _ = self.paragraph_lstm(inputs, init_hidden)
		
		lstm_out = self.get_pad_rnn_outputs(lstm_out, self.paragraph_size, idx_unsort)
		#print('lstm : ', lstm_out.size())
		lstm_vector = torch.mean(lstm_out, 1)

		return lstm_vector


	# return (batch, seq_len, tag_size)
	def forward(self, question, paragraph, question_length, paragraph_length):
		question_vector = self.get_question_lstm(question, question_length)
		paragraph_vector = self.get_paragraph_lstm(paragraph, paragraph_length, question_vector)  
		print('paragraph: ', paragraph_vector.size())

		start_space = self.start_net(paragraph_vector)
		start_score = F.log_softmax(start_space)
		
		end_space = self.end_net(paragraph_vector)
		end_score = F.log_softmax(end_space)

		return start_score, end_score

	
	# return (batch, seq_len)
	def get_answer(self, question, paragraph, question_length, paragraph_length):
		start_score, end_score = self.forward(question, paragraph, question_length, paragraph_length)
		_, tag = torch.max(score, dim = -1)
		return tag.data.cpu().tolist()
  
	
	# return one value
	def get_loss(self, question, paragraph, answer, question_length, paragraph_length):
		start_score, end_score = self.forward(question, paragraph, question_length, paragraph_length)

		answer = answer.transpose(0, 1)
		start_loss = self.loss_func(start_score, answer[0])
		end_loss = self.loss_func(end_score, answer[1])
		loss = torch.mean(torch.cat([start_loss, end_loss], -1))

		return loss
	
	
	
	


	