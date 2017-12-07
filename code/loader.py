import h5py
import math
import torch
import torch.utils.data as data

class loadTrainDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.nb_samples = len(self.file['question'][:])
        print('Dataset: ', self.nb_samples)

    def __getitem__(self, index):
        question = self.file['question'][index]
        paragraph = self.file['paragraph'][index]
        answer = self.file['answer'][index]
        question_length = self.file['question_length'][index]
        paragraph_length = self.file['paragraph_length'][index]
        return question, paragraph, answer, question_length, paragraph_length

    def __len__(self):
        return self.nb_samples

class loadValDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.nb_samples = len(self.file['question'][:])
        print('Dataset: ', self.nb_samples)

    def __getitem__(self, index):
        question_id = self.file['question_id'][index]
        question = self.file['question'][index]
        paragraphs = self.file['paragraphs'][index]
        question_length = self.file['question_length'][index]
        paragraph_lengths = self.file['paragraph_lengths'][index]
        return question, paragraphs, question_length, paragraph_lengths

    def __len__(self):
        return self.nb_samples

class loadTestDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.nb_samples = len(self.file['question'][:])
        print('Dataset: ', self.nb_samples)

    def __getitem__(self, index):
        question_id = self.file['question_id'][index]
        question = self.file['question'][index]
        paragraph = self.file['paragraph'][index]
        question_length = self.file['question_length'][index]
        paragraph_length = self.file['paragraph_length'][index]
        return question_id, question, paragraph, question_length, paragraph_length

    def __len__(self):
        return self.nb_samples
        