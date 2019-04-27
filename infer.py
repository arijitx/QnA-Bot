import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,BertTokenizer,whitespace_tokenize)

if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle

from infer_utils import *
import spacy
nlp = spacy.load('en_core_web_md')

def is_whitespace(c):
		if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
			return True
		return False

def is_punc(c):
	if c in '?,.!()[]-_\'"':
		return True
	return False

def punc_sep(s):
	tokens = []
	is_prev_white = True
	for c in s:
		if is_whitespace(c):
			is_prev_white = True
		else:	
			if is_punc(c):
				tokens.append(c)
				is_prev_white = True
			else:
				if is_prev_white:
					is_prev_white = False
					tokens.append(c)
				else:
					tokens[-1]+=c
	return ' '.join(tokens)

def str_to_coqa_example(contenxt, question, prev_ques, prev_answ):
	paragraph_text = contenxt
	doc_tokens = []
	char_to_word_offset = []
	prev_is_whitespace = True
	for c in paragraph_text:
		if is_whitespace(c):
			prev_is_whitespace = True
		else:
			if prev_is_whitespace:
				doc_tokens.append(c)
				prev_is_whitespace = False
			else:
				doc_tokens[-1] += c

		char_to_word_offset.append(len(doc_tokens) - 1)

	question_text = question

	example = CoQAExample(
				qas_id='random',
				question_text=question_text,
				doc_tokens=doc_tokens,
				orig_answer_text="",
				start_position=0,
				end_position=0,
				is_impossible=False,
				is_yes= False,
				is_no=False,
				answer_span="",
				prev_ques=prev_ques,
				prev_answ=prev_answ)
	return example

class InferCoQA():
	def __init__(self, model_path, lower_case = True):
		self.model_path = model_path
		self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=lower_case)
		self.model = BertForQuestionAnswering.from_pretrained(model_path)
		self.model.cuda()
		self.model.eval()

	def predict(self, contenxt, question, prev_ques, prev_answ):
		t = time.time()
		coqa_example = str_to_coqa_example(contenxt, question, prev_ques, prev_answ)
		coqa_features = convert_examples_to_features([coqa_example], self.tokenizer, max_seq_length=512,doc_stride=128, max_query_length=100, is_training=False)
		
		all_input_ids = torch.tensor([f.input_ids for f in coqa_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in coqa_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in coqa_features], dtype=torch.long)
		all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
		coqa_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
		
		coqa_sampler = SequentialSampler(coqa_data)
		coqa_dataloader = DataLoader(coqa_data, sampler=coqa_sampler, batch_size=1)
		all_results = []
		for input_ids, input_mask, segment_ids, example_indices in coqa_dataloader:
			input_ids = input_ids.cuda()
			input_mask = input_mask.cuda()
			segment_ids = segment_ids.cuda()

			
			with torch.no_grad():
				score = self.model(input_ids, segment_ids, input_mask)

			coqa_feature = coqa_features[example_indices[0].item()]
			unique_id = int(coqa_feature.unique_id)
			all_results.append(RawResult(unique_id=unique_id,score=score[0].cpu(),length=input_ids.size(1)))

		output_prediction_file = "predictions.json"
		output_nbest_file = "nbest_predictions.json"
		output_null_log_odds_file = "null_odds.json"
		write_predictions([coqa_example], coqa_features, all_results,
					1, 100,
					True, output_prediction_file,
					output_nbest_file, output_null_log_odds_file, False,
					False, 0.0)
		os.remove(output_nbest_file)
		res = json.loads(open(output_prediction_file).read())['random']
		os.remove(output_prediction_file)
		print('inference time :',time.time() - t )
		return res

# iq = InferCoQA('coqa_ynu_history_1')
# print('done loading model ..')
# context = input("Context : ")


# prev_q = ""
# prev_a = ""
# while True:
# 	q = input("Question : ")
# 	a = iq.predict(context,q,prev_q,prev_a)
# 	print("Answer :",a)
# 	prev_q = q
# 	prev_a = a


