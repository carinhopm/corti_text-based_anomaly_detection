#This is modified so that input and target are now the same
import os
import io
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import BertTokenizer # pip install transformers


class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 200)
        #self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'ptb.'+split+'.txt')
        self.data_file = 'ptb.'+split+'.json'
        #self.vocab_file = 'ptb.vocab.json'
        
        # Load pre-trained BERT model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }
    
    # For BERT pre-trained model hyperparameters check: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    @property
    def vocab_size(self):
        return 30522

    @property
    def pad_idx(self):
        return self.tokenizer.convert_tokens_to_ids('[PAD]')

    @property
    def sos_idx(self):
        return self.tokenizer.convert_tokens_to_ids('[CLS]')

    @property
    def eos_idx(self):
        return self.tokenizer.convert_tokens_to_ids('[SEP]')

    @property
    def unk_idx(self):
        return self.tokenizer.convert_tokens_to_ids('[UNK]')
    
    def idx2word(self, idx, pad_idx):
        sent_str = [str()]*len(idx)
        for i, sent in enumerate(idx):
            for word_id in sent:
                if word_id == pad_idx:
                    break
                sent_str[i] += self.tokenizer.convert_ids_to_tokens(word_id) + " "
            sent_str[i] = sent_str[i].strip()
        return sent_str


    def _load_data(self):
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
    
    def _create_data(self):

        data = defaultdict(dict)                        #A dictionary
        with io.open(self.raw_data_path, mode='r', encoding='utf-8') as file:     #Opens datafile

            for i, line in enumerate(file):             #lineindex, line in file

                # Split line into word-tokens with BERT tokenizer
                words = self.tokenizer.tokenize(line)
                
                input = ['[CLS]'] + words               #[CLS] in start
                input = input[:self.max_sequence_length-1] #making so that inputs and targets are missing end and start respectively.
                input = input + ['[SEP]']

                target = input.copy()
                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)                     #defining length of sentence
                
                input.extend(['[PAD]'] * (self.max_sequence_length-length)) #adds padding to end up till max sequence length
                target.extend(['[PAD]'] * (self.max_sequence_length-length))

                input = [self.tokenizer.convert_tokens_to_ids(w) for w in input]   #For each word in input search for word index in BERT vocabulary
                target = [self.tokenizer.convert_tokens_to_ids(w) for w in target] #Same for target.

                id = len(data)                                                #index of the line, could use i
                data[id]['input'] = input                                     #data[line-index]["input"] is input in the number format
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))
        
        self._load_data()
