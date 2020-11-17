#This is modified so that input and target are now the same
import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
#from nltk.tokenize import TweetTokenizer
from transformers import BertTokenizer # pip install transformers

#from utils import OrderedCounter

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
        
        # Always create data (at least as 1st approach)
        self._create_data()

        '''
        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()
        '''


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }
    
    '''
    @property
    def vocab_size(self):
        return len(self.w2i)
    
    # To start a sentence
    @property
    def cls_idx(self):
        return self.w2i['[CLS]']

    # To end a sentence
    @property
    def sep_idx(self):
        return self.w2i['[SEP]']

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
    '''
    
    def _create_data(self):

        data = defaultdict(dict)                        #A dictionary
        with open(self.raw_data_path, 'r') as file:     #Opens datafile

            for i, line in enumerate(file):             #lineindex, line in file

                # Split line into word-tokens with BERT tokenizer
                words = self.tokenizer.tokenize(line)
                
                input = ['[CLS]'] + words               #[CLS] in start
                input = input[:self.max_sequence_length-1] #making so that inputs and targets are missing end and start respectively.
                input = input + ['[SEP]']

                input = words
                target = input.clone()
                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)                     #defining length of sentence
                
                input.extend(['[PAD]'] * (self.max_sequence_length-length)) #adds padding to end up till max sequence length
                target.extend(['[PAD]'] * (self.max_sequence_length-length))

                input = [self.tokenizer.convert_tokens_to_ids(w) for w in input]   #For each word in input search for word index in BERT vocabulary
                target = [self.tokenizer.convert_tokens_to_ids(w) for w in target] #Same for target.

                id = len(data)                                                #index of the line, could use i
                data[id]['input'] = torch.tensor([input])                     #data[line-index]["input"] is input in the number format
                data[id]['target'] = torch.tensor([target])
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

    '''
    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()                        #need vocabulary before we can work
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False) #A splitter of sentences made for tweets

        data = defaultdict(dict)                        #A dictionary
        with open(self.raw_data_path, 'r') as file:     #Opens datafile

            for i, line in enumerate(file):             #lineindex, line in file

                words = tokenizer.tokenize(line)        #split line according to tweettokenizer into words.

                input = ['<sos>'] + words               #<sos> in start
                input = input[:self.max_sequence_length-1] #making so that inputs and targets are missing end and start respectively.
                input = input + ['<eos>']

                target = input.clone()

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)                     #defining length of sentence

                input.extend(['<pad>'] * (self.max_sequence_length-length)) #adds padding to end up till max sequence length
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]   #For each word in input search for word index in vocabulary or return 1 (for unknown) instead
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target] #Same for target.

                id = len(data)                                                #index of the line, could use i
                data[id]['input'] = input                                     #data[line-index]["input"] er inputtet i talformatet.
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)         #sentence splitter for tweets

        w2c = OrderedCounter()                                  #All three are dictionary like stuff
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>'] #padding, unknown, start, end
        for st in special_tokens:
            i2w[len(w2i)] = st                                #word of index 0,1,2,3 is special_tokens
            w2i[st] = len(w2i)                                #index of special token is 0,1,2,3

        with open(self.raw_data_path, 'r') as file:           #opening data file

            for i, line in enumerate(file):                  #index of line, line
                words = tokenizer.tokenize(line)             #line split
                w2c.update(words)                            #makes a dictionary of words with counts order by which words it first encountered.

            for w, c in w2c.items():                        #word, count
                if c > self.min_occ and w not in special_tokens: #IF not too few counts
                    i2w[len(w2i)] = w                       #word of index
                    w2i[w] = len(w2i)                       #index of word

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)                    #vocabulary consists of word to index and index to word.
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file: #Safe vocabulary
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()                                  #load it where you saved it.
    '''
