#This has been modified so that it no longer uses the input sequence in the output
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

from transformers import BertModel


class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, hidden_size2, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        #self.tensor = torch.cuda.FloatTensor if False else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        # For BERT pre-trained model hyperparameters check: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
        self.embedding_model = BertModel.from_pretrained('bert-base-uncased')
        self.embedding_model.eval()
        #self.embedding = nn.Embedding(vocab_size, embedding_size) #given our vocabulary size and a choice of embedding_size
                                                                   #we can make vectors for our vocabulary representing correlations and not assuming orthogonality
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.hidden_factor = (2 if bidirectional else 1) * self.num_layers

        self.encoder_embedding_BN = nn.BatchNorm1d(embedding_size)
        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.encoder_hidden_BN = nn.BatchNorm1d(hidden_size*self.hidden_factor)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.encoder_hid2 = nn.Linear(hidden_size * self.hidden_factor, hidden_size2)
        self.encoder_hid2_BN = nn.BatchNorm1d(hidden_size2)
        self.decoder_hid2 = nn.Linear(hidden_size2, hidden_size * self.hidden_factor)
        self.decoder_hidden_BN = nn.BatchNorm1d(hidden_size)






        self.hidden2mean = nn.Linear(hidden_size2, latent_size)
        self.hidden2logv = nn.Linear(hidden_size2, latent_size)

        self.latent2hidden2 = nn.Linear(latent_size, hidden_size2)
        self.decoder_hid2_BN = nn.BatchNorm1d(hidden_size2)

        self.outputs2vocab = nn.Linear(hidden_size * self.hidden_factor, vocab_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def encoder(self,input_embedding,batch_size,sorted_lengths):
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        if self.rnn_type == "lstm":
            _, (hidden, _) = self.encoder_rnn(packed_input)
        else:
            _, hidden = self.encoder_rnn(packed_input)
        #hidden = self.drop(hidden)
        hidden = self.relu(hidden)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.view(self.batch_size, self.hidden_size)
        hidden = self.encoder_hidden_BN(hidden)

        hidden2 = self.encoder_hid2(hidden)
        #hidden2 = self.encoder_hid2_BN(hidden2)

        mean = self.hidden2mean(hidden2)
        logv = self.hidden2logv(hidden2)

        return mean, logv

    def decoder(self,z,batch_size,sorted_lengths):
        hidden2 = self.latent2hidden(z)
        #hidden = self.relu(hidden)
        hidden = self.decoder_hid2(hidden2)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        #hidden = self.drop(hidden)
        hidden = self.decoder_hidden_BN(hidden.permute(1,2,0)).permute(2,0,1).contiguous()
        hidden = self.relu(hidden)

        decoder_input_sequence = to_var(torch.Tensor(batch_size,self.max_sequence_length).fill_(self.sos_idx).long())

        decoder_input_embedding = self.embedding_model(decoder_input_sequence.to(torch.int64)) #int64! Cudas mortal enemy! Only uses long!
        decoder_input_embedding = self.embedding_dropout(decoder_input_embedding[0])

        packed_input = rnn_utils.pack_padded_sequence(decoder_input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        if self.rnn_type == "lstm":
            outputs, _ = self.decoder_rnn(packed_input, (hidden,torch.zeros_like(hidden)))
        else:
            outputs, _ = self.decoder_rnn(packed_input, hidden)

        return outputs

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        embedding_output = self.embedding_model(input_sequence.to(torch.int64))
        input_embedding = embedding_output[0]
        mean, logv = self.encoder(input_embedding, batch_size, sorted_lengths)

        std = torch.exp(0.5 * logv)

        eps = to_var(torch.randn([batch_size, self.latent_size]))
        z = eps * std + mean #This is creating a number from the distribution, nice

        # DECODER
        outputs = self.decoder(z, batch_size, sorted_lengths)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        padded_outputs = self.drop(padded_outputs)
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.reshape(-1, padded_outputs.size(2))), dim=-1)
        #logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(b, s, self.hidden_size * self.hidden_factor)), dim=-1)
        logp = logp.view(b, s, self.vocab_size) #batch_size,max_sequence_length,vocab_size

        return logp, mean, logv, z

    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            #hidden = hidden.squeeze(0)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
