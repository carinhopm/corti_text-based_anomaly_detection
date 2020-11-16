#This has been modified so that it no longer uses the input sequence in the output
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        #self.tensor = torch.cuda.FloatTensor if False else torch.Tensor


        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size) #given our vocabulary size and a choice of embedding_size we can make vectors for our vocabulary representing correlations and not assuming orthogonality.
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()
        self.hidden_factor = (2 if self.bidirectional else 1) * num_layers
        self.encoder_rnn_out_size_full =  self.max_sequence_length*self.hidden_size*self.hidden_factor
        self.decoder_rnn_out_size = int(embedding_size/self.hidden_factor)
        self.decoder_rnn_in_size = hidden_size*self.hidden_factor


        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        #self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
        #                       batch_first=True)

        self.decoder_rnn = rnn(self.decoder_rnn_in_size, self.decoder_rnn_out_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        #self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        #self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2mean = nn.Linear(self.encoder_rnn_out_size_full, latent_size)
        self.hidden2logv = nn.Linear(self.encoder_rnn_out_size_full, latent_size)
        self.latent2hidden = nn.Linear(latent_size, self.encoder_rnn_out_size_full)
        self.outputs2vocab = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence.type(torch.long))   #Translates the words into vectors with correlations.

        mean,logv = encoder(input_embedding,batch_size)


        # REPARAMETERIZATION

        std = torch.exp(0.5 * logv)

        eps = to_var(torch.randn([batch_size, self.latent_size]))
        z = eps * std + mean #This is creating a number from the sitribution, nice

        # DECODER

        outputs = decoder(z,batch_size)

        logp = nn.functional.log_softmax(self.outputs2vocab(outputs), dim=-1)


        return logp, mean, logv, z

    def encoder(self,x,batch_size):
        ##What I have worked with in RNN
        encoder_rnn_out, _ = self.encoder_rnn(input_embedding)
        hidden = encoder_rnn_out
        hidden = torch.reshape(hidden,(batch_size, self.encoder_rnn_out_size_full))
        ##What I have worked with in RNN
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)

        return mean, logv

    def decoder(self,z,batch_size):
        decoder_hidden = self.latent2hidden(z)
        ##What I have worked with in RNN
        decoder_hidden = decoder_hidden.view(batch_size, self.max_sequence_length,self.hidden_size*self.hidden_factor)
        outputs, _ = self.decoder_rnn(decoder_hidden)
        ##What I have worked with in RNN

        return outputs

#I AM CURRENTLY EDITING INFERENCE TO WORK WITH THE NEW ARCHITECTURE.
    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

#        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
#            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

#        hidden = hidden.unsqueeze(0)

        hidden = hidden.view(batch_size, self.max_sequence_length,self.hidden_size*self.hidden_factor)

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

            #output, hidden = self.decoder_rnn(input_embedding, hidden)
            outputs, hidden = self.decoder_rnn(hidden)

            logits = self.outputs2vocab(output) #This is 1 sentence

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
