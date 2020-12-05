import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from ptb import PTB
from utils import to_var, expierment_name
from model import SentenceVAE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    t1 = time.time() #starting time

    splits = ['train'] + (['valid'] if args.valid else []) + (['test'] if args.test else [])

    datasets = OrderedDict()


    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ

        )

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )
    model = SentenceVAE(**params)

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)
    model_save_folder = "Saved_models"
    dump_folder = "dump"

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    #save_model_path = os.path.join(args.save_model_path, ts)
    save_model_path = args.save_model_path + "/" + model_save_folder

    #os.makedirs(save_model_path)
    #os.mkdir(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')
    NLL_full = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='none')
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp[:, :torch.max(length).item(),:].contiguous().view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target.type(torch.long))

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight
    
    def loss_fn_full(logp, target, length, mean, logv):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp[:, :torch.max(length).item(),:].contiguous().view(-1, logp.size(2))
        # shape (bs*seq_len*vocab)
        # Negative Log Likelihood
        NLL_loss = NLL_full(logp, target.type(torch.long))
        NLL_loss = NLL_loss.view(-1,torch.max(length).item())

        NLL_loss = torch.sum(NLL_loss,1)/length

        KL_loss = -0.5 * torch.sum((1 + logv - mean.pow(2) - logv.exp()),1)

        return NLL_loss, KL_loss

    def roccurve(val,test,N=50,lin=True):
        val=np.column_stack((val,np.zeros(len(val))))
        test=np.column_stack((test,np.ones(len(test))))
        total=np.append(val,test,axis=0)
        total=total[total[:,0].argsort()]
        nval=np.shape(val)[0]
        ntot=np.shape(total)[0]
        ntest=ntot-nval
        itest=0
        ival=0
        j=0
        testar=[]
        valar=[]

        for i in range(ntot):

            if total[ntot-i-1,1]==1:
                itest+=1
                if itest/ntest>j/N:
                    j=j+1
                    testar+=[itest]
                    valar+=[ival]
                if itest==ntest:
                    break

            else:
                ival+=1

        testar+=[itest]
        valar+=[ival]

        testar=np.array(testar)/ntest
        valar=np.array(valar)/nval
    
        # print("score:",score,"score2:",sum(valar)*(testar[1]-testar[0]))
        # print("10% true positive gives:",valar[int(N*0.1)], " false positive rate,\n50% true positive gives:",valar[int(N/2)],
        #       " false positive rate,\n 90% true positive gives:", valar[int(N*0.9)], " false positive rate")
        print("10 %% true positive gives: %4.2f %% false positive rate,\n50 %% true positive gives: %4.2f %% false positive rate,\n90 %% true positive rate gives: %4.2f %% false positive rate" % (valar[int(N*0.1)]*100,valar[int(N*0.5)]*100,valar[int(N*0.9)]*100))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    #tensor = torch.cuda.FloatTensor if False else torch.Tensor
    step = 0
    
    lossar_train = np.empty((0,5), float)
    lossar_validation = np.zeros((0,5), float)
    lossar_validation_acum = np.zeros((0,7), float)
    lossar_test = np.zeros((0,5), float)
    lossar_test_acum = np.zeros((0,7), float)
    
    for epoch in range(args.epochs):
        
        NLL_full_val = np.zeros((0,1), float)
        NLL_full_test = np.zeros((0,1), float)
        loss_full_val = np.zeros((0,1), float)
        loss_full_test = np.zeros((0,1), float)

        for split in splits:
            print("\n----- " + split + ", epoch " + str(epoch) + " --------------")
            
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['length'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                loss_norm = loss/torch.sum(batch['length'])*batch_size
                NLL_loss_norm = NLL_loss/torch.sum(batch['length'])*batch_size
                KL_loss_norm = KL_loss.detach().clone()

                
                if not model.training:
                    NLL_lossfull, KL_lossfull = loss_fn_full(logp, batch['target'], batch['length'], mean, logv)
                    if args.plt_total_loss == True:
                        lossfull = (NLL_lossfull + KL_weight * KL_lossfull)
                    
                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    lossar_train = np.append(lossar_train,np.array([[len(lossar_train),loss_norm.item(),NLL_loss_norm.item()/batch_size,KL_loss_norm.item()/batch_size,batch_size]]),axis=0)


                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, time passed %6.1f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                          KL_loss.item()/batch_size, KL_weight, time.time()-t1))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    
                    lossar_validation = np.append(lossar_validation,np.array([[epoch,loss_norm.item(),NLL_loss_norm.item()/batch_size,KL_loss_norm.item()/batch_size,batch_size]]),axis=0)                
                    NLL_full_val = np.append(NLL_full_val,NLL_lossfull.cpu().detach().numpy())
                    if args.plt_total_loss == True:
                        loss_full_val = np.append(loss_full_val,lossfull.cpu().detach().numpy())
                    
                if split == 'test':
                    lossar_test = np.append(lossar_test,np.array([[epoch,loss_norm.item(),NLL_loss_norm.item()/batch_size,KL_loss_norm.item()/batch_size,batch_size]]),axis=0)
                    NLL_full_test = np.append(NLL_full_test,NLL_lossfull.cpu().detach().numpy())
                    if args.plt_total_loss == True:
                        loss_full_test = np.append(loss_full_test,lossfull.cpu().detach().numpy())
                    
                    
            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', dump_folder)):
                    os.makedirs('dumps/'+dump_folder)
                with open(os.path.join('dumps/'+dump_folder+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)
                    
                current=lossar_validation[:,0]==epoch #Epoch,loss,NLL, KL, bs
                nevents=sum(lossar_validation[current,-1])

                loss_mu = np.sum(lossar_validation[current,1:4]*lossar_validation[current,-1][:,None],0)/nevents
                loss_sigma = np.sqrt(sum((lossar_validation[current,-1][:,None]*(lossar_validation[current,1:4]-loss_mu)**2)))/nevents #Uncertainty on mean
                print("Mean of total loss: %4.5f +- %2.5f" % (loss_mu[0], loss_sigma[0])) #Sigma here is uncertainty on mean not std! Multiply by sqrt(nevents) to get std
                print("Mean of NLL loss: %4.5f +- %2.5f" % (loss_mu[1], loss_sigma[1]))
                print("Mean of KL loss: %4.5f +- %2.5f \n" % (loss_mu[2], loss_sigma[2]))

                lossar_validation_acum = np.append(lossar_validation_acum,np.array([[lossar_train[-1,0],loss_mu[0],loss_mu[1],loss_mu[2],
                                                                                     loss_sigma[0],loss_sigma[1],loss_sigma[2]]]),axis=0)           
                    
            if split == 'test':
                current=lossar_test[:,0]==epoch #Epoch,loss,NLL, KL, bs
                nevents=sum(lossar_test[current,-1])

                loss_mu = np.sum(lossar_test[current,1:4]*lossar_test[current,-1][:,None],0)/nevents
                loss_sigma = np.sqrt(sum((lossar_test[current,-1][:,None]*(lossar_test[current,1:4]-loss_mu)**2)))/nevents #Uncertainty on mean
                print("Mean of total loss: %4.5f +- %2.5f" % (loss_mu[0], loss_sigma[0])) #Sigma here is uncertainty on mean not std! Multiply by sqrt(nevents) to get std
                print("Mean of NLL loss: %4.5f +- %2.5f" % (loss_mu[1], loss_sigma[1]))
                print("Mean of KL loss: %4.5f +- %2.5f \n" % (loss_mu[2], loss_sigma[2]))

                lossar_test_acum = np.append(lossar_test_acum,np.array([[lossar_train[-1,0],loss_mu[0],loss_mu[1],loss_mu[2],
                                                                                     loss_sigma[0],loss_sigma[1],loss_sigma[2]]]),axis=0)
                
                NLL_full_val=NLL_full_val[:len(NLL_full_test)]
                NLL_full_test=NLL_full_test[:len(NLL_full_val)]

                roccurve(NLL_full_val,NLL_full_test,N=100,lin=True)
                
                if args.plt_total_loss == True:
                    loss_full_val=loss_full_val[:len(loss_full_test)]
                    loss_full_test=loss_full_test[:len(loss_full_val)]
                    roccurve(loss_full_val,loss_full_test,N=100,lin=True)

            # save checkpoint
            if split == 'train':
                nevents = 0 #I made the formulars to take into account that last batch has different size
                loss_mu = np.array([0,0,0],dtype=float) #And I use last 200 of train (we can change this number) as the model used on the 
                i=0 # first batch is different from the next and so on.
                while nevents<=200:
                    i+=1
                    loss_mu += lossar_train[-i,1:4]*lossar_train[-i,-1]
                    nevents += lossar_train[-i,-1]
                    if i==len(lossar_train):
                        break
                loss_mu = loss_mu/nevents
                # loss_sigma = np.sqrt(sum((lossar[-i:,2]*(lossar[-i:,1]-loss_mu)**2))/nevents)
                loss_sigma = np.sqrt(sum((lossar_train[-i:,-1][:,None]*(lossar_train[-i:,1:4]-loss_mu)**2)))/nevents
                print("Mean loss and uncertainty of 200 last: %4.5f +- %2.5f" % (loss_mu[0], loss_sigma[0])) #Sigma here is uncertainty on mean not std! Multiply by sqrt(nevents) to get std
                print("Mean NLL loss of 200 last: %4.5f +- %2.5f" % (loss_mu[1], loss_sigma[1]))
                print("Mean KL loss of 200 last: %4.5f +- %2.5f" % (loss_mu[2], loss_sigma[2]))

                '''
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s \n" % checkpoint_path)
                '''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=30)
    parser.add_argument('--min_occ', type=int, default=1) # It's not been used
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=12)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    
    # For BERT pre-trained model hyperparameters check: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    parser.add_argument('-vs', '--vocab_size', type=int, default=30522) # Cannot be changed for the moment
    parser.add_argument('-eb', '--embedding_size', type=int, default=768) # Cannot be changed for the moment
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    #parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    
    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=4000)
    
    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')
    parser.add_argument('-plt_l', '--plt_total_loss', type=bool, default='False')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
