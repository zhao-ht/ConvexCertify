import torch as t
import torch.nn as nn
import torch.nn.functional as F


class emb_to_vocab(nn.Module):

    def __init__(self, W,normalize=True):
        super(emb_to_vocab, self).__init__()
        if normalize:
            self.W=t.nn.parameter.Parameter(F.normalize(W,p=2,dim=0),requires_grad =False)
        else:
            self.W = W

    def forward(self,X):
        return X@self.W


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_variable_size, rnn_size, rnn_num_layers, embed_size,W,args):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.latent_variable_size = latent_variable_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size
        self.rnn_num_layers = rnn_num_layers
        self.args=args
        self.out_length=args.recovery_length if args.inconsistent_recorvery_length else args.max_len
        if self.out_length == 10:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, self.embed_size, 4, 2, 0),
                # nn.BatchNorm1d(self.embed_size),
                nn.ELU(),

                nn.ConvTranspose1d(self.embed_size, self.embed_size, 4, 2, 0)
            )

        elif self.out_length==51:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, 4096, 4, 2, 0),
                nn.BatchNorm1d(4096),
                nn.ELU(),

                nn.ConvTranspose1d(4096, 2048, 4, 2, 0, 1),
                nn.BatchNorm1d(2048),
                nn.ELU(),

                nn.ConvTranspose1d(2048, 1024, 4, 2, 0),
                nn.BatchNorm1d(1024),
                nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                nn.ConvTranspose1d(1024, self.embed_size, 4, 2, 0)
            )
        elif self.out_length == 254:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, 4096, 4, 2, 0,1),
                nn.BatchNorm1d(4096),
                nn.ELU(),

                nn.ConvTranspose1d(4096, 2048, 4, 2, 0,1),
                nn.BatchNorm1d(2048),
                nn.ELU(),

                nn.ConvTranspose1d(2048, 1024, 4, 2, 0,1),
                nn.BatchNorm1d(1024),
                nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                nn.ConvTranspose1d(1024, 1024, 4, 2, 0,1),
                nn.BatchNorm1d(1024),
                nn.ELU(),
                nn.ConvTranspose1d(1024, 1024, 4, 2, 0,1),
                nn.BatchNorm1d(1024),
                nn.ELU(),
                nn.ConvTranspose1d(1024, self.embed_size, 4, 2, 0,1),
            )
        elif self.out_length == 191:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, 4096, 4, 2, 0),
                nn.BatchNorm1d(4096),
                nn.ELU(),

                nn.ConvTranspose1d(4096, 2048, 4, 2, 0),
                nn.BatchNorm1d(2048),
                nn.ELU(),

                nn.ConvTranspose1d(2048, 1024, 4, 2, 0),
                nn.BatchNorm1d(1024),
                nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                nn.BatchNorm1d(1024),
                nn.ELU(),
                nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                nn.BatchNorm1d(1024),
                nn.ELU(),
                nn.ConvTranspose1d(1024, self.embed_size, 4, 2, 0),
            )
        elif self.out_length == 257:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, self.latent_variable_size, 4, 4,0),
                nn.BatchNorm1d(self.latent_variable_size),
                nn.ELU(),

                nn.ConvTranspose1d(self.latent_variable_size, self.latent_variable_size, 4, 4, 0),
                nn.BatchNorm1d(self.latent_variable_size),
                nn.ELU(),

                nn.ConvTranspose1d(self.latent_variable_size, 2*self.latent_variable_size, 4, 4, 0),
                nn.BatchNorm1d(2*self.latent_variable_size),
                nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                nn.ConvTranspose1d(2*self.latent_variable_size, self.embed_size, 4, 4, 0),
            )
        elif self.out_length == 513:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, self.latent_variable_size, 4, 4, 0),
                # nn.BatchNorm1d(self.latent_variable_size),
                nn.ELU(),

                nn.ConvTranspose1d(self.latent_variable_size, self.latent_variable_size, 4, 4, 0),
                # nn.BatchNorm1d(self.latent_variable_size),
                nn.ELU(),

                nn.ConvTranspose1d(self.latent_variable_size, 2 * self.latent_variable_size, 4, 4, 0),
                # nn.BatchNorm1d(2 * self.latent_variable_size),
                nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
                # nn.BatchNorm1d(1024),
                # nn.ELU(),

                nn.ConvTranspose1d(2 * self.latent_variable_size, 2 * self.latent_variable_size, 4, 4, 0),
                nn.ELU(),

                nn.ConvTranspose1d(2 * self.latent_variable_size, self.embed_size, 2, 2, 0),
            )
        # self.cnn = nn.Sequential(
        #     nn.ConvTranspose1d(self.latent_variable_size, 200, 4, 2, 0),
        #     nn.BatchNorm1d(200),
        #     nn.ELU(),
        #
        #     nn.ConvTranspose1d(200, 400, 4, 2, 0, output_padding=1),
        #     nn.BatchNorm1d(400),
        #     nn.ELU(),
        #
        #     nn.ConvTranspose1d(400, 800, 4, 2, 0),
        #     nn.BatchNorm1d(800),
        #     nn.ELU(),
        #
        #     # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.ELU(),
        #
        #     # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
        #     # nn.BatchNorm1d(1024),
        #     # nn.ELU(),
        #
        #     nn.ConvTranspose1d(800, self.embed_size, 4, 2, 0)
        # )

        # if self.args.reconstruct_by_RNN:
        self.rnn = nn.GRU(input_size=self.embed_size + self.embed_size,
                          hidden_size=self.rnn_size,
                          num_layers=self.rnn_num_layers,
                          batch_first=True)
        self.latent_to_hidden = nn.Sequential(t.nn.ReLU(), nn.Linear(self.latent_variable_size, self.rnn_size))
        self.hidden_to_emb = nn.Sequential(t.nn.ReLU(),nn.Linear(self.rnn_size, self.embed_size))

        self.emb_to_emb = nn.Sequential(t.nn.ReLU(),nn.Linear(self.embed_size, self.embed_size))
        self.emb_to_vocab = nn.Linear(self.embed_size, self.vocab_size,bias=False)
        self.emb_to_vocab.weight=t.nn.parameter.Parameter(F.normalize(W, p=2, dim=0), requires_grad=args.to_vocab_trainable)

        self.hidden_to_label=nn.Sequential(t.nn.ReLU(),
                                           nn.Linear(self.latent_variable_size, self.latent_variable_size),
                                           t.nn.ReLU(),
                                           nn.Linear(self.latent_variable_size,self.args.num_classes))


    def forward(self, latent_variable, decoder_input):
        """
        :param latent_variable: An float tensor with shape of [batch_size, latent_variable_size]
        :param decoder_input: An float tensot with shape of [batch_size, max_seq_len, embed_size]
        :return: two tensors with shape of [batch_size, max_seq_len, vocab_size]
                    for estimating likelihood for whole model and for auxiliary target respectively
        """
        tem = t.tensor(0.0).to(latent_variable.device)
        if self.args.classify_on_hidden:
            clas = self.hidden_to_label(latent_variable)
        else:
            clas=tem.repeat(latent_variable.shape[0],self.args.num_classes)
        cnn_out = self.conv_decoder(latent_variable)
        if self.args.info_loss=='reconstruct':
            emb2_tem = cnn_out.contiguous().view(-1, self.embed_size)
            emb2_tem = self.emb_to_emb(emb2_tem)
            logits2 = self.emb_to_vocab(emb2_tem)
            logits2 = logits2.view(cnn_out.shape[0], cnn_out.shape[1], self.vocab_size)
            emb2 = cnn_out
        else:
            emb2=cnn_out
            logits2=tem
        if self.args.reconstruct_by_RNN:
            init=self.latent_to_hidden(latent_variable).unsqueeze(0)
            logits1, emb1 = self.rnn_decoder(cnn_out, decoder_input,initial_state=init)
        else:
            logits1=tem
            emb1=tem

        return logits1,logits2,emb1,emb2,clas


    def conv_decoder(self, latent_variable):
        latent_variable = latent_variable.unsqueeze(2)

        out = self.cnn(latent_variable)
        return t.transpose(out, 1, 2).contiguous()

    def rnn_decoder(self, cnn_out, decoder_input, initial_state=None,return_final=False):
        hidden, final_state = self.rnn(t.cat([cnn_out, decoder_input], 2), initial_state)

        [batch_size, seq_len, _] = hidden.size()
        hidden = hidden.contiguous().view(-1, self.rnn_size)

        emb1 = self.hidden_to_emb(hidden)
        emb1 = self.emb_to_emb(emb1)
        logits1 = self.emb_to_vocab(emb1)

        logits1 = logits1.view(batch_size, seq_len, self.vocab_size)

        if not return_final:

            return logits1, emb1.view(batch_size, seq_len, self.embed_size)
        else:
            return logits1,final_state





class Decoder_RNN(nn.Module):
    def __init__(self, vocab_size, latent_variable_size, rnn_size, rnn_num_layers, embed_size,W,args):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.latent_variable_size = latent_variable_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size
        self.rnn_num_layers = rnn_num_layers

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(self.latent_variable_size, 4096, 4, 2, 0),
            nn.BatchNorm1d(4096),
            nn.ELU(),

            nn.ConvTranspose1d(4096, 2048, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(2048),
            nn.ELU(),

            nn.ConvTranspose1d(2048, 1024, 4, 2, 0),
            nn.BatchNorm1d(1024),
            nn.ELU(),

            # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
            # nn.BatchNorm1d(1024),
            # nn.ELU(),

            # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
            # nn.BatchNorm1d(1024),
            # nn.ELU(),

            nn.ConvTranspose1d(1024, self.embed_size, 4, 2, 0)
        )

        # self.cnn = nn.Sequential(
        #     nn.ConvTranspose1d(self.latent_variable_size, 200, 4, 2, 0),
        #     nn.BatchNorm1d(200),
        #     nn.ELU(),
        #
        #     nn.ConvTranspose1d(200, 400, 4, 2, 0, output_padding=1),
        #     nn.BatchNorm1d(400),
        #     nn.ELU(),
        #
        #     nn.ConvTranspose1d(400, 800, 4, 2, 0),
        #     nn.BatchNorm1d(800),
        #     nn.ELU(),
        #
        #     # nn.ConvTranspose1d(1024, 1024, 4, 2, 0, output_padding=1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.ELU(),
        #
        #     # nn.ConvTranspose1d(1024, 1024, 4, 2, 0),
        #     # nn.BatchNorm1d(1024),
        #     # nn.ELU(),
        #
        #     nn.ConvTranspose1d(800, self.embed_size, 4, 2, 0)
        # )


        self.rnn = nn.GRU(input_size=self.embed_size + self.embed_size,
                          hidden_size=self.rnn_size,
                          num_layers=self.rnn_num_layers,
                          batch_first=True)

        self.hidden_to_emb = nn.Sequential(t.nn.ReLU(),nn.Linear(self.rnn_size, self.embed_size))
        self.latent_to_hidden = nn.Sequential(t.nn.ReLU(),nn.Linear(self.latent_variable_size, self.rnn_size))
        self.emb_to_emb = nn.Sequential(t.nn.ReLU(),nn.Linear(self.embed_size, self.embed_size))
        self.emb_to_vocab = nn.Linear(self.embed_size, self.vocab_size,bias=False)
        self.emb_to_vocab.weight=t.nn.parameter.Parameter(F.normalize(W, p=2, dim=0), requires_grad=args.to_vocab_trainable)

        self.hidden_to_label=nn.Sequential(t.nn.ReLU(),
                                           nn.Linear(self.latent_variable_size, self.latent_variable_size),
                                           t.nn.ReLU(),
                                           nn.Linear(self.latent_variable_size,2))


    def forward(self, latent_variable, decoder_input):
        """
        :param latent_variable: An float tensor with shape of [batch_size, latent_variable_size]
        :param decoder_input: An float tensot with shape of [batch_size, max_seq_len, embed_size]
        :return: two tensors with shape of [batch_size, max_seq_len, vocab_size]
                    for estimating likelihood for whole model and for auxiliary target respectively
        """
        clas=self.hidden_to_label(latent_variable)
        cnn_out = self.conv_decoder(latent_variable)
        init=self.latent_to_hidden(latent_variable).unsqueeze(0)
        logits1, logits2,emb1,emb2 = self.rnn_decoder(cnn_out, decoder_input,initial_state=init)

        return logits1,logits2,emb1,emb2,clas

    def conv_decoder(self, latent_variable):
        latent_variable = latent_variable.unsqueeze(2)

        out = self.cnn(latent_variable)
        return t.transpose(out, 1, 2).contiguous()

    def rnn_decoder(self, cnn_out, decoder_input, initial_state=None,return_final=False):
        hidden, final_state = self.rnn(t.cat([cnn_out, decoder_input], 2), initial_state)

        [batch_size, seq_len, _] = hidden.size()
        hidden = hidden.contiguous().view(-1, self.rnn_size)

        emb1 = self.hidden_to_emb(hidden)
        emb1 = self.emb_to_emb(emb1)
        logits1 = self.emb_to_vocab(emb1)

        logits1 = logits1.view(batch_size, seq_len, self.vocab_size)

        if not return_final:
            emb2=cnn_out.contiguous().view(-1, self.embed_size)
            emb2 = self.emb_to_emb(emb2)
            logits2=self.emb_to_vocab(emb2)
            logits2 = logits2.view(batch_size, seq_len, self.vocab_size)
            return logits1, logits2,emb1.view(batch_size, seq_len, self.embed_size),emb2.view(batch_size, seq_len, self.embed_size)
        else:
            return logits1,final_state