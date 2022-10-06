import torch as t
import torch.nn as nn
import torch.nn.functional as F

import model.ibp as ibp


class Encoder(nn.Module):
    def __init__(self, embed_size, latent_size):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.latent_size = latent_size

        self.cnn = nn.Sequential(
            nn.Conv1d(self.embed_size, 1024, 4, 2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # ibp.Conv1d(1024, 1024, 4, 2),
            # # ibp.BatchNorm1d(256),
            # ibp.Activation(F.relu),

            # ibp.Conv1d(1024, 1024, 4, 2),
            # # ibp.BatchNorm1d(256),
            # ibp.Activation(F.relu),

            nn.Conv1d(1024, 2048, 4, 2),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Conv1d(2048, 4096, 4, 2),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Conv1d(4096, self.latent_size, 4, 2),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )

        self.hidden2mean = nn.Linear(self.latent_size, self.latent_size)
        self.hidden2logv = nn.Linear(self.latent_size, self.latent_size)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """

        '''
        Transpose input to the shape of [batch_size, embed_size, seq_len]
        '''





        input = input.permute(0,2,1)

        res = self.cnn(input).squeeze(2)
        mu=self.hidden2mean(res)
        log_sigma=self.hidden2logv(res)

        return mu,log_sigma,None

class Encoder_IBP(nn.Module):
    def __init__(self, embed_size, latent_size):
        super(Encoder_IBP, self).__init__()

        self.embed_size = embed_size
        self.latent_size = latent_size

        self.cnn = nn.Sequential(
            ibp.Conv1d(self.embed_size, 1024, 4, 2),
            # ibp.BatchNorm1d(1024),
            ibp.Activation(F.relu),

            # ibp.Conv1d(1024, 1024, 4, 2),
            # # ibp.BatchNorm1d(256),
            # ibp.Activation(F.relu),

            # ibp.Conv1d(1024, 1024, 4, 2),
            # # ibp.BatchNorm1d(256),
            # ibp.Activation(F.relu),

            ibp.Conv1d(1024, 2048, 4, 2),
            # ibp.BatchNorm1d(2048),
            ibp.Activation(F.relu),

            ibp.Conv1d(2048, 4096, 4, 2),
            # ibp.BatchNorm1d(4096),
            ibp.Activation(F.relu),

            ibp.Conv1d(4096, self.latent_size, 4, 2),
            # ibp.BatchNorm1d(self.latent_size),
            ibp.Activation(F.relu)
        )

        self.hidden2mean = ibp.Linear(self.latent_size, self.latent_size)
        self.hidden2logv = ibp.Linear(self.latent_size, self.latent_size)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """

        '''
        Transpose input to the shape of [batch_size, embed_size, seq_len]
        '''

        x_vecs = input['input_emb']
        mask = input['mask']
        lengths = mask.sum(1)
        x_vecs = (x_vecs* mask.unsqueeze(-1)).permute(0,2,1)

        res = self.cnn(x_vecs)
        res = ibp.sum(res / (lengths.to(dtype=t.float).view(-1, 1, 1)), 2)
        mu=self.hidden2mean(res)
        log_sigma=self.hidden2logv(res)

        return mu,log_sigma,None


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes,
                 dropout):
        super().__init__()

        # self.embedding = nn.Embedding.from_pretrained(
        #     pretrained_embeddings, freeze=False)

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)

        # self.fc = Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.latent_size=len(filter_sizes)*n_filters
        self.hidden2mean = nn.Linear(self.latent_size, self.latent_size)
        self.hidden2logv = nn.Linear(self.latent_size, self.latent_size)
        self.fc = Linear(len(filter_sizes) * n_filters, 2)

    def forward(self, x):
        # text, _ = x
        # text: [sent len, batch size]
        # text = x.permute(1, 0)  # 维度换位,
        text = x.permute(0, 2, 1)
        # text: [batch size, sent len]

        # embedded = self.embedding(text)
        # embedded: [batch size, sent len, emb dim]

        # embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = self.convs(text)

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        res = self.dropout(t.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        # return self.fc(cat)
        # return res

        # mu = self.hidden2mean(res)
        # log_sigma = self.hidden2logv(res)

        return res,self.fc(res)









def attention_pool(x, mask, layer):
  """Attention pooling

  Args:
    x: batch of inputs, shape (B, n, h)
    mask: binary mask, shape (B, n)
    layer: Linear layer mapping h -> 1
  Returns:
    pooled version of x, shape (B, h)
  """
  attn_raw = layer(x).squeeze(2)  # B, n, 1 -> B, n
  attn_raw = ibp.add(attn_raw, (1 - mask) * -1e20 )
  attn_logsoftmax = ibp.log_softmax(attn_raw, 1)
  attn_probs = ibp.activation(t.exp, attn_logsoftmax)  # B, n
  return ibp.bmm(attn_probs.unsqueeze(1), x).squeeze(1)  # B, 1, n x B, n, h -> B, h

class CNNModel(nn.Module):
  """Convolutional neural network.

  Here is the overall architecture:
    1) Rotate word vectors
    2) One convolutional layer
    3) Max/mean pool across all time
    4) Predict with MLP

  """
  def __init__(self,word_vec_size, hidden_size, kernel_size,
               pool='max', dropout=0.2, no_wordvec_layer=False,
               early_ibp=False, relu_wordvec=True, unfreeze_wordvec=False):
    super(CNNModel, self).__init__()
    cnn_padding = (kernel_size - 1) // 2  # preserves size
    self.pool = pool
    # Ablations
    self.no_wordvec_layer = no_wordvec_layer
    self.early_ibp = early_ibp
    self.relu_wordvec = relu_wordvec
    self.unfreeze_wordvec=False
    # End ablations
    # self.embs = ibp.Embedding.from_pretrained(word_mat, freeze=not self.unfreeze_wordvec)
    if no_wordvec_layer:
      self.conv1 = ibp.Conv1d(word_vec_size, hidden_size, kernel_size,
                              padding=cnn_padding)
    else:
      self.linear_input = ibp.Linear(word_vec_size, hidden_size)
      self.conv1 = ibp.Conv1d(hidden_size, hidden_size, kernel_size,
                              padding=cnn_padding)
    if self.pool == 'attn':
      self.attn_pool = ibp.Linear(hidden_size, 1)
    self.dropout = ibp.Dropout(dropout)
    self.fc_hidden = ibp.Linear(hidden_size, hidden_size)
    # self.fc_output = ibp.Linear(hidden_size, 1)

    self.hidden2mean = ibp.Linear(hidden_size, hidden_size)
    self.hidden2logv = ibp.Linear(hidden_size, hidden_size)
    self.BN=ibp.BatchNorm1d(hidden_size)

  def forward(self,batch, compute_bounds=True, cert_eps=1.0):
    """
    Args:
      batch: A batch dict from a TextClassificationDataset with the following keys:
        - x: tensor of word vector indices, size (B, n, 1)
        - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
        - lengths: lengths of sequences, size (B,)
      compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
      cert_eps: Scaling factor for interval bounds of the input
    """
    x_vecs = batch['input_emb']
    # if compute_bounds:
    #   x = batch['x']
    # else:
    #   x = batch['x'].val
    mask = batch['mask']
    # lengths = batch['lengths']
    lengths = mask.sum(1)
    # x_vecs = self.embs(x)  # B, n, d
    # if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
    #   x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
    if not self.no_wordvec_layer:
      x_vecs = self.linear_input(x_vecs)  # B, n, h
    if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
      x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
    if self.no_wordvec_layer or not self.relu_wordvec:
      z = x_vecs
    else:
      z = ibp.activation(F.relu, x_vecs)  # B, n, h
    z_masked = z * mask.unsqueeze(-1)  # B, n, h
    z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
    c1 = ibp.activation(F.relu, self.conv1(z_cnn_in))  # B, h, n
    c1_masked = c1 * mask.unsqueeze(1)  # B, h, n
    if self.pool == 'mean':
      fc_in = ibp.sum(c1_masked / lengths.to(dtype=t.float).view(-1, 1, 1), 2)  # B, h
    elif self.pool == 'attn':
      fc_in = attention_pool(c1_masked.permute(0, 2, 1), mask, self.attn_pool)  # B, h
    else:
      # zero-masking works b/c ReLU guarantees that everything is >= 0
      fc_in = ibp.pool(t.max, c1_masked, 2)  # B, h

    fc_in = self.dropout(fc_in)
    fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in)) # B, h
    fc_hidden = self.dropout(fc_hidden)
    # output = self.fc_output(fc_hidden)  # B, 1
    mu = self.hidden2mean(fc_hidden)
    log_sigma = self.hidden2logv(fc_hidden)

    return mu, log_sigma, None

class CNNModel_bypass(nn.Module):
  """Convolutional neural network.

  Here is the overall architecture:
    1) Rotate word vectors
    2) One convolutional layer
    3) Max/mean pool across all time
    4) Predict with MLP

  """
  def __init__(self,word_vec_size, hidden_size, kernel_size,
               pool='max', dropout=0.2, no_wordvec_layer=False,
               early_ibp=False, relu_wordvec=True, unfreeze_wordvec=False):
    super(CNNModel_bypass, self).__init__()
    cnn_padding = (kernel_size - 1) // 2  # preserves size
    self.pool = pool
    # Ablations
    self.no_wordvec_layer = no_wordvec_layer
    self.early_ibp = early_ibp
    self.relu_wordvec = relu_wordvec
    self.unfreeze_wordvec=False
    # End ablations
    # self.embs = ibp.Embedding.from_pretrained(word_mat, freeze=not self.unfreeze_wordvec)
    if no_wordvec_layer:
      self.conv1 = ibp.Conv1d(word_vec_size, hidden_size, kernel_size,
                              padding=cnn_padding)
    else:
      self.linear_input = ibp.Linear(word_vec_size, hidden_size)
      self.conv1 = ibp.Conv1d(hidden_size, hidden_size, kernel_size,
                              padding=cnn_padding)
    if self.pool == 'attn':
      self.attn_pool = ibp.Linear(hidden_size, 1)
    self.dropout = ibp.Dropout(dropout)
    self.fc_hidden = ibp.Linear(hidden_size, hidden_size)
    # self.fc_output = ibp.Linear(hidden_size, 1)

    self.hidden2mean = ibp.Linear(hidden_size, hidden_size)
    self.hidden2logv = ibp.Linear(hidden_size, hidden_size)
    self.BN=ibp.BatchNorm1d(hidden_size)

  def forward(self,batch, compute_bounds=True, cert_eps=1.0):
    """
    Args:
      batch: A batch dict from a TextClassificationDataset with the following keys:
        - x: tensor of word vector indices, size (B, n, 1)
        - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
        - lengths: lengths of sequences, size (B,)
      compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
      cert_eps: Scaling factor for interval bounds of the input
    """
    x_vecs = batch['input_emb_hypothesis']
    # if compute_bounds:
    #   x = batch['x']
    # else:
    #   x = batch['x'].val
    mask = batch['mask_hypothesis']
    # lengths = batch['lengths']
    lengths = mask.sum(1)
    # x_vecs = self.embs(x)  # B, n, d
    # if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
    #   x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
    if not self.no_wordvec_layer:
      x_vecs = self.linear_input(x_vecs)  # B, n, h
    if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
      x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
    if self.no_wordvec_layer or not self.relu_wordvec:
      z = x_vecs
    else:
      z = ibp.activation(F.relu, x_vecs)  # B, n, h
    z_masked = z * mask.unsqueeze(-1)  # B, n, h
    z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
    c1 = ibp.activation(F.relu, self.conv1(z_cnn_in))  # B, h, n
    c1_masked = c1 * mask.unsqueeze(1)  # B, h, n
    if self.pool == 'mean':
      fc_in = ibp.sum(c1_masked / lengths.to(dtype=t.float).view(-1, 1, 1), 2)  # B, h
    elif self.pool == 'attn':
      fc_in = attention_pool(c1_masked.permute(0, 2, 1), mask, self.attn_pool)  # B, h
    else:
      # zero-masking works b/c ReLU guarantees that everything is >= 0
      fc_in = ibp.pool(t.max, c1_masked, 2)  # B, h

    fc_in = self.dropout(fc_in)
    fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in)) # B, h
    fc_hidden = self.dropout(fc_hidden)
    # output = self.fc_output(fc_hidden)  # B, 1
    mu = self.hidden2mean(fc_hidden)
    log_sigma = self.hidden2logv(fc_hidden)

    return mu, log_sigma, None



class BOWModel(nn.Module):
  """Bag of words + MLP"""

  def __init__(self, word_vec_size, hidden_size,
               dropout_prob=0.1, num_layers=2, no_wordvec_layer=False):
    super(BOWModel, self).__init__()
    # self.embs = ibp.Embedding.from_pretrained(word_mat)
    self.rotation = ibp.Linear(word_vec_size, hidden_size)
    self.sum_drop = ibp.Dropout(dropout_prob) if dropout_prob else None
    layers = []
    for i in range(num_layers):
      layers.append(ibp.Linear(2*hidden_size, 2*hidden_size))
      layers.append(ibp.Activation(F.relu))
      if dropout_prob:
        layers.append(ibp.Dropout(dropout_prob))
    # layers.append(ibp.Linear(2*hidden_size, len(EntailmentLabels)))
    # layers.append(ibp.LogSoftmax(dim=1))
    self.layers = nn.Sequential(*layers)
    self.hidden2mean = ibp.Linear(2*hidden_size, hidden_size)
    self.hidden2logv = ibp.Linear(2*hidden_size, hidden_size)

  def forward(self, batch, compute_bounds=True, cert_eps=1.0):
    """
    Forward pass of BOWModel.
    Args:
      batch: A batch dict from an EntailmentDataset with the following keys:
        - prem: tensor of word vector indices for premise (B, p, 1)
        - hypo: tensor of word vector indices for hypothesis (B, h, 1)
        - prem_mask: binary mask over premise words (1 for real, 0 for pad), size (B, p)
        - hypo_mask: binary mask over hypothesis words (1 for real, 0 for pad), size (B, h)
        - prem_lengths: lengths of premises, size (B,)
        - hypo_lengths: lengths of hypotheses, size (B,)
      compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
      cert_eps: float, scaling factor for the interval bounds.
    """
    def encode(sequence, mask):

      vecs = self.rotation(sequence)
      if isinstance(vecs, ibp.DiscreteChoiceTensor):
        vecs = vecs.to_interval_bounded(eps=cert_eps)
      z1 = ibp.activation(F.relu, vecs)
      z1_masked = z1 * mask.unsqueeze(-1)
      z1_pooled = ibp.sum(z1_masked, -2)
      return z1_pooled
    # if not compute_bounds:
    #     batch['prem']['x'] = batch['prem']['x'].val
    #     batch['hypo']['x'] = batch['hypo']['x'].val
    #     batch['prem']['x'] = batch['prem']['x'].val
    #     batch['hypo']['x'] = batch['hypo']['x'].val
    prem_encoded = encode(batch['input_emb_premises'], batch['mask_premises'])
    hypo_encoded = encode(batch['input_emb_hypothesis'], batch['mask_hypothesis'])
    input_encoded = ibp.cat([prem_encoded, hypo_encoded], -1)
    hidden = self.layers(input_encoded)
    mu = self.hidden2mean(hidden)
    log_sigma = self.hidden2logv(hidden)

    return mu, log_sigma, None



# class Encoder_RNN(nn.Module):
#     def __init__(self, embed_size, latent_size):
#         super(Encoder_RNN, self).__init__()
#
#         self.embed_size = embed_size
#         self.latent_size = latent_size
#
#         self.rnn = nn.GRU(input_size=self.embed_size,
#                           hidden_size=self.rnn_size,
#                           num_layers=self.rnn_num_layers,
#                           batch_first=True)
#
#         self.hidden2mean = nn.Linear(self.rnn_size, self.latent_size)
#         self.hidden2logv = nn.Linear(self.rnn_size, self.latent_size)
#
#     def forward(self, input):
#         """
#         :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
#         :return: An float tensor with shape of [batch_size, latent_variable_size]
#         """
#
#         '''
#         Transpose input to the shape of [batch_size, embed_size, seq_len]
#         '''
#
#
#         res,_ = self.rnn(input)
#         mu=self.hidden2mean(res)
#         log_sigma=self.hidden2logv(res)
#
#         return mu,log_sigma