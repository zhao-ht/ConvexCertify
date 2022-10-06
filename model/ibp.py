"""Interval bound propagation layers in pytorch."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

sys.path.append('..')
from utils import IntervalBoundedTensor, DiscreteChoiceTensor
from torch import Tensor


##########################################
def max_diff_norm(x): # calculate max|f(x)-f(x+delta)|
  ub = ((x.ub - x.val)**2).unsqueeze(1) #[b, h]
  lb = ((x.lb - x.val)**2).unsqueeze(1)
  bound = torch.cat([ub, lb], dim=1) #[b, h]
  bound_max = torch.max(bound, dim=1).values
  return torch.sqrt(torch.sum(bound_max, dim=-1))
  
  






class BatchNorm1d(nn.BatchNorm1d):

  def _check_input_dim(self, input):
    if input.val.dim() != 2 and input.val.dim() != 3:
      raise ValueError('expected 2D or 3D input (got {}D input)'
                       .format(input.val.dim()))

  def forward(self, input: IntervalBoundedTensor) -> IntervalBoundedTensor:
    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
      # TODO: if statement only here to tell the jit to skip emitting this when it is None
      if self.num_batches_tracked is not None:
        self.num_batches_tracked = self.num_batches_tracked + 1
        if self.momentum is None:  # use cumulative moving average
          exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
          exponential_average_factor = self.momentum

    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
      bn_training = True
    else:
      bn_training = (self.running_mean is None) and (self.running_var is None)

    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """
    _=F.batch_norm(
      input.val,
      # If buffers are not to be tracked, ensure that they won't be updated
      self.running_mean if not self.training or self.track_running_stats else None,
      self.running_var if not self.training or self.track_running_stats else None,
      self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
    val= F.batch_norm(
      input.val,
      # If buffers are not to be tracked, ensure that they won't be updated
      self.running_mean if not self.training or self.track_running_stats else None,
      self.running_var if not self.training or self.track_running_stats else None,
      self.weight, self.bias, False, exponential_average_factor, self.eps)
    ub=F.batch_norm(
      input.ub,
      # If buffers are not to be tracked, ensure that they won't be updated
      self.running_mean if not self.training or self.track_running_stats else None,
      self.running_var if not self.training or self.track_running_stats else None,
      self.weight, self.bias, False, exponential_average_factor, self.eps)
    lb=F.batch_norm(
      input.lb,
      # If buffers are not to be tracked, ensure that they won't be updated
      self.running_mean if not self.training or self.track_running_stats else None,
      self.running_var if not self.training or self.track_running_stats else None,
      self.weight, self.bias, False, exponential_average_factor, self.eps)

    return IntervalBoundedTensor(val, lb, ub)



##### nn.Module's for BoundedTensor #####

class Linear(nn.Linear):
  """Linear layer."""
  def forward(self, x):
    if isinstance(x, torch.Tensor):
      return super(Linear, self).forward(x)
    if isinstance(x, IntervalBoundedTensor):
      z = F.linear(x.val, self.weight, self.bias)
      weight_abs = torch.abs(self.weight)
      mu_cur = (x.ub + x.lb) / 2
      r_cur = (x.ub - x.lb) / 2
      mu_new = F.linear(mu_cur, self.weight, self.bias)
      r_new = F.linear(r_cur, weight_abs)
      return IntervalBoundedTensor(z, mu_new - r_new, mu_new + r_new)
    elif isinstance(x, DiscreteChoiceTensor):
      new_val = F.linear(x.val, self.weight, self.bias)
      new_choices = F.linear(x.choice_mat, self.weight, self.bias)
      return DiscreteChoiceTensor(new_val, new_choices, x.choice_mask, x.sequence_mask)
    elif isinstance(x, NormBallTensor):
      q = 1.0 / (1.0 - 1.0 / x.p_norm)  # q from Holder's inequality
      z = F.linear(x.val, self.weight, self.bias)
      q_norm = torch.norm(self.weight, p=q, dim=1)  # Norm along in_dims axis
      delta = x.radius * q_norm
      return IntervalBoundedTensor(z, z - delta, z + delta)  # Broadcast out_dims
    else:
      raise TypeError(x)


class LinearOutput(Linear):
  """Linear output layer.

  A linear layer, but instead of computing interval bounds, computes

      max_{z feasible} c^T z + d

  where z is the output of this layer, for given vector(s) c and scalar(s) d.
  Following Gowal et al. (2018), we can get a slightly better bound here
  than by doing normal bound propagation.
  """
  def forward(self, x_ibp, c_list=None, d_list=None):
    """Compute linear output layer and bound on adversarial objective.

    Args:
      x_ibp: an ibp.Tensor of shape (batch_size, in_dims)
      c_list: list of torch.Tensor, each of shape (batch_size, out_dims)
      d_list: list of torch.Tensor, each of shape (batch_size,)
    Returns:
      x: ibp.Tensor of shape (batch_size, out_dims)
      bounds: if c_list and d_list, torch.Tensor of shape (batch_size,)
    """
    x, x_lb, x_ub = x_ibp
    z = F.linear(x, self.weight, self.bias)
    if c_list and d_list:
      bounds = []
      mu_cur = ((x_lb + x_ub) / 2).unsqueeze(1)  # B, 1, in_dims
      r_cur = ((x_ub - x_lb) / 2).unsqueeze(1)  # B, 1, in_dims
      for c, d in zip(c_list, d_list):
        c_prime = c.matmul(self.weight).unsqueeze(2)  # B, in_dims, 1
        d_prime = c.matmul(self.bias) + d  # B,
        c_prime_abs = torch.abs(c_prime)  # B, in_dims, 1
        mu_new = mu_cur.matmul(c_prime).view(-1)  # B,
        r_cur = r_cur.matmul(c_prime_abs).view(-1)  # B,
        bounds.append(mu_new + r_cur + d)
      return z, bounds
    else:
      return z


class Embedding(nn.Embedding):
  """nn.Embedding for DiscreteChoiceTensor.

  Note that unlike nn.Embedding, this module requires that the last dimension
  of the input is size 1, and will squeeze it before calling F.embedding.
  This requirement is due to how DiscreteChoiceTensor requires a dedicated
  dimension to represent the dimension along which values can change.
  """
  def forward(self, x):
    if isinstance(x, torch.Tensor):
      return super(Embedding, self).forward(x.squeeze(-1))
    if isinstance(x, DiscreteChoiceTensor):
      if x.val.shape[-1] != 1:
        raise ValueError('Input tensor has shape %s, where last dimension != 1' % x.shape)
      new_val = F.embedding(
          x.val.squeeze(-1), self.weight, self.padding_idx, self.max_norm,
          self.norm_type, self.scale_grad_by_freq, self.sparse)
      new_choices = F.embedding(
          x.choice_mat.squeeze(-1), self.weight, self.padding_idx, self.max_norm,
          self.norm_type, self.scale_grad_by_freq, self.sparse)
      return DiscreteChoiceTensor(new_val, new_choices, x.choice_mask, x.sequence_mask)
    else:
      raise TypeError(x)


class Conv1d(nn.Conv1d):
  """One-dimensional convolutional layer.

  Works the same as a linear layer.
  """
  def forward(self, x):
    if isinstance(x, torch.Tensor):
      return super(Conv1d, self).forward(x)
    if isinstance(x, IntervalBoundedTensor):
      z = F.conv1d(x.val, self.weight, self.bias, self.stride,
                   self.padding, self.dilation, self.groups)
      weight_abs = torch.abs(self.weight)
      mu_cur = (x.ub + x.lb) / 2
      r_cur = (x.ub - x.lb) / 2
      mu_new = F.conv1d(mu_cur, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
      r_new = F.conv1d(r_cur, weight_abs, None, self.stride,
                       self.padding, self.dilation, self.groups)
      return IntervalBoundedTensor(z, mu_new - r_new, mu_new + r_new)
    else:
      raise TypeError(x)


class MaxPool1d(nn.MaxPool1d):
  """One-dimensional max-pooling layer."""
  def forward(self, x):
    if isinstance(x, torch.Tensor):
      return super(MaxPool1d, self).forward(x)
    elif isinstance(x, IntervalBoundedTensor):
      z = F.max_pool1d(x.val, self.kernel_size, self.stride, self.padding,
                       self.dilation, self.ceil_mode, self.return_indices)
      lb = F.max_pool1d(x.lb, self.kernel_size, self.stride, self.padding,
                        self.dilation, self.ceil_mode, self.return_indices)
      ub = F.max_pool1d(x.ub, self.kernel_size, self.stride, self.padding,
                        self.dilation, self.ceil_mode, self.return_indices)
      return IntervalBoundedTensor(z, lb, ub)
    else:
      raise TypeError(x)


class LSTM(nn.Module):
  """An LSTM."""
  def __init__(self, input_size, hidden_size, bidirectional=False):
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    self.i2h = Linear(input_size, 4 * hidden_size)
    self.h2h = Linear(hidden_size, 4 * hidden_size)
    if bidirectional:
      self.back_i2h = Linear(input_size, 4 * hidden_size)
      self.back_h2h = Linear(hidden_size, 4 * hidden_size)

  def _step(self, h, c, x_t, i2h, h2h, analysis_mode=False):
    preact = add(i2h(x_t), h2h(h))
    g_t = activation(torch.tanh, preact[:, 3 * self.hidden_size:])
    gates = activation(torch.sigmoid, preact[:, :3 * self.hidden_size])
    i_t = gates[:, :self.hidden_size]
    f_t = gates[:, self.hidden_size:2 * self.hidden_size]
    o_t = gates[:, 2 * self.hidden_size:]
    c_t = add(mul(c, f_t), mul(i_t, g_t))
    h_t = mul(o_t, activation(torch.tanh, c_t))
    if analysis_mode:
      return h_t, c_t, i_t, f_t, o_t
    return h_t, c_t

  def _process(self, h, c, x, i2h, h2h, reverse=False, mask=None, analysis_mode=False):
    B, T, d = x.shape  # batch_first=True
    idxs = range(T)
    if reverse:
      idxs = idxs[::-1]
    h_seq = []
    c_seq = []
    if analysis_mode:
      i_seq = []
      f_seq = []
      o_seq = []
    for i in idxs:
      x_t = x[:,i,:]  # B, d_in
      if analysis_mode:
        h_t, c_t, i_t, f_t, o_t = self._step(h, c, x_t, i2h, h2h, analysis_mode=True)
        i_seq.append(i_t)
        f_seq.append(f_t)
        o_seq.append(o_t)
      else:
        h_t, c_t = self._step(h, c, x_t, i2h, h2h)
      if mask is not None:
        # Don't update h or c when mask is 0
        mask_t = mask[:,i].unsqueeze(1)  # B,1
        h = h_t * mask_t + h * (1.0 - mask_t)
        c = c_t * mask_t + c * (1.0 - mask_t)
      h_seq.append(h)
      c_seq.append(c)
    if reverse:
      h_seq = h_seq[::-1]
      c_seq = c_seq[::-1]
      if analysis_mode:
        i_seq = i_seq[::-1]
        f_seq = f_seq[::-1]
        o_seq = o_seq[::-1]
    if analysis_mode:
      return h_seq, c_seq, i_seq, f_seq, o_seq
    return h_seq, c_seq

  def forward(self, x, s0, mask=None, analysis_mode=False):
    """Forward pass of LSTM

    Args:
      x: word vectors, size (B, T, d)
      s0: tuple of (h0, x0) where each is (B, d), or (B, 2d) if bidirectional=True
      mask: If provided, 0-1 mask of size (B, T)
    """
    h0, c0 = s0  # Each is (B, d), or (B, 2d) if bidirectional=True
    if self.bidirectional:
      h0_back = h0[:,self.hidden_size:]
      h0 = h0[:,:self.hidden_size]
      c0_back = c0[:,self.hidden_size:]
      c0 = c0[:,:self.hidden_size]
    if analysis_mode:
      h_seq, c_seq, i_seq, f_seq, o_seq = self._process(
          h0, c0, x, self.i2h, self.h2h, mask=mask, analysis_mode=True)
    else:
      h_seq, c_seq = self._process(h0, c0, x, self.i2h, self.h2h, mask=mask)
    if self.bidirectional:
      if analysis_mode:
        h_back_seq, c_back_seq, i_back_seq, f_back_seq, o_back_seq = self._process(
            h0_back, c0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask,
            analysis_mode=True)
        i_seq = [cat((f, b), dim=1) for f, b in zip(i_seq, i_back_seq)]
        f_seq = [cat((f, b), dim=1) for f, b in zip(f_seq, f_back_seq)]
        o_seq = [cat((f, b), dim=1) for f, b in zip(o_seq, o_back_seq)]
      else:
        h_back_seq, c_back_seq = self._process(
            h0_back, c0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask)
      h_seq = [cat((hf, hb), dim=1) for hf, hb in zip(h_seq, h_back_seq)]
      c_seq = [cat((cf, cb), dim=1) for cf, cb in zip(c_seq, c_back_seq)]
    h_mat = stack(h_seq, dim=1)  # list of (B, d) -> (B, T, d)
    c_mat = stack(c_seq, dim=1)  # list of (B, d) -> (B, T, d)
    if analysis_mode:
      i_mat = stack(i_seq, dim=1)
      f_mat = stack(f_seq, dim=1)
      o_mat = stack(o_seq, dim=1)
      return h_mat, c_mat, (i_mat, f_mat, o_mat)
    return h_mat, c_mat


class GRU(nn.Module):
  """A GRU."""
  def __init__(self, input_size, hidden_size, bidirectional=False):
    super(GRU, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    self.i2h = Linear(input_size, 3 * hidden_size)
    self.h2h = Linear(hidden_size, 3 * hidden_size)
    if bidirectional:
      self.back_i2h = Linear(input_size, 3 * hidden_size)
      self.back_h2h = Linear(hidden_size, 3 * hidden_size)

  def _step(self, h, x_t, i2h, h2h):
    i_out = i2h(x_t)
    h_out = h2h(h)
    preact = add(i_out[:, :2*self.hidden_size], h_out[:, :2*self.hidden_size])
    gates = activation(torch.sigmoid, preact)
    r_t = gates[:, :self.hidden_size]
    z_t = gates[:, self.hidden_size:]
    i_state = i_out[:, 2*self.hidden_size:]
    h_state = h_out[:, 2*self.hidden_size:]
    n_t = activation(torch.tanh, i_state + mul(r_t, h_state))
    if isinstance(z_t, torch.Tensor):
      ones = torch.ones_like(z_t)
    else:
      ones = torch.ones_like(z_t.val)
    h_t = add(mul(add(ones, - z_t), n_t), mul(z_t, h))
    return h_t

  def _process(self, h, x, i2h, h2h, reverse=False, mask=None):
    B, T, d = x.shape  # batch_first=True
    idxs = range(T)
    if reverse:
      idxs = idxs[::-1]
    h_seq = []
    for i in idxs:
      x_t = x[:,i,:]  # B, d_in
      h_t = self._step(h, x_t, i2h, h2h)
      if mask is not None:
        # Don't update h when mask is 0
        mask_t = mask[:,i].unsqueeze(1)  # B,1
        h = h_t * mask_t + h * (1.0 - mask_t)
      h_seq.append(h)
    if reverse:
      h_seq = h_seq[::-1]
    return h_seq

  def forward(self, x, h0, mask=None):
    """Forward pass of GRU

    Args:
      x: word vectors, size (B, T, d)
      h0: tuple of (h0, x0) where each is (B, d), or (B, 2d) if bidirectional=True
      mask: If provided, 0-1 mask of size (B, T)
    """
    if self.bidirectional:
      h0_back = h0[:,self.hidden_size:]
      h0 = h0[:,:self.hidden_size]
    h_seq = self._process(h0, x, self.i2h, self.h2h, mask=mask)
    if self.bidirectional:
      h_back_seq = self._process(
          h0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask)
      h_seq = [cat((hf, hb), dim=1) for hf, hb in zip(h_seq, h_back_seq)]
    h_mat = stack(h_seq, dim=1)  # list of (B, d) -> (B, T, d)
    return h_mat


class Dropout(nn.Dropout):
  def forward(self, x):
    if isinstance(x, torch.Tensor):
      return super(Dropout, self).forward(x)
    elif isinstance(x, IntervalBoundedTensor):
      if self.training:
        probs = torch.full_like(x.val, 1.0 - self.p)
        mask = torch.distributions.Bernoulli(probs).sample() / (1.0 - self.p)
        return IntervalBoundedTensor(mask * x.val, mask * x.lb, mask * x.ub)
      else:
        return x
    else:
      raise TypeError(x)


def add(x1, x2):
  """Sum two tensors."""
  # I think we have to do it this way and not as operator overloading,
  # to catch the case of torch.Tensor.__add__(IntervalBoundedTensor)
  if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
    return x1 + x2
  elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
    if isinstance(x2, torch.Tensor):
      x1, x2 = x2, x1  # WLOG x1 is torch.Tensor
    if isinstance(x2, IntervalBoundedTensor):
      return IntervalBoundedTensor(x2.val + x1, x2.lb + x1, x2.ub + x1)
    else:
      raise TypeError(x1, x2)
  else:
    if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
      return IntervalBoundedTensor(x1.val + x2.val, x1.lb + x2.lb, x1.ub + x2.ub)
    else:
      raise TypeError(x1, x2)


def mul(x1, x2):
  """Elementwise multiplication of two tensors."""
  if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
    return torch.mul(x1, x2)
  elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
    if isinstance(x2, torch.Tensor):
      x1, x2 = x2, x1  # WLOG x1 is torch.Tensor
    if isinstance(x2, IntervalBoundedTensor):
      z = torch.mul(x2.val, x1)
      lb_mul = torch.mul(x2.lb, x1)
      ub_mul = torch.mul(x2.ub, x1)
      lb_new = torch.min(lb_mul, ub_mul)
      ub_new = torch.max(lb_mul, ub_mul)
      return IntervalBoundedTensor(z, lb_new, ub_new)
    else:
      raise TypeError(x1, x2)
  else:
    if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
      z = torch.mul(x1.val, x2.val)
      ll = torch.mul(x1.lb, x2.lb)
      lu = torch.mul(x1.lb, x2.ub)
      ul = torch.mul(x1.ub, x2.lb)
      uu = torch.mul(x1.ub, x2.ub)
      stack = torch.stack((ll, lu, ul, uu))
      lb_new = torch.min(stack, dim=0)[0]
      ub_new = torch.max(stack, dim=0)[0]
      return IntervalBoundedTensor(z, lb_new, ub_new)
    else:
      raise TypeError(x1, x2)

def div(x1, x2):
  """Elementwise division of two tensors."""
  if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
    return torch.div(x1, x2)
  if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, torch.Tensor):
    z = torch.div(x1.val, x2)
    lb_div = torch.div(x1.lb, x2)
    ub_div = torch.div(x1.ub, x2)
    lb_new = torch.min(lb_div, ub_div)
    ub_new = torch.max(lb_div, ub_div)
    return IntervalBoundedTensor(z, lb_new, ub_new)
  else:
    raise TypeError(x1, x2)


def bmm(x1, x2):
  """Batched matrix multiply.

  Args:
    x1: tensor of shape (B, m, p)
    x2: tensor of shape (B, p, n)
  Returns:
    tensor of shape (B, m, n)
  """
  if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
    return torch.matmul(x1, x2)
  elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
    swap = False
    if isinstance(x2, torch.Tensor):
      swap = True
      x1, x2 = x2.permute(0, 2, 1), x1.permute(0, 2, 1)  # WLOG x1 is torch.Tensor
    if isinstance(x2, IntervalBoundedTensor):
      z = torch.matmul(x1, x2.val)
      x1_abs = torch.abs(x1)
      mu_cur = (x2.ub + x2.lb) / 2
      r_cur = (x2.ub - x2.lb) / 2
      mu_new = torch.matmul(x1, mu_cur)
      r_new = torch.matmul(x1_abs, r_cur)
      if swap:
        z = z.permute(0, 2, 1)
        mu_new = mu_new.permute(0, 2, 1)
        r_new = r_new.permute(0, 2, 1)
      return IntervalBoundedTensor(z, mu_new - r_new, mu_new + r_new)
    else:
      raise TypeError(x1, x2)
  else:
    if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
      z = torch.matmul(x1.val, x2.val)
      ll = torch.einsum('ijk,ikl->ijkl', x1.lb, x2.lb)  # B, m, p, n
      lu = torch.einsum('ijk,ikl->ijkl', x1.lb, x2.ub)  # B, m, p, n
      ul = torch.einsum('ijk,ikl->ijkl', x1.ub, x2.lb)  # B, m, p, n
      uu = torch.einsum('ijk,ikl->ijkl', x1.ub, x2.ub)  # B, m, p, n
      stack = torch.stack([ll, lu, ul, uu])
      mins = torch.min(stack, dim=0)[0]  # B, m, p, n
      maxs = torch.max(stack, dim=0)[0]  # B, m, p, n
      lb_new = torch.sum(mins, dim=2)  # B, m, n
      ub_new = torch.sum(maxs, dim=2)  # B, m, n
      return IntervalBoundedTensor(z, lb_new, ub_new)
    else:
      raise TypeError(x1, x2)


def matmul_nneg(x1, x2):
  """Matrix multiply for non-negative matrices (easier than the general case)."""
  if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
    if (x1 < 0).any(): raise ValueError('x1 has negative entries')
    if (x2 < 0).any(): raise ValueError('x2 has negative entries')
    return torch.matmul(x1, x2)
  elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
    swap = False
    if isinstance(x2, torch.Tensor):
      swap = True
      x1, x2 = x2.permute(0, 2, 1), x1.permute(0, 2, 1)  # WLOG x1 is torch.Tensor
    if isinstance(x2, IntervalBoundedTensor):
      if (x1 < 0).any(): raise ValueError('x1 has negative entries')
      if (x2.lb < 0).any(): raise ValueError('x2 has negative lower bounds')
      z = torch.matmul(x1, x2.val)
      lb_new = torch.matmul(x1, x2.lb)
      ub_new = torch.matmul(x1, x2.ub)
      if swap:
        lb_new = lb_new.permute(0, 2, 1)
        ub_new = ub_new.permute(0, 2, 1)
      return IntervalBoundedTensor(z, lb_new, ub_new)
    else:
      raise TypeError(x1, x2)
  else:
    if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
      if (x1.lb < 0).any(): raise ValueError('x1 has negative lower bounds')
      if (x2.lb < 0).any(): raise ValueError('x2 has negative lower bounds')
      z = torch.matmul(x1.val, x2.val)
      lb_new = torch.matmul(x1.lb, x2.lb)
      ub_new = torch.matmul(x1.ub, x2.ub)
      return IntervalBoundedTensor(z, lb_new, ub_new)
    else:
      raise TypeError(x1, x2)


def cat(tensors, dim=0):
  if all(isinstance(x, torch.Tensor) for x in tensors):
    return torch.cat(tensors, dim=dim)
  tensors_ibp = []
  for x in tensors:
    if isinstance(x, IntervalBoundedTensor):
      tensors_ibp.append(x)
    elif isinstance(x, torch.Tensor):
      tensors_ibp.append(IntervalBoundedTensor(x, x, x))
    else:
      raise TypeError(x)
  return IntervalBoundedTensor(torch.cat([x.val for x in tensors_ibp], dim=dim),
                               torch.cat([x.lb for x in tensors_ibp], dim=dim),
                               torch.cat([x.ub for x in tensors_ibp], dim=dim))

def stack(tensors, dim=0):
  if all(isinstance(x, torch.Tensor) for x in tensors):
    return torch.stack(tensors, dim=dim)
  tensors_ibp = []
  for x in tensors:
    if isinstance(x, IntervalBoundedTensor):
      tensors_ibp.append(x)
    elif isinstance(x, torch.Tensor):
      tensors_ibp.append(IntervalBoundedTensor(x, x, x))
    else:
      raise TypeError(x)
  return IntervalBoundedTensor(
      torch.stack([x.val for x in tensors_ibp], dim=dim),
      torch.stack([x.lb for x in tensors_ibp], dim=dim),
      torch.stack([x.ub for x in tensors_ibp], dim=dim))


def pool(func, x, dim):
  """Pooling operations (e.g. mean, min, max).

  For all of these, the pooling passes straight through the bounds.
  """
  if func not in (torch.mean, torch.min, torch.max, torch.sum):
    raise ValueError(func)
  if func in (torch.min, torch.max):
    func_copy = func
    func = lambda *args: func_copy(*args)[0]  # Grab first return value for min/max
  if isinstance(x, torch.Tensor):
    return func(x, dim)
  elif isinstance(x, IntervalBoundedTensor):
    return IntervalBoundedTensor(func(x.val, dim), func(x.lb, dim),
                                 func(x.ub, dim))
  else:
    raise TypeError(x)


def sum(x, *args, **kwargs):
  if isinstance(x, torch.Tensor):
    return torch.sum(x, *args)
  elif isinstance(x, IntervalBoundedTensor):
    return IntervalBoundedTensor(
        torch.sum(x.val, *args, **kwargs),
        torch.sum(x.lb, *args, **kwargs),
        torch.sum(x.ub, *args, **kwargs))
  else:
    raise TypeError(x)


class Activation(nn.Module):
  def __init__(self, func):
    super(Activation, self).__init__()
    self.func = func

  def forward(self, x):
    return activation(self.func, x)


def activation(func, x):
  """Monotonic elementwise activation functions (e.g. ReLU, sigmoid).

  Due to monotonicity, it suffices to evaluate the activation at the endpoints.
  """
  if func not in (F.relu, torch.sigmoid, torch.tanh, torch.exp):
    raise ValueError(func)
  if isinstance(x, torch.Tensor):
    return func(x)
  elif isinstance(x, IntervalBoundedTensor):
    return IntervalBoundedTensor(func(x.val), func(x.lb), func(x.ub))
  else:
    raise TypeError(x)


class LogSoftmax(nn.Module):
  def __init__(self, dim):
    super(LogSoftmax, self).__init__()
    self.dim = dim

  def forward(self, x):
    return log_softmax(x, self.dim)


def log_softmax(x, dim):
  """logsoftmax operation, requires |dim| to be provided.

  Have to do some weird gymnastics to get vectorization and stability.
  """
  if isinstance(x, torch.Tensor):
    return F.log_softmax(x, dim=dim)
  elif isinstance(x, IntervalBoundedTensor):
    out = F.log_softmax(x.val, dim)
    # Upper-bound on z_i is u_i - log(sum_j(exp(l_j)) + (exp(u_i) - exp(l_i)))
    ub_lb_logsumexp = torch.logsumexp(x.lb, dim, keepdim=True)
    ub_relu = F.relu(x.ub - x.lb)  # ReLU just to prevent cases where lb > ub due to rounding
    # Compute log(exp(u_i) - exp(l_i)) = u_i + log(1 - exp(l_i - u_i)) in 2 different ways
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf for further discussion
    # (1) When u_i - l_i <= log(2), use expm1
    ub_log_diff_expm1 = torch.log(-torch.expm1(-ub_relu))
    # (2) When u_i - l_i > log(2), use log1p
    use_log1p = (ub_relu > 0.693)
    ub_relu_log1p = torch.masked_select(ub_relu, use_log1p)
    ub_log_diff_log1p = torch.log1p(-torch.exp(-ub_relu_log1p))
    # NOTE: doing the log1p and then masked_select creates NaN's
    # I think this is likely to be a subtle pytorch bug that unnecessarily
    # propagates NaN gradients.
    ub_log_diff_expm1.masked_scatter_(use_log1p, ub_log_diff_log1p)
    ub_log_diff = x.ub + ub_log_diff_expm1

    ub_scale = torch.max(ub_lb_logsumexp, ub_log_diff)
    ub_log_partition = ub_scale + torch.log(
        torch.exp(ub_lb_logsumexp - ub_scale)
        + torch.exp(ub_log_diff - ub_scale))
    ub_out = x.ub - ub_log_partition

    # Lower-bound on z_i is l_i - log(sum_{j != i}(exp(u_j)) + exp(l_i))
    # Normalizing scores by max_j u_j works except when i = argmax_j u_j, u_i >> argmax_{j != i} u_j, and u_i >> l_i.
    # In this case we normalize by the second value
    lb_ub_max, lb_ub_argmax = torch.max(x.ub, dim, keepdim=True)

    # Make `dim` the last dim for easy argmaxing along it later
    dims = np.append(np.delete(np.arange(len(x.shape)), dim), dim).tolist()
    # Get indices to place `dim` back where it was originally
    rev_dims = np.insert(np.arange(len(x.shape) - 1), dim, len(x.shape) - 1).tolist()
    # Flatten x.ub except for `dim`
    ub_max_masked = x.ub.clone().permute(dims).contiguous().view(-1, x.shape[dim])
    # Get argmax along `dim` and set max indices to -inf
    ub_max_masked[np.arange(np.prod(x.shape) / x.shape[dim]), ub_max_masked.argmax(1)] = -float('inf')
    # Reshape to make it look like x.ub again
    ub_max_masked = ub_max_masked.view(np.array(x.shape).take(dims).tolist()).permute(rev_dims)

    lb_logsumexp_without_argmax = ub_max_masked.logsumexp(dim, keepdim=True)

    lb_ub_exp = torch.exp(x.ub - lb_ub_max)
    lb_cumsum_fwd = torch.cumsum(lb_ub_exp, dim)
    lb_cumsum_bwd = torch.flip(torch.cumsum(torch.flip(lb_ub_exp, [dim]), dim), [dim])
    # Shift the cumulative sums so that i-th element is sum of things before i (after i for bwd)
    pad_fwd = [0] * (2 * len(x.shape))
    pad_fwd[-2*dim - 2] = 1
    pad_bwd = [0] * (2 * len(x.shape))
    pad_bwd[-2*dim - 1] = 1
    lb_cumsum_fwd = torch.narrow(F.pad(lb_cumsum_fwd, pad_fwd), dim, 0, x.shape[dim])
    lb_cumsum_bwd = torch.narrow(F.pad(lb_cumsum_bwd, pad_bwd), dim, 1, x.shape[dim])
    lb_logsumexp_without_i = lb_ub_max + torch.log(lb_cumsum_fwd + lb_cumsum_bwd)  # logsumexp over everything except i
    lb_logsumexp_without_i.scatter_(dim, lb_ub_argmax, lb_logsumexp_without_argmax)
    lb_scale = torch.max(lb_logsumexp_without_i, x.lb)
    lb_log_partition = lb_scale + torch.log(
        torch.exp(lb_logsumexp_without_i - lb_scale)
        + torch.exp(x.lb - lb_scale))
    lb_out = x.lb - lb_log_partition
    return IntervalBoundedTensor(out, lb_out, ub_out)

  else:
    raise TypeError(x)


  
