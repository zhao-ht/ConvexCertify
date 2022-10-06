import numpy as np
import torch
import sys
DEBUG = True
TOLERANCE = 1e-2

##### BoundedTensor and subclasses #####

class BoundedTensor(object):
  """Contains a torch.Tensor plus bounds on it."""
  @property
  def shape(self):
    return self.val.shape

  def __add__(self, other):
    return add(self, other)

  def __mul__(self, other):
    return mul(self, other)

  def __truediv__(self, other):
    return div(self, other)


class IntervalBoundedTensor(BoundedTensor):
  """A tensor with elementwise upper and lower bounds.
  This is the main building BoundedTensor subclass.
  All layers in this library accept IntervalBoundedTensor as input,
  and when handed one will generate another IntervalBoundedTensor as output.
  """
  def __init__(self, val, lb, ub):
    self.val = val
    self.lb = lb
    self.ub = ub
    if DEBUG:
      # Sanity check lower and upper bounds when creating this
      # Note that there may be small violations on the order of 1e-5,
      # due to floating point rounding/non-associativity issues.
      # e.g. https://github.com/pytorch/pytorch/issues/9146
      max_lb_violation = torch.max(lb - val)
      if max_lb_violation > TOLERANCE:
        print('WARNING: Lower bound wrong (max error = %g)' % max_lb_violation.item(), file=sys.stderr)
        # raise ValueError
      max_ub_violation = torch.max(val - ub)
      if max_ub_violation > TOLERANCE:
        print('WARNING: Upper bound wrong (max error = %g)' % max_ub_violation.item(), file=sys.stderr)
        # raise ValueError

  ### Reimplementations of torch.Tensor methods
  def __neg__(self):
    return IntervalBoundedTensor(-self.val, -self.ub, -self.lb)

  def permute(self, *dims):
    return IntervalBoundedTensor(self.val.permute(*dims),
                                 self.lb.permute(*dims),
                                 self.ub.permute(*dims))

  def squeeze(self, dim=None):
    return IntervalBoundedTensor(self.val.squeeze(dim=dim),
                                 self.lb.squeeze(dim=dim),
                                 self.ub.squeeze(dim=dim))

  def unsqueeze(self, dim):
    return IntervalBoundedTensor(self.val.unsqueeze(dim),
                                 self.lb.unsqueeze(dim),
                                 self.ub.unsqueeze(dim))

  def to(self, device):
    self.val = self.val.to(device)
    self.lb = self.lb.to(device)
    self.ub = self.ub.to(device)
    return self

  # For slicing
  def __getitem__(self, key):
    return IntervalBoundedTensor(self.val.__getitem__(key),
                                 self.lb.__getitem__(key),
                                 self.ub.__getitem__(key))

  def __setitem__(self, key, value):
    if not isinstance(value, IntervalBoundedTensor):
      raise TypeError(value)
    self.val.__setitem__(key, value.val)
    self.lb.__setitem__(key, value.lb)
    self.ub.__setitem__(key, value.ub)

  def __delitem__(self, key):
    self.val.__delitem__(key)
    self.lb.__delitem__(key)
    self.ub.__delitem__(key)


class DiscreteChoiceTensor(BoundedTensor):
  """A tensor for which each row can take a discrete set of values.

  More specifically, each slice along the first d-1 dimensions of the tensor
  is allowed to take on values within some discrete set.
  The overall tensor's possible values are the direct product of all these
  individual choices.

  Recommended usage is only as an input tensor, passed to Linear() layer.
  Only some layers accept this tensor.
  """
  def __init__(self, val, choice_mat, choice_mask, sequence_mask):
    """Create a DiscreteChoiceTensor.

    Args:
      val: value, dimension (*, d).  Let m = product of first d-1 dimensions.
      choice_mat: all choices-padded with 0 where fewer than max choices are available, size (*, C, d)
      choice_mask: mask tensor s.t. choice_maks[i,j,k]==1 iff choice_mat[i,j,k,:] is a valid choice, size (*, C)
      sequence_mask: mask tensor s.t. sequence_mask[i,j,k]==1 iff choice_mat[i,j] is a valid word in a sequence and not padding, size (*)
    """
    self.val = val
    self.choice_mat = choice_mat
    self.choice_mask = choice_mask
    self.sequence_mask = sequence_mask

  def to_interval_bounded(self, eps=1.0):
    """
    Convert to an IntervalBoundedTensor.
    Args:
      - eps: float, scaling factor for the interval bounds
    """
    choice_mask_mat = (((1 - self.choice_mask).float() * 1e16)).unsqueeze(-1)  # *, C, 1
    seq_mask_mat = self.sequence_mask.unsqueeze(-1).unsqueeze(-1).float()
    lb = torch.min((self.choice_mat + choice_mask_mat) * seq_mask_mat, -2)[0]  # *, d
    ub = torch.max((self.choice_mat - choice_mask_mat) * seq_mask_mat, -2)[0]  # *, d
    val = self.val * self.sequence_mask.unsqueeze(-1)
    if eps != 1.0:
        lb = val - (val - lb) * eps
        ub = val + (ub - val) * eps
    return IntervalBoundedTensor(val, lb, ub)

  def to(self, device):
    """Moves the Tensor to the given device"""
    self.val = self.val.to(device)
    self.choice_mat = self.choice_mat.to(device)
    self.choice_mask = self.choice_mask.to(device)
    self.sequence_mask = self.sequence_mask.to(device)
    return self


class NormBallTensor(BoundedTensor):
  """A tensor for which each is within some norm-ball of the original value."""
  def __init__(self, val, radius, p_norm):
    self.val = val
    self.radius = radius
    self.p_norm = p_norm


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

