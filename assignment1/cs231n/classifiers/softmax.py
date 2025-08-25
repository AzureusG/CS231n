from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # normalized hinge loss plus regularization
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in xrange(num_train):
      scores = X[i].dot(W)
      shift_scores = scores - max(scores)
      loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
      loss += loss_i
      for j in xrange(num_classes):
          softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
          if j == y[i]:
              dW[:,j] += (-1 + softmax_output) *X[i] 
          else: 
              dW[:,j] += softmax_output *X[i] 

    loss /= num_train 
    loss +=  0.5* reg * np.sum(W * W)
    dW = dW/num_train + reg* W 

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    
    # 数值稳定性处理：减去最大值
    shift_scores = scores - np.max(scores, axis=1, keepdims=True)
    
    # 计算指数并避免除零
    exp_scores = np.exp(shift_scores)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
    
    # 避免除零：添加小的epsilon
    sum_exp[sum_exp == 0] = 1e-12
    softmax_output = exp_scores / sum_exp
    
    # 计算损失（避免log(0)）
    correct_probs = np.clip(softmax_output[np.arange(num_train), y], 1e-12, 1.0)
    loss = -np.sum(np.log(correct_probs)) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    # 计算梯度
    dscores = softmax_output.copy()
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
