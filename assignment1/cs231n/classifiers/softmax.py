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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # data loss
    data_loss = 0.0
    for i, x in enumerate(X):

      # forward
      score = x.dot(W) # 1x3073 dot 3073x10 -> 1x10 : row
      score_exp = np.exp(score)
      score_exp_sum = np.sum(score_exp)
      score_normalized = np.array([s / score_exp_sum for s in score_exp])
      loss += -np.log(score_normalized[y[i]])

      # backward
      dL = 1
      dMinusLog_dSoftmax = -1.0 / score_normalized
      dL_dSoftmax = dL * dMinusLog_dSoftmax

      # dSoftmax_dScores = (-1 * score_exp * score_exp[y[i]]) / np.square(score_exp_sum)
      dSoftmax_dScores = -1 * score_normalized[y[i]] * score_normalized
      # dSoftmax_dScores[y[i]] = np.exp(np.sum(score)) / np.square(score_exp_sum)
      dSoftmax_dScores[y[i]] = score_normalized[y[i]] * (1 - score_normalized[y[i]])
      dL_dScores = dL_dSoftmax * dSoftmax_dScores # [1x10]

      dScores_dW = x # [1x3073]
      # print('dScores_dW shape: ', np.shape(dScores_dW))
      # print('dL_dScores shape: ', np.shape(dL_dScores))
      # dL_dW = dScores_dW[np.newaxis, :].T.dot(dL_dScores[np.newaxis, :]) # [3073x1 (dot) 1x10]
      dL_dW = np.outer(dScores_dW, dL_dScores) # [1x3073] X [1x10]

      dW += dL_dW

    data_loss = loss / X.shape[0]
    dW /= X.shape[0]
  
    # regularization
    reg_loss = reg * np.sum(np.square(W))

    # Full Loss
    loss = data_loss + reg_loss
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    X_num = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) # [Nx3073] x [3073x10] = [Nx10]
    scores_exp = np.exp(scores) # [Nx10]
    scores_exp_sum = np.sum(scores_exp, axis=1) # [Nx1]
    scores_exp_norm = scores_exp / scores_exp_sum.reshape(-1, 1) # [Nx10]
    # scores_exp_norm = scores_exp / scores_exp_sum[:, np.newaxis] # [Nx10]
    
    # fancy indexing: 배열에 인덱스 배열을 전달하여 여러 인덱스를 한번에 선택
    softmax_losses = -np.log(scores_exp_norm[np.arange(X_num), y]) # [Nx1]
    loss = np.sum(softmax_losses)

    dL = np.array([1]).reshape(-1, 1)
  
    dMinusLog_dSoftmax = -1. / scores_exp_norm # [Nx10]
    dL_dSoftmax = dL * dMinusLog_dSoftmax #[Nx10]

    dSoftmax_dScores = -1 * scores_exp_norm[np.arange(X_num)] * scores_exp_norm[np.arange(X_num), y].reshape(-1, 1)
    dSoftmax_dScores[np.arange(X_num), y] = scores_exp_norm[np.arange(X_num), y] * (1 - scores_exp_norm[np.arange(X_num), y])
    dL_dScores = dL_dSoftmax * dSoftmax_dScores # [Nx10]

    dScores_dW = X # [Nx3073]
    dL_dW = np.dot(dScores_dW.T, dL_dScores) # [3073xN] dot [Nx10] = [3073x10]

    loss /= X_num
    dW = dL_dW / X_num

    loss += reg*np.sum(np.square(W)) # regularization
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
