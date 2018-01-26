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
  # here, it is easy to run into numeric]nstability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_trains):
    scores = X[i].dot(W)
    scores -= scores[np.argmax(scores)]
    scores = np.exp(scores)
    loss += -np.log(scores[y[i]] / np.sum(scores))

    dW[:,y[i]] += -X[i]
    for j in range(num_classes):
        dW[:,j] += X[i]*scores[j]/np.sum(scores)
  
  loss /= num_trains
  dW /= num_trains

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_demensions = X.shape[1]
  num_classes = W.shape[1]
  
  y_ar = np.zeros((num_trains,num_classes))
  for i in range(num_trains):
      y_ar[i,y[i]]=1

  scores = X.dot(W)
  scores = (scores.T - np.amax(scores, axis = 1)).T
  scores = np.exp(scores)
  loss_ar = np.zeros(num_trains)
  loss_ar = -np.log(np.sum(scores * y_ar, axis =1) / np.sum(scores, axis = 1))

  loss = np.sum(loss_ar)
  dW = X.T.dot(-y_ar + (scores.T/np.sum(scores,axis=1)).T)

  loss /= num_trains
  dW /= num_trains

  loss += reg * np.sum(W*W)
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

