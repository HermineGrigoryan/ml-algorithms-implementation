import numpy as np
import matplotlib.pyplot as plt

def LR_loss(y, y_pred):
  SSE = np.sum((y - y_pred)**2)
  return SSE

def LR_MSE(weights, X, y):
  mse = np.mean(y - np.sum(X*weights, axis=1))**2
  return mse

def LR_loss_gradient(LR_weights, LR_X, LR_Y):
  dw = -2*np.sum((LR_Y - np.sum(LR_X*LR_weights, axis=1)).reshape(-1, 1)*LR_X, axis=0)
  return dw

def r_squared(y, y_hat, n_features = None, adjusted = False):
  RSS = np.sum((y - y_hat)**2)
  TSS = np.sum((y - np.mean(y))**2)
  if adjusted:
    n = len(y)
    return 1 - (RSS/(n - n_features - 1))/(TSS/(n-1))
  else:
    return 1 - RSS/TSS

class SimpleLinearRegression():
  
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def fit(self):
    X_bar = self.X.mean()
    y_bar = self.y.mean()

    self.X = self.X.flatten()
    print("Shape of X", self.X.shape)
    print("Shape of y", self.y.shape)

    self.b1 = np.sum((self.X - X_bar)*(self.y - y_bar)) / np.sum((self.X - X_bar)**2)
    self.b0 = y_bar - self.b1*X_bar

    return self

  @property
  def coeffs(self):
    return np.hstack([self.b0, self.b1])

  def predict(self, new_X):
    pred = self.b0 + self.b1*np.array(new_X)

    self.pred = pred
    self.new_X = new_X
    
    return self.pred

  def plot(self):
    plt.scatter(self.X, self.y)
    plt.plot(self.new_X, self.pred, c='red')
    plt.show()




class GradientDescent():
  def __init__(self, X, y):
    self.X = X
    self.y = y
    print(self.X.shape, self.y.shape)

  def fit(self, n_iter, alpha, starting_point=1):
    self.X = np.c_[np.ones((self.y.shape[0])), self.X]
    weights = np.ones(self.X.shape[1])*starting_point
    mse = []
    self.iter = np.array(range(n_iter))
    for i in range(n_iter):
      # print(weights)
      weights = weights - alpha*LR_loss_gradient(weights, self.X, self.y)
      mse.append(LR_MSE(weights, self.X, self.y))

    self.weights = weights
    self.mse = mse

    return self
  
  @property
  def coeffs(self):
    return self.weights[0], self.weights[1:]

  @property
  def square_metrics(self):
    pass

  @property
  def r_squared(self, adjusted = True):
    pass

  def predict(self, new_df):
    X_new = np.c_[np.ones(len(self.y)), new_df]
    pred = np.sum(X_new*self.weights, axis=1)
    self.pred = pred
    self.X_new = new_df
    return pred

  def plot_mse(self):
    plt.plot(self.iter, self.mse)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.show()

class StochasticGradientDescent():
  def __init__(self, X, y):
    self.X = X
    self.y = y
    print(X.shape, y.shape)

  def fit(self, n_iter, alpha, starting_point=1, batch_size=1):
    self.X = np.c_[np.ones((self.y.shape[0])), self.X]
    weights = np.ones(self.X.shape[1])*starting_point
    mse = []
    self.iter = np.array(range(n_iter))
    for i in range(n_iter):
      samples = np.random.randint(0, len(self.y), batch_size)
      sample_X = self.X[samples]
      sample_Y = self.y[samples]
      weights = weights - alpha*LR_loss_gradient(weights, sample_X, sample_Y)
      mse.append(LR_MSE(weights, sample_X, sample_Y))

    self.weights = weights
    self.mse = mse

    return self
  
  @property
  def coeffs(self):
    return self.weights[0], self.weights[1:]

  @property
  def square_metrics(self):
    pass

  @property
  def r_squared(self, adjusted = True):
    pass

  def predict(self, new_df):
    X_new = np.c_[np.ones((self.y.shape[0])), new_df]
    pred = np.sum(X_new*self.weights, axis=1)
    self.pred = pred
    self.X_new = new_df
    return pred

  def plot_mse(self):
    plt.plot(self.iter, self.mse)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.show()

