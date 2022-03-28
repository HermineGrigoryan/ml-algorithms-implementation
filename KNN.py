import numpy as np
import matplotlib.pyplot as plt

def distance_metric(x1, x2, dist_type='euclidean'):
    if dist_type == 'euclidean':
        X = np.array(x1) - np.array(x2) 
        dist = np.sqrt(np.sum(X**2))
    return dist


class KnnClassifier:
    def __init__(self, X, y, k_neighbors):
        self.X = np.array(X)
        self.y = np.array(y)
        self.k_neighbors = k_neighbors
    
    def __repr__(self):
        return f'KnnClassifier({self.k_neighbors})'
    
    def predict(self, new_x):
        self.new_x = new_x
        
        pred_y = []
        
        for i in self.new_x:
            dists = {ind: distance_metric(x, i) for ind, x in enumerate(self.X)}
            nearest_neighbors = sorted(dists, key=dists.get)[0:self.k_neighbors]  
            target_neighbors = self.y[nearest_neighbors]
            labels, counts = np.unique(target_neighbors, return_counts=True)
            pred_y.append(labels[counts.argmax()])
        
        self.pred_y = pred_y
        
        return pred_y
    
    def plot_2d(self, x1=0, x2=1):
        plt.scatter(self.new_x[:, x1], self.new_x[:, x2], c=self.pred_y, alpha=0.1)
        plt.scatter(self.X[:, x1], self.X[:, x2], c=self.y)
        plt.show()