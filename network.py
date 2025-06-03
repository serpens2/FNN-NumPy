from reader import read_data
from warnings import warn
import os
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(legacy='1.25')
'''
Basic feed-forward neural network made entirely with numpy.
Architecture:
    - 1 input layer with 784 neurons (one per image pixel) and ReLU activation function
    - 1 hidden layers with 10 neurons and ReLU activation function
    - 1 output layer with 10 neurons and softmax actiovation function
'''
class neural_network:
    def __init__(self,batch_size=32,epochs=1,learning_rate=0.001):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.W1 = np.random.uniform(size=(10,784))
        self.b1 = np.random.uniform(size=(10,))
        self.W2 = np.random.uniform(size=(10,10))
        self.b2 = np.random.uniform(size=(10,))
    def ReLU(self,z):
        return np.array([ np.max(z_j,0) for z_j in z ]).reshape(-1,)
    def softmax(self,z):
        z -= np.max(z) # using identity softmax(x + c) = softmax(x) to prevent overflow
        a = np.exp(z)
        a /= np.sum(a)
        return a
    def ReLU_derivative(self,z):
        return np.array([ int(z_j > 0) for z_j in z ]).reshape(-1,)
    def softmax_derivative(self,z):
        a = self.softmax(z)
        a -= a*a
        return a
    def forward(self,x, full=False):
        '''
        If full == True, returns weighted inputs z1,z2
        and their corresponding activation vectors a1,a2 at each layer.
        Otherwise returns the output a2
        '''
        z1 = np.dot(self.W1,x).reshape(-1,) + self.b1
        a1 = self.ReLU(z1)
        z2 = np.dot(self.W2,a1) + self.b2
        a2 = self.softmax(z2)
        if full == True:
            return [ [z1,a1],[z2,a2] ]
        else:
            return a2
    def backpropagate(self,x,y_true):
        L1, L2 = self.forward(x, full = True)
        z1,a1 = L1
        z2,a2 = L2
        d2 = a2 - y_true # using cross-entropy cost function
        d1 = ( (self.W2).T @ d2 ) * self.ReLU_derivative(z1)
        b_grad = [d1,d2]
        n2, m2 = self.W2.shape
        W2_grad = np.array([ [ a1[k]*d2[j] for k in range(m2) ] for j in range(n2) ])                
        n1, m1 = self.W1.shape
        # a0 is the input layer x
        W1_grad = np.array([ [ x[k]*d1[j] for k in range(m1) ] for j in range(n1) ])                
        W_grad = [W1_grad, W2_grad]
        return W_grad, b_grad
    def label_to_vec(self,lbl : int):
        y = np.zeros(10)
        y[lbl] = 1
        return y.reshape(-1,)
    def learn_batch(self,batch,batch_labels):
        # each column of a batch is the sample
        W_grad = [np.zeros(self.W1.shape), np.zeros(self.W2.shape)]
        b_grad = [np.zeros(self.b1.shape), np.zeros(self.b2.shape) ]
        batch_y = [ self.label_to_vec(lbl) for lbl in batch_labels ]
        for i in range(self.batch_size):
            dW, db = self.backpropagate(batch[:,i].reshape(-1,),batch_y[i])
            for j in range(len(b_grad)):
                W_grad[j] += dW[j]  
                b_grad[j] += db[j]
        self.W1 -= self.learning_rate * W_grad[0] / self.batch_size
        self.W2 -= self.learning_rate * W_grad[1] / self.batch_size
        self.b1 -= self.learning_rate * b_grad[0] / self.batch_size
        self.b2 -= self.learning_rate * b_grad[1] / self.batch_size
    def learn(self,X,y,show_graph = False, eval_each_batch: int | None = None, eval_test_size: int | None = None):
        '''
        Calculates accuracy after each eval_each_batch batch over eval_test_size randomly sampled points from X,y
        and plots results in the end if show_grapth == True
        '''
        # each column of X is a sample
        cols = X.shape[1]
        r = 0
        if cols % self.batch_size != 0:
            while (cols - r) % self.batch_size != 0:
                r += 1
            warn(f'number of samples {X.shape[1]} does not match batch size {self.batch_size}, last {r} samples were removed', DeprecationWarning, stacklevel=2)
            X = X[:,:cols-r]
        # default values for eval_each_batch and eval_test_size
        if (eval_each_batch is None):
            eval_each_batch = int( np.ceil( ( self.epochs*X.shape[1]/self.batch_size )/10 ) ) # bad for small datasets
        if (eval_test_size is None):
            eval_test_size = self.batch_size
        if show_graph == True:
            acc = []
            batch_acc = []
        for epoch in range(self.epochs):
            for j in range( int(cols/self.batch_size) ):
                if j % 100 == 0:
                    print(f'batch {j}')
                X_batch = X[:,j*self.batch_size:(j+1)*self.batch_size]
                y_batch = y[j*self.batch_size:(j+1)*self.batch_size]
                self.learn_batch(X_batch,y_batch)
                if (j+1) % eval_each_batch == 0:
                    idx = np.random.randint(cols - r, size = eval_test_size)
                    X_test = np.array([ X[:,i] for i in idx ]).T
                    y_test = np.array([ y[i] for i in idx ])
                    acc_j = self.get_accuracy(X_test,y_test)
                    print('')
                    print(f'accuracy = {acc_j}')
                    print('')
                    if show_graph == True:
                        acc.append(acc_j)
                        batch_acc.append( j+1 + epoch*(cols - r) )
            if self.epochs != 1:
                print(f'##### Epoch {epoch+1} complete #####')
        if show_graph == True:
            plt.plot(batch_acc,acc,'-o',markersize=5)
            plt.grid()
            plt.xlabel('№ of batch')
            plt.ylabel('accuracy')
            plt.show()
    def get_accuracy(self,X_test,y_test):
        N = X_test.shape[1]
        outputs = [ self.forward(X_test[:,i],full=False) for i in range(N)]
        y_pred = [ np.argmax(output) for output in outputs]
        acc = 0
        for i in range(N):
            if y_test[i] == y_pred[i]:
                acc += 1
        acc /= N
        return acc
    def save_params(self,folder_name):
        current_dir = os.getcwd()
        folder = os.path.join(current_dir,folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        W1_file = os.path.join(folder,'W1.txt')
        np.savetxt(W1_file,self.W1)
        W2_file = os.path.join(folder,'W2.txt')
        np.savetxt(W2_file,self.W2)
        b1_file = os.path.join(folder,'b1.txt')
        np.savetxt(b1_file,self.b1)
        b2_file = os.path.join(folder,'b2.txt')
        np.savetxt(b2_file,self.b2)
    def load_params(self,folder_name):
        current_dir = os.getcwd()
        folder = os.path.join(current_dir,folder_name)
        W1_file = os.path.join(folder,'W1.txt')
        W1 = np.loadtxt(W1_file)
        self.W1 = W1
        W2_file = os.path.join(folder,'W2.txt')
        W2 = np.loadtxt(W2_file)
        self.W2 = W2
        b1_file = os.path.join(folder,'b1.txt')
        b1 = np.loadtxt(b1_file)
        self.b1 = b1
        b2_file = os.path.join(folder,'b2.txt')
        b2 = np.loadtxt(b2_file)
        self.b2 = b2
            
##########################################
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = read_data()
    X_train = np.array([ Xi.ravel() for Xi in X_train ]).T
    X_test = np.array([ Xi.ravel() for Xi in X_test ]).T
    # normalizing images
    X_train = np.array([ X/255 for X in X_train ])
    X_test = np.array([X/255 for X in X_test])
    print('Data is read successfully')
    network = neural_network(epochs=5)
#    network.load_params('learned_params')
    network.learn(X_train, y_train, show_graph = False)
    acc = network.get_accuracy(X_test,y_test)
    print(f'Achieved accuracy: {acc}')
    network.save_params('learned_params')
