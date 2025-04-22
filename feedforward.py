import numpy as np

class Network:
    def __init__(self, input=784, hidden1=200, hidden2=80, output=10, alpha=1e-4, batch_size=32):
        
        #layer sizes
        self.input = input          #784
        self.hidden1 = hidden1      #200
        self.hidden2 = hidden2      #80
        self.output = output        #10

        #weights and biases
        self.w1 = np.random.rand(self.input, self.hidden1)         #(784,200)
        self.w2 = np.random.rand(self.hidden1, self.hidden2)       #(200,80)
        self.w3 = np.random.rand(self.hidden2, self.output)        #(80,10)
        self.b1 = np.random.rand(1, self.hidden1)                     #(1,200)
        self.b2 = np.random.rand(1, self.hidden2)                     #(1,80)
        self.b3 = np.random.rand(1, self.output)                      #(1,10)

        #gradients
        self.dw1 = np.zeros((self.input, self.hidden1))         #(784,200)
        self.dw2 = np.zeros((self.hidden1, self.hidden2))       #(200,80)
        self.dw3 = np.zeros((self.hidden2, self.output))        #(80,10)
        self.db1 = np.zeros((1, self.hidden1))                   #(1,200)
        self.db2 = np.zeros((1, self.hidden2))                   #(1,80)
        self.db3 = np.zeros((1, self.output))                    #(1,10)

        #activations
        self.z1 = np.zeros((1, self.hidden1))         #(1,200)
        self.z2 = np.zeros((1, self.hidden2))         #(1,80)
        self.z3 = np.zeros((1, self.output))          #(1,10)
        self.a1 = np.zeros((1, self.hidden1))         #(1,200)
        self.a2 = np.zeros((1, self.hidden2))         #(1,80)
        self.a3 = np.zeros((1, self.output))          #(1,10)

        #normalized activations
        self.z1_hat = np.zeros((1, self.hidden1))     #(1,200)
        self.z2_hat = np.zeros((1, self.hidden2))     #(1,80)
        self.mu1 = 0
        self.mu2 = 0
        self.sigma1 = 0
        self.sigma2 = 0

        #hyperparameters
        self.alpha = alpha
        self.batch_size = batch_size

        self.cost = []

    def normalize_layer(self, x):
        # Normalize the input layer and handle edge cases
        stdx = np.std(x)
        meanx = np.mean(x)
        if np.abs(stdx) < 1e-10:
            return meanx, stdx, x
        else:
            x = (x - meanx) / stdx
            return meanx, stdx, x
    
    def relu(self, x):
        return np.maximum(0.0, x)
    
    def drelu_square(self, z):
        # Derivative of ReLU: 1 if z > 0, else 0
        drelu = np.zeros((z.shape[1], z.shape[1]))
        for i in range(z.shape[1]):
            drelu[i, i] = 1 if z[0,i] > 0 else 0
        return drelu
    
    def drelu(self, z):
        # Derivative of ReLU: 1 if z > 0, else 0
        drelu = np.zeros((1, z.shape[1]))
        for i in range(z.shape[1]):
            drelu[0, i] = 1 if z[0,i] > 0 else 0
        return drelu
    
    def softmax(self, x):
        # Compute the softmax of the input
        numerator = np.exp(x)
        denominator = np.sum(numerator)
        return numerator / denominator
    
    def norm_backprop2(self, z_hat, z, sigma, mu):
        assert z.shape == z_hat.shape, f"Shapes of z {z.shape} and z_hat {z_hat.shape} do not match"
        m = z.shape[1]

        dz_hat_dz = np.zeros((m, m))

        #dz_hat_i/dz_j = d/dz_j(1/sig) * (zi-mu) + 1/sig * (d/dz_j(zi-mu))
        #dz_hat_i/dz_j = 1/(sqrt(m)*sig)*2(sum(z-mu)*(del_ij - 1/m))*(zi-mu) + 1/(sig)*(del_ij - 1/m)

        for i in range(m):
            for j in range(m):
                del_ij = 1 if i==j else 0
                d_1_over_sig_dzj = (1/(np.sqrt(m)*sigma))*2*(np.sum(z-mu)*(del_ij - 1/m))
                dz_hat_dz[i,j] = d_1_over_sig_dzj*(z[0,i]-mu) + (1/sigma)*(del_ij - 1/m)

        return dz_hat_dz
    
    def norm_backprop(self, z_hat, z, sigma, mu):
        assert z.shape == z_hat.shape, f"Shapes of z {z.shape} and z_hat {z_hat.shape} do not match"
        m = z.shape[1]

        dz_hat_dz = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                del_ij = 1 if i==j else 0
                dz_hat_dz[i,j] = (del_ij - 1/m)/(sigma) - (z[0,i]-mu)*(z[0,j]-mu)/(m*sigma**3)

        return dz_hat_dz
    
    def forward(self, x):
        x = x.reshape(1, self.input)
        # Forward pass
        self.z1 = x @ self.w1 + self.b1
        self.mu1, self.sigma1, self.z1_hat = self.normalize_layer(self.z1)
        self.a1 = self.relu(self.z1_hat)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.mu2, self.sigma2, self.z2_hat = self.normalize_layer(self.z2)
        self.a2 = self.relu(self.z2_hat)
        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward(self, y, x):
        y = y.reshape(1, self.output)
        x = x.reshape(1, self.input)

        #Cross entropy loss = -sum(y*log(y_pred))
        self.cost.append(-np.sum(y * np.log(self.a3 + 1e-10)) / y.shape[0])  # Avoid log(0) by adding a small constant

        #input to hidden1:      in*w1+b1-->z1--norm-->z1_hat-->relu-->a1
        #hidden1 to hidden2:    a1*w2+b2-->z2--norm-->z2_hat-->relu-->a2
        #hidden2 to output:     a2*w3+b3-->z3--softmax-->a3-->cost

        #-------------------------------output to hidden2--------------------------------------
        dz3_db3 = 1                         #(1,)
        dz3_da2 = self.w3.T                 #(10,80)
        dC_dz3 = self.a3 - y                #(1,10)
        dC_dw3 = self.a2.T @ dC_dz3         #(80,10)
        dC_db3 = dz3_db3 * dC_dz3           #(1,10)
        dC_da2 = dC_dz3 @ dz3_da2           #(1,80)

        #-------------------------------hidden2 to hidden1--------------------------------------
        da2_dz2_hat = self.drelu(self.z2_hat)   #(1,80)
        dz2_hat_dz2 = self.norm_backprop(self.z2_hat, self.z2, self.sigma2, self.mu2)  #(80,80)
        dz2_db2 = 1                             #(1,)
        dz2_da1 = self.w2.T                     #(80,200)
        dC_dz2_hat = dC_da2 * da2_dz2_hat       #(1,80)
        dC_dz2 = dC_dz2_hat @ dz2_hat_dz2       #(1,80)
        dC_dw2 = self.a1.T @ dC_dz2             #(200,80)
        dC_db2 = dz2_db2 * dC_dz2               #(1,80)
        dC_da1 = dC_dz2 @ dz2_da1               #(1,200)

        #-------------------------------hidden1 to input--------------------------------------
        da1_dz1_hat = self.drelu(self.z1_hat)
        dz1_hat_dz1 = self.norm_backprop(self.z1_hat, self.z1, self.sigma1, self.mu1)
        dC_dz1_hat = dC_da1 * da1_dz1_hat       #(1,200)
        dC_dz1 = dC_dz1_hat @ dz1_hat_dz1       #(1,200)
        dz1_db1 = 1                             #(1,)
        dC_dw1 = x.T @ dC_dz1                  #(784,200)
        dC_db1 = dz1_db1 * dC_dz1               #(1,200)

        self.dw1 += dC_dw1
        self.dw2 += dC_dw2
        self.dw3 += dC_dw3
        self.db1 += dC_db1
        self.db2 += dC_db2
        self.db3 += dC_db3

        return self.cost[-1]
    
    def update(self, batch_size):
        # Update weights and biases using gradients
        self.w1 -= self.alpha * self.dw1 / batch_size
        self.w2 -= self.alpha * self.dw2 / batch_size
        self.w3 -= self.alpha * self.dw3 / batch_size
        self.b1 -= self.alpha * self.db1 / batch_size
        self.b2 -= self.alpha * self.db2 / batch_size
        self.b3 -= self.alpha * self.db3 / batch_size

    def zero_gradients(self):
        # Reset gradients to zero after updating
        self.dw1.fill(0)
        self.dw2.fill(0)
        self.dw3.fill(0)
        self.db1.fill(0)
        self.db2.fill(0)
        self.db3.fill(0)

