import numpy as np
import util
import math


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_eval)
    util.plot(x_eval,y_eval,clf.theta,save_path+".png")
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == y_eval[i]:
            right+=1
    print("accuracy: ", right/len(y_eval))
    np.savetxt(save_path,predictions) 


    # *** START CODE HERE ***

    # Train a GDA classifier

    # Plot decision boundary on validation set

    # Use np.savetxt to save outputs from validation set to save_path

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        phi = 0
        sigma = None#np.matrix([[0.0,0.0], [0.0,0.0]])
        if self.theta == None:
            self.theta = np.zeros(len(x[0]))

        totalzeroY = 0
        totalOneY = 0

        mu0 = np.zeros(len(x[0]))
        mu1 = np.zeros(len(x[0]))
        for i in range(len(x)):                   
            if y[i] == 0:
                totalzeroY += 1
                mu0 += x[i]      
            else:
                totalOneY += 1
                mu1+= x[i]
        mu0/= totalzeroY
        mu1/= totalOneY
        phi = sum(y)/len(y)
        if y[0] == 1:
            sigma = np.outer((x[0] - mu1),(x[0] - mu1).T)
        else:
            sigma = np.outer((x[0] - mu0),(x[0] - mu0).T)
        for i in range(1,len(x)):
            if y[i] == 1:
                sigma += np.outer((x[i] - mu1),(x[i] - mu1).T)
            else:
                sigma += np.outer((x[i] - mu0),(x[i] - mu0).T)
        sigma/= len(x)
        print("mu0: ",mu0)
        print("mu1: ",mu1)
        print("phi: ",phi)
        print("sigma: ",sigma)
        self.theta = np.dot((mu1-mu0).T,np.linalg.inv(sigma))
        tmp = np.dot((mu1-mu0).T,np.linalg.inv(sigma))
        numerator = np.dot(tmp,(mu1+mu0))
        theta_0 = np.log(phi/(1-phi)) - numerator/2
        self.theta = np.append(theta_0,self.theta)




            

            # if np.linalg.norm((newTheta - self.theta), ord=1) <= self.eps:
            #     self.theta = newTheta
            #     #self.intercept = theta_0
            #     break
            # else:
            #     self.theta = newTheta
            #     #self.intercept = theta_0

            



        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # *** END CODE HERE ***

    def predict(self, x):
        predictions = []
        for inp in x:
            h = 1/(1+np.exp(-(np.dot(self.theta.T[1:],inp)+self.theta[0])))
            if h > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        print("THETA: ",self.theta)

        return predictions



        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')



