import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)


    # *** START CODE HERE ***
    # Fit a LWR model
    assert tau == 0.5
    locally_weighted_regression = LocallyWeightedLinearRegression(tau)
    locally_weighted_regression.fit(x_train,y_train)

    y_pred = locally_weighted_regression.predict(x_eval)

    # Get MSE value on the validation set
    mean_squared_error = ((y_eval-y_pred) ** 2).mean()
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(x_train, y_train, 'bx')
    ax.plot(x_eval, y_pred, 'ro')
    ax.set_title(f'tau = {tau}, MSE = {mean_squared_error}')
    fig.savefig('output/p05b_lwr.png')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x 
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        m,n = self.x.shape
        

        y_pred = np.zeros(x.shape[0])


        #there's probably a more efficient way to iterate and predict each y using batch numpy operations but my tiny brain is burned out and can't think about this anymore :(
        for i, x_i in enumerate(x):
            # each diagonal W is unique to the current data point(x_i or x[i]) we are attempting to predict
            # based on the current datapoint's distance from each point in our training set (self.x[j])
            W = np.diag( np.exp( - np.linalg.norm(x_i-self.x,  ord=2, axis=1)**2 / (2 * self.tau**2) ) )

            # X (capitalized) refers to our training set, self.x
            # only the final predication step and the earlier calculation of W makes use of our input (x)

            #solve for theta via  X^T W X theta = X^T W y
            theta = np.linalg.solve(self.x.T @ W @ self.x, self.x.T @ W @ self.y)
            y_pred[i] = theta.T @ x_i



        
        # omiga =np.exp(-np.sum((self.x - x[188])**2/(2*self.tau**2), axis = 1))
        # print(omiga)
        # print(np.sum((self.x - x[188])**2/(2*self.tau**2), axis=1).shape)
        # # print(   (np.linalg.norm(self.x - x[188], 2)**2)/(2*self.tau**2)  )
        # print ( -(np.linalg.norm(x[188]-self.x, ord=2, axis=1)**2 )/ (2 * self.tau**2)   )

        
        # W = np.exp( -(np.linalg.norm(x[188]-self.x,  ord=2, axis=1)**2 )/ (2 * self.tau**2) )
        # print(W)
        # print(np.array_equal(omiga, W))
        
       
        return y_pred
        