import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)


    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    gda = GDA(verbose=True)
    gda.fit(x_train, y_train)

    prediction = gda.predict(x_eval)

    plot_path = pred_path + ".plot.png"
    util.plot(x_eval, y_eval, gda.theta, plot_path, correction=1.0)

    np.savetxt(pred_path, prediction)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        m = x.shape[0]
        n = x.shape[1]

        y_is_1 = y==1
        y_is_0 = y==0
        phi = (y_is_1).sum() / m
        mu_0 = x[y_is_0].sum(axis=0)/y_is_0.sum()
        mu_1 = x[y_is_1].sum(axis=0)/y_is_1.sum()
        x_minus_mu = x.copy()
        x_minus_mu[y_is_0] -= mu_0
        x_minus_mu[y_is_1] -= mu_1
        sigma = (x_minus_mu.T @ x_minus_mu)

        #to build theta and theta_0 defired in problem 1c
        sigma_inv = np.linalg.inv(sigma)
        theta = (mu_1 - mu_0).T @ sigma_inv
        theta_0 = 1/2 * (mu_0.T @ sigma_inv @ mu_0 - mu_1.T @ sigma_inv @ mu_1) + np.log((1-phi)/phi)

        self.theta = np.hstack([theta_0, theta])
        return self.theta    
        

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (1/(1+np.exp(-np.inner(self.theta,x))) >= 0.5).astype(np.int)
        # *** END CODE HERE
