import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path

    initial_theta = np.zeros(x_train.shape[1])
    #took 2008 iterations to converge using the given learning rate and default epsilon value
    poisson_reg = PoissonRegression(step_size=lr, max_iter=10000, theta_0=initial_theta, verbose=True)
    poisson_reg.fit(x_train, y_train)

    prediction = poisson_reg.predict(x_eval)

    np.savetxt(pred_path, prediction)

    # comparing prediction and y_eval copied from:
    # https://scikit-learn.org/0.16/auto_examples/plot_cv_predict.html
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.scatter(y_eval, prediction)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.savefig(pred_path + ".comparision.png")
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        iterations = 0
        # use initial theta
        theta = self.theta
        #arbitary addition to ensure we fail the L1 norm < epislon check on first iteration
        theta_prev = self.theta + 1

        while iterations < self.max_iter and np.linalg.norm(theta-theta_prev,1) >= self.eps:
            theta_prev = theta
            # this is a batch gradient ascent operation that will operate on all m datapoints
            # dot product here performs the summation
            theta = theta + self.step_size / m * (y-self._h(theta, x)) @ x
            iterations += 1
        self.theta = theta
        # # *** END CODE HERE ***
        
    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self._h(self.theta, x)
        # *** END CODE HERE ***

    def _h(self, theta, x):
        return np.exp(np.inner(theta, x))