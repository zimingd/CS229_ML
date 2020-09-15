import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)


    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path

    initial_theta = np.zeros(x_train.shape[1])
    log_reg = LogisticRegression(step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=initial_theta, verbose=True)
    log_reg.fit(x_train, y_train)

    prediction = log_reg.predict(x_eval)

    plot_path = pred_path + ".plot.png"
    util.plot(x_eval, y_eval, log_reg.theta, plot_path, correction=1.0)

    np.savetxt(pred_path, prediction)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        iterations = 0
        # use initial theta
        theta = self.theta
        #arbitary addition to ensure we fail the L1 norm < epislon check on first iteration
        theta_prev = self.theta + 1

        while iterations < self.max_iter and np.linalg.norm(theta-theta_prev,1) >= self.eps:
            gradient = np.zeros(n)
            hessian = np.zeros((n,n))

            for i in range(m):
                x_i = x[i]

                h_theta = self._h(theta, x_i)  
                gradient += (y[i]-h_theta) * x_i
                hessian += h_theta * (1-h_theta) * np.outer(x_i, x_i)
            #apply the 1/m at end of loop
            gradient /= -m # negative b/c we did y[i] - h_theta instead of h_theta - y_i 
            hessian /= m
            
            #update theta
            theta_prev = theta
            theta = theta - np.linalg.solve(hessian, gradient) # same as inv(hessian) x gradient

            iterations += 1
        self.theta = theta
        print(theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self._h(self.theta, x)
        # *** END CODE HERE ***

    def _h(self, theta, x):
        return 1/(1+np.exp(-np.inner(theta,x)))