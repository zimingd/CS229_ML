import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression

import matplotlib.pyplot as plt


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)



    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data

    min_mse_index = 0
    min_mse = 999999999999

    for i, tau in enumerate(tau_values):
        locally_weighted_regression = LocallyWeightedLinearRegression(tau)
        locally_weighted_regression.fit(x_train,y_train)
        y_pred = locally_weighted_regression.predict(x_valid)
        mean_squared_error = ((y_valid-y_pred) ** 2).mean()
        
        if(mean_squared_error < min_mse):
            min_mse = mean_squared_error
            min_mse_index = i

        plot(tau, mean_squared_error, x_train, y_train, x_valid, y_pred)

    tau = tau_values[min_mse_index]
    locally_weighted_regression = LocallyWeightedLinearRegression(tau)
    locally_weighted_regression.fit(x_train,y_train)
    y_pred = locally_weighted_regression.predict(x_test)
    mean_squared_error = ((y_test-y_pred) ** 2).mean()
    plot(tau, mean_squared_error, x_train, y_train, x_valid, y_pred, test_set=True)
    np.savetxt(pred_path, y_pred)


    # *** END CODE HERE ***

def plot(tau, mean_squared_error, x_train, y_train, x_valid, y_pred, * , test_set=False):
    fig,ax = plt.subplots()
    ax.plot(x_train, y_train, 'bx')
    ax.plot(x_valid, y_pred, 'ro')
    ax.set_title(f'tau = {tau}, MSE = {mean_squared_error}')
    fig.savefig(f'output/p05c_tau_{tau}{"_final" if test_set else ""}.png')