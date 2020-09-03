import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    initial_theta = np.zeros(x_train.shape[1])
    log_reg_c = LogisticRegression(theta_0=initial_theta)
    log_reg_c.fit(x_train, y_train)
    util.plot(x_test, y_test, log_reg_c.theta, pred_path_c + ".png")
    y_pred = log_reg_c.predict(x_test)
    np.savetxt(pred_path_c, y_pred)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    initial_theta = np.zeros(x_train.shape[1])
    log_reg_d = LogisticRegression(theta_0=initial_theta)
    log_reg_d.fit(x_train, y_train)
    util.plot(x_test, y_test, log_reg_d.theta, pred_path_d + ".png")
    y_pred = log_reg_d.predict(x_test)
    np.savetxt(pred_path_d, y_pred)

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
    
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    alpha = log_reg_d.predict(x_valid[y_valid == 1]).mean()
    
    util.plot(x_test, y_test, log_reg_d.theta, pred_path_e + ".png", correction=alpha)

    y_pred_corrected = y_pred/alpha
    np.savetxt(pred_path_e, y_pred_corrected)

