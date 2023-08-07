import pandas as pd 
import numpy as np 
import math
import json
from numpy.polynomial import polynomial


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.5f}'.format


# The colors class defines constants for different terminal colors.
class colors:
    HEADER = '\033[95m'       # pink
    OKBLUE = '\033[94m'       # blue
    OKCYAN = '\033[96m'       # cyan
    OKGREEN = '\033[92m'      # green
    WARNING = '\033[93m'      # yellow
    FAIL = '\033[91m'         # red
    ENDC = '\033[0m'          # reset color
    BOLD = '\033[1m'          # bold
    UNDERLINE = '\033[4m'     # underline


class OLPS:
    """
    A class for online portfolio selection (OLPS) functions.
    make sure to add: from numpy.polynomial import polynomial.

    Toolkits available:
    1. Exponential Gradient (EG): exponential_gradient_weights(df)
    2. Online Newton Step (ONS): ONS_weights(df)
    3. Covers Two Stocks: CoversUP_Two_Stocks(df)
    4. Exponentiated Gradient Tilting (EGT): egt_weights(df)
    5. Anti-Correlation Dynamic Time Warping: (AC-DTW) anticor_dwt_weights(df)
    6. Follow The Linearized Leader (FTLR): ftlr_weights(df)
    7. Follow The Linearized Stable Leader: ftlr_stable_weights(df)
    """
    
    def exponential_gradient_weights(self, original_df, peta=0.08):
        """
        A function that computes the exponential gradient algorithm for online portfolio selection.
        
        Parameters:
        original_df (pd.DataFrame): a dataframe of prices for each asset.
        peta (float): the learning rate for the algorithm.
        
        Returns:
        pd.DataFrame: a dataframe of portfolio weights for each asset at each time step.
        """
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1))
        n_SYMBOLS = len(prices_df.columns)
        b_t = np.ones(n_SYMBOLS) / n_SYMBOLS

        eta = peta 
        reg_param = lambda b, b_t: np.sum(b * np.log(b / b_t))

        def update_weights(x_t, b_t, eta, reg_param):
            grad = x_t / np.dot(b_t, x_t)
            b_new = b_t * np.exp(eta * grad - reg_param(b_t, b_t))
            return b_new / np.sum(b_new)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))
        weights[0] = b_t
        for i in range(1, n_days):
            weights[i] = update_weights(log_returns.iloc[i], weights[i-1], eta, reg_param)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df


    def ONS_weights(self, original_df, delta=0.125, beta=1.0, eta=0.0):
        """
        A function that computes the online Newton step algorithm for online portfolio selection.
        
        Parameters:
        original_df (pd.DataFrame): a dataframe of prices for each asset.
        delta (float): the regularization parameter for the algorithm (default: 0.125).
        beta (float): the scaling parameter for the update step (default: 1.0).
        eta (float): the weight for the uniform prior (default: 0.0).
        
        Returns:
        pd.DataFrame: a dataframe of portfolio weights for each asset at each time step.
        """
        prices_df = original_df.copy()
        log_returns = np.log(prices_df / prices_df.shift(1))
        n_SYMBOLS = len(prices_df.columns)

        m = n_SYMBOLS
        A = np.mat(np.eye(m))
        b = np.mat(np.zeros(m)).T

        def frank_wolfe_quadratic_programming(P, q):
            m = q.shape[0]
            x = np.zeros((m, 1))
            max_iter = 1000

            for _ in range(max_iter):
                gradient = P @ x + q
                idx = np.argmin(gradient)
                d = np.zeros((m, 1))
                d[idx] = 1.0
                step_size = 2.0 / (_ + 2)
                x = (1 - step_size) * x + step_size * d

            return x

        def projection_in_norm(x, M):
            P = 2 * M
            q = -2 * M * x
            result = frank_wolfe_quadratic_programming(P, q)
            return np.squeeze(np.asarray(result))

        def update_weights(r, p):
            grad = np.mat(r / np.dot(p, r)).T
            A_new = A + grad * grad.T
            b_new = b + (1 + 1.0 / beta) * grad

            pp = projection_in_norm(delta * A_new.I * b_new, A_new)
            return np.squeeze(np.asarray(pp * (1 - eta) + np.ones(len(r)) / float(len(r)) * eta))

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))
        weights[0] = np.ones(n_SYMBOLS) / n_SYMBOLS

        for i in range(1, n_days):
            weights[i] = update_weights(log_returns.iloc[i], weights[i-1])

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df


    def CoversUP_Two_Stocks(self, r):
        """
        A function that implements the CoversUP algorithm for two stocks.
        
        Parameters:
        r (ndarray): a numpy array of stock returns for two assets.
        
        Returns:
        ndarray: a numpy array of portfolio weights for two assets.
        """
        T = len(r)
        n = len(r[0])
        assert n == 2, "CoversUP_Two_Stocks function can only be applied to a numpy array with two assets."
        x = np.zeros((T, n))
        prod_returns = np.array([1.0])
        m = 1
        for t in range(T):
            numerator = polynomial.polymulx(prod_returns)
            numerator_integral = polynomial.polyint(numerator)

            denominator_integral = polynomial.polyint(prod_returns)

            num = polynomial.polyval(1, numerator_integral) - polynomial.polyval(0, numerator_integral)
            denum = polynomial.polyval(1, denominator_integral) - polynomial.polyval(0, denominator_integral)

            p = num / denum
            x[t][0] = p
            x[t][1] = 1.0 - p

            new_return_polynomial = np.array([r[t][1], r[t][0] - r[t][1]])
            prod_returns = polynomial.polymul(prod_returns, new_return_polynomial)
        return x


    def covers_up_weights(self, original_df):
        """
        A function that computes the CoversUP algorithm for two stocks on a DataFrame of prices.
        
        Parameters:
        original_df (pd.DataFrame): a dataframe of prices for two assets.
        
        Returns:
        pd.DataFrame: a dataframe of portfolio weights for two assets at each time step.
        """
        prices_df = original_df.copy()
        assert len(prices_df.columns) == 2, "CoversUP_Two_Stocks function can only be applied to a DataFrame with two assets."

        pct_returns = prices_df.pct_change().dropna().values
        T, n = pct_returns.shape

        weights = np.zeros((T, n))

        for t in range(T):
            x = self.CoversUP_Two_Stocks(pct_returns[:t + 1])
            weights[t] = x[-1]

        weights_df = pd.DataFrame(weights, columns=prices_df.columns, index=prices_df.index[1:])
        return weights_df

    def exp_normalize(self, x):
        """
        Applies exponential normalization to an input array.

        Parameters:
        x (numpy.ndarray): An array to be normalized.

        Returns:
        numpy.ndarray: The normalized array.
        """
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()

    def EG_tilde(self,r):
        """
        Computes the Exponentiated Gradient Tilting (EGT) algorithm for online portfolio selection.

        Parameters:
        r (numpy.ndarray): A 2D array of stock returns.

        Returns:
        numpy.ndarray: A 2D array of portfolio weights for each stock at each time step.
        """
        T = r.shape[0]
        n = r.shape[1]
        if T==0:
            return np.ones(n)/n

        x = np.zeros((T,n))
        tilde_x_t_plus_1 = np.ones(n)/n
        x_t_plus_1 = np.ones(n)/n
        theta = 0
        for t in range(T):
            x[t] = tilde_x_t_plus_1

            alpha_t = (n*n*np.log(n)/(t+1))**0.25

            eta_t = alpha_t/n * np.sqrt(np.log(n)/(t+1))

            tilde_r = (1-alpha_t/n)*r[t] + alpha_t/n

            gradient = - tilde_r/np.dot(tilde_r,x_t_plus_1)

            theta = theta - gradient

            x_t_plus_1 = self.exp_normalize(eta_t*theta)
            tilde_x_t_plus_1 = (1-alpha_t)*x_t_plus_1 + alpha_t/n

        return x[-1]


    def egt_weights(self, original_df):
        """
        Computes the Exponentiated Gradient Tilting (EGT) algorithm for online portfolio selection on a DataFrame of prices.

        Parameters:
        original_df (pandas.DataFrame): A DataFrame of prices for each asset.

        Returns:
        pandas.DataFrame: A DataFrame of portfolio weights for each asset at each time step.
        """
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.EG_tilde(log_returns.iloc[:i].values)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df

    def euclidean_distances(self,X, Y):
        X = X[:, np.newaxis, :]
        Y = Y[np.newaxis, :, :]
        return np.sqrt(np.sum((X - Y) ** 2, axis=-1))

    def euclidean_cdist(self,XA, XB):
        """
        Description: This function computes the Euclidean distance between 
        pairs of points in two arrays. It takes in two arrays, XA and XB, as 
        input and returns a 2D array, dm, where the element dm[i,j] is the 
        Euclidean distance between XA[i] and XB[j].

        Parameters:

        XA: numpy array of shape (m, n) representing the first set of points
        XB: numpy array of shape (p, n) representing the second set of points
        Returns:

        dm: 2D numpy array of shape (m, p) representing the pairwise Euclidean 
        distances between points in XA and XB

        """
        XA = np.asarray(XA, order='c')
        XB = np.asarray(XB, order='c')

        if len(XA.shape) == 1:
            XA = XA[:, None]
        if len(XB.shape) == 1:
            XB = XB[:, None]

        mA, n = XA.shape
        mB, n = XB.shape

        dm = np.empty((mA, mB), dtype=np.double)

        for i in range(mA):
            for j in range(mB):
                dm[i, j] = np.linalg.norm(XA[i] - XB[j])

        return dm

    def dtw_distance(self,X, Y):
        """
        Description: This function computes the Dynamic Time Warping (DTW) distance 
        between two time series X and Y. It takes in two arrays, X and Y, as input 
        and returns a scalar value representing the DTW distance between them.

        Parameters:

        X: numpy array of shape (n,) representing the first time series
        Y: numpy array of shape (m,) representing the second time series
        Returns:

        cost[-1,-1]: scalar value representing the DTW distance between X and Y
        """
        n, m = len(X), len(Y)
        cost = np.zeros((n+1, m+1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        #cost[1:, 1:] = self.euclidean_cdist(X, Y)
        cost[1:, 1:] = self.euclidean_distances(X,Y)
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost[i, j] += min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        return cost[-1, -1]

    def anticor_dtw(self,r, window):
        """
        Description: This function computes the Anti-Correlation Dynamic Time Warping (AC-DTW) 
        weights for a given time series. It takes in a 2D numpy array, r, representing the log 
        returns of a set of assets, and a window size, and returns a 2D numpy array, x, 
        representing the AC-DTW weights for the time series.

        Parameters:

        r: 2D numpy array of shape (T, n) representing the log returns of a set of assets
        window: integer value representing the size of the window used to compute the AC-DTW weights
        Returns:

        x[-1]: 1D numpy array of shape (n,) representing the AC-DTW weights for the last time 
        step in the time series
        """
        T, n = r.shape
        x = np.zeros((T, n))

        for t in range(T):
            if t == 0:
                x[t] = np.ones(n) / n
            else:
                c = np.zeros(n)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            X = r[max(0, t - window):t, i].reshape(-1, 1)
                            Y = r[max(0, t - window):t, j].reshape(-1, 1)
                            c[i] += self.dtw_distance(X, Y)
                x[t] = np.exp(-c) / np.sum(np.exp(-c))
        return x[-1]

    def anticor_dwt_weights(self,original_df ,window):
        """
        Description: This function computes the Anti-Correlation Dynamic Time Warping (AC-DTW) weights 
        for a given set of asset prices. It takes in a pandas DataFrame, original_df, representing the 
        asset prices, and a window size, and returns a pandas DataFrame, weights_df, representing the 
        AC-DTW weights for the time series.

        Parameters:

        r: pandas DataFrame of shape (T, n) representing the asset prices
        original_df: pandas DataFrame of shape (T, n) representing the original asset prices
        window: integer value representing the size of the window used to compute the AC-DTW weights
        Returns:

        weights_df: pandas DataFrame of shape (T-1, n) representing the AC-DTW weights for the time series, 
        excluding the first time step.
        """
        prices_df = original_df.copy()
        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.anticor_dtw(log_returns.iloc[:i].values, window)
            
        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df

    def Follow_The_Linearized_Leader(self,r):
        """
        Description: The Follow_The_Linearized_Leader function implements the Follow-The-Leader (FTL) 
        algorithm by selecting the asset with the largest value of an estimated parameter. The function 
        takes a 2-dimensional numpy array r as input, where each row represents the log returns of the assets 
        for a single day. The function returns a 2-dimensional numpy array x of the same shape as r, where
        each row represents the portfolio weights for the corresponding day.

        Parameters: 
        r: 2-dimensional numpy array of shape (T, n), where T is the number of days 
        n:  The number of assets. The array contains the log returns of the assets for each day.

        Returns:
        x : 2-dimensional numpy array of shape (T, n), where each row represents the portfolio weights for the corresponding day.
        """
        T = len(r)
        n = len(r[0])
        x = np.zeros((T,n))
        theta = 0
        gradients = 0
        for t in range(T):
            x[t][np.argmax(theta)]=1.0
            gradient = - r[t]/np.dot(r[t],x[t])
            theta = theta - gradient
        return x[-1]

    def ftlr_weights(self, original_df):
        """
        Description: The ftlr_weights function uses the Follow_The_Linearized_Leader function to compute portfolio
        weights based on the FTL algorithm. The function takes a pandas DataFrame original_df as input, where 
        each column represents the daily prices of a single asset. The function returns a pandas DataFrame weights_df 
        of the same shape as original_df, where each column represents the portfolio weights for the corresponding asset, 
        and each row represents the date for which the weights are computed.

        Parameters: 
        original_df: pandas DataFrame of shape (T+1, n), where T is the number of days and n is the 
        number of assets. The DataFrame contains the daily prices of the assets.

        Returns:
        pandas DataFrame of shape (T, n), where each column represents the portfolio weights for the 
        corresponding asset, and each row represents the date for which the weights are computed.
        """
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.Follow_The_Linearized_Leader(log_returns.iloc[:i].values)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df

    def Follow_The_Linearized_Stable_Leader(self, r, usdt_index, bearish_threshold=0.8):
        """
        Description: The Follow_The_Linearized_Stable_Leader function implements Strategic Follow-The-Leader (FTL) 
        algorithm by selecting the asset with the largest value of an estimated parameter and allocating to stables on downturn.
        The function takes a 2-dimensional numpy array r as input, where each row represents the log returns of the assets 
        for a single day. The function returns a 2-dimensional numpy array x of the same shape as r, where
        each row represents the portfolio weights for the corresponding day.

        Parameters: 
        r: 2-dimensional numpy array of shape (T, n), where T is the number of days 
        n:  The number of assets. The array contains the log returns of the assets for each day.

        Returns:
        x : 2-dimensional numpy array of shape (T, n), where each row represents the portfolio weights for the corresponding day.
        """
        T = len(r)
        n = len(r[0])
        x = np.zeros((T, n))
        theta = np.zeros(n)

        gradients = 0
        
        for t in range(T):
            total_return = np.sum(r[t])
            
            if total_return < 0:  # Check if the total return for the day is negative
                x_no_usdt = np.zeros(n - 1)
                x_no_usdt[np.argmax(theta[:-1])] = 1.0
                x[t, :-1] = (1 - bearish_threshold) * x_no_usdt
                x[t, usdt_index] = bearish_threshold
            else:
                x[t][np.argmax(theta)] = 1.0
                
            gradient = - r[t] / np.dot(r[t], x[t])
            theta = theta - gradient
            
        return x[-1]

    def ftlr_stable_weights(self, original_df):
        """
        Description: The ftlr_stable_weights function uses the Follow_The_Linearized_Stable_Leader function to compute portfolio
        weights based on the FTL algorithm. The function takes a pandas DataFrame original_df as input, where 
        each column represents the daily prices of a single asset. The function returns a pandas DataFrame weights_df 
        of the same shape as original_df, where each column represents the portfolio weights for the corresponding asset, 
        and each row represents the date for which the weights are computed.

        Parameters: 
        original_df: pandas DataFrame of shape (T+1, n), where T is the number of days and n is the 
        number of assets. The DataFrame contains the daily prices of the assets.

        Returns:
        pandas DataFrame of shape (T, n), where each column represents the portfolio weights for the 
        corresponding asset, and each row represents the date for which the weights are computed.
        """
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.Follow_The_Linearized_Stable_Leader(log_returns.iloc[:i].values, usdt_index=original_df.columns.get_loc('BUSDUSDT'))

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df

    def proj_simplex(self,y, A):
        m = len(y)
        p = np.ones(m) / m
        max_iter = 1000
        eps = 1e-8

        for _ in range(max_iter):
            grad = A @ (y - p)
            idx = np.argmax(grad)
            step_size = min(p[idx], grad[idx] / A[idx, idx])

            if step_size < eps:
                break

            p -= step_size * np.eye(m)[idx]

        return p

    def ONS_SI(self,r, y, c, eta, beta, delta):
        T = r.shape[0]
        m = r.shape[1]

        p_tilde = np.ones(m) / m
        p_bar_prev = np.zeros(m)
        W_c_prev = 1
        wealth = []

        for t in range(T):
            y_t = y[t]
            p_hat_t = p_tilde if y_t == 1 else np.zeros(m)

            transaction_cost = c * np.sum(np.abs(p_hat_t - p_bar_prev))
            W_c_t = W_c_prev * (1 - transaction_cost)

            r_t = r[t]

            if y_t == 1:
                w_t = np.dot(p_hat_t, r_t)
                p_bar_t = (1 / w_t) * (p_hat_t * r_t)
            else:
                w_t = 1
                p_bar_t = p_bar_prev

            W_c_t = W_c_t * w_t

            wealth.append(W_c_t)

            b_t = (1 + 1 / beta) * np.sum(r[:t+1], axis=0)
            A_t = np.dot(r[:t+1].T, r[:t+1]) + np.eye(m)

            p_tilde_next = (1 - eta) * proj_simplex(p_bar_t + delta * A_t @ b_t, A_t) + eta / m * np.ones(m)

            p_tilde = p_tilde_next
            p_bar_prev = p_bar_t
            W_c_prev = W_c_t

        return wealth[-1]

    def ONS_SI_weights(self, original_df,r, y, c, eta, beta, delta):
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.ONS_SI(r, y, c, eta, beta, delta)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df
