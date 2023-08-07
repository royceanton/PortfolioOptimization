import numpy as np
import pandas as pd

class PortfolioOptimizer:
    """
    Class PortfolioOptimizer:
        This class provides methods for optimizing portfolios using various 
        techniques, such as following the leader and modified smooth prediction.

        Method Best_Constantly_Rebalanced_Portfolio:
        This method computes the best constantly rebalanced portfolio given a matrix 
        of historical returns. It uses an augmented Lagrangian method to optimize a 
        logarithmic objective function subject to a constraint on the sum of the portfolio weights.

        Method Follow_The_Leader:
        This method implements the "follow-the-leader" algorithm, which constructs a 
        portfolio that mimics the best constantly rebalanced portfolio at each time step. 
        It uses the Best_Constantly_Rebalanced_Portfolio method to compute the portfolio weights.

        Method Follow_The_Leader_weights:
        This method computes the portfolio weights for the "follow-the-leader" strategy 
        using the log returns of a dataframe of asset prices. It uses the Follow_The_Leader 
        method to construct the portfolio weights.

        Method Modified_Smooth_Prediction:
        This method implements the "modified smooth prediction" algorithm, which constructs
        a portfolio that balances the best constantly rebalanced portfolio and a modified 
        version of the return vector. It uses the Best_Constantly_Rebalanced_Portfolio 
        method to compute the portfolio weights.

        Method Modified_Smooth_Prediction_weights:
        This method computes the portfolio weights for the "modified smooth prediction" 
        strategy using the log returns of a dataframe of asset prices. It uses the 
        Modified_Smooth_Prediction method to construct the portfolio weights.

    """
    def __init__(self):
        pass

    def Best_Constantly_Rebalanced_Portfolio(self, r, max_iter=10):
        T = r.shape[0]
        n = r.shape[1]
        if T == 0:
            return np.ones(n) / n

        def objective(x, r):
            return -np.sum(np.log(r @ x))

        def gradient(x, r):
            if r.shape[0] == 0:
                return np.zeros_like(x)
            return -np.sum(r / (r @ x)[:, np.newaxis], axis=0)

        def constraint(x):
            return np.sum(x) - 1

        def augmented_lagrangian(x, r, mu, nu):
            return objective(x, r) + mu * constraint(x) + 0.5 * nu * constraint(x) ** 2

        def update_x(x, r, mu, nu, alpha):
            grad = gradient(x, r) + mu * nu * constraint(x)
            return x - alpha * grad

        x = np.ones(n) / n
        mu = 1.0
        nu = 1.0
        alpha_init = 0.1
        tol = 1e-6
        obj_tol = 1e-6
        constraint_tol = 1e-6

        for i in range(max_iter):
            alpha = alpha_init / (1 + i)  # Adaptive learning rate
            x_prev = x.copy()
            x = update_x(x, r, mu, nu, alpha)
            x = np.maximum(x, 0)
            x = x / np.sum(x)

            if np.linalg.norm(x - x_prev) < tol and np.abs(constraint(x)) < constraint_tol and np.abs(objective(x, r) - objective(x_prev, r)) < obj_tol:
                break

        return x

    def Follow_The_Leader(self, r,max_iter):
        T = len(r)
        n = len(r[0])
        x = np.zeros((T, n))
        for t in range(T):
            x[t] = self.Best_Constantly_Rebalanced_Portfolio(r[:t])
        return x[-1]

    def Follow_The_Leader_weights(self, original_df):
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.Follow_The_Leader(log_returns.iloc[:i].values, max_iter=100)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df

    def Modified_Smooth_Prediction(self,r, max_iter=10):
        T = r.shape[0]
        n = r.shape[1]
        if T==0:
            return np.ones(n)/n

        tilde_r = np.ones((T,n))
        x = np.zeros((T,n))
        I = np.identity(n)
        for t in range(T):
            alpha_t = n*(t+1)**(-1/3)
            y = self.Best_Constantly_Rebalanced_Portfolio(np.concatenate((I, tilde_r[:t]) ), max_iter=10 )
            tilde_r[t] = (1-alpha_t/n)*r[t] + alpha_t/n
            x[t] = (1-alpha_t)*y + alpha_t/n

        return x[-1]

    def Modified_Smooth_Prediction_weights(self, original_df):
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.Modified_Smooth_Prediction(log_returns.iloc[:i].values, max_iter=10)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df

    def Modified_Smooth_Prediction_short(self, r, max_iter=10):
        T = r.shape[0]
        n = r.shape[1]
        if T == 0:
            return np.ones(n) / n

        tilde_r = np.ones((T, n))
        x = np.zeros((T, n))
        I = np.identity(n)
        for t in range(T):
            alpha_t = n * (t + 1) ** (-1 / 3)
            y = self.Best_Constantly_Rebalanced_Portfolio(np.concatenate((I, tilde_r[:t])), max_iter=10)
            
            # Change the direction of the weights for short selling
            y = -y
            
            tilde_r[t] = (1 - alpha_t / n) * r[t] + alpha_t / n
            x[t] = (1 - alpha_t) * y + alpha_t / n

        return x[-1]

    def Modified_Smooth_Prediction_short_weights(self, original_df):
        prices_df = original_df.copy()

        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        n_SYMBOLS = len(prices_df.columns)

        n_days = len(log_returns)
        weights = np.zeros((n_days, n_SYMBOLS))

        for i in range(1, n_days):
            weights[i] = self.Modified_Smooth_Prediction_short(log_returns.iloc[:i].values, max_iter=10)

        weights_df = pd.DataFrame(weights, columns=prices_df.columns)
        return weights_df
