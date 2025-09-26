import cvxpy as cp
import numpy as np
from sklearn.preprocessing import StandardScaler

class Portfolio():
    def __init__(self, reg=0, verbose=False, solver=cp.MOSEK):
        self.reg = reg
        self.verbose = verbose
        self.solver = solver

    def fit(self, params):
        self.prepare_model(params)
        self.lda.value = self.reg
        self.prob.solve(solver=self.solver, verbose=self.verbose)
        self.coef_ = self.x.value
        self.solve_time_ = self.prob.solver_stats.solve_time

    def prepare_model(self, params):
        R = params['returns']      # shape (N, d): scenario-wise asset returns
        weight = params['weight']  # shape (N,): scenario weights
        N, d = R.shape
        R = R / 100
        R = R.astype(float)

        # Variables
        self.x = cp.Variable(d, nonneg=True)                # portfolio weights
        self.alpha = cp.Variable()             # dual objective upper bound
        self.beta = cp.Variable(N)
        self.z = cp.Variable(N)
        self.nu = cp.Variable(nonneg=True)

        self.lda = cp.Parameter(nonneg=True)

        # Constraints
        sqrt_w = np.sqrt(weight)
        cons = []
        for i in range(N):
            portfolio_return_i = R[i] @ self.x
            cons.append(self.z[i] >= -portfolio_return_i)
            cons.append(self.alpha * sqrt_w[i] >= self.z[i] * sqrt_w[i] + self.beta[i])

        cons.append(cp.SOC(self.nu, self.beta))
        cons.append(cp.sum(self.x) == 1)  # Fully invested
        # Objective
        obj = cp.Minimize(
            self.alpha
            - cp.sum(cp.multiply(sqrt_w, self.beta))
            + self.lda * self.nu / np.sqrt(2)
        )
        self.prob = cp.Problem(obj, cons)

class Portfolio_CVaR():
    def __init__(self, reg=0, verbose=False, solver=cp.MOSEK):
        self.reg = reg
        self.verbose = verbose
        self.solver = solver

    def fit(self, params):
        self.prepare_model(params)
        self.lda.value = self.reg
        self.prob.solve(solver=self.solver, verbose=self.verbose)
        self.coef_ = self.x.value
        self.solve_time_ = self.prob.solver_stats.solve_time

    def prepare_model(self, params):
        R = params['returns']      # shape (N, d): scenario-wise asset returns
        weight = params['weight']  # shape (N,): scenario weights
        alpha_ = params['alpha_']
        N, d = R.shape
        R = R / 100
        R = R.astype(float)

        # Variables
        self.x = cp.Variable(d, nonneg=True)                # portfolio weights
        self.alpha = cp.Variable()             # dual objective upper bound
        self.beta = cp.Variable(N)
        self.z_1 = cp.Variable(N)
        self.z_2 = cp.Variable(N)
        self.nu = cp.Variable(nonneg=True)
        self.b = cp.Variable()

        self.lda = cp.Parameter(nonneg=True)
        # Constraints
        cons = []
        sqrt_weight = np.sqrt(np.array(weight))  # 제약문 밖에서 한 번만 계산

        for i in range(N):
            portfolio_return_i = R[i] @ self.x
            cons.append(self.z_1[i] >= - (1 / alpha_) * portfolio_return_i)
            cons.append(self.z_2[i] >= self.b)
            cons.append(self.alpha * sqrt_weight[i] >= self.z_1[i] * sqrt_weight[i] + self.beta[i])
            cons.append(self.alpha * sqrt_weight[i] >= self.z_2[i] * sqrt_weight[i] + self.beta[i])

        cons.append(cp.SOC(self.nu, self.beta))
        cons.append(cp.sum(self.x) == 1)        # Fully invested

        # Objective
        obj = cp.Minimize(
            self.alpha
            - cp.sum([np.sqrt(weight[i]) * self.beta[i] for i in range(N)])
            + self.lda * self.nu/np.sqrt(2)
        )

        self.prob = cp.Problem(obj, cons)
