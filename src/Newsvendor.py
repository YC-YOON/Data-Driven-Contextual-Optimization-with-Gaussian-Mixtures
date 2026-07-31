import numpy as np
import cvxpy as cp

class Newsvendor():
    def __init__(self, reg=0, verbose=False, solver=cp.MOSEK):
        self.reg = reg
        self.verbose = verbose
        self.solver = solver

    def fit(self, params):
        self.prepare_model(params)
        self.lda.value = self.reg / self.scale

        self.prob.solve(solver=self.solver, verbose=self.verbose)
        self.coef_ = self.q.value * self.scale
        self.time_ = self.prob.solver_stats.solve_time

    def prepare_model(self, params):
        scale = params.get("scale", 100.0)

        b = params['b']
        h = params['h']
        s_test = params['s_test']


        xi_is = np.asarray(params['xi_is'], dtype=float) / scale
        weight = np.asarray(params['weight'], dtype=float)
        N = len(weight)

        self.scale = scale
        self.alpha = cp.Variable(1)
        self.beta = cp.Variable(N)
        self.z = cp.Variable(N)
        self.s_p = cp.Variable(N, nonneg=True)
        self.s_m = cp.Variable(N, nonneg=True)
        self.q = cp.Variable(1, nonneg=True)
        self.nu = cp.Variable(1, nonneg=True)
        self.lda = cp.Parameter(nonneg=True)
        cons = []

        for i in range(N):
            cons.append(self.alpha * np.sqrt(weight[i]) >= self.z[i] * np.sqrt(weight[i]) + self.beta[i])
            cons.append(self.s_p[i] >= self.q - xi_is[i])
            cons.append(self.s_m[i] >= xi_is[i] - self.q)
            cons.append(self.z[i] >= h * self.s_p[i] + b * self.s_m[i])
        cons.append(cp.SOC(self.nu, self.beta))
        
        obj = cp.Minimize(self.alpha - cp.sum(cp.multiply(np.sqrt(weight), self.beta)) + self.lda * self.nu / np.sqrt(2))

        self.prob = cp.Problem(obj, cons)
        

        