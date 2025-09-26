import numpy as np
import cvxpy as cp

class Newsvendor():
    def __init__(self,reg=0,verbose=False,solver=cp.MOSEK):
        self.reg=reg
        self.verbose=verbose
        self.solver=solver
        
    def fit(self,params):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        """
        
        self.prepare_model(params)
        self.lda.value = self.reg
        self.prob.solve(solver=self.solver, verbose=self.verbose) 
        self.coef_ = self.q.value
        self.time_ = self.prob.solver_stats.solve_time
#         self.time_ = self.prob.solver_stats.setup_time

    def prepare_model(self, params):
        b=params['b']
        h=params['h']
        gamma=params['gamma']
        xi=params['xi']
        weight=params['weight']
        
        N=len(weight)
        
        #define variables
        self.alpha=cp.Variable(1,nonneg=False)
        self.beta=cp.Variable(N,nonneg=False)
        self.z=cp.Variable(N,nonneg=False)
        self.s_p=cp.Variable(N,nonneg=True)
        self.s_m=cp.Variable(N,nonneg=True)
        self.q=cp.Variable(1,nonneg=True)
        self.nu=cp.Variable(1,nonneg=True)
        
        #define parameters
        self.lda=cp.Parameter(nonneg=True)
        
        # define constraints
        cons=[]
        
        for i in range(N):
            # cons.append(self.alpha >= self.z[i]+self.beta[i]/np.sqrt(weight[i]))
            cons.append(self.alpha * np.sqrt(weight[i]) >= self.z[i] * np.sqrt(weight[i])+self.beta[i])
            cons.append(self.s_p[i] >= self.q-xi[i])
            cons.append(self.s_m[i] >= xi[i]-self.q)
            cons.append(self.z[i] >= h*self.s_p[i]+b*self.s_m[i])
        cons.append(cp.SOC(self.nu,self.beta))
        
        
        obj=cp.Minimize(self.alpha-sum(np.sqrt(weight[i])*self.beta[i] for i in range(N))+self.lda*self.nu/np.sqrt(2))
        self.prob = cp.Problem(obj, cons)
        
class NewsvendorTrue():
    def __init__(self,reg=0,verbose=False,solver=cp.MOSEK):
        self.reg=reg
        self.verbose=verbose
        self.solver=solver
        
    def fit(self,params):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        """
        
        self.prepare_model(params)
        self.lda.value = self.reg
        
        self.prob.solve(solver=self.solver, verbose=self.verbose) 
        self.time_ = self.prob.solver_stats.solve_time
        self.coef_ = self.q.value
#         self.time_ = self.prob.solver_stats.setup_time

    def prepare_model(self, params):
        b=params['b']
        h=params['h']
        xi=params['xi']
        weight=params['weight']

        N=len(xi)
        
        #define variables
        self.q=cp.Variable(1,nonneg=True)
        
        #define parameters
        self.lda=cp.Parameter(nonneg=True)

        
        # define constraints
        cons=[]
        # obj_fun = 0
        # for i in range(N):
        #     obj_fun += h * cp.pos(self.q-xi[i]) + b * cp.pos(xi[i] - self.q)
            # obj_fun += h * (self.q-xi[i])
        obj=cp.Minimize(sum(weight[i] * (h * cp.pos(self.q-xi[i]) + b * cp.pos(xi[i] - self.q)) for i in range(N))/N + self.lda * self.q)
        # obj=cp.Minimize(sum(h * cp.pos(self.q-xi[i]) + b * cp.pos(xi[i] - self.q) for i in range(N))/N )
        # obj=cp.Minimize(obj_fun/N)
        self.prob = cp.Problem(obj, cons)
        
        