import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime
from scipy.optimize import minimize 


class risk_parity(object):
    
    def __init__(self, w, V):
        self.V = V
        self.w = w
    
    def get_portfolio_var(self, w,V):
        w = np.matrix(w)
        return (w*V*w.transpose())[0,0]
    
    def get_risk_contribution(self, w,V):
        w = np.matrix(w)
        sigma = np.sqrt(get_portfolio_var(w,V))
        # Marginal Risk Contribution
        MRC = w.transpose()*V
        # Risk Contribution
        RC = np.multiply(w.transpose(), MRC)/sigma
        return RC
    
    # calculate portfolio risk
    #w_t is the target risk weights
    def risk_budget_objective(self, w, args):
        V = args[0]
        w_t = args[1]
        sig_p =  np.sqrt(calculate_portfolio_var(w,V)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p,w_t))
        asset_RC = calculate_risk_contribution(w,V)
        J = sum(np.square(asset_RC-risk_target.transpose()))[0,0] # sum of squared error
        return J
    
    def get_risk_parity_weights(self, V, w_t, w):
        # Restrictions to consider in the optimisation: only long positions whose
        # sum equals 100%
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                      {'type': 'ineq', 'fun': lambda x: x})
        
        # Optimisation process in scipy
        optimize_result = minimize(fun=risk_budget_objective,
                                   x0=w,
                                   args=[V, w_t],
                                   method='SLSQP',
                                   constraints=constraints,
                                   tol=1e-10,
                                   options={'disp': False})
        
        # Recover the weights from the optimised object
        weights = optimize_result.x
        # It returns the optimised weights
        return weights

  