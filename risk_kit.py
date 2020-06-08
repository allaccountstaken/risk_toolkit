import pandas as pd
import numpy as np
import scipy.stats as ss
import scipy




def compute_drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns 
    Returns a DataFrame with columns for wealth index, peaks, and drawdons
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    return pd.DataFrame({
        'Wealth': wealth_index, 
        'Peaks': previous_peaks, 
        'Drawdowns': drawdowns
                        })


def get_ffme_returns():
    """ 
    Load Fama-French dataset for the returns of the top and bottom deciles by market cap
    """
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m%').to_period('M')
    return rets
    

def get_hfi_returns():
    """
    Load and format hedge fund index data
    """
    filename = 'data/edhec-hedgefundindices.csv'
    hfi=pd.read_csv(filename, header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi
                 
def get_ind_returns():
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def semideviation(r):
    """
    Returns negative semideviation of r
    Input r must be Series or DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes skew of Series or DataFrame
    Returns a Float or a Series
    """
    demeaned_r = r - r.mean()
    # use population std so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes kurtosis of Series or DataFrame
    Returns a Float or a Series
    """
    demeaned_r = r - r.mean()
    # use population std so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4

def is_normal(r, level=0.01):
    """
    Level 0.01
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_hist(r, level=5):
    
    """
    Return historic VaR
    Default level = 5
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_hist, level=level)
    elif isinstance(r, pd.Series):
        return - np.percentile(r, level)
    else:
        raise TypeError("Expected input Series or DataFrame")

        
def var_gaussian(r, level=5, modified=False):
    """
    Returns parametric Gaussian VaR at a certain level
    Input r of Series or DataFrame type
    """
    # Computer z-score based on Gaussian
    z = ss.norm.ppf(level/100)
    # Modify z-score based on emperically observed skewness and kurtosis
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36
             
        
  
    return (r.mean() + z * r.std(ddof=0))
    

def cvar_historic(r, level=5):
    """
    Computes conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond  = r <= -var_hist(r, level=level)
        return - r[is_beyond].mean()
    else:
        raise TypeError("Expected input Series or DataFrame")


def annualize_ret(r, periods_per_year):
    """
    Annualizes a set of returns
    We shoudl infer the periods per year...
    """
    
    compound_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compound_growth ** (periods_per_year / n_periods) - 1

def annualize_vol(r, periods_per_year):
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, rf_rate, periods_per_year):
    """
    Computes annualized sharpe ratio of a set of returns
    """
    rf_per_period = (1 + rf_rate) ** (1 / periods_per_year) - 1
    excess_return = r - rf_per_period
    ann_excess_return = annualize_ret(excess_return, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_excess_return /  ann_vol



def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns 

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights) ** 0.5



def plot_ef2(n_points, er, covmat, style='.-'):
    """ Plots 2-asset efficient frontier"""
    if er.shape[0] !=2 or er.shape[0] != 2:
        raise ValueError('can only prlot 2-asset frontier')
        
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, covmat) for w in weights]
    ef = pd.DataFrame({'Return':rets, 'Vol': vols})
    
    return ef.plot.line(x='Vol', y='Return', style=style)


def msr(riskfree_rate, er, cov):
    #from scipy.optimize import minimize
    """
    Returns the weights of the portfolio that gives the maximum sharpe ratio
    given the riskfree rate and expected return and a covariance matrix
    """
  
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    
    
    weights_sum_to_1 = {
        'type' : 'eq',
        'fun' : lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
        
    
    results = minimize(neg_sharpe_ratio, init_guess, 
                       args=(riskfree_rate, er, cov,), method="SLSQP", 
                       options={'disp' : False}, 
                       constraints=(weights_sum_to_1), 
                       bounds=bounds
                      )
    
    return results.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Vol portfolio
    given covariace matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def plot_ef(n_points, er, cov, show_cml=False, show_ew=False, show_gmv=False,
            style='.-', riskfree_rate=0):
    """ Plots N-asset efficient frontier"""
   
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({'Return':rets, 
                       'Vol': vols
                      })
    
    ax = ef.plot.line(x='Vol', y='Return', style=style)
    
    if show_ew: #Equally weighted portfolio
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display EW portfolio
        ax.plot([vol_ew], [r_ew], color='goldenrod', 
                marker='o', markersize=12)
    
    if show_gmv: #Global minimul variance
        
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display GMV portfolio
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', 
                marker='o', markersize=10)
    
    if show_cml: #Capital Markets Line
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', 
                marker='o', linestyle='dashed', 
                markersize=12, linewidth=2)
        
        return ax
        

def optimal_weights(n_points, er, cov):
    """
    list of weights to run the optimizer on 
    """
    target_rets = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rets]
    return weights

from scipy.optimize import minimize


def minimize_vol(target_return, er, cov):
    #from scipy.optimize import minimize
    """
    target return -> weights vector
    """
  
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    
    return_is_target = {
        'type' : 'eq',
        'args' : (er,),
        'fun' : lambda weights, er: target_return - portfolio_return(weights, er)
    }
    
    weights_sum_to_1 = {
        'type' : 'eq',
        'fun' : lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(portfolio_vol, init_guess, 
                       args=(cov,), method="SLSQP", 
                       options={'disp' : False}, 
                       constraints=(return_is_target, weights_sum_to_1), 
                       bounds=bounds
                      )
    
    return results.x





    
    