import pandas as pd
import numpy as np
import scipy.stats as ss
import scipy
import pandas as pd
import numpy as np
import math



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


def get_ind_size():
    ind = pd.read_csv('data/ind30_m_size.csv', header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    ind = pd.read_csv('data/ind30_m_nfirms.csv', header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_return():
    pass


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



def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate = 0.03, drawdown=None):
    """
    Runs a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary with Asset Value, Risk Budget and Risk Weight histories
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor 
    peak = account_value

    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12

    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1-drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        # update the account value for this time step
        account_value = risky_alloc *(1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        # save values to look at the history 
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    
    
    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth" : account_history,
        "Risky Wealth" : risky_wealth,
        "Risk Budget" : cushion_history,
        "Risk Allocation" : risky_w_history,
        "m" : m,
        "start" : start,
        "floor" : floor,
        "risky_r" : risky_r,
        "safe_r" : safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    
    return backtest_result

    
    
    
    
    
# Instructors code below:

def run_cppi1(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result




def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val



# def discount(t, r):
#     """Compute teh price of a pure disocunt bond that 
#     pays 1$ at time t given interest rate r
#     """
#     return (1+r)**(-t)


# def pv(l, r):
#     """
#     Computes PV of a sequence of liabilities
#     l is indexed by the time and teh values are the amounts
#     of each liability.
#     Returns the present value
#     """
#     dates = l.index
#     discounts = discount(dates, r)
#     return (discounts * l).sum()


# def funding_ratio(assets, liabilities, r):
#     """Computes the funding ratio"""
#     return assets / pv(liabilities, r)


def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices