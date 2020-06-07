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















    
    