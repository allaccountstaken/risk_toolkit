import pandas as pd


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
    

def get_hfi_returns(filename: str):
    """
    Load and format hedge fund index data
    """
    #filename = filename
    hfi=pd.read_csv(filename, header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi


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


