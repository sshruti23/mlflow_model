import datetime

from pandas_datareader import data as pdr
import yfinance as yf


def download_yfinance_data():
    yf.pdr_override()
    y_symbols = ["BTC-USD"]

    startdate = datetime.datetime(2022, 1, 1)
    enddate = datetime.datetime(2022, 12, 31)
    df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)
    df.to_csv('/dbfs/data/raw.csv', mode='x')


if __name__ == "__main__":
    download_yfinance_data()
