#!/usr/bin/env python3
"""
Visualize a Crypto currency DataFrame after cleaning and resampling it daily.
"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

def visualize():
    # Load the data
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    # 1) Remove the 'Weighted_Price' column
    df = df.drop(columns=['Weighted_Price'])

    # 2) Rename 'Timestamp' to 'Date'
    df = df.rename(columns={'Timestamp': 'Date'})

    # 3) Convert timestamp values to date values
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # 4) Set 'Date' as the index
    df = df.set_index('Date')

    # 5) Fill missing values in 'Close' with the previous row's value
    df['Close'] = df['Close'].fillna(method='ffill')

    # 6) Fill missing High, Low, and Open with the same row's Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 7) Set missing values in Volume columns to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # 8) Select data from 2017-01-01 onward
    df = df.loc['2017-01-01':]

    # Group by day ('D') and resample:
    #  High: max, Low: min, Open: mean, Close: mean,
    #  Volume_(BTC): sum, Volume_(Currency): sum
    df_daily = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    # Print the resulting daily DataFrame
    print(df_daily)

    # Plot all columns in one figure
    df_daily.plot()
    plt.show()

    # Return the transformed (daily) DataFrame
    return df_daily

if __name__ == "__main__":
    visualize()
