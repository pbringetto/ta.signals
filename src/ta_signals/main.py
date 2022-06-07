import pandas as pd
import pandas_ta as ta
import numpy as np
import cfg_load
from scipy.stats import linregress
from scipy.signal import argrelextrema
import math
pd.options.mode.chained_assignment = None

class TaSignals:
    def __init__(self, window):
        self.window = window

    def on_balance_volume(self, df, n, key):
        """Calculate On-Balance Volume for given data.

        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        i = 0
        OBV = [0]
        while i < df.index[-1]:
            if df.loc[i + 1, key] - df.loc[i, key] > 0:
                OBV.append(df.loc[i + 1, 'volume'])
            if df.loc[i + 1, key] - df.loc[i, key] == 0:
                OBV.append(0)
            if df.loc[i + 1, key] - df.loc[i, key] < 0:
                OBV.append(-df.loc[i + 1, 'volume'])
            i = i + 1
        OBV = pd.Series(OBV)
        OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='obv')
        df = df.join(OBV_ma)
        data = {
                    'on_balance_volume': False,
                }
        return data, df

    def bollinger(self, df, key):

        tp = (df[key] + df['low'] + df['high'])/3
        df['std'] = tp.rolling(20).std(ddof=0)
        matp = tp.rolling(20).mean()
        df['bollinger_high'] = matp + 2*df['std']
        df['bollinger_low'] = matp - 2*df['std']
        data = {'bollinger_low': df['bollinger_low'].iloc[-1], 'bollinger_high': df['bollinger_high'].iloc[-1]}
        data = [
            {'key': 'bollinger_above_matp', 'value': 'Above bollinger matp' if (df[key].iloc[-1] > matp.iloc[-1] + 25) else False},
            {'key': 'bollinger_below_matp', 'value': 'Below bollinger matp' if (df[key].iloc[-1] < matp.iloc[-1] - 25) else False},
            {'key': 'bollinger_at_matp', 'value': 'At bollinger matp' if math.isclose(df[key].iloc[-1], matp.iloc[-1], abs_tol=25) else False},
            {'key': 'below_bollinger_low', 'value': 'Below bollinger Low' if df[key].iloc[-1] < (df['bollinger_low'].iloc[-1] - 25) else False},
            {'key': 'above_bollinger_high', 'value': 'Above bollinger high' if df[key].iloc[-1] > (df['bollinger_high'].iloc[-1] + 25) else False},
            {'key': 'at_bollinger_low', 'value': 'At Bollinger low' if math.isclose(df[key].iloc[-1], df['bollinger_low'].iloc[-1], abs_tol=25) else False},
            {'key': 'at_bollinger_high', 'value': 'At Bollinger high' if math.isclose(df[key].iloc[-1], df['bollinger_high'].iloc[-1], abs_tol=25) else False},
        ]
        return data, df

    def ma(self, df, key):
        df['ma20'] = df[key].rolling(20).mean()
        df['ma50'] = df[key].rolling(50).mean()
        df['ma100'] = df[key].rolling(100).mean()
        df['ma200'] = df[key].rolling(200).mean()

        data = {'ma20': df['ma20'].iloc[-1], 'ma50': df['ma50'].iloc[-1], 'ma100': df['ma100'].iloc[-1], 'ma200': df['ma200'].iloc[-1]}
        data = [
            {'key': 'above_ma_20', 'value': 'Above 20 period moving average' if df[key].iloc[-1] > df['ma20'].iloc[-1] else False},
            {'key': 'above_ma_50', 'value': 'Above 50 period moving average' if df[key].iloc[-1] > df['ma50'].iloc[-1] else False},
            {'key': 'above_ma_100', 'value': 'Above 100 period moving average' if df[key].iloc[-1] > df['ma100'].iloc[-1] else False},
            {'key': 'above_ma_200', 'value': 'Above 200 period moving average' if df[key].iloc[-1] > df['ma200'].iloc[-1] else False},
            {'key': 'below_ma_20', 'value': 'Below 20 period moving average' if df[key].iloc[-1] < df['ma20'].iloc[-1] else False},
            {'key': 'below_ma_50', 'value': 'Below 50 period moving average' if df[key].iloc[-1] < df['ma50'].iloc[-1] else False},
            {'key': 'below_ma_100', 'value': 'Below 100 period moving average' if df[key].iloc[-1] < df['ma100'].iloc[-1] else False},
            {'key': 'below_ma_200', 'value': 'Below 200 period moving average' if df[key].iloc[-1] < df['ma200'].iloc[-1] else False},
        ]
        return data, df


    def ema(self, df, key):
        df['ema20'] = df[key].ewm(span=20, adjust=False).mean()
        df['ema50'] = df[key].ewm(span=50, adjust=False).mean()
        df['ema100'] = df[key].ewm(span=100, adjust=False).mean()
        df['ema200'] = df[key].ewm(span=200, adjust=False).mean()

        data = {'ema20': df['ema20'].iloc[-1], 'ema50': df['ema50'].iloc[-1], 'ema100': df['ema100'].iloc[-1], 'ema200': df['ema200'].iloc[-1]}
        data = [
            {'key': 'above_ema_20', 'value': 'Above 20 period exponential moving average' if df[key].iloc[-1] > df['ema20'].iloc[-1] else False},
            {'key': 'above_ema_50', 'value': 'Above 50 period exponential moving average' if df[key].iloc[-1] > df['ema50'].iloc[-1] else False},
            {'key': 'above_ema_100', 'value': 'Above 100 period exponential moving average' if df[key].iloc[-1] > df['ema100'].iloc[-1] else False},
            {'key': 'above_ema_200', 'value': 'Above 200 period exponential moving average' if df[key].iloc[-1] > df['ema200'].iloc[-1] else False},
            {'key': 'below_ema_20', 'value': 'Below 20 period exponential moving average' if df[key].iloc[-1] > df['ema20'].iloc[-1] else False},
            {'key': 'below_ema_50', 'value': 'Below 50 period exponential moving average' if df[key].iloc[-1] > df['ema50'].iloc[-1] else False},
            {'key': 'below_ema_100', 'value': 'Below 100 period exponential moving average' if df[key].iloc[-1] > df['ema100'].iloc[-1] else False},
            {'key': 'below_ema_200', 'value': 'Below 200 period exponential moving average' if df[key].iloc[-1] > df['ema200'].iloc[-1] else False},
        ]
        return data, df

    def rsi(self, df, key):
        df = self.get_rsi(df, key)
        df['rsi_slope'] = df['rsi'].rolling(window=self.window).apply(self.get_slope, raw=True)
        data = {'rsi': df['rsi'].iloc[-1]}
        data = [
            {'key': 'rsi_rising', 'value': 'RSI is rising' if df['rsi_slope'].iloc[-1] > 0 and df['rsi_slope'].iloc[-1] < 10 else False},
            {'key': 'rsi_rising_fast', 'value': 'RSI is rising fast' if df['rsi_slope'].iloc[-1] > 10 else False},
            {'key': 'rsi_dropping', 'value': 'RSI is dropping' if df['rsi_slope'].iloc[-1] < 0 and df['rsi_slope'].iloc[-1] > 10 else False},
            {'key': 'rsi_dropping_fast', 'value': 'RSI is dropping fast' if df['rsi_slope'].iloc[-1] < -10  else False},
            {'key': 'rsi_oversold', 'value': 'RSI is oversold' if df['rsi'].iloc[-1] <= 30 and df['rsi'].iloc[-1] >= 20 else False},
            {'key': 'rsi_extremely_oversold', 'value': 'RSI is extremely oversold' if df['rsi'].iloc[-1] <= 20 else False},
            {'key': 'rsi_overbought', 'value': 'RSI is overbought' if df['rsi'].iloc[-1] >= 70  and df['rsi'].iloc[-1] <= 80 else False},
            {'key': 'rsi_extremely_overbought', 'value': 'RSI is extremely overbought' if df['rsi'].iloc[-1] >= 80 else False},
        ]
        return data, df

    def macd_slope(self, df, key):
        k = df[key].ewm(span=12, adjust=False, min_periods=12).mean()
        d = df[key].ewm(span=26, adjust=False, min_periods=26).mean()
        macd = k - d
        macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_h = macd - macd_s
        df['MACD_12_26_9'] = df.index.map(macd)
        df['MACDs_12_26_9'] = df.index.map(macd_s)
        df['MACDh_12_26_9'] = df.index.map(macd_h)
        #df.ta.macd(close=key, fast=12, slow=26, signal=9, append=True)
        df['macd_slope'] = df['MACD_12_26_9'].rolling(window=self.window).apply(self.get_slope, raw=True)
        df['macd_sig_slope'] = df['MACDs_12_26_9'].rolling(window=self.window).apply(self.get_slope, raw=True)
        df['macd_hist_slope'] = df['MACDh_12_26_9'].rolling(window=self.window).apply(self.get_slope, raw=True)
        data = {'macd_slope': df['macd_slope'].iloc[-1], 'macd_signal': df['MACDs_12_26_9'].iloc[-1], 'macd_hist': df['MACDh_12_26_9'].iloc[-1]}
        bullish_cross = math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=25) and df['macd_slope'].iloc[-1] > 0
        bearish_cross = math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=25) and df['macd_slope'].iloc[-1] < 0
        data = [
            {'key': 'macd_rising_fast', 'value': 'MACD is rising fast' if df['macd_slope'].iloc[-1] > 20 else False},
            {'key': 'macd_rising', 'value': 'MACD is rising' if df['macd_slope'].iloc[-1] > 0  and df['macd_slope'].iloc[-1] < 20 else False},
            {'key': 'macd_dropping', 'value': 'MACD is dropping' if df['macd_slope'].iloc[-1] < 0 and df['macd_slope'].iloc[-1] > -20 else False},
            {'key': 'macd_dropping_fast', 'value': 'MACD is dropping fast' if df['macd_slope'].iloc[-1] < -20 else False},
            {'key': 'bullish_macd_cross', 'value': 'Bullish MACD Crossover soon' if bullish_cross else False},
            {'key': 'bearish_macd_cross', 'value': 'Bearish MACD Crossover soon' if bearish_cross else False},
            {'key': 'macd_over_signal', 'value': 'MACD over Signal' if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1] else False},
            {'key': 'macd_over_centerline', 'value': 'MACD over centerline' if df['MACD_12_26_9'].iloc[-1] > 0 else False},
            {'key': 'macd_under_signal', 'value': 'MACD under Signal' if df['MACD_12_26_9'].iloc[-1] < df['MACDs_12_26_9'].iloc[-1] else False},
            {'key': 'macd_under_centerline', 'value': 'MACD under centerline' if df['MACD_12_26_9'].iloc[-1] < 0 else False},
        ]
        return data, df

    def low_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.less_equal, order=order)[0]

    def high_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.greater_equal, order=order)[0]

    def peak_slope(self, df, key):
        lows, highs = self.range_slopes(df, key, 21, 21)
        lows[f'{key}_lows_slope'] = lows[key].rolling(window=4).apply(self.get_slope, raw=True)
        highs[f'{key}_highs_slope'] = highs[key].rolling(window=4).apply(self.get_slope, raw=True)
        return lows[f'{key}_lows_slope'].iloc[-1], highs[f'{key}_highs_slope'].iloc[-1], df

    def rsi_close_div_peak_slopes(self, df, key):
        close_low_slope, close_high_slope, df = self.peak_slope(df, key)
        rsi_low_slope, rsi_high_slope, df = self.peak_slope(df, 'rsi')
        df.merge(df)
        return close_low_slope, close_high_slope, rsi_low_slope, rsi_high_slope, df

    def high_low_idx(self, df, key, low_window, high_window):
        return self.low_idx(df, key, low_window), self.high_idx(df, key, high_window)

    def range_less_than_prev(self, h, l, prev_h, prev_l):
        return (h - l) < (prev_h - prev_l)

    def is_compressing(self, highs, lows, i = 3):
         r = False
         while i > 0:
             if self.range_less_than_prev(highs[key].iloc[-i], lows[key].iloc[-i], highs[key].iloc[-(i-1)], lows[key].iloc[-(i-1)]):
                 r = True
             i -= 1
         return r

    def range_slopes(self, df, key, low_window, high_window):
        low_idx, high_idx = self.high_low_idx(df, key, low_window, high_window)
        lows, highs = df.iloc[low_idx].copy(), df.iloc[high_idx].copy()
        lows[f'{key}_lows_slope'] = lows[key].rolling(window=5).apply(self.get_slope, raw=True)
        highs[f'{key}_highs_slope'] = highs[key].rolling(window=self.window).apply(self.get_slope, raw=True)
        return lows, highs

    def divergence(self, df, key):
        close_low_slope, close_high_slope, rsi_low_slope, rsi_high_slope, df = self.rsi_close_div_peak_slopes(df, key)
        bullish_regular = (close_low_slope < 0) and (rsi_low_slope > 0)
        bearish_regular = (close_high_slope > 0) and (rsi_high_slope < 0)
        bullish_hidden = (rsi_high_slope < 0) and math.isclose(close_low_slope, 0, abs_tol=1)
        bearish_hidden = (close_high_slope < 0) and math.isclose(close_high_slope, 0, abs_tol=1)
        data = {'close_low_slope': close_low_slope, 'close_high_slope': close_high_slope, 'rsi_low_slope': rsi_low_slope, 'rsi_high_slope': rsi_high_slope}
        data = [
            {'key': 'bullish_regular', 'value': 'Bullish divergence' if bullish_regular else False},
            {'key': 'bullish_hidden', 'value': 'Hidden Bullish divergence' if bullish_hidden else False},
            {'key': 'bearish_regular', 'value': 'Bearish divergence' if bearish_regular else False},
            {'key': 'bearish_hidden', 'value': 'Hidden Bearish divergence' if bearish_hidden else False},
        ]
        return data, df

    def get_slope(self, array):
        y = np.array(array)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return slope

    def get_rsi(self, df, key):
        window_length = 14
        df['diff'] = df[key].diff(1)
        df['gain'] = df['diff'].clip(lower=0).round(2)
        df['loss'] = df['diff'].clip(upper=0).abs().round(2)
        df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
        df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
        for i, row in enumerate(df['avg_gain'].iloc[window_length+1:]):
            df['avg_gain'].iloc[i + window_length + 1] = (df['avg_gain'].iloc[i + window_length] * (window_length - 1) + df['gain'].iloc[i + window_length + 1]) / window_length
        for i, row in enumerate(df['avg_loss'].iloc[window_length+1:]):
            df['avg_loss'].iloc[i + window_length + 1] = (df['avg_loss'].iloc[i + window_length] * (window_length - 1) + df['loss'].iloc[i + window_length + 1])  / window_length
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1.0 + df['rs']))
        df.drop(['diff', 'gain', 'loss', 'avg_gain', 'avg_gain', 'rs'], axis=1, inplace=True)
        return df