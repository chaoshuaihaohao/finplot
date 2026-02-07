#!/usr/bin/env python3

from collections import defaultdict
import dateutil.parser
import finplot as fplt
import numpy as np
import pandas as pd
import requests
import os

baseurl = 'https://www.bitmex.com/api'
fplt.timestamp_format = '%m/%d/%Y %H:%M:%S.%f'


def local2timestamp(s):
    return int(dateutil.parser.parse(s).timestamp())


def download_price_history(symbol='XBTUSD', start_time='2023-01-01', end_time='2023-10-29', interval_mins=60):
    csv_file_path = r'D:\业余\github\stockdata\A_data\002738_qfq_A_data.csv'
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"本地CSV文件不存在：{csv_file_path}，请检查路径是否正确！")
    # 读取CSV文件（跳过股票代码列，直接使用所需列）
    df = pd.read_csv(csv_file_path, usecols=['date', 'open', 'close', 'high', 'low', 'volume'])

    # 数据格式适配（完全匹配你的CSV结构）
    # 1. 日期列转换为时间戳（time列）
    df['time'] = df['date'].apply(lambda x: local2timestamp(x))

    start_time = local2timestamp(start_time)
    end_time = local2timestamp(end_time)
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    # 3. 移除空值，确保数据完整性
    df = df.dropna(subset=['time', 'open', 'close', 'high', 'low', 'volume'])

    # 4. 设置time为索引，返回和原函数一致的格式
    return df.set_index('time')[['open', 'close', 'high', 'low', 'volume']]

def plot_accumulation_distribution(df, ax):
    ad = (2*df.close-df.high-df.low) * df.volume / (df.high - df.low)
    ad.cumsum().ffill().plot(ax=ax, legend='Accum/Dist', color='#f00000')


def plot_bollinger_bands(df, ax):
    mean = df.close.rolling(20).mean()
    stddev = df.close.rolling(20).std()
    df['boll_hi'] = mean + 2.5*stddev
    df['boll_lo'] = mean - 2.5*stddev
    p0 = df.boll_hi.plot(ax=ax, color='#808080', legend='BB')
    p1 = df.boll_lo.plot(ax=ax, color='#808080')
    fplt.fill_between(p0, p1, color='#bbb')


def plot_ema(df, ax):
    df.close.ewm(span=9).mean().plot(ax=ax, legend='EMA')


def plot_heikin_ashi(df, ax):
    df['h_close'] = (df.open+df.close+df.high+df.low) / 4
    ho = (df.open.iloc[0] + df.close.iloc[0]) / 2
    for i,hc in zip(df.index, df['h_close']):
        df.loc[i, 'h_open'] = ho
        ho = (ho + hc) / 2
    df['h_high'] = df[['high','h_open','h_close']].max(axis=1)
    df['h_low'] = df[['low','h_open','h_close']].min(axis=1)
    df[['h_open','h_close','h_high','h_low']].plot(ax=ax, kind='candle')


def plot_heikin_ashi_volume(df, ax):
    df[['h_open','h_close','volume']].plot(ax=ax, kind='volume')


def plot_on_balance_volume(df, ax):
    obv = df.volume.copy()
    obv[df.close < df.close.shift()] = -obv
    obv[df.close==df.close.shift()] = 0
    obv.cumsum().plot(ax=ax, legend='OBV', color='#008800')


def plot_rsi(df, ax):
    diff = df.close.diff().values
    gains = diff.copy()
    losses = -diff.copy()
    with np.errstate(invalid='ignore'):
        gains[(gains<0)|np.isnan(gains)] = 0.0
        losses[(losses<=0)|np.isnan(losses)] = 1e-10 # we don't want divide by zero/NaN
    n = 14
    m = (n-1) / n
    ni = 1 / n
    g = gains[n] = np.nanmean(gains[:n])
    l = losses[n] = np.nanmean(losses[:n])
    gains[:n] = losses[:n] = np.nan
    for i,v in enumerate(gains[n:],n):
        g = gains[i] = ni*v + m*g
    for i,v in enumerate(losses[n:],n):
        l = losses[i] = ni*v + m*l
    rs = gains / losses
    df['rsi'] = 100 - (100/(1+rs))
    df.rsi.plot(ax=ax, legend='RSI')
    fplt.set_y_range(0, 100, ax=ax)
    fplt.add_horizontal_band(30, 70, ax=ax)


def plot_vma(df, ax):
    df.volume.rolling(20).mean().plot(ax=ax, color='#c0c030')


symbol = '002738'
df = download_price_history(
    symbol=symbol,
    start_time='2014-12-30',  # 你的数据起始日期
    end_time='2025-01-01'     # 数据结束日期
)

ax, axv, ax2, ax3, ax4 = fplt.create_plot(f'A股 {symbol} 平均K线图', rows=5)
ax.set_visible(xgrid=True, ygrid=True)

# price chart
plot_heikin_ashi(df, ax)
plot_bollinger_bands(df, ax)
plot_ema(df, ax)

# volume chart
plot_heikin_ashi_volume(df, axv)
plot_vma(df, ax=axv)

# some more charts
plot_accumulation_distribution(df, ax2)
plot_on_balance_volume(df, ax3)
plot_rsi(df, ax4)

# restore view (X-position and zoom) when we run this example again
fplt.autoviewrestore()

fplt.show()