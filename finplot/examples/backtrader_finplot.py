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

def plot_ma(df, ax):
    # 多周期均线（MA5 / 10 / 20 / 60 / 200）
    # 使用局部变量避免污染原df（可选）
    close = df['close']

    # 关键！！！所有均线 加 .dropna()，删除前面无效NaN，不撑大Y轴灰色框
    ma5 = close.rolling(window=5).mean().dropna()
    ma10 = close.rolling(window=10).mean().dropna()
    ma20 = close.rolling(window=20).mean().dropna()
    ma60 = close.rolling(window=60).mean().dropna()
    ma200 = close.rolling(window=200).mean().dropna()


    # 定义颜色（若未定义全局常量，则直接使用十六进制）
    fplt.plot(ma5, ax=ax, color='black', legend='MA5')
    fplt.plot(ma10, ax=ax, color='#FFA500', legend='MA10')
    fplt.plot(ma20, ax=ax, color='#0000FF', legend='MA20')
    fplt.plot(ma60, ax=ax, color='#9933FF', legend='MA60')  # 紫色
    fplt.plot(ma200, ax=ax, color='#009999', legend='MA200')  # 青色

def plot_candlestick(df, ax):
    """
    绘制标准（原始）K线图（非Heikin-Ashi）
    要求 df 包含 'open', 'close', 'high', 'low' 列
    """
    # ===== 1. 绘制K线 =====
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=ax)


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
    """计算并绘制三个不同周期的RSI指标（RSI6/12/24）"""
    periods = [6, 12, 24]
    colors = ['black', 'orange', 'red']

    for i, period in enumerate(periods):
        # ===== 为每个周期独立初始化数据（避免跨周期污染）=====
        diff = df['close'].diff().values
        gains = diff.copy()
        losses = -diff.copy()

        with np.errstate(invalid='ignore'):
            gains[(gains < 0) | np.isnan(gains)] = 0.0
            losses[(losses <= 0) | np.isnan(losses)] = 1e-10

        m = (period - 1) / period
        ni = 1 / period

        # ===== 安全处理空切片警告 =====
        # 前period个值设为NaN（RSI有效值从第period+1根K线开始）
        gains[:period] = np.nan
        losses[:period] = np.nan

        # 安全计算初始平均值（过滤NaN后计算）
        valid_gains = gains[period:2 * period]  # 取后续有效数据段
        valid_losses = losses[period:2 * period]
        avg_gain = np.nanmean(valid_gains) if not np.all(np.isnan(valid_gains)) else 0.0
        avg_loss = np.nanmean(valid_losses) if not np.all(np.isnan(valid_losses)) else 1e-10

        # 初始化平滑起点
        if len(gains) > period:
            gains[period] = avg_gain
            losses[period] = avg_loss

        # Wilder's 平滑计算
        for idx in range(period + 1, len(gains)):
            gains[idx] = ni * gains[idx] + m * gains[idx - 1]
            losses[idx] = ni * losses[idx] + m * losses[idx - 1]

        # 计算RSI
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))

        # 存储并绘制（使用当前周期的列名）
        col_name = f'rsi_{period}'
        df[col_name] = rsi

        df[col_name].plot(ax=ax, legend=f'RSI({period})', color=colors[i])

    # 设置坐标轴范围
    fplt.set_y_range(0, 100, ax=ax)

    # 超买超卖区域
    fplt.add_horizontal_band(30, 70, ax=ax)
    fplt.plot([50] * len(df), ax=ax, color='gray', style='--', width=1.0,
              legend='强弱线(50)')


def plot_macd(df, ax):
    # plot macd with standard colors first
    macd = df.close.ewm(span=12).mean() - df.close.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    df['macd_diff'] = macd - signal
    fplt.volume_ocv(df[['open', 'close', 'macd_diff']], ax=ax, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax, legend='MACD', color='black')
    fplt.plot(signal, ax=ax, legend='Signal', color='red')
    fplt.legend_text_color = 'yellow'
    fplt.legend_background_color = 'green'

def plot_vma(df, ax):
    df.volume.rolling(20).mean().plot(ax=ax, color='#c0c030')


def plot_kdj(df, ax):
    """
    使用 calculate_kdj_bt 计算 KDJ 并绘图（完全对齐A股券商逻辑）
    """

    def calculate_kdj_bt(df_plot, period=9, k_period=3, d_period=3):
        """
        完全对齐【A股券商版KDJ】递推逻辑（2/3前值+1/3当前值），适配所有pandas版本
        核心：和backtrader的_KDJBase类next()逻辑1:1复刻，彻底解决数据不一致
        :param df_plot: 原始K线df（含open/high/low/close列，小写）
        :param period: KDJ核心周期（默认9）
        :param k_period: 兼容参数（无实际作用，保留为了传参一致）
        :param d_period: 兼容参数（无实际作用，保留为了传参一致）
        :return: 带kdj_k/kdj_d/kdj_j列的df
        """
        df = df_plot.copy()
        # 若数据量不足，直接填充50（和券商版KDJ初始化一致）
        if len(df) < 1:
            df['kdj_k'] = 50.0
            df['kdj_d'] = 50.0
            df['kdj_j'] = 50.0
            return df

        # Step1：计算RSV - 完全对齐券商版逻辑（手动取N周期高低，除零保护RSV=50）
        df['n_high'] = df['high'].rolling(window=period, min_periods=1).max()  # 至少1个数据，避免空值
        df['n_low'] = df['low'].rolling(window=period, min_periods=1).min()
        # 除零保护：高低价相同时RSV=50（券商标准）
        df['rsv'] = np.where(
            df['n_high'] == df['n_low'],
            50.0,
            100.0 * (df['close'] - df['n_low']) / (df['n_high'] - df['n_low'])
        )

        # Step2：计算K/D - 券商版核心递推逻辑（2/3前值 + 1/3当前RSV），初始化K/D=50
        df['kdj_k'] = 50.0  # 第一根K线初始化K=50
        df['kdj_d'] = 50.0  # 第一根K线初始化D=50
        for i in range(1, len(df)):
            # K线递推：2/3*前一根K + 1/3*当前RSV
            df.loc[df.index[i], 'kdj_k'] = (2 / 3) * df.loc[df.index[i - 1], 'kdj_k'] + (1 / 3) * df.loc[
                df.index[i], 'rsv']
            # D线递推：2/3*前一根D + 1/3*当前K
            df.loc[df.index[i], 'kdj_d'] = (2 / 3) * df.loc[df.index[i - 1], 'kdj_d'] + (1 / 3) * df.loc[
                df.index[i], 'kdj_k']

        # Step3：计算J线 - 券商通用公式 3K - 2D
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        # 清理中间列，避免干扰后续绘图
        df.drop(columns=['n_high', 'n_low', 'rsv'], inplace=True)

        # 适配新版pandas的空值填充（无method参数，直接用bfill()），兼容所有版本
        df['kdj_k'] = df['kdj_k'].bfill().fillna(50.0)
        df['kdj_d'] = df['kdj_d'].bfill().fillna(50.0)
        df['kdj_j'] = df['kdj_j'].bfill().fillna(50.0)
        return df
    # 1. 调用您提供的计算函数（默认参数 period=9, k/d_period=3）
    df_with_kdj = calculate_kdj_bt(df, period=9, k_period=3, d_period=3)

    # 2. 绘制 K/D/J 线（使用鲜明且专业的颜色）
    df_with_kdj['kdj_k'].plot(ax=ax, legend='K', color='black')
    df_with_kdj['kdj_d'].plot(ax=ax, legend='D', color='orange')
    df_with_kdj['kdj_j'].plot(ax=ax, legend='J', color='red')

    # 5. 可选：添加超买/超卖区域（半透明）
    fplt.add_horizontal_band(20, 80, ax=ax)


def draw_trade_signals(df_plot, ax, buy_signals=None, sell_signals=None):
    """绘制买卖信号 - 兼容周/月线，动态偏移"""
    if buy_signals is None:
        buy_signals = []
    if sell_signals is None:
        sell_signals = []

    if len(df_plot) < 1 or (not buy_signals and not sell_signals):
        return

    # ✅ 修复1: price_offset 应为 0.005 (0.5%) 而非 0.05 (5%)
    price_offset = 0.005
    size_reduction = 0.5

    # 买入信号（↓ 三角形，放在最低价下方）
    if buy_signals:
        buy_x, buy_y = [], []
        for dt_str, _ in buy_signals:  # 价格参数未使用，用下划线忽略
            match_rows = df_plot[df_plot['date_str'] == dt_str]
            if not match_rows.empty:
                row = match_rows.iloc[0]
                signal_price = row['low'] * (1 - price_offset)
                buy_x.append(row.name)  # 使用时间戳索引
                buy_y.append(signal_price)
        if buy_x:
            fplt.plot(buy_x, buy_y, ax=ax, color='#FFD700',
                      style='v', width=2.5, legend='买入')

    # 卖出信号（↑ 三角形，放在最高价上方）
    if sell_signals:
        sell_x, sell_y = [], []
        for dt_str, _ in sell_signals:
            match_rows = df_plot[df_plot['date_str'] == dt_str]
            if not match_rows.empty:
                row = match_rows.iloc[0]
                signal_price = row['high'] * (1 + price_offset)
                sell_x.append(row.name)
                sell_y.append(signal_price)
        if sell_x:
            fplt.plot(sell_x, sell_y, ax=ax, color='#1E90FF',
                      style='^', width=2.5, legend='卖出')


#######################################################
## update crosshair and legend when moving the mouse ##

def update_legend_text(x, y):
    ts_sec = int(x // 1_000_000_000)
    if ts_sec not in df.index:
        if hover_label is not None:
            hover_label.setText('')  # 清空文本
        return
    row = df.loc[ts_sec]
    fmt = '<span style="color:#%s">%%.2f</span>' % ('red' if (row.open < row.close) else 'green')
    rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; 开%s 收%s 高%s 低%s' % (fmt, fmt, fmt, fmt)
    values = [row.open, row.close, row.high, row.low]
    if hover_label is not None:
        hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

def update_crosshair_text(x, y, xtext, ytext):
    ts_sec = int(x // 1_000_000_000)
    if ts_sec in df.index:
        close_price = df.loc[ts_sec, 'close']
        ytext = '%s (close%+.2f)' % (ytext, (y - close_price))
    return xtext, ytext



#################################### 以下是主函数部分 ##########################################

symbol = '002738'
interval = 'd'  # 日线（必须定义！）
df = download_price_history(
    symbol=symbol,
    start_time='2014-12-30',  # 你的数据起始日期
    end_time='2025-01-01'     # 数据结束日期
)

# change to b/w coloring templates for next plots
fplt.candle_bull_color = fplt.volume_bull_color = 'red'                     # K线/成交量阳线边框颜色
fplt.candle_bull_body_color = fplt.volume_bull_body_color = 'white'         # K线/成交量阳线实体颜色
fplt.candle_bear_color = fplt.candle_bear_body_color = 'green'
fplt.volume_bear_color = fplt.volume_bear_body_color = 'green'

ax, axv, ax2, ax3, ax4, ax5, ax6, ax7 = fplt.create_plot(f'A股 {symbol} 平均K线图', rows=8)
# 不显示第一个ax的网格线
ax.set_visible(xgrid=False, ygrid=False)
hover_label = fplt.add_legend('', ax=ax)  # ✅ 全局 hover_label

# price chart
plot_candlestick(df, ax=ax)
plot_ma(df, ax)

plot_heikin_ashi(df, ax7)
plot_bollinger_bands(df, ax7)
plot_ema(df, ax7)

# volume chart
plot_heikin_ashi_volume(df, axv)
plot_vma(df, ax=axv)

# some more charts
plot_accumulation_distribution(df, ax2)
plot_on_balance_volume(df, ax3)
plot_rsi(df, ax4)
plot_macd(df, ax5)
plot_kdj(df, ax6)


fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
fplt.add_crosshair_info(update_crosshair_text, ax=ax)

# restore view (X-position and zoom) when we run this example again
fplt.autoviewrestore()

fplt.show()