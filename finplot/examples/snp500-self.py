#!/usr/bin/env python3

import finplot as fplt
import pandas as pd

# 配置本地 CSV 文件路径和股票代码
csv_path = r'D:\业余\github\stockdata\A_data\002738_qfq_A_data.csv'
symbol = '002738'
interval = '1d'

# 读取本地 CSV 数据
df = pd.read_csv(
    csv_path,
    # 指定日期列并转换为 datetime 类型
    parse_dates=['date'],
    # 将日期列设为索引（finplot 需要时间索引）
    index_col='date'
)

# 数据列名映射（将 CSV 中的列名转换为代码需要的列名）
# CSV: open, close, high, low, volume
# 目标: Open, Close, High, Low, Volume (首字母大写，保持和原代码兼容)
df.rename(columns={
    'open': 'Open',
    'close': 'Close',
    'high': 'High',
    'low': 'Low',
    'volume': 'Volume'
}, inplace=True)

# 检查数据是否读取成功
if df.empty:
    raise ValueError(f"未读取到 {csv_path} 中的数据，请检查文件路径是否正确")
print(f"成功读取 {symbol} 数据，共 {len(df)} 行")
print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")

# 创建绘图窗口（修改标题为股票代码）
ax, ax2 = fplt.create_plot(f'{symbol} MACD', rows=2)

# plot macd with standard colors first
macd = df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean()
signal = macd.ewm(span=9).mean()
df['macd_diff'] = macd - signal
fplt.volume_ocv(df[['Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
fplt.plot(macd, ax=ax2, legend='MACD')
fplt.plot(signal, ax=ax2, legend='Signal')

# change to b/w coloring templates for next plots
fplt.candle_bull_color = fplt.candle_bear_color = fplt.candle_bear_body_color = '#000'
fplt.volume_bull_color = fplt.volume_bear_color = '#333'
fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

# plot price and volume
fplt.candlestick_ochl(df[['Open','Close','High','Low']], ax=ax)
hover_label = fplt.add_legend('', ax=ax)
axo = ax.overlay()
fplt.volume_ocv(df[['Open','Close','Volume']], ax=axo)
fplt.plot(df.Volume.ewm(span=24).mean(), ax=axo, color=1)

#######################################################
## update crosshair and legend when moving the mouse ##

def update_legend_text(x, y):
    row = df.loc[pd.to_datetime(x, unit='ns')]
    # format html with the candle and set legend
    fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open<row.Close).all() else 'a00')
    rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
    values = [row.Open, row.Close, row.High, row.Low]
    hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

def update_crosshair_text(x, y, xtext, ytext):
    ytext = '%s (Close%+.2f)' % (ytext, (y - df.iloc[x].Close))
    return xtext, ytext

fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
fplt.add_crosshair_info(update_crosshair_text, ax=ax)

fplt.show()