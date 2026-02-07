#!/usr/bin/env python3

import finplot as fplt
import numpy as np
import pandas as pd
import dateutil.parser
import os


# 时间戳转换函数
def local2timestamp(s):
    return int(dateutil.parser.parse(s).timestamp())


# 读取本地CSV数据函数
def download_local_data(csv_file_path, start_time='2014-12-30', end_time='2025-01-01'):
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"本地CSV文件不存在：{csv_file_path}，请检查路径是否正确！")
    # 读取CSV文件（使用所需列）
    df = pd.read_csv(csv_file_path, usecols=['date', 'open', 'close', 'high', 'low', 'volume'])

    # 关键修复：先确保日期列是字符串类型，避免转换错误
    df['date'] = df['date'].astype(str)

    # 日期列转换为时间戳并筛选时间范围
    df['time'] = df['date'].apply(lambda x: local2timestamp(x))
    start_ts = local2timestamp(start_time)
    end_ts = local2timestamp(end_time)
    df = df[(df['time'] >= start_ts) & (df['time'] <= end_ts)]

    # 移除空值，确保数据完整性
    df = df.dropna(subset=['time', 'open', 'close', 'high', 'low', 'volume'])

    # 确保数值列是浮点型
    for col in ['open', 'close', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 再次移除转换失败的行
    df = df.dropna(subset=['open', 'close', 'high', 'low', 'volume'])

    # 重命名列以匹配原有代码的列名
    df = df.rename(columns={
        'open': 'Open',
        'close': 'Close',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume'
    })

    # 设置日期为索引
    df['Date'] = pd.to_datetime(df['date'])
    df = df.set_index('Date')

    # 添加Adj Close列（和Close保持一致）
    df['Adj Close'] = df['Close']

    # 终极修复：重新创建整个DataFrame，确保所有数据都是可写的
    new_df = pd.DataFrame()
    new_df.index = df.index.copy()

    # 只处理数值列，跳过日期相关列
    numeric_cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close']
    for col in numeric_cols:
        # 强制创建可写的float64数组
        new_df[col] = np.array(df[col].values, dtype=np.float64).copy()

    return new_df


# 读取本地002738股票数据
csv_file_path = r'D:\业余\github\stockdata\A_data\002738_qfq_A_data.csv'
try:
    btc = download_local_data(
        csv_file_path=csv_file_path,
        start_time='2014-09-01',
        end_time='2025-01-01'
    )
except Exception as e:
    print(f"读取本地数据失败: {e}")
    exit(1)

# 检查数据是否为空
if btc.empty or len(btc) == 0:
    print("错误：本地数据为空，无法继续执行！")
    exit(1)

# 修复finplot的只读数组问题：替换内部的数组处理逻辑
# 先创建时间戳数组（finplot内部使用）
timestamps = np.array([t.timestamp() for t in btc.index], dtype=np.float64).copy()

# 原有分析图表逻辑（修改数据传递方式）
ax1, ax2, ax3, ax4, ax5 = fplt.create_plot('002738 长期分析', rows=5, maximize=False)
fplt.set_y_scale(ax=ax1, yscale='log')

# 1. 价格和均线绘制（使用时间戳+数值的方式传递）
close = np.array(btc.Close.values, dtype=np.float64).copy()
fplt.plot(timestamps, close, color='#000', legend='Log price', ax=ax1)

# 计算均线
ma200 = np.array(btc.Close.rolling(200).mean().values, dtype=np.float64).copy()
ma50 = np.array(btc.Close.rolling(50).mean().values, dtype=np.float64).copy()
fplt.plot(timestamps, ma200, legend='MA200', ax=ax1)
fplt.plot(timestamps, ma50, legend='MA50', ax=ax1)

# 成交量OCV（特殊处理：避免finplot内部修改数据）
one = np.ones_like(close, dtype=np.float64).copy()
volume_data = np.column_stack([ma200, ma50, one])
# 创建临时DataFrame用于volume_ocv
temp_df = pd.DataFrame(volume_data, index=btc.index, columns=['ma200', 'ma50', 'one'])
temp_df = temp_df.copy(deep=True)
fplt.volume_ocv(temp_df, candle_width=1, ax=ax1.overlay(scale=0.02))

# 2. 日收益率
daily_ret = np.array(btc.Close.pct_change() * 100, dtype=np.float64).copy()
fplt.plot(timestamps, daily_ret, width=3, color='#000', legend='Daily returns %', ax=ax2)

# 3. 日收益率直方图（过滤NaN）
fplt.add_legend('Daily % returns histogram', ax=ax3)
valid_daily_ret = daily_ret[~np.isnan(daily_ret)]
fplt.hist(valid_daily_ret, bins=60, ax=ax3)

# 4. 年收益率
yearly_data = btc.Close.resample('YE').last()
yearly_ret = np.array(yearly_data.pct_change().dropna() * 100, dtype=np.float64).copy()
yearly_timestamps = np.array([t.timestamp() for t in yearly_data.pct_change().dropna().index], dtype=np.float64).copy()
fplt.bar(yearly_timestamps, yearly_ret, ax=ax4)
fplt.add_legend('Yearly returns in %', ax=ax4)

# 5. 月度收益率热力图
monthly_data = btc['Adj Close'].resample('ME').last()
monthly_ret = monthly_data.pct_change().dropna() * 100
months_index = monthly_ret.index.month_name().to_list()
mnames = months_index[months_index.index('January'):][:12] if 'January' in months_index else months_index[:12]

# 计算月度均值并确保数组可写
mrets = []
for mname in mnames:
    month_vals = monthly_ret[monthly_ret.index.month_name() == mname].values
    if len(month_vals) > 0:
        mrets.append(np.mean(month_vals))
    else:
        mrets.append(0.0)
mrets = np.array(mrets, dtype=np.float64).copy()

# 创建热力图数据
hmap_data = mrets.reshape((3, 4)).T if len(mrets) >= 12 else np.zeros((4, 3))
hmap = pd.DataFrame(hmap_data, columns=[2, 1, 0])
hmap = hmap.reset_index()
# 确保热力图数据可写
for col in hmap.columns:
    hmap[col] = np.array(hmap[col].values, dtype=np.float64).copy()

colmap = fplt.ColorMap([0.3, 0.5, 0.7], [[255, 110, 90], [255, 247, 0], [60, 255, 50]])
fplt.heatmap(hmap, rect_size=1, colmap=colmap, colcurve=lambda x: x, ax=ax5)

# 添加月度文本标签（增加边界检查）
mnames_reshaped = np.array(mnames).reshape((3, 4)) if len(mnames) >= 12 else np.array(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']).reshape((3, 4))
for j, mrow in enumerate(mnames_reshaped):
    for i, month in enumerate(mrow):
        if i < len(hmap) and (2 - j) in hmap.columns:
            val = hmap.loc[i, 2 - j] if i < len(hmap) else 0.0
            s = month + ' %+.2f%%' % val
            fplt.add_text((i, 2.5 - j), s, anchor=(0.5, 0.5), ax=ax5)

ax5.set_visible(crosshair=False, xaxis=False, yaxis=False)

# 关键修复：强制刷新前确保所有数据都是可写的
fplt.refresh()

fplt.show()