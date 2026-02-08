#!/usr/bin/env python3

import finplot as fplt
from functools import lru_cache
from PyQt6.QtWidgets import QApplication, QGridLayout, QGraphicsView, QComboBox, QLabel
from threading import Thread
import yfinance as yf
import os
import pandas as pd
from datetime import datetime
import time

def local2timestamp(date_str):
    """将 'YYYY-MM-DD' 格式的日期字符串转换为时间戳"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return int(time.mktime(dt.timetuple())) * 1000  # finplot expects milliseconds

app = QApplication([])
win = QGraphicsView()
win.setWindowTitle('TradingView wannabe')
layout = QGridLayout()
win.setLayout(layout)
win.resize(600, 500)

combo = QComboBox()
combo.setEditable(True)
[combo.addItem(i) for i in 'AMRK META REVG TSLA WMT CT=F GC=F ^GSPC ^FTSE ^N225 EURUSD=X ETH-USD'.split()]
layout.addWidget(combo, 0, 0, 1, 1)
info = QLabel()
layout.addWidget(info, 0, 1, 1, 1)

ax = fplt.create_plot(init_zoom_periods=100)
win.axs = [ax] # finplot requres this property
axo = ax.overlay()
layout.addWidget(ax.vb.win, 1, 0, 1, 2)


@lru_cache(maxsize=15)
def download(symbol='XBTUSD', start_time='2023-01-01', end_time='2023-10-29', interval_mins=60):
    csv_file_path = r'D:\outsidework\github\stockdata\A_data\002738_qfq_A_data.csv'
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

@lru_cache(maxsize=100)
def get_name(symbol):
    return yf.Ticker(symbol).info.get('shortName') or symbol

plots = []
def update(txt):
    df = download(txt)
    if len(df) < 20: # symbol does not exist
        return
    info.setText('Loading symbol name...')
    price = df['open close high low'.split()]
    ma20 = df.close.rolling(20).mean()
    ma50 = df.close.rolling(50).mean()
    volume = df['open close volume'.split()]
    ax.reset() # remove previous plots
    axo.reset() # remove previous plots
    fplt.candlestick_ochl(price)
    fplt.plot(ma20, legend='MA-20')
    fplt.plot(ma50, legend='MA-50')
    fplt.volume_ocv(volume, ax=axo)
    fplt.refresh() # refresh autoscaling when all plots complete
    Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

combo.currentTextChanged.connect(update)
update(combo.currentText())


fplt.show(qt_exec=False) # prepares plots when they're all setup
win.show()
app.exec()
