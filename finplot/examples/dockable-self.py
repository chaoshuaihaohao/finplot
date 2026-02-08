#!/usr/bin/env python3

import finplot as fplt
from functools import lru_cache
from PyQt6.QtWidgets import QApplication, QGridLayout, QMainWindow, QGraphicsView, QComboBox, QLabel
from pyqtgraph.dockarea import DockArea, Dock
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
win = QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1600,800)
win.setWindowTitle("Docking charts example for finplot")

# Set width/height of QSplitter
win.setStyleSheet("QSplitter { width : 20px; height : 20px; }")

# Create docks
dock_0 = Dock("dock_0", size = (1000, 100), closable = True)
dock_1 = Dock("dock_1", size = (1000, 100), closable = True)
dock_2 = Dock("dock_2", size = (1000, 100), closable = True)
area.addDock(dock_0)
area.addDock(dock_1)
area.addDock(dock_2)

# Create example charts
combo = QComboBox()
combo.setEditable(True)
[combo.addItem(i) for i in 'AMRK META REVG TSLA TWTR WMT CT=F GC=F ^GSPC ^FTSE ^N225 EURUSD=X ETH-USD'.split()]
dock_0.addWidget(combo, 0, 0, 1, 1)
info = QLabel()
dock_0.addWidget(info, 0, 1, 1, 1)

# Chart for dock_0
ax0,ax1,ax2 = fplt.create_plot_widget(master=area, rows=3, init_zoom_periods=100)
area.axs = [ax0, ax1, ax2]
dock_0.addWidget(ax0.ax_widget, 1, 0, 1, 2)
dock_1.addWidget(ax1.ax_widget, 1, 0, 1, 2)
dock_2.addWidget(ax2.ax_widget, 1, 0, 1, 2)

# Link x-axis
ax1.setXLink(ax0)
ax2.setXLink(ax0)
win.axs = [ax0,ax1,ax2]

@lru_cache(maxsize = 15)
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

@lru_cache(maxsize = 100)
def get_name(symbol):
    return yf.Ticker(symbol).info.get("shortName") or symbol

def update(txt):
    df = download(txt)
    if len(df) < 20: # symbol does not exist
        return
    info.setText("Loading symbol name...")
    price = df [["open", "close", "high", "low"]]  # 修改列名以匹配CSV
    ma20 = df.close.rolling(20).mean()  # 修改列名
    ma50 = df.close.rolling(50).mean()  # 修改列名
    volume = df [["open", "close", "volume"]]  # 修改列名
    ax0.reset() # remove previous plots
    ax1.reset() # remove previous plots
    ax2.reset() # remove previous plots
    fplt.candlestick_ochl(price, ax = ax0)
    fplt.plot(ma20, legend = "MA-20", ax = ax1)
    fplt.plot(ma50, legend = "MA-50", ax = ax1)
    fplt.volume_ocv(volume, ax = ax2)
    fplt.refresh() # refresh autoscaling when all plots complete
    Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

combo.currentTextChanged.connect(update)
update(combo.currentText())

fplt.show(qt_exec = False) # prepares plots when they're all setup
win.show()
app.exec()