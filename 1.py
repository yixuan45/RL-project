import sys
import threading
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
                             QFileDialog, QLabel, QLineEdit, QComboBox, QMessageBox,
                             QSplitter, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPalette
from mitmproxy import proxy, options
from mitmproxy.tools.dump import DumpMaster
from mitmproxy.http import HTTPFlow
import matplotlib.dates as mdates

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ProxyAddon:
    """mitmproxy 插件：捕获 HTTP/HTTPS 请求日志"""
    def __init__(self, log_callback):
        self.log_callback = log_callback  # 日志回调函数（传递给主窗口）
        self.start_time = time.time()

    def response(self, flow: HTTPFlow):
        """响应处理：提取请求/响应信息"""
        if flow.request and flow.response:
            try:
                # 提取核心字段
                log_data = {
                    "timestamp": datetime.fromtimestamp(flow.request.timestamp_start).strftime("%Y-%m-%d %H:%M:%S"),
                    "method": flow.request.method,
                    "url": flow.request.url.split('?')[0],  # 去除查询参数
                    "full_url": flow.request.url,
                    "status_code": flow.response.status_code,
                    "remote_ip": flow.server_conn.peername[0] if flow.server_conn.peername else "",
                    "local_ip": flow.client_conn.peername[0] if flow.client_conn.peername else "",
                    "user_agent": flow.request.headers.get("User-Agent", ""),
                    "content_type": flow.response.headers.get("Content-Type", ""),
                    "response_time": round((flow.response.timestamp_end - flow.request.timestamp_start) * 1000, 2),  # 毫秒
                    "bytes_sent": flow.request.headers.get("Content-Length", 0),
                    "bytes_received": flow.response.headers.get("Content-Length", 0)
                }
                # 回调传递日志数据
                self.log_callback(log_data)
            except Exception as e:
                print(f"日志提取失败：{e}")

class ProxyThread(QThread):
    """代理服务器线程（避免阻塞 GUI）"""
    error_signal = pyqtSignal(str)

    def __init__(self, port=8080, log_callback=None):
        super().__init__()
        self.port = port
        self.log_callback = log_callback
        self.is_running = False

    def run(self):
        try:
            # 配置 mitmproxy
            opts = options.Options(listen_host='0.0.0.0', listen_port=self.port)
            pconf = proxy.config.ProxyConfig(opts)
            self.master = DumpMaster(opts, with_termlog=False, with_dumper=False)
            self.master.server = proxy.server.ProxyServer(pconf)
            # 添加自定义插件
            self.master.addons.add(ProxyAddon(self.log_callback))
            self.is_running = True
            self.master.run()  # 启动代理
        except Exception as e:
            self.error_signal.emit(f"代理启动失败：{str(e)}")
            self.is_running = False

    def stop(self):
        if self.is_running and hasattr(self, 'master'):
            self.master.shutdown()
            self.is_running = False
            self.wait()

class LogAnalyzerWindow(QMainWindow):
    """主窗口：整合代理、表格、图表"""
    log_add_signal = pyqtSignal(dict)  # 跨线程传递日志信号

    def __init__(self):
        super().__init__()
        self.setWindowTitle("网络日志分析系统")
        self.setGeometry(100, 100, 1400, 800)
        self.log_data = []  # 存储所有日志
        self.proxy_thread = None  # 代理线程
        self.init_ui()
        self.init_signal()

    def init_ui(self):
        """初始化界面布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 顶部控制栏
        control_layout = QHBoxLayout()
        # 代理控制
        self.proxy_btn = QPushButton("启动代理")
        self.proxy_btn.clicked.connect(self.toggle_proxy)
        self.proxy_label = QLabel(f"代理状态：未运行 | 端口：8080")
        control_layout.addWidget(self.proxy_btn)
        control_layout.addWidget(self.proxy_label)
        control_layout.addSpacing(20)

        # 日志操作
        self.import_btn = QPushButton("导入日志")
        self.export_btn = QPushButton("导出日志")
        self.clear_btn = QPushButton("清空日志")
        self.import_btn.clicked.connect(self.import_logs)
        self.export_btn.clicked.connect(self.export_logs)
        self.clear_btn.clicked.connect(self.clear_logs)
        control_layout.addWidget(self.import_btn)
        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addStretch()

        # 筛选控件
        self.filter_label = QLabel("筛选状态码：")
        self.status_filter = QComboBox()
        self.status_filter.addItems(["全部", "2xx", "3xx", "4xx", "5xx"])
        self.status_filter.currentTextChanged.connect(self.filter_logs)
        control_layout.addWidget(self.filter_label)
        control_layout.addWidget(self.status_filter)

        main_layout.addLayout(control_layout)

        # 2. 主体内容（拆分面板：表格 + 图表）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧：日志表格
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "时间", "方法", "URL", "状态码", "远程IP", "响应时间(ms)", "用户代理", "内容类型"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        splitter.addWidget(self.table)

        # 右侧：图表标签页
        self.chart_tab = QTabWidget()
        # 初始化4个图表面板
        self.trend_chart = self.create_chart_widget("请求趋势分析")
        self.status_chart = self.create_chart_widget("状态码分布")
        self.domain_chart = self.create_chart_widget("Top 10 访问域名")
        self.time_chart = self.create_chart_widget("响应时间分布")
        self.chart_tab.addTab(self.trend_chart, "请求趋势")
        self.chart_tab.addTab(self.status_chart, "状态码分布")
        self.chart_tab.addTab(self.domain_chart, "Top域名")
        self.chart_tab.addTab(self.time_chart, "响应时间")
        splitter.addWidget(self.chart_tab)

        # 3. 底部统计信息
        self.stats_label = QLabel("日志总数：0 | 成功请求：0 | 失败请求：0 | 平均响应时间：0ms")
        main_layout.addWidget(self.stats_label)

    def init_signal(self):
        """初始化信号槽"""
        self.log_add_signal.connect(self.add_log_to_table)  # 接收日志信号

    def create_chart_widget(self, title):
        """创建图表容器"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"<h3>{title}</h3>"))
        # 用于显示图表的标签（matplotlib 绘图后转为图片）
        self.__dict__[f"{title}_label"] = QLabel()
        layout.addWidget(self.__dict__[f"{title}_label"])
        return widget

    def toggle_proxy(self):
        """启动/停止代理服务器"""
        if self.proxy_thread and self.proxy_thread.is_running:
            # 停止代理
            self.proxy_thread.stop()
            self.proxy_btn.setText("启动代理")
            self.proxy_label.setText("代理状态：未运行 | 端口：8080")
            QMessageBox.information(self, "提示", "代理已停止！")
        else:
            # 启动代理
            self.proxy_thread = ProxyThread(port=8080, log_callback=self.on_log_received)
            self.proxy_thread.error_signal.connect(self.on_proxy_error)
            self.proxy_thread.start()
            # 等待代理启动（简单延时）
            time.sleep(0.5)
            if self.proxy_thread.is_running:
                self.proxy_btn.setText("停止代理")
                self.proxy_label.setText("代理状态：运行中 | 端口：8080")
                QMessageBox.information(self, "提示", 
                    "代理已启动！\n请在浏览器中设置代理：127.0.0.1:8080\n"
                    "HTTPS 需安装 mitmproxy 证书（运行 mitmdump 后自动生成）")
            else:
                QMessageBox.warning(self, "警告", "代理启动失败！")

    def on_log_received(self, log_data):
        """代理捕获日志后的回调（运行在代理线程）"""
        self.log_data.append(log_data)
        self.log_add_signal.emit(log_data)  # 发送信号到主线程

    def add_log_to_table(self, log_data):
        """添加日志到表格（运行在主线程）"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        # 填充表格数据
        columns = [
            log_data["timestamp"], log_data["method"], log_data["url"],
            str(log_data["status_code"]), log_data["remote_ip"],
            str(log_data["response_time"]), log_data["user_agent"][:50] + "..." if len(log_data["user_agent"]) > 50 else log_data["user_agent"],
            log_data["content_type"]
        ]
        for col, value in enumerate(columns):
            item = QTableWidgetItem(value)
            # 状态码颜色标记（2xx绿色，4xx/5xx红色）
            if col == 3:
                if str(value).startswith("2"):
                    item.setForeground(QColor("green"))
                elif str(value).startswith(("4", "5")):
                    item.setForeground(QColor("red"))
            self.table.setItem(row, col, item)
        # 更新统计和图表
        self.update_stats()
        self.update_charts()

    def filter_logs(self):
        """根据状态码筛选日志"""
        filter_text = self.status_filter.currentText()
        # 清空表格
        self.table.setRowCount(0)
        # 筛选后的数据
        filtered_data = self.log_data
        if filter_text != "全部":
            prefix = filter_text[0]
            filtered_data = [log for log in self.log_data if str(log["status_code"]).startswith(prefix)]
        # 重新填充表格
        for log in filtered_data:
            self.add_log_to_table(log)

    def update_stats(self):
        """更新底部统计信息"""
        total = len(self.log_data)
        success = len([log for log in self.log_data if str(log["status_code"]).startswith("2")])
        failed = len([log for log in self.log_data if str(log["status_code"]).startswith(("4", "5"))])
        avg_time = round(sum([log["response_time"] for log in self.log_data])/total, 2) if total > 0 else 0
        self.stats_label.setText(
            f"日志总数：{total} | 成功请求：{success} | 失败请求：{failed} | 平均响应时间：{avg_time}ms"
        )

    def update_charts(self):
        """更新所有图表"""
        if len(self.log_data) == 0:
            return
        df = pd.DataFrame(self.log_data)
        self.plot_trend_chart(df)
        self.plot_status_chart(df)
        self.plot_domain_chart(df)
        self.plot_time_chart(df)

    def plot_trend_chart(self, df):
        """请求趋势图（按分钟统计）"""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["minute"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        trend_data = df.groupby("minute").size()

        plt.figure(figsize=(8, 4))
        trend_data.plot(kind="line", color="#1f77b4", marker="o", linewidth=2)
        plt.title("请求趋势（按分钟）")
        plt.xlabel("时间")
        plt.ylabel("请求次数")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 保存为图片并显示在 GUI
        self.save_chart_to_label(plt, self.trend_chart_label)

    def plot_status_chart(self, df):
        """状态码分布图（饼图）"""
        status_counts = df["status_code"].astype(str).str[:1].value_counts()
        status_labels = [f"{code}xx ({count}次)" for code, count in status_counts.items()]
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

        plt.figure(figsize=(6, 6))
        plt.pie(status_counts.values, labels=status_labels, colors=colors[:len(status_counts)], autopct="%1.1f%%")
        plt.title("状态码分布")
        plt.axis("equal")
        self.save_chart_to_label(plt, self.status_chart_label)

    def plot_domain_chart(self, df):
        """Top 10 访问域名（柱状图）"""
        # 提取域名（从 URL 中）
        df["domain"] = df["url"].apply(lambda x: x.split("//")[-1].split("/")[0])
        domain_counts = df["domain"].value_counts().head(10)

        plt.figure(figsize=(8, 4))
        domain_counts.plot(kind="bar", color="#ff7f0e")
        plt.title("Top 10 访问域名")
        plt.xlabel("域名")
        plt.ylabel("访问次数")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.save_chart_to_label(plt, self.domain_chart_label)

    def plot_time_chart(self, df):
        """响应时间分布（直方图）"""
        # 过滤异常值（大于1000ms的按1000ms处理）
        response_times = df["response_time"].clip(upper=1000)

        plt.figure(figsize=(8, 4))
        plt.hist(response_times, bins=20, color="#2ca02c", alpha=0.7)
        plt.title("响应时间分布（ms）")
        plt.xlabel("响应时间")
        plt.ylabel("请求次数")
        plt.axvline(response_times.mean(), color="red", linestyle="--", label=f"平均值：{response_times.mean():.1f}ms")
        plt.legend()
        plt.tight_layout()
        self.save_chart_to_label(plt, self.time_chart_label)

    def save_chart_to_label(self, plt_obj, label):
        """将 matplotlib 图表保存为图片并显示在 QLabel"""
        from io import BytesIO, StringIO
        from PIL import Image, ImageQt

        buf = BytesIO()
        plt_obj.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        qt_img = ImageQt.ImageQt(img)
        label.setPixmap(qt_img.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        plt_obj.close()

    def import_logs(self):
        """导入本地日志文件（JSON格式）"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择日志文件", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    imported_data = json.load(f)
                # 验证日志格式
                if isinstance(imported_data, list) and len(imported_data) > 0 and "url" in imported_data[0]:
                    self.log_data.extend(imported_data)
                    # 刷新表格和图表
                    self.filter_logs()
                    self.update_stats()
                    self.update_charts()
                    QMessageBox.information(self, "成功", f"导入 {len(imported_data)} 条日志！")
                else:
                    QMessageBox.warning(self, "错误", "日志文件格式无效！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导入失败：{str(e)}")

    def export_logs(self):
        """导出日志到 JSON/Excel 文件"""
        if len(self.log_data) == 0:
            QMessageBox.warning(self, "警告", "没有日志可导出！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存日志文件", "", "JSON Files (*.json);;Excel Files (*.xlsx)")
        if file_path:
            try:
                if file_path.endswith(".json"):
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(self.log_data, f, ensure_ascii=False, indent=2)
                elif file_path.endswith(".xlsx"):
                    df = pd.DataFrame(self.log_data)
                    df.to_excel(file_path, index=False)
                QMessageBox.information(self, "成功", "日志导出完成！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导出失败：{str(e)}")

    def clear_logs(self):
        """清空所有日志"""
        if QMessageBox.question(self, "确认", "是否清空所有日志？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.log_data.clear()
            self.table.setRowCount(0)
            self.update_stats()
            # 清空图表
            for tab_name in ["请求趋势分析", "状态码分布", "Top 10 访问域名", "响应时间分布"]:
                self.__dict__[f"{tab_name}_label"].clear()

    def on_proxy_error(self, error_msg):
        """代理启动错误处理"""
        QMessageBox.critical(self, "错误", error_msg)
        self.proxy_btn.setText("启动代理")
        self.proxy_label.setText("代理状态：未运行 | 端口：8080")

    def closeEvent(self, event):
        """窗口关闭时停止代理"""
        if self.proxy_thread and self.proxy_thread.is_running:
            self.proxy_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LogAnalyzerWindow()
    window.show()
    sys.exit(app.exec())