#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 前端：参数输入 + DRL训练可视化
- Page1: 参数输入
- Page2: 训练 & 可视化
- Page1 -> Page2 参数传递 via dict
- 训练调用 drl_train.py 中的训练函数
"""

import sys
import csv
import time
from typing import List, Tuple, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QMessageBox,
    QTextEdit, QFileDialog, QFrame, QStackedWidget, QGroupBox, QProgressBar
)
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ---------------- Styling ----------------
APP_STYLE = """
QWidget{font-family: 'Segoe UI', Roboto, Helvetica, Arial; font-size:13px}
QGroupBox { border: none; }
#header { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4e9af1, stop:1 #6dd3b2); color: white; padding: 12px; border-radius: 8px; }
#title { font-weight: 700; font-size: 18px; }
#subtitle { color: rgba(255,255,255,0.95); }
QFrame.card { background: white; border-radius: 8px; border: 1px solid #e6e6e6; padding: 12px; }
QLineEdit[readonly="true"] { background: #f5f6fb; }
QPushButton#calcBtn{ background: #4e9af1; color: white; padding: 8px 14px; border-radius: 6px; }
QPushButton#calcBtn:hover{ background: #3a7ad9; }
QPushButton#clearBtn{ background: #f0f0f0; color: #333; padding: 8px 12px; border-radius: 6px; }
QPushButton#gotoTrainBtn{ background:#ffb86b; color:#222; padding:8px 12px; border-radius:6px;}
QPushButton#exportBtn{ background: #6dd3b2; color: white; padding: 8px 12px; border-radius: 6px; }
QLabel.section{ font-weight: 600; margin-bottom: 6px }
#footer { color: #666; font-size: 12px }
"""
print("drl_train.py loaded")  # 确认 Python 真正加载了这个文件

# ---------------- DRL 后端调用 ----------------
from drl_train import TrainingWorkerBackend  # 你需在 drl_train.py 中提供 QThread 兼容的 TrainingWorkerBackend


# ---------------- Training worker (线程包装 DRL 后端) ----------------
class TrainingWorker(QThread):
    """
    前端线程，调用 drl_train.py 中训练函数
    """
    log_msg = pyqtSignal(str)
    epoch_result = pyqtSignal(int, float, float)
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.worker_backend = TrainingWorkerBackend(config)

        # 绑定 backend 信号到前端信号
        self.worker_backend.log_msg.connect(self.log_msg)
        self.worker_backend.epoch_result.connect(self.epoch_result)
        self.worker_backend.progress.connect(self.progress)
        self.worker_backend.finished_signal.connect(self.finished_signal)

    def run(self):
        self.worker_backend.run_training()


# ---------------- Page 1: 参数输入 ----------------
class ParamInputPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # header
        header = QFrame()
        header.setObjectName("header")
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(10, 6, 10, 6)
        title = QLabel("耦合器参数输入工具")
        title.setObjectName("title")
        subtitle = QLabel("高频 / 大功率场景 — 交互式参数设置")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h_layout.addWidget(title)
        h_layout.addWidget(subtitle)
        header.setLayout(h_layout)
        layout.addWidget(header)

        # card
        card = QFrame()
        card.setProperty("class", "card")
        card_layout = QVBoxLayout()
        card_layout.setSpacing(8)

        scene_label = QLabel("场景与输入")
        scene_label.setProperty("class", "section")
        card_layout.addWidget(scene_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.scene_combo = QComboBox()
        self.scene_combo.addItems(["高频模式", "大功率模式"])
        form.addRow("场景:", self.scene_combo)

        dval = QDoubleValidator(bottom=-1e12, top=1e12, decimals=9)
        ival = QIntValidator(0, 1000000)

        self.v_in_edit = QLineEdit(); self.v_in_edit.setValidator(dval); self.v_in_edit.setPlaceholderText("例如：48")
        form.addRow("V_in (V)", self.v_in_edit)
        self.v_out_edit = QLineEdit(); self.v_out_edit.setValidator(dval); self.v_out_edit.setPlaceholderText("例如：12")
        form.addRow("V_out (V)", self.v_out_edit)
        self.N_ph_edit = QLineEdit(); self.N_ph_edit.setValidator(ival); self.N_ph_edit.setPlaceholderText("例如：4")
        form.addRow("N_ph", self.N_ph_edit)
        self.P_out_edit = QLineEdit(); self.P_out_edit.setValidator(dval); self.P_out_edit.setPlaceholderText("例如：1600")
        form.addRow("P_out (W)", self.P_out_edit)
        self.L_target_edit = QLineEdit(); self.L_target_edit.setValidator(dval); self.L_target_edit.setPlaceholderText("例如：2.0")
        form.addRow("L_target (uH)", self.L_target_edit)
        self.M_target_edit = QLineEdit(); self.M_target_edit.setValidator(dval); self.M_target_edit.setPlaceholderText("例如：1.0")
        form.addRow("M_target (uH)", self.M_target_edit)

        card_layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.calc_btn = QPushButton("计算输出参数"); self.calc_btn.setObjectName("calcBtn")
        self.clear_btn = QPushButton("清除"); self.clear_btn.setObjectName("clearBtn")
        self.goto_train_btn = QPushButton("进入训练"); self.goto_train_btn.setObjectName("gotoTrainBtn")
        btn_row.addWidget(self.calc_btn); btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(); btn_row.addWidget(self.goto_train_btn)
        card_layout.addLayout(btn_row)

        # 输出调试
        out_label = QLabel("调试 / 快速预览"); out_label.setProperty("class", "section")
        card_layout.addWidget(out_label)
        self.quick_debug = QTextEdit(); self.quick_debug.setReadOnly(True); self.quick_debug.setMaximumHeight(140)
        card_layout.addWidget(self.quick_debug)

        card.setLayout(card_layout)
        layout.addWidget(card)
        self.setLayout(layout)

        self.calc_btn.clicked.connect(self.on_calculate)
        self.clear_btn.clicked.connect(self.on_clear)

    def on_clear(self):
        for w in [self.v_in_edit, self.v_out_edit, self.N_ph_edit, self.P_out_edit, self.L_target_edit, self.M_target_edit]:
            w.clear()
        self.quick_debug.clear()

    def on_calculate(self):
        """点击‘计算输出参数’后计算 I_ph 并显示"""
        try:
            scene = self.scene_combo.currentText()
            V_in = float(self.v_in_edit.text())
            V_out = float(self.v_out_edit.text())
            N_ph = float(self.N_ph_edit.text())
            P_out = float(self.P_out_edit.text())
            L_target = float(self.L_target_edit.text())
            M_target = float(self.M_target_edit.text())

            # ✅ 计算相电流 I_ph
            I_ph = P_out / V_out / N_ph

            # ✅ 输出到 quick_debug 区域
            self.quick_debug.clear()
            self.quick_debug.append(f"场景={scene}")
            self.quick_debug.append(f"V_in={V_in:.2f}, V_out={V_out:.2f}, N_ph={N_ph:.0f}, P_out={P_out:.1f}")
            self.quick_debug.append(f"L_target={L_target:.2f}, M_target={M_target:.2f}")
            self.quick_debug.append(f"🔹计算得到: I_ph = {I_ph:.3f} A\n")

        except Exception as e:
            self.quick_debug.append(f"⚠️ 输入错误: {e}")

    def get_parameters(self) -> Dict[str, Any]:
        def val(e): return float(e.text()) if e.text() else None
        return {
            "scene": self.scene_combo.currentText(),
            "v_in": val(self.v_in_edit),
            "v_out": val(self.v_out_edit),
            "N_ph": val(self.N_ph_edit),
            "P_out": val(self.P_out_edit),
            "L_target": val(self.L_target_edit),
            "M_target": val(self.M_target_edit)
        }


# ---------------- Page2: 训练 & 可视化 ----------------
class TrainingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[TrainingWorker] = None
        self.train_records: List[Tuple[int, float, float]] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # header
        header = QFrame(); header.setObjectName("header")
        h_layout = QHBoxLayout(); h_layout.setContentsMargins(10,6,10,6)
        title = QLabel("训练与可视化"); title.setObjectName("title")
        subtitle = QLabel("深度强化学习训练 — 可视化与日志"); subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h_layout.addWidget(title); h_layout.addWidget(subtitle); header.setLayout(h_layout)
        layout.addWidget(header)

        # 参数显示
        self.param_display = QTextEdit(); self.param_display.setReadOnly(True); self.param_display.setMaximumHeight(120)
        layout.addWidget(QLabel("接收到的输入参数"))
        layout.addWidget(self.param_display)

        # 配置组
        cfg_group = QGroupBox()
        cfg_layout = QFormLayout()
        # === 训练参数设置（中文 + 示例提示） ===
        self.epochs_edit = QLineEdit();
        self.epochs_edit.setMaximumWidth(200)
        self.epochs_edit.setPlaceholderText("例如：200") # 初始训练设为200，正式训练设为1000

        self.lr_edit = QLineEdit();
        self.lr_edit.setMaximumWidth(200)
        self.lr_edit.setPlaceholderText("例如：0.001")

        self.batch_edit = QLineEdit();
        self.batch_edit.setMaximumWidth(200)
        self.batch_edit.setPlaceholderText("例如：64")

        cfg_layout.addRow("训练轮数：", self.epochs_edit)
        cfg_layout.addRow("学习率：", self.lr_edit)
        cfg_layout.addRow("批次更新大小：", self.batch_edit)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        # 按钮
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("开始训练"); self.export_btn = QPushButton("导出 CSV"); self.back_btn = QPushButton("返回输入界面")
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.export_btn); btn_row.addStretch(); btn_row.addWidget(self.back_btn)
        layout.addLayout(btn_row)

        # 日志
        layout.addWidget(QLabel("训练日志"))
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setMaximumHeight(160)
        layout.addWidget(self.log_area)

        # 进度条
        self.progress = QProgressBar(); layout.addWidget(self.progress)

        # 图表
        layout.addWidget(QLabel("训练曲线（Reward 与 各性能指标）"))

        # --- 两个子图：上面画 Reward，下面画六个误差 ---
        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)

        self.ax_reward = self.fig.add_subplot(211)
        self.ax_reward.set_title("Average Reward")
        self.ax_reward.set_xlabel("Epoch")
        self.ax_reward.set_ylabel("Reward")
        self.ax_reward.grid(True)

        self.ax_metrics = self.fig.add_subplot(212)
        self.ax_metrics.set_title("Metrics (L_err, k_err, Ripple, Vol, Loss, Temp)")
        self.ax_metrics.set_xlabel("Epoch")
        self.ax_metrics.set_ylabel("Value")
        self.ax_metrics.grid(True)

        # 定义七条线
        self.lines = {
            "Reward": self.ax_reward.plot([], [], label="Reward", marker="o")[0],
            "L_err": self.ax_metrics.plot([], [], label="L_err", marker="s")[0],
            "k_err": self.ax_metrics.plot([], [], label="k_err", marker="^")[0],
            "Ripple": self.ax_metrics.plot([], [], label="Ripple", marker="v")[0],
            "Vol": self.ax_metrics.plot([], [], label="Vol", marker="d")[0],
            "Loss": self.ax_metrics.plot([], [], label="Loss", marker="x")[0],
            "Temp": self.ax_metrics.plot([], [], label="Temp", marker="+")[0],
        }

        #self.ax_reward.legend(loc="upper left")
        #self.ax_metrics.legend(loc="upper left")
        self.ax_reward.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        self.ax_metrics.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.start_btn.clicked.connect(self.on_start_training)
        self.export_btn.clicked.connect(self.on_export_csv)

    def load_parameters(self, params: Dict[str, Any]):
        txt_lines = [f"{k}: {v}" for k,v in params.items()]
        self.param_display.setPlainText("\n".join(txt_lines))
        self.base_params = params

    def on_start_training(self):
        config = self.base_params.copy()
        config.update({
            "epochs": int(self.epochs_edit.text()),
            "lr": float(self.lr_edit.text()),
            "batch_size": int(self.batch_edit.text())
        })
        self.train_records.clear()

        # 不清空已有曲线，只清空数据
        for line in self.lines.values():
            line.set_data([], [])
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()
        self.ax_metrics.relim()
        self.ax_metrics.autoscale_view()
        self.canvas.draw()

        self.log_area.clear()
        self.progress.setValue(0)

        self.worker = TrainingWorker(config)
        self.worker.log_msg.connect(self.append_log)
        self.worker.epoch_result.connect(self.update_plot)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_signal.connect(lambda: self.append_log("✅ 训练完成"))

        self.worker.worker_backend.top3_signal.connect(self.show_top3_results)
        self.worker.start()




    def append_log(self, msg: str):
        self.log_area.append(msg)

    def update_plot(self, epoch: int, loss: float, reward: float):
        # 记录当前 epoch 的基本数据
        self.train_records.append((epoch, loss, reward))

        # 从日志中解析出最新一行包含指标的数据
        last_log = self.log_area.toPlainText().split("\n")[-1]
        import re
        metrics = re.findall(r"(\w+)=(-?\d+(?:\.\d+)?(?:e-?\d+)?)", last_log)

        metric_dict = {k: float(v) for k, v in metrics}


        # 获取横坐标（epoch）
        epochs = [x[0] for x in self.train_records]

        # 更新各曲线数据
        rewards = [x[2] for x in self.train_records]
        self.lines["Reward"].set_data(epochs, rewards)

        # 其他6个指标——从日志中提取（如果有则更新）
        for key in ["L_err", "k_err", "Ripple", "Vol", "Loss", "Temp"]:
            y_val = metric_dict.get(key, None)
            if y_val is not None:
                x_data = list(self.lines[key].get_xdata())
                y_data = list(self.lines[key].get_ydata())

                x_data.append(epoch)
                y_data.append(y_val)
                self.lines[key].set_data(x_data, y_data)

        self.ax_reward.relim()
        self.ax_reward.autoscale_view()
        self.ax_metrics.relim()
        self.ax_metrics.autoscale_view()

        # 刷新绘图
        self.canvas.draw()

    def on_export_csv(self):
        if not self.train_records:
            QMessageBox.warning(self, "警告", "暂无训练记录可导出")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV", "", "CSV Files (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch","Loss","Reward"])
                writer.writerows(self.train_records)
            QMessageBox.information(self, "提示", f"已导出到 {path}")

    def show_top3_results(self, top3_results):
        # 切换到结果展示页面
        main_window = self.parent().parent()  # 获取主窗口
        main_window.page3.show_results(top3_results)
        main_window.central_widget.setCurrentWidget(main_window.page3)





class ResultPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        header = QFrame(); header.setObjectName("header")
        h_layout = QHBoxLayout(); h_layout.setContentsMargins(10,6,10,6)
        title = QLabel("策略评估结果"); title.setObjectName("title")
        subtitle = QLabel("奖励最高的前三组参数"); subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h_layout.addWidget(title); h_layout.addWidget(subtitle); header.setLayout(h_layout)
        layout.addWidget(header)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.back_btn = QPushButton("返回训练界面")
        layout.addWidget(self.back_btn)
        self.back_btn.clicked.connect(lambda: self.parent().setCurrentIndex(1))  # 回到训练页

        self.setLayout(layout)

    def show_results(self, top3_results):
        text = ""
        for i, (reward, params) in enumerate(top3_results, 1):
            text += f"🏆 Top-{i} | Reward={reward:.3f}\n"
            for k, v in params.items():
                text += f"  {k}: {v:.5f}\n"
            text += "\n"
        self.result_text.setPlainText(text)




# ---------------- 主窗口 ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("耦合器 DRL 优化前端")
        self.resize(900, 700)
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.page1 = ParamInputPage()
        self.page2 = TrainingPage()
        self.central_widget.addWidget(self.page1)
        self.central_widget.addWidget(self.page2)

        self.page1.goto_train_btn.clicked.connect(self.goto_training_page)
        self.page2.back_btn.clicked.connect(self.goto_input_page)

        self.page3 = ResultPage()
        self.central_widget.addWidget(self.page3)


    def goto_training_page(self):
        params = self.page1.get_parameters()
        self.page2.load_parameters(params)
        self.central_widget.setCurrentWidget(self.page2)

    def goto_input_page(self):
        self.central_widget.setCurrentWidget(self.page1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
