import sys
import json
import os
import cv2
import numpy as np
import pyautogui
from PIL import Image
from paddleocr import PaddleOCR
import requests
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit,
    QListWidget, QListWidgetItem, QHBoxLayout, QVBoxLayout, QWidget,
    QGroupBox, QInputDialog, QDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QLabel, QComboBox, QLineEdit,QCheckBox
)

HISTORY_FILE = "history.json"
CONFIG_FILE = "config.json"

class GraphProblemSolver(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图论问题解答器")
        self.setGeometry(100, 100, 1000, 600)
        self.problem_text = ""
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')
        self.model_map = {
            "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
            "Qwen (通义千问)": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "Ollama 本地": "http://localhost:11434/api/chat/completions"
        }
        self.api_model = "DeepSeek"
        self.api_url = self.model_map[self.api_model]
        self.api_key = ""
        self.load_config()
        self.init_ui()
        self.load_history_from_file()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        box = QGroupBox("输入方式")
        b_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("导入图片")
        self.screenshot_btn = QPushButton("截图识别")
        b_layout.addWidget(self.load_image_btn)
        b_layout.addWidget(self.screenshot_btn)
        box.setLayout(b_layout)
        self.load_image_btn.clicked.connect(self.load_image)
        self.screenshot_btn.clicked.connect(self.take_screenshot)

        self.input_text = QTextEdit()
        self.input_text.setFont(QFont("Arial", 12))
        self.input_text.setPlaceholderText("识别结果将在此显示，可修改后发送")

        self.send_btn = QPushButton("发送问题")
        self.send_btn.clicked.connect(self.send_to_model)
        self.export_btn = QPushButton("导出历史记录")
        self.export_btn.clicked.connect(self.export_history)
        self.adj_btn = QPushButton("输入邻接矩阵")
        self.adj_btn.clicked.connect(self.open_adj_matrix)
        self.api_btn = QPushButton("设置API")
        self.api_btn.clicked.connect(self.set_api)
        self.delete_btn = QPushButton("删除选中历史")
        self.delete_btn.clicked.connect(self.delete_selected_history)
        self.clear_btn = QPushButton("清空输入")
        self.clear_btn.clicked.connect(self.clear_input)

        self.result_text = QTextEdit()
        self.result_text.setFont(QFont("Arial", 12))
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("模型回答将在此显示")

        left.addWidget(box)
        left.addWidget(self.input_text)
        left.addWidget(self.send_btn)
        left.addWidget(self.adj_btn)
        left.addWidget(self.api_btn)
        left.addWidget(self.export_btn)
        left.addWidget(self.delete_btn)
        left.addWidget(self.result_text)
        left.addWidget(self.clear_btn)

        self.history_list = QListWidget()
        self.history_list.setFont(QFont("Courier New", 10))
        self.history_list.itemClicked.connect(self.display_history_item)
        right.addWidget(QLabel("搜索历史"))
        right.addWidget(self.history_list)

        main_layout.addLayout(left, 3)
        main_layout.addLayout(right, 1)
        central.setLayout(main_layout)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg)")
        if fname:
            img = cv2.imread(fname)
            if img is None:
                self.input_text.setPlainText("无法加载图片")
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.problem_text = self.ocr_image(img_rgb)
            self.input_text.setPlainText(self.problem_text)

    def take_screenshot(self):
        img = pyautogui.screenshot()
        img_np = np.array(img)
        dlg = CropDialog(img_np)
        if dlg.exec_() and dlg.crop_rect:
            rect = dlg.crop_rect
            crop = img_np[int(rect.y()):int(rect.y()+rect.height()), int(rect.x()):int(rect.x()+rect.width())]
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            self.problem_text = self.ocr_image(img_rgb)
            self.input_text.setPlainText(self.problem_text)
        else:
            self.input_text.setPlainText("未选中有效区域")

    def ocr_image(self, img):
        res = self.ocr_model.ocr(img, cls=True)
        return "\n".join([line[1][0] for line in res[0]])

    def send_to_model(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            return
        self.send_btn.setEnabled(False)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一名图论专家，给出解题过程(不超200字)"},
                {"role": "user", "content": text}
            ],
            "temperature": 0.5,
            "max_tokens": 512
        }
        try:
            r = requests.post(self.api_url, headers=headers, json=data)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"]
            self.result_text.setPlainText(ans)
            self.add_to_history(text, ans)
        except Exception as e:
            self.result_text.setPlainText(f"调用失败: {e}")
        finally:
            self.send_btn.setEnabled(True)

    def set_api(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("设置 API")
        layout = QVBoxLayout(dlg)

        model_box = QComboBox()
        model_box.addItems(self.model_map.keys())
        model_box.setCurrentText(self.api_model)

        url_input = QLineEdit()
        url_input.setText(self.api_url)

        key_input = QLineEdit()
        key_input.setText(self.api_key)
        key_input.setPlaceholderText("可选：API Key")

        model_box.currentTextChanged.connect(lambda name: url_input.setText(self.model_map.get(name, "")))

        layout.addWidget(QLabel("选择模型："))
        layout.addWidget(model_box)
        layout.addWidget(QLabel("API 地址："))
        layout.addWidget(url_input)
        layout.addWidget(QLabel("API Key："))
        layout.addWidget(key_input)

        btns = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        ok_btn.clicked.connect(lambda: (self.set_api_config(model_box.currentText(), url_input.text(), key_input.text()), dlg.accept()))
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def set_api_config(self, name, url, key):
        self.api_model = name
        self.api_url = url
        self.api_key = key
        self.save_config()

    def save_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({"model": self.api_model, "url": self.api_url, "key": self.api_key}, f)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    self.api_model = d.get("model", self.api_model)
                    self.api_url = d.get("url", self.api_url)
                    self.api_key = d.get("key", self.api_key)
            except:
                pass

    def open_adj_matrix(self):
        n, ok = QInputDialog.getInt(self, "邻接矩阵", "顶点数:", min=1, value=5)
        if not ok:
            return
        dlg = AdjMatrixDialog(n)
        if dlg.exec_():
            mat = dlg.get_matrix()
            disp = f"邻接矩阵(对角=∞):\n" + "\n".join([" ".join([str(x) for x in row]) for row in mat])
            self.input_text.setPlainText(disp)

    def add_to_history(self, q, a):
        item = QListWidgetItem(q[:30] + "...")
        item.setData(Qt.UserRole, (q, a))
        self.history_list.addItem(item)

    def delete_selected_history(self):
        row = self.history_list.currentRow()
        if row >= 0:
            self.history_list.takeItem(row)

    def display_history_item(self, item):
        q, a = item.data(Qt.UserRole)
        self.input_text.setPlainText(q)
        self.result_text.setPlainText(a)

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出", "", "JSON (*.json);;TXT (*.txt)")
        if not path:
            return
        data = []
        for i in range(self.history_list.count()):
            q, a = self.history_list.item(i).data(Qt.UserRole)
            data.append((q, a))
        try:
            if path.endswith('.json'):
                json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    for i, (q, a) in enumerate(data, 1):
                        f.write(f"问题{i}\n{q}\n回答{i}\n{a}\n\n")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def load_history_from_file(self):
        if os.path.exists(HISTORY_FILE):
            try:
                for q, a in json.load(open(HISTORY_FILE, 'r', encoding='utf-8')):
                    self.add_to_history(q, a)
            except:
                pass

    def closeEvent(self, e):
        json.dump([self.history_list.item(i).data(Qt.UserRole) for i in range(self.history_list.count())],
                  open(HISTORY_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        self.save_config()
        super().closeEvent(e)
    
    def clear_input(self):
        self.input_text.clear()
        self.result_text.clear()

class CropDialog(QDialog):
    def __init__(self, img_np):
        super().__init__()
        self.img_np = img_np
        h, w, _ = img_np.shape
        img = QImage(img_np.data, w, h, QImage.Format_BGR888)
        self.pix = QPixmap.fromImage(img)
        self.label = QLabel()
        self.label.setPixmap(self.pix)
        self.crop_rect = None
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.label.mousePressEvent = self.mp
        self.label.mouseMoveEvent = self.mm
        self.label.mouseReleaseEvent = self.mr

    def mp(self, e): self.start = e.pos(); self.end = e.pos()
    def mm(self, e): self.end = e.pos(); self.update_rect()
    def mr(self, e): self.end = e.pos(); self.crop_rect = QRect(self.start, self.end).normalized(); self.accept()

    def update_rect(self):
        pix = self.pix.copy()
        p = QPainter(pix)
        p.setPen(QPen(Qt.red, 2))
        p.drawRect(QRect(self.start, self.end))
        p.end()
        self.label.setPixmap(pix)

class AdjMatrixDialog(QDialog):
    def __init__(self, n):
        super().__init__()
        self.setWindowTitle('邻接矩阵')
        self.resize(400, 400)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(n, n)
        layout.addWidget(self.table)
        self.table.setHorizontalHeaderLabels([str(i) for i in range(n)])
        self.table.setVerticalHeaderLabels([str(i) for i in range(n)])
        cell_size = 50
        for i in range(n):
            self.table.setColumnWidth(i, cell_size)
            self.table.setRowHeight(i, cell_size)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setStretchLastSection(False)

        # 新增：无向图复选框
        self.undirected_box = QCheckBox("无向图（自动对称）")
        layout.addWidget(self.undirected_box)
        self.table.itemChanged.connect(self.sync_undirected)

        btns = QHBoxLayout()
        ok = QPushButton('确定')
        cancel = QPushButton('取消')
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        self.n = n

    def sync_undirected(self, item):
        if self.undirected_box.isChecked():
            row, col = item.row(), item.column()
            if row != col:
                self.table.blockSignals(True)
                self.table.setItem(col, row, QTableWidgetItem(item.text()))
                self.table.blockSignals(False)

    def get_matrix(self):
        mat = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                it = self.table.item(i, j)
                if i == j:
                    row.append(float('inf'))
                    continue
                if it and it.text():
                    try:
                        row.append(float(it.text()))
                    except:
                        row.append(float('inf'))
                else:
                    row.append(float('inf'))
            mat.append(row)
        return mat
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GraphProblemSolver()
    win.show()
    sys.exit(app.exec_())