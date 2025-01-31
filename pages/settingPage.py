from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class settingPage(QWidget):
    def __init__(self):
        super(settingPage, self).__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Settings Page"))
        layout.addWidget(QPushButton("Save Settings"))
        self.setLayout(layout)
