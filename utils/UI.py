import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.timer = None
        self.stop_button = None
        self.start_button = None
        self.image_label = None
        self.initUI()
        self.cap = cv2.VideoCapture(0)

    def initUI(self):
        self.setWindowTitle('Camera App')

        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.timer.start(20)

    def stop_camera(self):
        self.timer.stop()
        self.cap.release()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CameraApp()
    ex.show()
    sys.exit(app.exec_())
