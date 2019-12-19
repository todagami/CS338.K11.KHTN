from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(775, 456)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selectImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.selectImageBtn.setGeometry(QtCore.QRect(10, 140, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.selectImageBtn.setFont(font)
        self.selectImageBtn.setObjectName("selectImageBtn")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(120, 30, 271, 271))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(470, 320, 271, 51))
        self.listWidget.setObjectName("listWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(400, 140, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.imageLbl_2 = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl_2.setGeometry(QtCore.QRect(470, 30, 271, 271))
        self.imageLbl_2.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl_2.setText("")
        self.imageLbl_2.setObjectName("imageLbl_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(410, 330, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 21))
        self.menubar.setObjectName("menubar")
        self.menuDEMO = QtWidgets.QMenu(self.menubar)
        self.menuDEMO.setObjectName("menuDEMO")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuDEMO.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #################################################

        self.selectImageBtn.clicked.connect(self.setImage)
        self.pushButton.clicked.connect(self.addItem)

        #################################################
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selectImageBtn.setText(_translate("MainWindow", "Select Image"))
        self.pushButton.setText(_translate("MainWindow", "=>"))
        self.label.setText(_translate("MainWindow", "Result:"))
        self.menuDEMO.setTitle(_translate("MainWindow", "DEMO"))
    
    def setImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)")  # Ask for file
        if fileName:  # If the user gives a file
            # Setup pixmap with the provided image
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(
            ), QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.imageLbl.setPixmap(pixmap)  # Set the pixmap onto the label
            # Align the label to center
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
            
        # load the traffic sign recognizer model
        print("[INFO] loading model...")
        model = load_model("output\\trafficsignnet.model")

        # load the label names
        labelNames = open("signnames.csv").read().strip().split("\n")[1:]
        labelNames = [l.split(",")[1] for l in labelNames]

        # grab the paths to the input images, shuffle them, and grab a sample
        print("[INFO] predicting...")
        imagePaths = fileName

        # load the image, resize it to 32x32 pixels, and then apply
        # Contrast Limited Adaptive Histogram Equalization (CLAHE),
        # just like we did during training
        image = io.imread(imagePaths)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        # preprocess the image by scaling it to the range [0, 1]
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        # make predictions using the traffic sign recognizer CNN
        preds = model.predict(image)
        j = preds.argmax(axis=1)[0]
        label = labelNames[j]

        # load the image using OpenCV, resize it, and draw the label
        # on it
        image = cv2.imread(imagePaths)
        image = imutils.resize(image, width=128)
        cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)
        with open('result.txt', 'w') as f:
            f.write(label)
        # save the image to disk
        p = os.path.sep.join(["examples", "result.png"])
        cv2.imwrite(p, image)

    def addItem(self):
        fileName1 = "examples\\result.png"
        if fileName1:  # If the user gives a file
            # Setup pixmap with the provided image
            pixmap = QtGui.QPixmap(fileName1)
            pixmap = pixmap.scaled(self.imageLbl_2.width(), self.imageLbl_2.height(
            ), QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.imageLbl_2.setPixmap(pixmap)  # Set the pixmap onto the label
            # Align the label to center
            self.imageLbl_2.setAlignment(QtCore.Qt.AlignCenter)

        with open('result.txt', 'r') as f:
            value = f.read()
        self.listWidget.clear() # Clear the pre result
        self.listWidget.addItem(value) # Add the value we got to the list

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

