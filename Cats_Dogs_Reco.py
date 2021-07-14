# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HandWrittenDigit.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import numpy as np
import os
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(610, 364)
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 320, 111, 28))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(480, 140, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.predict_animal)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 281))
        self.label.setText("")
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(400, 70, 201, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_2.setGeometry(QtCore.QRect(400, 310, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_2.setFont(font)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.retranslateUi(Dialog)
        self.pushButton_2.clicked.connect(self.label.clear)
        self.pushButton_2.clicked.connect(self.textBrowser.clear)
        self.pushButton_2.clicked.connect(self.textBrowser_2.clear)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.filename = None
        self.textBrowser.setText('Select a Photo')
        self.textBrowser_2.setText('Processing !!')

    def load_Image_Path(self):
        filename = QFileDialog.getOpenFileName( filter="Image files (*.jpg *.gif)")
        imagePath = filename[0]
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        return imagePath

    def predict_animal(self):

        model = load_model('Dogs_Cats_Model.h5')

        im=Image.open(self.load_Image_Path())
        im=im.resize((32,32))

        im=np.expand_dims(im,axis=0)
        im=np.array(im)
        im=im/255
        pred = model.predict([im]) > 0.75

        if pred == 1:
            self.textBrowser.setText('Its a Dog.')
            self.textBrowser_2.setText('Completed !!')
        else:
            self.textBrowser.setText('Its a Cat.')
            self.textBrowser_2.setText('Completed !!')


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Cats_And_Dogs_Recognizer"))
        self.pushButton_2.setText(_translate("Dialog", "Clear"))
        self.pushButton_3.setText(_translate("Dialog", "Analyse"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())