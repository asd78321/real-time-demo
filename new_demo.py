import serial
import time
import sys
import os
import numpy as np
import struct
from matplotlib.figure import Figure
from keras.models import load_model

sys.path.append("{}\\DEMO.py".format(os.getcwd()))
from DEMO import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, \
    QGridLayout, QFormLayout, QLineEdit, QTextEdit

# initInf

configFileName = "./6843_pplcount_debug.cfg"
dataPortName = "COM5"
userPortName = "COM10"
pose_name = ["fall","get_up","stand or walk"]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.mainLayout = QVBoxLayout()
        self.ui.pushButton.toggle()
        self.ui.pushButton.clicked.connect(lambda: Radar_func.demo(self, self.ui))
        self.ui.pushButton_2.toggle()
        self.ui.pushButton_2.clicked.connect(lambda: Radar_func.exit_UI(self,self.ui))
        self.show()




class Radar_func:
    def exit_UI(self, ui):
        ui.graphicsView.clear()
        ui.graphicsView_2.clear()
        app.closeAllWindows()
        sys.exit()

    def serialConfig(self, configFileName, dataPortName, userPortName):
        try:
            cliPort = serial.Serial(userPortName, 115200)
            dataPort = serial.Serial(dataPortName, 921600,
                                     timeout=0.1)  # this timeout for buffer's updating transfer rate too slowly by serial port
        except serial.SerialException as se:
            print("Serial Port 0ccupied,error = ")
            print(str(se))
            return

        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            cliPort.write((i + '\n').encode())
            print(i)
            time.sleep(0.04)
        print("-----------------------------------------------------------------------")
        print('---------------------------已啟動雷達！---------------------------------')
        return cliPort, dataPort

    def readAndParseData(self, Dataport, ui, draw_heigth_y, draw_doopler_y, draw_x,prediction_data,model):
        # Constants
        magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
        readBuffer = Dataport.read(Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')

        # packet Constants
        HeaderLength = 52
        Typelength = 8
        PointcloudLength = 20

        if len(byteVec) > 8:
            if np.all(byteVec[0:8] == magicWord) and len(readBuffer) > 60:
                subFrameNum = struct.unpack('I', readBuffer[24:28])[0]
                numTLVs = struct.unpack('h', readBuffer[48:50])[0]
                typeTLV = struct.unpack('I', readBuffer[52:56])[0]
                lenTLV = struct.unpack('I', readBuffer[56:60])[0]  # include length of tlvHeader(8bytes)
                numPoints = (lenTLV - 8) // 20
                print("frames: ", subFrameNum, "numTLVs:", numTLVs, "type:", typeTLV, "length:", lenTLV, 'numPoints:',
                      numPoints)
                with open('./test.bin', 'a+b') as file:
                    file.write(readBuffer)
                if numTLVs != 0:
                    if typeTLV == 6 and numPoints > 0:
                        # init data list
                        x = []
                        y = []
                        z = []
                        data = []
                        pointClouds = []
                        range_list = []
                        azimuth_list = []
                        elevation_list = []
                        doppler_list = []

                        for numP in range(numPoints):
                            try:
                                Prange = struct.unpack('f', readBuffer[
                                                            HeaderLength + Typelength + numP * PointcloudLength:HeaderLength + Typelength + numP * PointcloudLength + 4])
                                azimuth = struct.unpack('f', readBuffer[
                                                             HeaderLength + Typelength + numP * PointcloudLength + 4:HeaderLength + Typelength + numP * PointcloudLength + 8])
                                elevation = struct.unpack('f', readBuffer[
                                                               HeaderLength + Typelength + numP * PointcloudLength + 8:HeaderLength + Typelength + numP * PointcloudLength + 12])
                                doppler = struct.unpack('f', readBuffer[
                                                             HeaderLength + Typelength + numP * PointcloudLength + 12:HeaderLength + Typelength + numP * PointcloudLength + 16])
                            except:  # Because sometimes the packet's length will not same with the packet's lenTLV
                                continue
                            range_list.append(Prange)  # range
                            azimuth_list.append(azimuth)  # azimuth
                            elevation_list.append(elevation)  # elevation
                            doppler_list.append(doppler)  # doppler

                        r = np.multiply(range_list[:], np.cos(elevation_list))
                        x = np.multiply(r[:], np.sin(azimuth_list[:]))
                        y = np.multiply(r[:], np.cos(azimuth_list[:]))
                        z = np.multiply(range_list[:], np.sin(elevation_list))
                        doppler_list = np.array(doppler_list)
                        centroid_height = np.sum(y) / len(y)
                        centroid_doopler = np.sum(doppler_list) / len(doppler_list)

                    ui.lcdNumber.setDigitCount(len(str(numPoints)))
                    ui.lcdNumber.display(numPoints)
                    ui.lcdNumber_2.setDigitCount(len(str(subFrameNum)))
                    ui.lcdNumber_2.display(subFrameNum)

                    if subFrameNum % 70 == 0 :
                        count = 70
                    else:
                        count = subFrameNum % 70
                    # print(count)
                    draw_heigth_y[count - 1] = centroid_height
                    draw_doopler_y[count - 1] = centroid_height
                    prediction_data[count - 1] = centroid_height

                    predition = np.argmax(model.predict(prediction_data.reshape(1,-1)))
                    ui.label_6.setText("{}".format(pose_name[predition]))
                    print(pose_name[predition])
                    if subFrameNum % 1 == 0:
                        ui.graphicsView.clear()
                        ui.graphicsView_2.clear()
                        ui.graphicsView.plot(draw_x, draw_heigth_y)
                        ui.graphicsView_2.plot(draw_x, draw_doopler_y)

                    app.processEvents()

    def plot_init(self):
        draw_heigth_y = np.zeros([70])
        draw_doopler_y = np.zeros([70])
        draw_x = range(1, 71)
        return draw_heigth_y, draw_doopler_y, draw_x



    def demo(self,ui):
        cliPort, dataPort = Radar_func.serialConfig(self, configFileName, dataPortName, userPortName)
        draw_heigth_y, draw_doopler_y, draw_x = Radar_func.plot_init(self)
        model = load_model("./model_cnn.h5")
        prediction_data = np.zeros([70])
        while True:
            time.sleep(0.1)
            try:
                Radar_func.readAndParseData(self, dataPort, ui, draw_heigth_y, draw_doopler_y, draw_x,prediction_data,model)
            except KeyboardInterrupt:
                dataPort.close()  # 清除序列通訊物件
                cliPort.write(('sensorStop\n').encode())
                cliPort.close()
                print('---------------------------已中斷連線！----------------------------------')
                break


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    sys.exit(app.exec_())
