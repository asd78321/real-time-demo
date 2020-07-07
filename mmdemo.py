import serial
import time
import numpy as np
import struct

import tensorflow as tf

# global CLIport, Dataport
# CLIport = {}
# Dataport = {}






# ------------------------------------------------------------------


def serialConfig(configFileName, dataPortName, userPortName):
    try:
        cliPort = serial.Serial(userPortName, 115200)
        dataPort = serial.Serial(dataPortName, 921600, timeout=0.08)# this timeout for buffer's updating transfer rate too slowly by serial port
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


# 定義空間
def voxalize(x_points, y_points, z_points, x, y, z):
    # voxel切割大小修改爲（50,30,50） 叠加frame數量修改為12 sliding window's offset為3
    # 定義房間大小
    x_min = -3
    x_max = 3

    y_min = 0.0
    y_max = 2.5

    z_max = 3
    z_min = -3

    z_res = (z_max - z_min) / z_points
    y_res = (y_max - y_min) / y_points
    x_res = (x_max - x_min) / x_points

    # new method for Feature-matrix

    pixel_x_y = np.zeros([x_points * y_points])
    pixel_y_z = np.zeros([z_points * y_points])
    pixel_x_z = np.zeros([x_points * z_points])

    for i in range(len(y)):

        x_pix = (x[i] - x_min) // x_res
        y_pix = (y[i] - y_min) // y_res
        z_pix = (z[i] - z_min) // z_res

        if x_pix > x_points:
            continue
        if y_pix > y_points:
            continue
        if z_pix > z_points:
            continue

        if x_pix == x_points:
            x_pix = x_points - 1
        if y_pix == y_points:
            y_pix = y_points - 1
        if z_pix == z_points:
            z_pix = z_points - 1

        pixel_x_y[int((y_pix) * x_points + x_pix)] = pixel_x_y[int((y_pix) * x_points + x_pix)] + 1
        pixel_y_z[int((y_pix) * z_points + z_pix)] = pixel_y_z[int((y_pix) * z_points + z_pix)] + 1
        pixel_x_z[int((z_pix) * x_points + x_pix)] = pixel_x_z[int((z_pix) * x_points + x_pix)] + 1

    pixel_x_y = np.array(pixel_x_y).reshape(x_points, y_points)
    pixel_y_z = np.array(pixel_y_z).reshape(y_points, z_points)
    pixel_x_z = np.array(pixel_x_z).reshape(x_points, z_points)

    return pixel_x_y, pixel_y_z, pixel_x_z


# ----------------------------------------------------------------------------------------------------------------------

def readAndParseData(Dataport):
    global byteBuffer, byteBufferLength

    # Constants
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    framedata = []
    # print(len(byteVec))

    # --------------------------------For point cloud-----------------------------------------------------------------------
    if np.all(byteVec[0:8] == magicWord) and len(readBuffer)>52:
        subFrameNum = struct.unpack('I', readBuffer[24:28])[0]
        numTLVs = struct.unpack('h', readBuffer[48:50])[0]
        typeTLV = struct.unpack('I', readBuffer[52:56])[0]
        lenTLV = struct.unpack('I', readBuffer[56:60])[0]  # include length of tlvHeader(8bytes)
        numPoints = (lenTLV - 8) // 20
        # print("frames: ",subFrameNum,"numTLVs:",numTLVs,"type:",typeTLV,"length:",lenTLv,'numPoints:',numPoints)
        PointcloudLength = 20

        # TLVpointCLOUD start index
        HeaderLength = 52
        Typelength = 8

        if typeTLV == 6 and numPoints > 0:
            # Initialize variables
            x = []
            y = []
            z = []
            pointClouds = []
            range_list = []
            azimuth_list = []
            elevation_list = []
            doppler_list = []
            for numP in range(numPoints):
                try:
                    Prange = struct.unpack('f', readBuffer[HeaderLength+Typelength+numP*PointcloudLength:HeaderLength+Typelength+numP*PointcloudLength + 4])
                    azimuth = struct.unpack('f', readBuffer[HeaderLength+Typelength+numP*PointcloudLength + 4:HeaderLength+Typelength+numP*PointcloudLength + 8])
                    elevation = struct.unpack('f', readBuffer[HeaderLength+Typelength+numP*PointcloudLength + 8:HeaderLength+Typelength+numP*PointcloudLength + 12])
                    doppler = struct.unpack('f', readBuffer[HeaderLength+Typelength+numP*PointcloudLength + 12:HeaderLength+Typelength+numP*PointcloudLength + 16])
                    framedata.append(pointClouds)
                except:# Because sometimes the packet's length will not same with the packet's lenTLV
                    continue
                # spherical coordinate system

                range_list.append(Prange) # range
                azimuth_list.append(azimuth) # azimuth
                elevation_list.append(elevation) # elevation
                doppler_list.append(doppler) # doppler

                # Pack to List with pointCloud(spherical)
                pointClouds.append([range_list, azimuth_list, elevation_list, doppler_list])

            # Cartesian coordinate system
            r = np.multiply(range_list[:], np.cos(elevation_list))
            x = np.multiply(r[:], np.sin(azimuth_list[:]))
            y = np.multiply(r[:], np.cos(azimuth_list[:]))
            z = np.multiply(range_list[:], np.sin(elevation_list))

            # Feature Matrix preprocess(Voxalize)
            p_x_y, p_y_z, p_z_x = voxalize(50, 30, 50, x, y, z)

            # Frame Data not null from Serial-port
            isnull = 0

        return subFrameNum, p_x_y, p_y_z, p_z_x, isnull
    else:
        isnull = 1
        return [],[],[],[], isnull

    # ----------------------------------------------------------------------------------------------------------------------


def stack_data(frames, x, y,input_1,input_2):

    x = np.reshape(x, (50, 30)) # Input1 Matrix setup
    y = np.reshape(y, (30, 50))# Input2 Matrix setup



    if frames < 12:
        input_1[frames, :, :] = x
        input_2[frames, :, :] = y
        print("Frames:{},stacking...".format(frames))
    else:

        input_1[0:11, :, :] = input_1[1:12, :, :]
        input_1[11, :, :] = x

        input_2[0:11, :, :] = input_2[1:12, :, :]
        input_2[11, :, :] = y


    return input_1,input_2

def model_init():

    # Define Name List of Classified results
    classes = ["st_sit", "sit_st", "sit_lie", "lie_sit", "fall", "grow_up", "other"]

    # create Interpreter for model
    interpreter = tf.lite.Interpreter(model_path="./converted_model2.tflite")
    interpreter.allocate_tensors()

    # Model Input/Output Matrix format
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Output's shape:", output_details[0]['shape'])
    print("Intput's shape:{}\n{}\n".format(input_details[0]['shape'], input_details[1]['shape']))

    # Define Input dimension
    input_1 = np.zeros([12, 50, 30])
    input_2 = np.zeros([12, 30, 50])

    return classes,interpreter,input_1,input_2


def prediction(input_1, input_2,interpreter,classes):
    # Reshape Input dimension and astype to float32
    input_1 = np.reshape(input_1, (1, 12, 50, 30, 1)).astype('float32')
    input_2 = np.reshape(input_2, (1, 12, 30, 50, 1)).astype('float32')


    #Set tensor (2 input)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_1)
    interpreter.set_tensor(interpreter.get_input_details()[1]['index'], input_2)

    #inference
    interpreter.invoke()

    #get results(probability)
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    return (classes[np.argmax(output_data)]),output_data[0,np.argmax(output_data)]


def demo():
    configFileName = "./6843_pplcount_debug.cfg"
    dataPortName = "COM5"
    userPortName = "COM10"

    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName, dataPortName, userPortName)

    # Initialize interpreter and Print Input/Output'shape of model
    classes,interpreter,input_1,input_2= model_init()

    # Main process
    while True:
        try:
            numframes, x, y, z, isnull = readAndParseData(Dataport)
            if isnull == 0:
                input_1,input_2=stack_data(numframes, x, y,input_1,input_2)
                if numframes>12 and numframes%3==0:
                    results,probabilty=prediction(input_1, input_2, interpreter, classes)
                    print("Frames:{}(Offset = 3)\nResults:{}\nProbability:{:.2f}%".format(numframes,results,probabilty*100))
            else:
                continue
            # print(sum(input_1.flatten()),sum(input_2.flatten()))

            time.sleep(0.08)  # Sampling frequency of 30 Hz
        except KeyboardInterrupt:
            Dataport.close()  # 清除序列通訊物件
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            print('---------------------------已中斷連線！----------------------------------')
            break


if __name__ == "__main__":
    demo()
