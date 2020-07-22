import serial
import time
import numpy as np
import struct
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import *

# global parameter
offset = 3


# ------------------------------------------------------------------
def dbfilter(raw_data):
    index_erro_point = np.squeeze(np.where(np.array(raw_data) == -1)).tolist()
    estimator = cluster.DBSCAN(eps=0.5, min_samples=8, metric='euclidean')
    estimator.fit(raw_data*[1,0.5,1])# 增加y軸聚合
    point_labels=estimator.labels_
    index_erro_point = np.squeeze(np.where(np.array(point_labels) == -1)).tolist()
    try:

        print("點雲數量:{}, 雜點數量：{}".format(len(point_labels),len(index_erro_point)))
    except:
        print((index_erro_point).tolist())
    # print(point_labels)
    # print("datalen:{},noiselen:{}".format(len(raw_data),len(point_labels)))


def serialConfig(configFileName, dataPortName, userPortName):
    try:
        cliPort = serial.Serial(userPortName, 115200)
        dataPort = serial.Serial(dataPortName, 921600,
                                 timeout=0.08)  # this timeout for buffer's updating transfer rate too slowly by serial port
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
    if np.all(byteVec[0:8] == magicWord) and len(readBuffer) > 52:
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
                    Prange = struct.unpack('f', readBuffer[
                                                HeaderLength + Typelength + numP * PointcloudLength:HeaderLength + Typelength + numP * PointcloudLength + 4])
                    azimuth = struct.unpack('f', readBuffer[
                                                 HeaderLength + Typelength + numP * PointcloudLength + 4:HeaderLength + Typelength + numP * PointcloudLength + 8])
                    elevation = struct.unpack('f', readBuffer[
                                                   HeaderLength + Typelength + numP * PointcloudLength + 8:HeaderLength + Typelength + numP * PointcloudLength + 12])
                    doppler = struct.unpack('f', readBuffer[
                                                 HeaderLength + Typelength + numP * PointcloudLength + 12:HeaderLength + Typelength + numP * PointcloudLength + 16])
                    framedata.append(pointClouds)
                except:  # Because sometimes the packet's length will not same with the packet's lenTLV
                    continue
                # spherical coordinate system

                range_list.append(Prange)  # range
                azimuth_list.append(azimuth)  # azimuth
                elevation_list.append(elevation)  # elevation
                doppler_list.append(doppler)  # doppler

                # Pack to List with pointCloud(spherical)
                pointClouds.append([range_list, azimuth_list, elevation_list, doppler_list])


            # Cartesian coordinate system
            r = np.multiply(range_list[:], np.cos(elevation_list))
            x = np.multiply(r[:], np.sin(azimuth_list[:]))
            y = np.multiply(r[:], np.cos(azimuth_list[:]))
            z = np.multiply(range_list[:], np.sin(elevation_list))

            data=np.concatenate((x,y,z),axis=1)
            # print(data.shape)
            dbfilter(data)

            # Feature Matrix preprocess(Voxalize)
            p_x_y, p_y_z, p_z_x = voxalize(50, 30, 50, x, y, z)

            # Frame Data not null from Serial-port
            isnull = 0
        else:
            return [], [], [], [], 0, 1

        return subFrameNum, p_x_y, p_y_z, p_z_x, numPoints, isnull
    else:
        isnull = 1
        return [], [], [], [], 0, isnull

    # ----------------------------------------------------------------------------------------------------------------------


def stack_data(frames, x, y, input_1, input_2):
    x = np.reshape(x, (50, 30))  # Input1 Matrix setup
    y = np.reshape(y, (30, 50))  # Input2 Matrix setup

    if frames < 12:
        input_1[frames, :, :] = x
        input_2[frames, :, :] = y
        print("Frames:{},stacking...".format(frames))
    else:

        input_1[0:11, :, :] = input_1[1:12, :, :]
        input_1[11, :, :] = x

        input_2[0:11, :, :] = input_2[1:12, :, :]
        input_2[11, :, :] = y

    return input_1, input_2


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

    return classes, interpreter, input_1, input_2


def prediction(input_1, input_2, interpreter, classes):
    # Reshape Input dimension and astype to float32
    input_1 = np.reshape(input_1, (1, 12, 50, 30, 1)).astype('float32')
    input_2 = np.reshape(input_2, (1, 12, 30, 50, 1)).astype('float32')

    # Set tensor (2 input)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_1)
    interpreter.set_tensor(interpreter.get_input_details()[1]['index'], input_2)

    # inference
    interpreter.invoke()

    # get results(probability)
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    return (classes[np.argmax(output_data)]), output_data[0, np.argmax(output_data)]


def camera_init():
    cap = cv2.VideoCapture(0)
    results = "stacking..."
    probabilty = 0
    pos_state = "normal"
    fontcolor = (255, 255, 255)
    return cap, results, probabilty, pos_state, fontcolor


def demo_camera(cap):
    while True:
        ret, Videoframe = cap.read()
        cv2.imshow("demo", Videoframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def check_results(new_results, new_probability, results, probability, pos_state, fontcolor):
    classes = ["st_sit", "sit_st", "sit_lie", "lie_sit", "fall", "grow_up", "other"]

    pos_states = ["normal", "Warning", "alert"]
    fontcolors = [(255, 255, 255), (0, 255, 255), (0, 0, 255)]  # white/ yellow/ red
    if new_probability >= 0.6:
        results = new_results
        probability = new_probability
        if new_results == classes[1] or new_results == classes[5]:
            pos_state = pos_states[1]
            fontcolor = fontcolors[1]
        elif new_results == classes[4]:
            pos_state = pos_states[2]
            fontcolor = fontcolors[2]
        else:
            pos_state = pos_states[0]
            fontcolor = fontcolors[0]

    return results, probability, pos_state, fontcolor


def plot_state(plot_data, plot_raw_data, savename, classes):
    plt.plot(np.arange(len(plot_data)), plot_data, color='red', label='checked')
    plt.plot(np.arange(len(plot_raw_data)), plot_raw_data, color='b', label='Unhecked')
    plt.yticks(range(1, 8), classes)
    plt.ylim([0.0, 8.0])
    plt.xlabel("Frames")
    plt.legend()
    plt.grid()
    plt.savefig(savename)


def demo():
    configFileName = "./6843_pplcount_debug.cfg"
    dataPortName = "COM5"
    userPortName = "COM10"
    plot_data = []  # checked results
    plot_raw_data = []  # unchecked results
    savename = './plot.png'

    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName, dataPortName, userPortName)

    # Initialize interpreter and Print Input/Output'shape of model
    classes, interpreter, input_1, input_2 = model_init()
    # Initialize Camera and cv2
    cap, results, probabilty, pos_state, fontcolor = camera_init()

    # Main process
    while True:
        try:
            numframes, x, y, z, numPoints, isnull = readAndParseData(Dataport)
            if isnull == 0:

                input_1, input_2 = stack_data(numframes, x, y, input_1, input_2)
                ret, Videoframe = cap.read()

                # define Picture/Frame's information
                cv2.putText(Videoframe, "Frames:{} Points:{}".format(numframes, numPoints), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(Videoframe, "Results:", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(Videoframe, "{}({:.2f}%) {}".format(results, probabilty * 100, pos_state), (140, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, fontcolor, 2, cv2.LINE_AA)

                if numframes > 11 and numframes % offset == 0:
                    cv2.imshow("demo", Videoframe)
                    new_results, new_probabilty = prediction(input_1, input_2, interpreter, classes)
                    plot_raw_data.append((np.where(np.array(classes) == new_results)[0][0]) + 1)
                    results, probabilty, pos_state, fontcolor = check_results(new_results, new_probabilty, results,
                                                                              probabilty, pos_state, fontcolor)
                    plot_data.append((np.where(np.array(classes) == results)[0][0]) + 1)
                else:
                    cv2.imshow("demo", Videoframe)
                    if results == "stacking...":
                        plot_data.append(0)
                        plot_raw_data.append(0)
                    else:
                        plot_data.append((np.where(np.array(classes) == results)[0][0]) + 1)
                        plot_raw_data.append((np.where(np.array(classes) == new_results)[0][0]) + 1)
                # leave loop and shutdown
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    plot_state(plot_data, plot_raw_data, savename, classes)
                    cap.release()
                    cv2.destroyAllWindows()
                    Dataport.close()  # 清除序列通訊物件
                    CLIport.write(('sensorStop\n').encode())
                    CLIport.close()
                    break

            else:
                continue
            # print(sum(input_1.flatten()),sum(input_2.flatten()))
            time.sleep(0.1)  # Sampling frequency of 30 Hz
        except KeyboardInterrupt:
            Dataport.close()  # 清除序列通訊物件
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            print('---------------------------已中斷連線！----------------------------------')
            break


if __name__ == "__main__":
    demo()
