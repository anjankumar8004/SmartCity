import pickle
from keras.models import load_model
import sim
from time import sleep as delay
import numpy as np
import cv2
import sys
import tensorflow as tf
import keras
import os
import threading
import time
from scipy import ndimage
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Program started')
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
clientID5 = sim.simxStart('127.0.0.1', 19990, True, True, 5000, 5)


if (clientID != -1):
    print('Connected to remote API server')
else:
    sys.exit('Failed connecting to remote API server')

if (clientID5 != -1):
    print('Connected to remote API server')
else:
    sys.exit('Failed connecting to remote API server')


delay(1)
model = tf.keras.models.load_model("model_AI_Car.h5")
model.load_weights("./model_AI_Car_Weights.h5")

dict_file = open("./ai_car.pkl", "rb")
category_dict = pickle.load(dict_file)

def car0():   
    try:
##########################CAR 0###############################
        lSpeed = 0
        rSpeed = 0
        #Vehicle Left/Righthandle
        error_code,left_motor_handle=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[0]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code,right_motor_handle=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[0]/rightMotor",sim.simx_opmode_oneshot_wait)
        #Camera Handle
        error_code,camera_handle=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[0]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode, resolution, image = sim.simxGetVisionSensorImage(
            clientID, camera_handle, 0, sim.simx_opmode_streaming)
        delay(1)


############################ CAR5 #############################
        lSpeed5 = 0
        rSpeed5 = 0
        error_code5,left_motor_handle5=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[5]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code5,right_motor_handle5=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[5]/rightMotor",sim.simx_opmode_oneshot_wait)

        error_code5,camera_handle5=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[5]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode5, resolution5, image5 = sim.simxGetVisionSensorImage(
            clientID, camera_handle5, 0, sim.simx_opmode_streaming)

############################# CAR1 ############################
        lSpeed1 = 0
        rSpeed1 = 0
        error_code1,left_motor_handle1=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[1]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code1,right_motor_handle1=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[1]/rightMotor",sim.simx_opmode_oneshot_wait)

        error_code1,camera_handle1=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[1]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode1, resolution1, image1 = sim.simxGetVisionSensorImage(
            clientID, camera_handle1, 0, sim.simx_opmode_streaming)


############################### CAR2 ##########################
        lSpeed2 = 0
        rSpeed2 = 0
        error_code2,left_motor_handle2=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[2]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code2,right_motor_handle2=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[2]/rightMotor",sim.simx_opmode_oneshot_wait)

        error_code2,camera_handle2=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[2]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode2, resolution2, image2 = sim.simxGetVisionSensorImage(
            clientID, camera_handle2, 0, sim.simx_opmode_streaming)


################################# CAR3 ########################
        lSpeed3 = 0
        rSpeed3 = 0
        error_code3,left_motor_handle3=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[3]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code3,right_motor_handle3=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[3]/rightMotor",sim.simx_opmode_oneshot_wait)

        error_code3,camera_handle3=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[3]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode3, resolution3, image3 = sim.simxGetVisionSensorImage(
            clientID, camera_handle3, 0, sim.simx_opmode_streaming)


##################################  CAR4 #######################
        lSpeed4 = 0
        rSpeed4 = 0
        error_code4,left_motor_handle4=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[4]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code4,right_motor_handle4=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[4]/rightMotor",sim.simx_opmode_oneshot_wait)

        error_code4,camera_handle4=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[4]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode4, resolution4, image4 = sim.simxGetVisionSensorImage(
            clientID, camera_handle4, 0, sim.simx_opmode_streaming)



###################################### CAR6  ###################
        lSpeed6 = 0
        rSpeed6 = 0
        error_code6,left_motor_handle6=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[6]/leftMotor",sim.simx_opmode_oneshot_wait)
        error_code6,right_motor_handle6=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[6]/rightMotor",sim.simx_opmode_oneshot_wait)

        error_code6,camera_handle6=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[6]/cam1",sim.simx_opmode_oneshot_wait)
        delay(1)

        returnCode6, resolution6, image6 = sim.simxGetVisionSensorImage(
            clientID, camera_handle6, 0, sim.simx_opmode_streaming)

        #################################################################
        while (1):
############################# CAR0 ####################################
            returnCode, resolution, image = sim.simxGetVisionSensorImage(
                clientID, camera_handle, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im = np.array(image).astype(np.uint8)
            im.resize([resolution[0], resolution[1], 3])

            # im = cv2.flip(im, 0)
            # 
            #rotation angle in degree
            im = ndimage.rotate(im, 90)
            im = cv2.flip(im, 1)
            im = cv2.resize(im, (512, 512))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            errorCode = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle, lSpeed, sim.simx_opmode_streaming)
            errorCode = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle, rSpeed, sim.simx_opmode_streaming)

            test_img = cv2.resize(im, (50, 50))
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            test_img = test_img/255
            test_img = test_img.reshape(1, 50, 50, 1)

            results = model.predict(test_img)
            label = np.argmax(results, axis=1)[0]
            acc = int(np.max(results, axis=1)[0]*100)

            # print(f"Moving 1: {category_dict[label]} with {acc}% accuracy.")

############################# CAR1 ####################################
            returnCode1, resolution1, image1 = sim.simxGetVisionSensorImage(
                clientID, camera_handle1, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im1 = np.array(image1).astype(np.uint8)
            im1.resize([resolution1[0], resolution1[1], 3])

            # im = cv2.flip(im, 0)
            

            #rotation angle in degree
            im1 = ndimage.rotate(im1, 90)
            im1 = cv2.flip(im1, 1)
            im1 = cv2.resize(im1, (512, 512))
            im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)

            errorCode1 = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle1, lSpeed1, sim.simx_opmode_streaming)
            errorCode = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle1, rSpeed1, sim.simx_opmode_streaming)

            test_img1 = cv2.resize(im1, (50, 50))
            test_img1 = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)
            test_img1 = test_img1/255
            test_img1 = test_img1.reshape(1, 50, 50, 1)

            results1 = model.predict(test_img1)
            label1 = np.argmax(results1, axis=1)[0]
            acc1 = int(np.max(results1, axis=1)[0]*100)

            # print(f"Moving 1: {category_dict[label]} with {acc}% accuracy.")


############################# CAR2 ####################################
            returnCode2, resolution2, image2 = sim.simxGetVisionSensorImage(
                clientID, camera_handle2, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im2 = np.array(image2).astype(np.uint8)
            im2.resize([resolution2[0], resolution2[1], 3])

            # im = cv2.flip(im, 0)


            #rotation angle in degree
            im2 = ndimage.rotate(im2, 90)
            im2 = cv2.flip(im2, 1)
            im2 = cv2.resize(im2, (512, 512))
            im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)

            errorCode2 = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle2, lSpeed2, sim.simx_opmode_streaming)
            errorCode2 = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle2, rSpeed2, sim.simx_opmode_streaming)

            test_img2 = cv2.resize(im2, (50, 50))
            test_img2 = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
            test_img2 = test_img2/255
            test_img2 = test_img2.reshape(1, 50, 50, 1)

            results2 = model.predict(test_img2)
            label2 = np.argmax(results2, axis=1)[0]
            acc2 = int(np.max(results2, axis=1)[0]*100)

            # print(f"Moving 1: {category_dict[label]} with {acc}% accuracy.")


############################# CAR3 ####################################
            returnCode3, resolution3, image3 = sim.simxGetVisionSensorImage(
                clientID, camera_handle3, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im3 = np.array(image3).astype(np.uint8)
            im3.resize([resolution3[0], resolution3[1], 3])

            # im = cv2.flip(im, 0)


            #rotation angle in degree
            im3 = ndimage.rotate(im3, 90)
            im3 = cv2.flip(im3, 1)
            im3 = cv2.resize(im3, (512, 512))
            im3 = cv2.cvtColor(im3, cv2.COLOR_RGB2BGR)

            errorCode3 = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle3, lSpeed3, sim.simx_opmode_streaming)
            errorCode3 = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle3, rSpeed3, sim.simx_opmode_streaming)

            test_img3 = cv2.resize(im3, (50, 50))
            test_img3 = cv2.cvtColor(test_img3, cv2.COLOR_BGR2GRAY)
            test_img3 = test_img3/255
            test_img3 = test_img3.reshape(1, 50, 50, 1)

            results3 = model.predict(test_img3)
            label3 = np.argmax(results3, axis=1)[0]
            acc3 = int(np.max(results3, axis=1)[0]*100)

            # print(f"Moving 1: {category_dict[label]} with {acc}% accuracy.")


############################# CAR4 ####################################
            returnCode4, resolution4, image4 = sim.simxGetVisionSensorImage(
                clientID, camera_handle4, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im4 = np.array(image4).astype(np.uint8)
            im4.resize([resolution4[0], resolution4[1], 3])

            # im = cv2.flip(im, 0)

            #rotation angle in degree
            im4 = ndimage.rotate(im4, 90)
            im4 = cv2.flip(im4, 1)
            im4= cv2.resize(im4, (512, 512))
            im4 = cv2.cvtColor(im4, cv2.COLOR_RGB2BGR)

            errorCode4 = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle4, lSpeed4, sim.simx_opmode_streaming)
            errorCode4 = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle4, rSpeed4, sim.simx_opmode_streaming)

            test_img4 = cv2.resize(im4, (50, 50))
            test_img4 = cv2.cvtColor(test_img4, cv2.COLOR_BGR2GRAY)
            test_img4 = test_img4/255
            test_img4 = test_img4.reshape(1, 50, 50, 1)

            results4 = model.predict(test_img4)
            label4 = np.argmax(results4, axis=1)[0]
            acc4 = int(np.max(results4, axis=1)[0]*100)

            # print(f"Moving 1: {category_dict[label]} with {acc}% accuracy.")

############################# CAR5 ####################################
            returnCode5, resolution5, image5 = sim.simxGetVisionSensorImage(
                clientID, camera_handle5, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im5 = np.array(image5).astype(np.uint8)
            im5.resize([resolution5[0], resolution5[1], 3])

            # im = cv2.flip(im, 0)
            #rotation angle in degree
            im5 = ndimage.rotate(im5, 90)
            im5= cv2.flip(im5, 1)
            im5 = cv2.resize(im5, (512, 512))
            im5 = cv2.cvtColor(im5, cv2.COLOR_RGB2BGR)

            errorCode5 = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle5, lSpeed5, sim.simx_opmode_streaming)
            errorCode5 = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle5, rSpeed5,sim.simx_opmode_streaming)

            test_img5 = cv2.resize(im5, (50, 50))
            test_img5 = cv2.cvtColor(test_img5, cv2.COLOR_BGR2GRAY)
            test_img5 = test_img5/255
            test_img5 = test_img5.reshape(1, 50, 50, 1)

            results5 = model.predict(test_img5)
            label5 = np.argmax(results5, axis=1)[0]
            acc5 = int(np.max(results5, axis=1)[0]*100)

            # print(f"Moving 5 : {category_dict[label5]} with {acc5}% accuracy.")


############################# CAR6 ####################################
            returnCode6, resolution6, image6 = sim.simxGetVisionSensorImage(
                clientID, camera_handle6, 0, sim.simx_opmode_buffer)
            #im = np.array(image, dtype=np.uint8)
            im6 = np.array(image6).astype(np.uint8)
            im6.resize([resolution6[0], resolution6[1], 3])

            # im = cv2.flip(im, 0)

            #rotation angle in degree
            im6 = ndimage.rotate(im6, 90)
            im6 = cv2.flip(im6, 1)
            im6 = cv2.resize(im6, (512, 512))
            im6 = cv2.cvtColor(im6, cv2.COLOR_RGB2BGR)

            errorCode6 = sim.simxSetJointTargetVelocity(
                clientID, left_motor_handle6, lSpeed6, sim.simx_opmode_streaming)
            errorCode6 = sim.simxSetJointTargetVelocity(
                clientID, right_motor_handle6, rSpeed6, sim.simx_opmode_streaming)

            test_img6 = cv2.resize(im6, (50, 50))
            test_img6 = cv2.cvtColor(test_img6, cv2.COLOR_BGR2GRAY)
            test_img6 = test_img6/255
            test_img6 = test_img6.reshape(1, 50, 50, 1)

            results6 = model.predict(test_img6)
            label6 = np.argmax(results6, axis=1)[0]
            acc6 = int(np.max(results6, axis=1)[0]*100)

            # print(f"Moving 1: {category_dict[label]} with {acc}% accuracy.")

######################################################################
            if (label == 0):
                lSpeed = 3
                rSpeed = 3
            elif (label == 1):
                lSpeed = -0.1
                rSpeed = 3
            elif (label == 2):
                lSpeed = 3
                rSpeed = -0.1
            elif (label == 3):
                lSpeed = -0.1
                rSpeed = -0.1
            else:
                lSpeed = 0
                rSpeed = 0
            label = -1
################## CAR1 #################
            
            if (label1 == 0):
                lSpeed1 = 3
                rSpeed1 = 3
            elif (label1 == 1):
                lSpeed1 = -0.1
                rSpeed1 = 3
            elif (label1 == 2):
                lSpeed1 = 3
                rSpeed1 = -0.1
            elif (label1 == 3):
                lSpeed1 = -0.1
                rSpeed1 = -0.1
            else:
                lSpeed1 = 0
                rSpeed1 = 0
            label1 = -1
            # cv2.imshow("data1", im1)
            # cv2.imshow("data", im)

################## CAR2 #################
            if (label2 == 0):
                lSpeed2 = 3
                rSpeed2 = 3
            elif (label2 == 1):
                lSpeed2 = -0.1
                rSpeed2 = 3
            elif (label2 == 2):
                lSpeed2 = 3
                rSpeed2 = -0.1
            elif (label2 == 3):
                lSpeed2 = -0.1
                rSpeed2 = -0.1
            else:
                lSpeed2 = 0
                rSpeed2 = 0
            label2 = -1
            # cv2.imshow("data5", im5)
##########################  CAR3 ############################################
            if (label3 == 0):
                lSpeed3 = 3
                rSpeed3 = 3
            elif (label3 == 1):
                lSpeed3 = -0.1
                rSpeed3 = 3
            elif (label3 == 2):
                lSpeed3 = 3
                rSpeed3 = -0.1
            elif (label3 == 3):
                lSpeed3 = -0.1
                rSpeed3 = -0.1
            else:
                lSpeed3 = 0
                rSpeed3 = 0
            label3 = -1
#############################  CAR4 #########################################
            if (label4 == 0):
                lSpeed4 = 3
                rSpeed4 = 3
            elif (label4 == 1):
                lSpeed4 = -0.1
                rSpeed4 = 3
            elif (label4 == 2):
                lSpeed4 = 3
                rSpeed4 = -0.1
            elif (label4 == 3):
                lSpeed4 = -0.1
                rSpeed4 = -0.1
            else:
                lSpeed4 = 0
                rSpeed4 = 0
            label4 = -1

############################## CAR4 ########################################
            if (label5 == 0):
                lSpeed5 = 3
                rSpeed5 = 3
            elif (label5 == 1):
                lSpeed5 = -0.1
                rSpeed5 = 3
            elif (label5 == 2):
                lSpeed5 = 3
                rSpeed5 = -0.1
            elif (label5 == 3):
                lSpeed5 = -0.1
                rSpeed5 = -0.1
            else:
                lSpeed5 = 0
                rSpeed5 = 0
            label5 = -1
#################################  CAR6 #####################################
            if (label6 == 0):
                lSpeed6 = 3
                rSpeed6 = 3
            elif (label6 == 1):
                lSpeed6 = -0.1
                rSpeed6 = 3
            elif (label6 == 2):
                lSpeed6 = 3
                rSpeed6 = -0.1
            elif (label6 == 3):
                lSpeed6 = -0.1
                rSpeed6 = -0.1
            else:
                lSpeed6 = 0
                rSpeed6 = 0
            label6 = -1

            com = cv2.waitKey(1)
            if (com == ord('q')):
                lSpeed = 0
                rSpeed = 0
                break

        cv2.destroyAllWindows()

####### CAR 0
        errorCode = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle, lSpeed, sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle, rSpeed, sim.simx_opmode_streaming)

####### CAR 1
        errorCode1 = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle1, lSpeed1, sim.simx_opmode_streaming)
        errorCode1 = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle1, rSpeed1, sim.simx_opmode_streaming)

 ####### CAR 2
        errorCode2 = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle2, lSpeed2, sim.simx_opmode_streaming)
        errorCode2 = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle2, rSpeed2, sim.simx_opmode_streaming)

 ####### CAR 3
        errorCode3 = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle3, lSpeed3, sim.simx_opmode_streaming)
        errorCode3 = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle3, rSpeed3, sim.simx_opmode_streaming)

 ####### CAR 4
        errorCode4 = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle4, lSpeed4, sim.simx_opmode_streaming)
        errorCode4 = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle4, rSpeed4, sim.simx_opmode_streaming)

 ####### CAR 5
        errorCode5 = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle5, lSpeed5, sim.simx_opmode_streaming)
        errorCode5 = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle5, rSpeed5, sim.simx_opmode_streaming)

 ####### CAR 6
        errorCode6 = sim.simxSetJointTargetVelocity(
            clientID, left_motor_handle6, lSpeed6, sim.simx_opmode_streaming)
        errorCode6 = sim.simxSetJointTargetVelocity(
            clientID, right_motor_handle6, rSpeed6, sim.simx_opmode_streaming)
    except Exception as e:
        cv2.destroyAllWindows()
        print("An exception occurred: " + str(e))
        traceback.print_exc()

thread1 = threading.Thread(target=car0())
#thread2 = threading.Thread(target=car5())

# Start the threads
#thread2.start()
thread1.start()

# Wait for both threads to finish
# thread1.join()
# thread2.join()