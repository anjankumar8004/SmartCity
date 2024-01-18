
from time import sleep as delay
import sim
import sys
import cv2
import numpy as np

print("Program Started")
sim.simxFinish(-1)
clientID=sim.simxStart("127.0.0.1",19999,True,True,5000,5)
if(clientID!=-1):
    print("Connected Successfully")
else:
    sys.exit("Failed to Connect")

## Motor controlling code
lspeed=0
rspeed=0

delay(1)

#Vehicle Left/Righthandle
error_code,left_motor_handle=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[5]/leftMotor",sim.simx_opmode_oneshot_wait)
error_code,right_motor_handle=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[5]/rightMotor",sim.simx_opmode_oneshot_wait)
#Camera Handle
error_code,camera_handle=sim.simxGetObjectHandle(clientID,"/PioneerP3DX[5]/cam1",sim.simx_opmode_oneshot_wait)

delay(1)
returnCode,resolution,image=sim.simxGetVisionSensorImage(clientID,camera_handle,0,sim.simx_opmode_streaming)
delay(1)

try:
    iter=1
    while(1):
        
        returnCode,resolution,image=sim.simxGetVisionSensorImage(clientID,camera_handle,0,sim.simx_opmode_buffer)
        img=np.array(image,dtype=np.uint8)
        img.resize([resolution[0], resolution[1], 3])

        img=cv2.flip(img,0)
        img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img=cv2.resize(img,(512,512))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        error_code=sim.simxSetJointTargetVelocity(clientID,left_motor_handle,lspeed,sim.simx_opmode_streaming)
        error_code=sim.simxSetJointTargetVelocity(clientID,right_motor_handle,rspeed,sim.simx_opmode_streaming)
        
        cv2.imshow("data",img)
        com=cv2.waitKey(1)
        if(com==ord('0')):
            break
        elif(com==ord('w')):
            lspeed=10
            rspeed=10
            cv2.imwrite(f"E:/AI Projects/AI_SmartCity/api_0.2/DataCollection/Forward/fwd_{iter}.jpg",img)
            iter=iter+1
        elif(com==ord('a')):
            lspeed=-0.1
            rspeed=10
            cv2.imwrite(f"E:/AI Projects/AI_SmartCity/api_0.2/DataCollection/Left/left_{iter}.jpg",img)
            iter=iter+1
        elif(com==ord('d')):
            lspeed=10
            rspeed=-0.1
            cv2.imwrite(f"E:/AI Projects/AI_SmartCity/api_0.2/DataCollection/Right/right_{iter}.jpg",img)
            iter=iter+1
        elif(com==ord('s')):
            lspeed=-0.1
            rspeed=-0.1
            cv2.imwrite(f"E:/AI Projects/AI_SmartCity/api_0.2/DataCollection/Stop/stop{iter}.jpg",img)
            iter=iter+1
        else:
            lspeed=0
            rspeed=0
    cv2.destroyAllWindows()
except Exception as e:
    print("An exception occurred: " + str(e))
    # exc_type, exc_value, exc_traceback = sys.exc_info()
    # # Get the exception message
    # exception_message = str(e)

    # # Get the line number
    # line_number = exc_traceback.extract_tb(e.__traceback__)[-1][1]

    # # Get the file name
    # file_name = exc_traceback.extract_tb(e.__traceback__)[-1][0]

    # Print the exception details
    #print(f"Exception in file '{file_name}', line {line_number}: {exception_message}")
    cv2.destroyAllWindows()
