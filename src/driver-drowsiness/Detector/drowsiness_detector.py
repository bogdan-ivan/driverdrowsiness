# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sqlite3
from data_base import DataBase
import playsound
from threading import Thread
from kNeighborsWrapper import KNeighborsWrapper
import datetime


# construct the argument parse and parse the arguments
print("[INFO] parsing arguments...")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
ap.add_argument("-a", "--alarm", type=str, default="alarms\Loud_Alarm_Clock.mp3",
    help="path to input video file")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
ear = 0
EYE_EAR_THRESH = 0.25
EYE_DROWSY_FRAMES = 30
EYE_BLINK_FRAMES = 2
originalEar = 0.25
highPitchEar = originalEar - 0.08
mediumPitchEar = originalEar - 0.05
lowPitchEar = originalEar - 0.03

# constants for yawn detection
MAR = 0
MOUTH_AR_THRESH = 0.35
MOUTH_AR_THRESH_ALERT = 0.30
MOUTH_AR_CONSEC_FRAMES = 20
yawn_rate = 0

# initialize the frame counters and the total number of blinks
COUNTER = 0
MCOUNTER = 0
MTOTAL=0
TOTAL_BLINKS=0
ALARM_ON = False

#-----------some counters-------
vigilant_counter =0
l_vigilant_counter =0
drowsy_counter =0
frame_counter=0

#head
x = 0
y = 0
h = 0
w = 0
Roll = 0
xNose = 0
yNose = 0
avgNormalX = 0
avgNormalY = 0
avgNormalZ = 0
rollCalibrationAverage = 0
noseXAverage = 0
noseYAverage = 0
yaw = 0
pitch = 0

# arrays for keeping the MAR and EAR values
earArray = np.array([])
marArray = np.array([])
rollArray = np.array([])
pitchArray = np.array([])
yawArray = np.array([])
yawnArray = np.array([])
xArray = np.array([])
yArray = np.array([])
zArray = np.array([])
driver_state_array = np.array([])

#variables for keeping averages
avgRoll = 0
avgPitch = 0
avgYaw = 0
avgEar = 0
avgMar = 0
avgYawn = 0

# variables
sunglassesCheck = False
sunglassesOn = False
yawnTotal = 0
time_from_last_blink=0
time_between_blinks=0
blink_duration=0
frames_until_dsa_cheked=300
predicted_class = 0
blink_rate=0
blink_rate_fix = 0
last_total_blinks=0
blink_amplitude = 1
last_ear = 1
time_between_yaws =0
last_yaw_time=0
recalibrate = False
alarm_counter = 0
recalibrate_start = 0
recalibrate_seconds = 0
one_time = True
currentTime = 0
currentBlinks = 0
sunglasses_flag = False
drowsy_driver_flag = False
drowsy_value = 0
toggle_info = True
toggle_hulls = True

#model required in order to classifie the data colected from a video
kNeighborsModel=KNeighborsWrapper()
print (time.process_time())

def count_the_prediction(predict):
    global vigilant_counter,l_vigilant_counter,drowsy_counter,frame_counter
    if predict==0:
        vigilant_counter+=1
        frame_counter=0
    elif predict==5:
        l_vigilant_counter+=1
        frame_counter=0
    elif predict ==10:
        drowsy_counter+=1
        frame_counter+=1

def get_vld_mean():
    global vigilant_counter,l_vigilant_counter,drowsy_counter
    return (l_vigilant_counter*5+drowsy_counter*10)/(vigilant_counter+l_vigilant_counter+drowsy_counter)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    if C < 0.1:  # practical finetuning due to possible numerical issue as a result of optical flow
        ear = 0.3
    else:
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
    if ear > 0.45:  # practical finetuning due to possible numerical issue as a result of optical flow
        ear = 0.45
    # return the eye aspect ratio
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[14], mouth[18])

    C = dist.euclidean(mouth[12], mouth[16])

    if C < 0.1:  # practical finetuning
        mar = 0.2
    else:
        # compute the mouth aspect ratio
        mar = (A) / (C)

    # return the mouth aspect ratio
    return mar

def sound_alarm(path):
    # play alarm
    playsound.playsound(path)

def get_head_roll(leftEye,rightEye):
    diference=leftEye-rightEye

    return np.arctan(diference[1]/diference[0])

def check_driver_state(arrayOfDataClasses):
    return True if arrayOfDataClasses.mean()>8.0 else False, arrayOfDataClasses.mean()

def auto_gamma_adjustamnet(gray_mean):
    if(gray_mean < 80):
        return 2.5, -1
    elif (gray_mean < 105):
        return 2.0, 0
    elif (gray_mean < 120):
        return 1.5, 1
    elif (gray_mean < 160):
        return 1.0, 2
    return 0.5, 3

def contour_mask(gray,cnt):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    #return np.transpose(np.nonzero(mask))
    return  mask

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def sunglasses_detection(gray,left_eye,right_eye):
    left_mask = contour_mask(gray, left_eye)
    left_mean = cv2.mean(gray, mask=left_mask)

    right_mask = contour_mask(gray, right_eye)
    right_mean = cv2.mean(gray, mask=right_mask)

    return (left_mean[0]+right_mean[0])/2

def print_info(frame):
        cv2.putText(frame, "BLINKS: {:.0f}".format(TOTAL_BLINKS), (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(MAR), (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWNS: {:.0f}".format(MTOTAL), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "yawn_rate: {:.2f}".format(yawn_rate), (500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "X: {:.0f}".format(x), (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Y: {:.0f}".format(y), (30, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Z: {:.0f}".format(w), (30, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Roll: {:.3f}".format(Roll), (570, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "AvgRoll: {:.3f}".format(avgRoll), (570, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EarAvg: {:.5f}".format(avgEar), (570, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MarAvg: {:.5f}".format(avgMar), (570, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Time: {:.1f}".format(elapsed), (30, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Predicted_class: {}".format(predicted_class), (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR THRESH: {:.2f}".format(EYE_EAR_THRESH), (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "normalX: {:.0f}".format(avgNormalX), (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "normalY: {:.0f}".format(avgNormalY), (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "normalZ: {:.0f}".format(avgNormalZ), (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "normalRoll: {:.3f}".format(rollCalibrationAverage), (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "normalXNose: {:.0f}".format(noseXAverage), (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "normalYNose: {:.0f}".format(noseYAverage), (30, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2),
        cv2.putText(frame, "Yaw: {}".format(yaw), (570, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2),
        cv2.putText(frame, "AvgYaw: {:.03f}".format(avgYaw), (570, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2),
        cv2.putText(frame, "Pitch: {:.0f}".format(pitch), (570, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "AvgPitch: {:.03f}".format(avgPitch), (570, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2),
        cv2.putText(frame, "Last blink time: {:.0f}".format(time_from_last_blink), (500, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Time_b_blinks : {:.0f}".format(time_between_blinks), (500, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink_rate : {:.0f}".format(blink_rate), (500, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink_amplitude : {:.2f}".format(blink_amplitude), (500, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Time_b_yawns : {:.0f}".format(time_between_yaws), (570, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink_duration : {:.0f}".format(blink_duration), (500, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Sunglasses: {}".format(sunglasses_flag), (570, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Eye Gamma: {:.2f}".format(sunglasses_value), (570, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Gamma Mode: {}".format(gammaMode), (570, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Sungl Tresh: {:.2f}".format(sunglasses_threshold), (570, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "D State Size: {}".format(driver_state_array.size), (30, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "D State Average: {:.2f}".format(drowsy_value), (30, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink_duration : {:.0f}".format(blink_duration), (500, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def print_hulls(leye, reye, mouth, x, y, w, h, noseXAverage, noseYAverage, xNose, yNose):
    cv2.drawContours(frame, [leye], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [reye], -1, (0, 255, 0), 1)
    MouthHull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [MouthHull], -1, (255, 0, 0), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.line(frame, (noseXAverage, noseYAverage), (xNose, yNose), (0, 255, 0), 2)
    cv2.circle(frame, (xNose, yNose), 3, (0, 0, 255), -1)

def average_calculator(data_array, data, elapsed):
    data_array = np.append(data_array, data)
    if elapsed > 1800:
        data_array = np.delete(data_array, 0)
        return data_array.sum() / data_array.size, data_array
    else:
        return data_array.sum() / data_array.size, data_array


# initialize dlib's face detector and then create
# the facial landmark predictor
print("[INFO] loading frontal face detector...")
detector = dlib.get_frontal_face_detector()
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
print("[INFO] loading facial landmark indexes...")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
if args["video"] != "":
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
else:
    vs = VideoStream(src=0).start()
    fps = FPS().start()
    fileStream = False
#vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

# ------ database -----------

print("[INFO] loading data base...")
data_B = DataBase()

#data_B.execute_query23()
#data_B.execute_query("""DROP TABLE '%s' ""","features")

name = input("Please insert your name: ")
name = str(name)
Name_flag = data_B.select_name(name)

if Name_flag == False:
    data_B.insert_name(name)

DBFlag = False

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    #frame = imutils.rotate_bound(frame,-90)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gamma, gammaMode = auto_gamma_adjustamnet(gray.mean())
    gray = adjust_gamma(gray, gamma=gamma)
    #frame = adjust_gamma(frame, gamma=gamma)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    elapsed = time.process_time()
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # get face coordonates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        z = w

        ###############YAWNING##################

        Mouth = shape[mStart:mEnd]
        MAR = mouth_aspect_ratio(Mouth)

        #calculating average for MAR in the last 30  mins
        avgMar, marArray = average_calculator(marArray, MAR, elapsed)

        # Calculate roll value
        Roll = get_head_roll(shape[45], shape[36])

        #calculating average for ROLL in the last 30mins
        avgRoll, rollArray = average_calculator(rollArray, Roll, elapsed)

        if MAR > MOUTH_AR_THRESH:
            MCOUNTER += 1

        elif MAR < MOUTH_AR_THRESH_ALERT:

            if MCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                MTOTAL += 1
                time_between_yaws =elapsed-last_yaw_time
                last_yaw_time = elapsed


            MCOUNTER = 0


        #number of yawns in the last 30 mins
        zero = 0
        if yawnTotal < MTOTAL:
            yawnTotal = MTOTAL
            yawnArray = np.append(yawnArray, 1)
            yawn_rate = yawnArray.sum()
        else:
            yawnArray = np.append(yawnArray, zero)
            yawn_rate = yawnArray.sum()

        if elapsed > 1800 & yawnArray.size > 1:
            if yawnTotal < MTOTAL:
                yawnTOTAL = MTOTAL
                yawnArray = np.delete(yawnArray, 1)
                yawn_rate = yawnArray.sum()
            else:
                yawnArray = np.delete(yawnArray, 1)
                yawnArray = np.append(yawnArray, zero)
                yawn_rate = yawnArray.sum()


        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # ------------- Calibration -------------
        #useful arrays
        earCalibrationArray = np.array([])
        normalX = np.array([])
        normalY = np.array([])
        normalZ = np.array([])
        normalNoseX = np.array([])
        normalNoseY = np.array([])
        normalPitch = np.array([])
        normalYaw = np.array([])
        rollCalibrationArray = np.array([])
        (xNose, yNose) = shape[30]


        if recalibrate_seconds > 5:
            recalibrate = False
            one_time = True

        #first seconds of running
        if elapsed < 10 or recalibrate:
            if one_time == True and recalibrate == True:
                recalibrate_start = elapsed
                one_time = False
            if elapsed > 10:
                recalibrate_seconds = elapsed - recalibrate_start
        #calculating averages for EYE EAR THRESH
            if ear > 0.19:
                earCalibrationArray = np.append(earCalibrationArray, ear)
                earCalibrationAverage = earCalibrationArray.sum() / earCalibrationArray.size



        #calculating averages for head position
            normalX = np.append(normalX, x)
            normalX = np.append(normalX, x)
            avgNormalX = normalX.sum() / normalX.size
            normalY = np.append(normalY, y)
            avgNormalY = normalY.sum() / normalY.size
            normalZ = np.append(normalZ, w)
            avgNormalZ = normalZ.sum() / normalZ.size

        #calculating averages for Roll
            rollCalibrationArray = np.append(rollCalibrationArray, Roll)
            rollCalibrationAverage = rollCalibrationArray.sum() / rollCalibrationArray.size

        #calculating averages for Nose positions
            normalNoseX = np.append(normalNoseX, xNose)
            noseXAverage = normalNoseX.sum() / normalNoseX.size
            normalNoseY = np.append(normalNoseY, yNose)
            noseYAverage = normalNoseY.sum() / normalNoseY.size


        #showing in real time
            cv2.putText(frame, "Please keep a normal position and look at the road", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2),
            cv2.putText(frame, "Roll: {:.3f}".format(Roll), (350, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            EYE_EAR_THRESH = earCalibrationAverage - 0.05
            originalEar = earCalibrationAverage - 0.05
            highPitchEar = earCalibrationAverage - 0.12
            mediumPitchEar = earCalibrationAverage - 0.10
            lowPitchEar = earCalibrationAverage - 0.08

        if Name_flag == False:
            if DBFlag == False and elapsed > 10:
                DBFlag = data_B.insert_ear(EYE_EAR_THRESH, name)
        elif Name_flag == True and elapsed > 10:
            originalEar = data_B.get_ear(name)

        # -----------------------------------------

        # calculating average for EAR in the last 30 mins
        avgEar, earArray = average_calculator(earArray,ear,elapsed)
        #calculating yaw
        yaw = xNose - noseXAverage
        #yaw average
        avgYaw, yawArray = average_calculator(yawArray,yaw,elapsed)
        #calculating pitch
        pitch = noseYAverage - yNose
        #pitch average
        avgPitch, pitchArray = average_calculator(pitchArray,pitch,elapsed)


        # EAR threshold based on pitch
        if pitch > 20:
            EYE_EAR_THRESH = lowPitchEar
            if pitch > 30:
                EYE_EAR_THRESH = mediumPitchEar
                if pitch > 40:
                    EYE_EAR_THRESH = highPitchEar
        else:
            EYE_EAR_THRESH = originalEar


        # active calibration

        if abs(y-avgNormalY) > 50 and abs(z-avgNormalZ) > 50 or abs(x-avgNormalX) > 60 or pitch < -60:
            avgNormalY = y
            avgNormalZ = z
            avgNormalX = x
            calibrateNose = shape[30]
            noseXAverage = calibrateNose[0]
            noseYAverage = calibrateNose[1]


        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # -------------Check for sunglasses -------------

        sunglasses_value = sunglasses_detection(gray, leftEyeHull, rightEyeHull)
        sunglasses_threshold = gray.mean()/2

        if sunglasses_value < sunglasses_threshold:
            sunglasses_flag = True
        else:
            sunglasses_flag = False

        # disable the eyes if sunglasses are on
        if sunglasses_flag == True:
            ear = 0.45


        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        # we consider 0.05 to be the smallest value that EYE_THRESH_HOLD when the driver is tired

        if ear <= (EYE_EAR_THRESH) or (sunglasses_flag and abs(Roll) > 0.40):
            COUNTER += 1
            if ear < last_ear:
                blink_amplitude = EYE_EAR_THRESH - ear
                last_ear = ear
            if COUNTER >= EYE_DROWSY_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if COUNTER >= EYE_BLINK_FRAMES:
                TOTAL_BLINKS += 1
                time_between_blinks =elapsed-time_from_last_blink
                time_from_last_blink=elapsed
            # if the eyes were closed for a sufficient number of
            # then sound the alarm


        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        #else:
            if COUNTER!=blink_duration and COUNTER!=0:
                blink_duration=COUNTER

            COUNTER = 0
            last_ear = 1
            ALARM_ON = False


        predicted_class=kNeighborsModel.predict(np.array([avgEar,avgMar,avgRoll,avgYaw,avgPitch,avgYawn,MTOTAL,blink_duration,blink_rate,blink_amplitude,time_between_blinks,time_between_yaws]).reshape(1,-1))

        driver_state_array =np.append(driver_state_array,int(predicted_class))

        if int(elapsed)%60 == 0 and blink_rate_fix != int(elapsed):
            blink_rate_fix = int(elapsed)
            blink_rate =TOTAL_BLINKS-last_total_blinks
            last_total_blinks=TOTAL_BLINKS
        if int(elapsed) > blink_rate_fix:
            blink_rate_fix = 0
        if elapsed < 60 and alarm_counter > 0:
            recalibrate = True

        if (driver_state_array.size > frames_until_dsa_cheked):
            drowsy_driver_flag, drowsy_value = check_driver_state(driver_state_array)
            if (drowsy_driver_flag==True):
                cv2.putText(frame, "You should get a break", (50, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ear = EYE_EAR_THRESH - 0.02
                COUNTER = 40
                #driver_state_array = np.delete(driver_state_array, np.s_[0:-1])
            #else:
                #driver_state_array = np.delete(driver_state_array, np.s_[0:-1])
            frames_until_dsa_cheked = frames_until_dsa_cheked + 300
            

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters

    count_the_prediction(predicted_class)

    if elapsed > 10:
        noseYAverage = int(noseYAverage)
        noseXAverage = int(noseXAverage)
        if(toggle_info == True):
            print_info(frame)
        if(toggle_hulls == True):
            print_hulls(leftEyeHull, rightEyeHull, Mouth, x, y, w, h, noseXAverage, noseYAverage, xNose, yNose)

        # show the frame
    cv2.imshow("Frame", frame)
    fps.update()
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        data_B.update_features(data_B.get_user_id(name),(vigilant_counter,l_vigilant_counter,drowsy_counter,elapsed,get_vld_mean(),datetime.date.today()))
        break
    if key == ord("i"):
        toggle_info = not toggle_info
    if key == ord("h"):
        toggle_hulls = not toggle_hulls


fps.stop()
print("[INFO] FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
print("[INFO] destroying all windows...")
cv2.destroyAllWindows()
print("[INFO] stopping video thread...")
vs.stop()