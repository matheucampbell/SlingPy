import cv2 as cv  # Open-source computer vision library
import math
import numpy as np
import picamera  # Interface module for Raspberry Pi Camera
import picamera.array
from threading import Thread
from time import sleep
import wiringpi as wp

game = True  # Whether or not a round is currently happening
show = True  # Whether or not the program should show the camera feed
first = True  # Whether or not the current frame is the detection frame
end = False  # Whether or not game has ended
det = False  # Whether or not a detection exists for the car
tap = False  # Whether or not the program should be tapping
cur_rec = False
no_rec = False

crop_hist = None
last_ang = None
front = None
rec = None
recs = None
current_frame = None
recent_frame = None
show_frame = None

tap_val = 116
rel_val = 108
cutoff = 13

frames = 0
streak = 0
base_height = 0

crit = (cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 10, 1)  # Criteria to run CamShift function with
null_rec = ((0.0, 0.0), (0.0, 0.0), 0.0)  # Empty rectangle placeholder

# Location of cascade classifier object; used to initially detect car's location
casc_path = '/home/pi/CV_Projects/cascades/newcascade.xml'
casc_obj = cv.CascadeClassifier(casc_path)

# Establishes PWM for micro-servo
wp.wiringPiSetupGpio()
wp.pinMode(18, wp.GPIO.PWM_OUTPUT)
wp.pwmSetMode(wp.GPIO.PWM_MODE_MS)
wp.pwmSetClock(192)
wp.pwmSetRange(2000)
wp.pwmWrite(18, rel_val)

# To be run in a separate thread to fetch frames
def get_frames():
    global recent_frame
    global end
    global thread_frames
    thread_frames = 0
    with picamera.PiCamera() as cam:
        cam.resolution = (352, 512)
        with picamera.array.PiRGBArray(cam) as raw:
            for frame in cam.capture_continuous(raw, 'bgr', use_video_port=True):
                thread_frames += 1
                recent_frame = raw.array
                raw.truncate(0)
                if end:
                    break

# To show frames
def show():
    global show_frame
    global end
    while not end:
        if show_frame is not None:
            cv.imshow('Current Frame', show_frame)
            cv.waitKey(1)

# Returns the quadrant of a given rectangle relative to the car's rectangle
def get_quad(center_rec, test_rec, angle):
    f_x, f_y = test_rec[0]
    rot_ang = angle
    
    if f_x >= center_rec[0][0]:
        if f_y >= center_rec[0][1]:  # fourth quadrant
            quadrant = 4
            rot_ang = 180 - rot_ang
        else:  # first quadrant
            quadrant = 1
    else:
        if f_y >= center_rec[0][1]:  # third quadrant
            quadrant = 3
            rot_ang += 180
        else:  # second quadrant
            quadrant = 2
            rot_ang = 360 - rot_ang

    return quadrant, rot_ang


frm_thread = Thread(target=get_frames, args=())
frm_thread.start()

show_thread = Thread(target=show, args=())
show_thread.start()

sleep(1)
print('Game Started')
# Initial tap to start the game
wp.pwmWrite(18, tap_val)
sleep(.25)
wp.pwmWrite(18, rel_val)

while game:
    current_frame = recent_frame.copy()
    frames += 1
    
    if frames <= cutoff:  # Run cascade classifier on first straightaway
        recs = casc_obj.detectMultiScale(current_frame)  # Detects car with a cascade classifier
        
        if type(recs) != tuple:  # If cascade classifier finds a match
            recent = recs
            # Defines search window to be used by CamShift
            search_window = (recs[0][0], recs[0][1], recs[0][2], recs[0][3])

            # Creates a version of current_frame that's cropped around the bounding box of the detected
            # rectangle in order to eventually create a mask from certain values in the cropped rectangle
            crop = current_frame[recs[0][1]:recs[0][1] + recs[0][3], recs[0][0]:recs[0][0] + recs[0][2]].copy()
            crop_hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)

            # Grabs a pixel close to the top left that's likely to be a road value and defines minimum
            # saturation and value numbers for the mask as 50 greater than those of that pixel
            top_left = crop_hsv[0:2, 0:2]
            sat_lim = np.average(top_left[:, :, 1]) + 50
            value_lim = np.average(top_left[:, :, 2]) + 50

            # Generates a 32x32 hue-saturation histogram from crop_hsv, excluding pixels deleted by the mask,
            # then normalizes it.
            mask = cv.inRange(crop_hsv, np.array((0., sat_lim, value_lim)), np.array((180., 255., 255.,)))
            crop_hist = cv.calcHist([crop_hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
            cv.normalize(crop_hist, crop_hist, 0., 255., cv.NORM_MINMAX)

            det = True
            cur_rec = True
        else:
            cur_rec = False

    if det and frames > cutoff:  # If cascade classifier period has elapsed and there's been a detection
        # CAMShift for current frame
        cur_hsv = cv.cvtColor(current_frame, cv.COLOR_BGR2HSV)  # Converts frame to HSV colorspace
        # Calculates a back projection of the current frame based on the last generated histogram.
        # On a back projection, every pixel of the current frame has a brightness that correlates
        # with the value of the histogram bin it fits into on crop_hist.
        cur_bp = cv.calcBackProject([cur_hsv], [0, 1], crop_hist, [0, 180, 0, 256], 1)

        if rec is not None and rec[0] != (0., 0.) and frames >= 50:  # CamShift starts after 50 frames.
            if first:  # Defines the dimensions of the rectangle to not mask out
                height = rec[1][1] * .7
                width = height
                base_height = rec[1][1]
                first = False

            bp_mask = np.zeros(cur_bp.shape, dtype=np.uint8)
            cen_x = rec[0][0]
            cen_y = rec[0][1]
            fin_x = cen_x - (width/2)
            fin_y = cen_y - (height/2)

            # Removes values from the back projection besides those in the car's detection range to increase 
            # effectivenes of CamShift
            bp_mask[int(fin_y):int(fin_y + height), int(fin_x):int(fin_x + width)] = 255
            cur_bp = cv.bitwise_and(cur_bp, cur_bp, mask=bp_mask)

        # CamShift returns a rectangle defined by a center point, width/height, and rotation angle
        # CamShift also returns a search window
        rec, search_window = cv.CamShift(cur_bp, search_window, crit)

        if rec == null_rec or rec[1][1] <= (base_height * .55):  # If CamShift made an error, reset from last success
            no_rec = True
            if streak == 0:
                x, y, w, h = search_window
            else:
                search_window = (x, y, w, h)

            streak += 1
            rec = null_rec
        else:
            if streak >= 5:
                anchor_frame = frames
                streak = 0
            
            streak = 0

            # Trigonometry
            # Generates two rectangles that are in front of and behind the car.
            # After determining which is in the front, the program will check 
            # if there are nonroad pixels within this rectangle to decide if it
            # should tap the screen.
            
            h_one = rec[1][1]
            spc = 15

            if tap and h_two >= base_height * .4:
                h_two = h_two * .85
            elif not tap:
                h_two = rec[1][1] * 2.75
                
            w_two = rec[1][0] * 3.5
            
            dis = (h_one/2) + (h_two/2) + spc
            ang = np.radians(rec[2])
            i = dis * math.cos(ang)
            j = math.sqrt((dis*dis) - (i*i))
            og_x, og_y = rec[0][0], rec[0][1]

            new_y = og_y - i
            new_x = og_x + j
            newer_y = og_y + i
            newer_x = og_x - j

            box_1 = ((new_x, new_y), (w_two, h_two), rec[2])
            box_2 = ((newer_x, newer_y), (w_two, h_two), rec[2])

            # Determine front rectangle
            if front is None:
                if box_1[0][1] < rec[0][1]:
                    front = box_1
                    B_1 = True
                else:
                    front = box_2
                    B_1 = False
            else:
                one_dis = math.sqrt((last_cen[0] - box_1[0][0])*(last_cen[0] - box_1[0][0]) +
                                    (last_cen[1] - box_1[0][1])*(last_cen[1] - box_1[0][1]))
                two_dis = math.sqrt((last_cen[0] - box_2[0][0])*(last_cen[0] - box_2[0][0]) +
                                    (last_cen[1] - box_2[0][1])*(last_cen[1] - box_2[0][1]))
                if one_dis < two_dis:
                    front = box_1
                    B_1 = True
                else:
                    front = box_2
                    B_1 = False
            
            rot_ang = math.degrees(math.asin(j/dis))
            quad, rot_ang = get_quad(rec, front, rot_ang)

            if last_ang is None:
                last_ang = rot_ang
            else:
                dif = abs(last_ang - rot_ang)
                if 75 <= dif <= 300:
                    if B_1:
                        front = box_2
                    else:
                        front = box_1

                    rot_ang = math.degrees(math.asin(j/dis))
                    quad, rot_ang = get_quad(rec, front, rot_ang)

            last_cen = front[0]
            last_ang = rot_ang
            
            # Extract from front
            # Takes the values from the rectangle in front of the car for analysis
            cur_rot = current_frame.copy()
            
            try:
                cur_size = (current_frame.shape[1], current_frame.shape[0])
                rec_size = (int(front[1][0]), int(front[1][1]))
                rec_cen = front[0]
                rec_ang = front[2]
                rot_mat = cv.getRotationMatrix2D(rec_cen, rec_ang, 1)
                rotated = cv.warpAffine(cur_rot, rot_mat, cur_size, cv.INTER_CUBIC)
                rot_crop = cv.getRectSubPix(rotated, rec_size, rec_cen)

            except ValueError:
                end = True
                cv.destroyAllWindows()
                wp.pwmWrite(18, rel_val)
                break

            # Calculate brightness difference
            # Large brightness difference between minimum and maximum indicates that
            # there is road and nonroad within the front rectangle
            if rot_crop is not None:
                rot_crop_hsv = cv.cvtColor(rot_crop, cv.COLOR_BGR2HSV)
                val_max = rot_crop_hsv[:, :, 2].max()
                val_min = rot_crop_hsv[:, :, 2].min()
                front_range = val_max - val_min
                
                if front_range >= 80:
                    wp.pwmWrite(18, tap_val)
                    tap = True
                else:
                    wp.pwmWrite(18, rel_val)
                    tap = False

        # Drawing
        if tap:
            cv.drawMarker(current_frame, (50, 50), (0, 0, 255), 3)

        points = cv.boxPoints(rec)
        cv.polylines(current_frame, np.int32([points]), True, (255, 0, 0))

        front_points = cv.boxPoints(front)
        cv.polylines(current_frame, np.int32([front_points]), True, (0, 255, 255))

        cv.rectangle(current_frame, (search_window[0], search_window[1]), ((search_window[0] + search_window[2]),
                     (search_window[1] + search_window[3])), (0, 255, 0))
            
    if cur_rec and frames <= cutoff:
        for x in range(len(recs)):
            x_one = recs[x][0]
            y_one = recs[x][1]
            x_two = x_one + recs[x][2]
            y_two = y_one + recs[x][3]
            cv.rectangle(current_frame, (x_one, y_one), (x_two, y_two), (0, 0, 255))

    show_frame = current_frame.copy()
