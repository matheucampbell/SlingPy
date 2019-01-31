import cv2 as cv
import numpy as np
import math

game = True
first = True
det = False
tap = False
crop_hist = None
last_ang = None
front = None
recs = None
rec = None
frames = 0
streak = 0
base_height = 0
crit = (cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 10, 1)
null_rec = ((0.0, 0.0), (0.0, 0.0), 0.0)

vid = cv.VideoCapture('C:\\Users\\Matheu Campbell\\Videos\\Tracking\\PiVid.h264')
casc_path = 'C:\\LocalPos\\PiClassifier\\Classifier\\cascade.xml'
casc_obj = cv.CascadeClassifier(casc_path)

while game:
    if vid.read()[0]:

        current_frame = vid.read()[1]
        frames += 1

        if current_frame is None:
            break

        if not first and 385 <= frames <= 750:
            block_points = cv.boxPoints(block_rec)
            cv.polylines(current_frame, np.int32([block_points]), True, (0, 0, 0), thickness=16)

        if frames <= 30:  # Only look for car in first 30 frames
            recs = casc_obj.detectMultiScale(current_frame)

            if type(recs) != tuple:  # Redefining car histogram if object detector finds object
                search_window = (recs[0][0], recs[0][1], recs[0][2], recs[0][3])

                crop = current_frame[recs[0][1]:recs[0][1] + recs[0][3], recs[0][0]:recs[0][0] + recs[0][2]].copy()
                crop_hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)

                top_left = crop_hsv[0:2, 0:2]
                sat_lim = np.average(top_left[:, :, 1]) + 50
                value_lim = np.average(top_left[:, :, 2]) + 50

                mask = cv.inRange(crop_hsv, np.array((0., sat_lim, value_lim)), np.array((180., 255., 255.,)))
                crop_hist = cv.calcHist([crop_hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
                cv.normalize(crop_hist, crop_hist, 0., 255., cv.NORM_MINMAX)

                det = True

        if det:
            # CAMShift for current frame
            cur_hsv = cv.cvtColor(current_frame, cv.COLOR_BGR2HSV)
            cur_bp = cv.calcBackProject([cur_hsv], [0, 1], crop_hist, [0, 180, 0, 256], 1)

            if rec is not None and rec[1] != (0., 0.) and frames >= 25:
                if first:
                    height = rec[1][1] * 1.5
                    width = height
                    base_height = rec[1][1]
                    base_width = rec[1][0]
                    first = False

                bp_mask = np.zeros(cur_bp.shape, dtype=np.uint8)
                cen_x = rec[0][0]
                cen_y = rec[0][1]
                fin_x = cen_x - (width/2)
                fin_y = cen_y - (height/2)

                bp_mask[int(fin_y):int(fin_y + height), int(fin_x):int(fin_x + width)] = 255
                cur_bp = cv.bitwise_and(cur_bp, cur_bp, mask=bp_mask)

            rec, search_window = cv.CamShift(cur_bp, search_window, crit)

            block_rec = ((rec[0][0], rec[0][1]), (rec[1][0] * 3, rec[1][1] * 1.65), rec[2])

            if rec == null_rec or rec[1][1] <= (base_height * .75):
                if streak == 0:
                    x, y, w, h = search_window
                else:
                    search_window = (x, y, w, h)
                streak += 1
                rec = null_rec
            else:
                streak = 0

            # Trigonometry
            if rec[1] != (0.0, 0.0):
                h_one = rec[1][1]
                spc = 10
                h_two = rec[1][1] * 2.5
                w_two = rec[1][0] * 2

                dis = (h_one / 2) + (h_two / 2) + spc
                ang = np.radians(rec[2])
                i = dis * math.cos(ang)
                j = math.sqrt((dis*dis) - (i*i))
                og_x = rec[0][0]
                og_y = rec[0][1]

                new_y = og_y - i
                new_x = og_x + j
                newer_y = og_y + i
                newer_x = og_x - j

                box_1 = ((new_x, new_y), (w_two, h_two), rec[2])
                box_2 = ((newer_x, newer_y), (w_two, h_two), rec[2])

                if front is None:
                    if box_1[0][1] < rec[0][1]:
                        front = box_1
                    else:
                        front = box_2
                else:
                    one_dis = math.sqrt((last_cen[0] - box_1[0][0])*(last_cen[0] - box_1[0][0]) +
                                        (last_cen[1] - box_1[0][1])*(last_cen[1] - box_1[0][1]))
                    two_dis = math.sqrt((last_cen[0] - box_2[0][0])*(last_cen[0] - box_2[0][0]) +
                                        (last_cen[1] - box_2[0][1])*(last_cen[1] - box_2[0][1]))
                    if one_dis < two_dis:
                        front = box_1
                    else:
                        front = box_2

                f_x, f_y = front[0]
                rot_ang = math.degrees(math.asin(j/dis))

                if f_x >= rec[0][0]:
                    if f_y >= rec[0][1]:  # fourth quadrant
                        quadrant = 4
                        rot_ang = 180 - rot_ang
                    else:
                        quadrant = 1
                else:
                    if f_y >= rec[0][1]:  # third quadrant
                        quadrant = 3
                        rot_ang += 180
                    else:  # second quadrant
                        quadrant = 2
                        rot_ang = 360 - rot_ang

                last_cen = front[0]

                # Extract from front
                cur_rot = current_frame.copy()

                cur_size = (current_frame.shape[1], current_frame.shape[0])
                rec_size = (int(front[1][0]), int(front[1][1]))
                rec_cen = front[0]
                rec_ang = front[2]
                rot_mat = cv.getRotationMatrix2D(rec_cen, rec_ang, 1)
                cv.warpAffine(cur_rot, rot_mat, cur_size, cv.INTER_CUBIC)
                rot_crop = cv.getRectSubPix(cur_rot, rec_size, rec_cen)

                # Calculate value range
                if rot_crop is not None:
                    rot_crop_hsv = cv.cvtColor(rot_crop, cv.COLOR_BGR2HSV)
                    val_max = rot_crop_hsv[:, :, 2].max()
                    val_min = rot_crop_hsv[:, :, 2].min()
                    front_range = val_max - val_min

                    if front_range >= 80:
                        tap = True
                    else:
                        tap = False

            # Drawing
            if tap and rec[1] != (0.0, 0.0):
                cv.drawMarker(current_frame, (50, 30), (0, 0, 255), 3)

            points = cv.boxPoints(rec)
            cv.polylines(current_frame, np.int32([points]), True, (255, 0, 0))

            front_points = cv.boxPoints(front)
            cv.polylines(current_frame, np.int32([front_points]), True, (0, 255, 255))

            cv.rectangle(current_frame, (search_window[0], search_window[1]), ((search_window[0] + search_window[2]),
                         (search_window[1] + search_window[3])), (0, 255, 0))

            if frames <= 30 and det is True:
                for x in range(len(recs)):
                    x_one = recs[x][0]
                    y_one = recs[x][1]
                    x_two = x_one + recs[x][2]
                    y_two = y_one + recs[x][3]
                    cv.rectangle(current_frame, (x_one, y_one), (x_two, y_two), (0, 0, 255))

            cv.imshow('Back Projection', cur_bp)
            cv.waitKey(1)

        cv.imshow('Video', current_frame)
        cv.waitKey()

    else:
        cv.destroyWindow('Video')
        vid.release()
        game = False
