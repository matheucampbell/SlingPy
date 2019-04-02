# Hardware
**Raspberry Pi Model 3 B+ and Raspberry Pi Camera Module V2**

The Raspberry Pi is connected to a its camera, which provides live video feed to the program. The Raspberry Pi runs all of the software.
Originally, this product was intended to be a portable machine which controls a servo motor and runs OpenCV functions, 
making Raspberry Pi ideal. Portability wasn't ever achieved due to the machine's inability to play Sling Drift very well anyway.

**Micro Servo Motor**

The motor's end effector was comprised of foil wrapped around a plastic arm connected to a spool of wire via jumper wire. This allowed
the foil to disrupt the capacitance of the phone enough to trigger a response. The motor only needed to be able to tap the screen. This 
requires very little torque or range of motion, and micro-servos are cheap.

# Software
**Fetching Frames**
The Raspberry Pi's camera module cannot fetch frames very quickly, so the program resolves frames in a thread separate from the main one 
to maximize framerate. Additionally, the PiCamera is run at the lowest framerate at which the car is discernible. The built-in
Python module PiCamera was used to interface with the camera. Frames are also shown in a separate thread to prevent unnecessary 
processinglag.

**Initial Detection (Cascade Classifier)**
The program uses a cascade classifier to identify the bounding box of the car within the first fifteen frames. It's computationally
expensive and cannot detect the orientation of the car, so it's not used to track the car. Instead, it provides the bounding box for a 
more versatile function to utilize. Though the classifier is run on every one of the first 13 frames, only the 13th matters. This last
frame is used to generate a histogram of values for CAMShift. 

The classifier was trained using the following parameters with OpenCV's opencv_createsamples.exe and opencv_tran_cascade.exe.

*VEC Samples*
Training Samples: 1000
Width: 20
Height: 40

*Training Cascade*
numPos: 66
numNeg: 200
numStages: 15
precalcValBufSize[Mb] : 62000
precalcIdxBufSize[Mb] : 62000
stageType: BOOST
featureType: HAAR
minHitRate: 0.995
maxFalseAlarmRate: 0.5
weightTrimRate: 0.95
maxDepth: 1
mode: ALL


**Motion and Orientation Tracking (CAMShift)**
CAMShift works by tracking the centroid of areas of pixels with high values on a back projection. The histogram used to generate the 
back projection for every frame of the video feed is created from the bounding box of the car in the 13th frame by the cascade classifier
. After filtering out values from the bounding box that are probably part of the road using a mask that excludes pixels that are too 
dark or too unsaturated, the program generates a histogram for the HSV color space.

CAMShift is applied to this back projection using the cascade classifier's original bounding box as a search window. The function returns
a rectangle defined by its center point, width, height, and angle of rotation.

**Determining Front**
After the original detection, some trigonometric functions are used to determine the centerpoint of a rectangle in front and behind 
the car. Though the program is only interested in the front rectangle, the ambiguity in the definition of the rectangle means that both
rectangles must be defined, as both may be the front of the car. The rectangle that denotes a car rotated at 1 degree will be exactly the
same as a car rotated 181 degrees.

To determine which is the true front, the program assumes the first ever detection's front's centerpoint will have a lower Y 
value than the car's centerpoint (origin is top left). Subsequent fronts are determined by distance to previous front. A lower distance
will likely be the front.

**Extracting Front**
After the front rectangle is defined, the program uses a rotation matrix to extract the array that represents that area. This rectangle
is analyzed for its overall value difference. If there is a value difference of greater than 80, the car is determined to be approaching
a spot that is not the road, so it must turn. The program must tap.

**Why doesn't it work very well?**
Once the orange car reaches the orange zone, the areas around the road appear to CAMShift to be part of the car. This disrupts the 
CAMShift. Though the area immediately around the search window is masked out, the CAMShift can be thrown off if the car approaches the
wall too closely.

Additionally, the car has trouble navigating loops. Loops can be detected without disruption of car tracking, but there is no way to 
tell how long to hold down a tap on a loop, and continuous adjustment does not work as it does on turns.
