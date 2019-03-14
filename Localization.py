import sys
import cv2
import numpy as np


def degree_to_radian(degree):
    return degree * np.pi / 180


def calculate_distance(FOV, object_width, frame_object_width, frame_width):
    """
    :param FOV: Field Of View of camera. Check your camera specification
    :param object_width: Real width of target ( cm or m )
    :param frame_object_width: Target's width in frame ( pixels )
    :param frame_width: Frame's width ( pixels )
    :return: Real distance between camera and target

    If you can calculate target's rotated angle, you can caluclate more precisely ( frame_object_width => frame_object_width * cos(rotated angle) )
    """
    object_FOV = (frame_object_width / frame_width) * degree_to_radian(FOV)
    L = (object_width/2.) * (1./np.tan(object_FOV))
    return L


def tracking_with_distance_estimation(FOV, object_width, tracker_type='MEDIANFLOW'):
    """
    Firstly, you have to draw your tracking target's bounding box. And then your target's distance will be calculated automatically.
    You can choose trackers' type, but in my case, MEDIANFLOW was best case.
    """
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture(0)

    # Save result frames
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    if not video.isOpened():
        print("Please check your webcam connection!")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read webcam')
        sys.exit()

    bbox = cv2.selectROI(frame, False)

    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            L = calculate_distance(FOV, object_width, int(bbox[2]), frame.shape[1])
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            L = -1

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "Object Distance : " + str(int(L)), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow("Tracking", frame)

        # Save result frames
        # out.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            video.release()
            # out.release()
            break


if __name__ == '__main__':
    tracking_with_distance_estimation(FOV=69, object_width=9)
