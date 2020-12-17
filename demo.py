#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
import tensorflow
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
from os.path import join
from collections import OrderedDict
from operator import sub

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.InteractiveSession(config=config)

warnings.filterwarnings('ignore')

rect_endpoint_tmp = []
rect_bbox = []
bbox_list_rois = []
drawing = False


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def select_object(img):
    """
    Interactive select rectangle ROIs and store list of bboxes.

    Parameters
    ----------
    img :
           image 3-dim.

    Returns
    -------
    bbox_list_rois : list of list of int
           List of bboxes of rectangle rois.
    """

    # mouse callback function
    def draw_rect_roi(event, x, y, flags, param):
        # grab references to the global variables
        global rect_bbox, rect_endpoint_tmp, drawing

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that drawing is being
        # performed. set rect_endpoint_tmp empty list.
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_endpoint_tmp = []
            rect_bbox = [(x, y)]
            drawing = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # drawing operation is finished
            rect_bbox.append((x, y))
            drawing = False

            # draw a rectangle around the region of interest
            p_1, p_2 = rect_bbox
            cv2.rectangle(img, p_1, p_2, color=(0, 255, 0), thickness=1)
            cv2.imshow('image', img)

            # for bbox find upper left and bottom right points
            p_1x, p_1y = p_1
            p_2x, p_2y = p_2

            lx = min(p_1x, p_2x)
            ty = min(p_1y, p_2y)
            rx = max(p_1x, p_2x)
            by = max(p_1y, p_2y)

            # add bbox to list if both points are different
            if (lx, ty) != (rx, by):
                bbox = [lx, ty, rx, by]
                bbox_list_rois.append(bbox)

        # if mouse is drawing set tmp rectangle endpoint to (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_endpoint_tmp = [(x, y)]

    # clone image img and setup the mouse callback function
    img_copy = img.copy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1200, 600)
    cv2.setMouseCallback('image', draw_rect_roi)

    # keep looping until the 'c' key is pressed
    while True:
        # display the image and wait for a keypress
        if not drawing:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1200, 600)
            cv2.imshow('image', img)
        elif drawing and rect_endpoint_tmp:
            rect_cpy = img.copy()
            start_point = rect_bbox[0]
            end_point_tmp = rect_endpoint_tmp[0]
            cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
            cv2.imshow('image', rect_cpy)

        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord('c'):
            break
    # close all open windows
    cv2.destroyAllWindows()

    return bbox_list_rois


def find_centroid(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


class Counter:
    def __init__(self, counter_in, counter_out, track_id):
        self.people_init = OrderedDict()
        self.people_bbox = OrderedDict()
        self.cur_bbox = OrderedDict()
        # self.dissappeared_frames = OrderedDict()
        self.counter_in = counter_in
        self.counter_out = counter_out
        self.track_id = track_id

    def obj_initialized(self, track_id):
        self.people_init[track_id] = 0

    def get_in(self):
        self.counter_in += 1

    def get_out(self):
        self.counter_out += 1

    def show_counter(self):
        return self.counter_in, self.counter_out


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    output_format = 'mp4'
    video_name = 'bus5.mp4'
    file_path = join('data_files', video_name)
    output_name = 'save_data/out_' + video_name[0:-3] + output_format
    initialize_door_by_yourself = False
    door_array = None
    enter_array = None

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    counter = Counter(counter_in=0, counter_out=0, track_id=0)

    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_name, fourcc, 15, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    ret, first_frame = video_capture.read()

    if door_array is None:
        if initialize_door_by_yourself:
            door_array = select_object(first_frame)[0]
            print(door_array)
        else:
            # [681, 9, 1123, 750] # bus1
            # door_array = [712, 10, 1468, 613] # bus 4
            door_array = [715, 86, 1380, 799] #  bus 5
        door_centroid = find_centroid(door_array)

    # if enter_array is None:
    #     if initialize_enter_by_yourself:
    #         enter_array = select_object(first_frame)[0]
    #         print(enter_array)
    #     else:
    #         enter_array = [712, 10, 1468, 613]
    #     enter_centroid = find_centroid(enter_array)
    border_door = door_array[3]
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for det in detections:
            bbox = det.to_tlbr()
            if show_detections and len(classes) > 0:
                score = "%.2f" % (det.confidence * 100) + "%"

                iou_val = str(round(bb_intersection_over_union(bbox, door_array), 3))
                cv2.putText(frame, score + " iou: " + iou_val, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 100, 255), 5)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # first appearence of object with id=track.id

            if track.track_id not in counter.people_init or counter.people_init[track.track_id] == 0:
                counter.obj_initialized(track.track_id)
                #     was initialized in door, probably going in
                if (bb_intersection_over_union(bbox, door_array) >= 0.03 and bbox[3] < border_door) or (
                        0.2 < bb_intersection_over_union(bbox, door_array)):
                    counter.people_init[track.track_id] = 1
                #     initialized in the bus, mb going out
                elif bb_intersection_over_union(bbox, door_array) < 0.45 and bbox[3] > border_door:
                    counter.people_init[track.track_id] = 2
                counter.people_bbox[track.track_id] = bbox
            counter.cur_bbox[track.track_id] = bbox

            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 5)

            if not show_detections:
                track_cls = track.cls
                cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 255, 0),
                            1)
                cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        id_get_lost = [track.track_id for track in tracker.tracks if track.time_since_update >= 19
                       and track.age >= 25 ]
        id_inside_tracked = [track.track_id for track in tracker.tracks if track.age > 300]
        for val in counter.people_init.keys():
            # check bbox also

            if val in id_get_lost and counter.people_init[val] != -1 :
                iou_door = bb_intersection_over_union(counter.cur_bbox[val], door_array)
                vector_person = map(sub, find_centroid(counter.cur_bbox[val]), find_centroid(counter.people_bbox[val]))
                print(vector_person)
                imaggg = cv2.line(frame, find_centroid(counter.cur_bbox[val]), find_centroid(counter.people_bbox[val]), (234,33,0), 7)
                cv2.imshow('frame', imaggg)
                cv2.waitKey(0)
                if counter.people_init[val] == 1 and iou_door <= 0.45 and counter.people_bbox[val][3] > border_door:
                    counter.get_in()
                elif counter.people_init[val] == 2 and iou_door > 0.03 and counter.people_bbox[val][3] < border_door:
                    counter.get_out()
                counter.people_init[val] = -1
                # del counter.people_bbox[val]
                del val
            elif val in id_inside_tracked and counter.people_init[val] == 1 \
                    and bb_intersection_over_union(counter.cur_bbox[val], door_array) <= 0.55 and counter.people_bbox[val][3] > border_door:
                counter.get_in()
                counter.people_init[val] = -1
                del val
        ins, outs = counter.show_counter()
        cv2.putText(frame, "in: {}, out: {} ".format(ins, outs), (10, 30), 0,
                    1e-3 * frame.shape[0], (255, 0, 0), 5)


        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1400, 800)
        cv2.imshow('image', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
