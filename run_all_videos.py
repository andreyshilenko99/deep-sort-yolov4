#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
import tensorflow
import os
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
from draw_enter import select_object, read_door_info

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


def find_centroid(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class CountTruth:
    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside


def get_truth(video_name, read_name='data_files/labels_counted.csv'):
    with open('data_files/labels_counted.csv', 'r') as file:
        lines = file.readlines()

    TruthArr = CountTruth(0, 0)
    for line in lines:
        line = line.split(",")
        if line[1] == video_name:
            TruthArr.inside = int(line[2])
            TruthArr.outside = int(line[3])
    return TruthArr


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

    def return_total_count(self):
        return (self.counter_in + self.counter_out)


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 1.0

    output_format = 'mp4'

    initialize_door_by_yourself = False
    door_array = None
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    error_values = []

    files = os.listdir('data_files/videos')
    for file in files:
        video_name = file
        print("opening video: {}".format(file))
        file_path = join('data_files/videos', video_name)
        output_name = 'save_data/out_' + video_name[0:-3] + output_format
        counter = Counter(counter_in=0, counter_out=0, track_id=0)
        truth = get_truth(video_name)

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
                all_doors = read_door_info('data_files/doors_info.csv')
                door_array = all_doors[video_name]

            door_centroid = find_centroid(door_array)
        border_door = door_array[3]
        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if not ret:
                y1 = (counter.counter_in - truth.inside) ** 2
                y2 = (counter.counter_out - truth.outside) ** 2
                total_count = counter.return_total_count()
                true_total = truth.inside + truth.outside
                err = abs(total_count - true_total) / true_total
                mse = (y1 + y2) / 2
                log_res = "in video: {}\n predicted / true\n counter in: {} / {}\n counter out: {} / {}\n" \
                          " total: {} / {}\n error: {}\n mse error: {}\n______________\n".format(video_name,
                                                                                                 counter.counter_in,
                                                                                                 truth.inside,
                                                                                                 counter.counter_out,
                                                                                                 truth.outside,
                                                                                                 total_count,
                                                                                                 true_total, err, mse)
                with open('log_results.txt', 'w') as file:
                    file.write(log_res)
                print(log_res)
                error_values.append(err)
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

            cv2.rectangle(frame, (int(door_array[0]), int(door_array[1])), (int(door_array[2]), int(door_array[3])),
                          (23, 158, 21), 3)
            for det in detections:
                bbox = det.to_tlbr()
                if show_detections and len(classes) > 0:
                    score = "%.2f" % (det.confidence * 100) + "%"

                    iou_val = str(round(bb_intersection_over_union(bbox, door_array), 3))
                    cv2.putText(frame, score + " iou: " + iou_val, (int(bbox[0]), int(bbox[3])), 0,
                                1e-3 * frame.shape[0], (0, 100, 255), 5)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                # first appearence of object with id=track.id

                if track.track_id not in counter.people_init or counter.people_init[track.track_id] == 0:
                    counter.obj_initialized(track.track_id)
                    #     was initialized in door, probably going in
                    # if (bb_intersection_over_union(bbox, door_array) >= 0.03 and bbox[3] < border_door) or (
                    if 0.1 < bb_intersection_over_union(bbox, door_array):
                        counter.people_init[track.track_id] = 1
                    #     initialized in the bus, mb going out
                    elif bb_intersection_over_union(bbox, door_array) < 0.1:  # and bbox[3] > border_door:
                        counter.people_init[track.track_id] = 2
                    counter.people_bbox[track.track_id] = bbox
                counter.cur_bbox[track.track_id] = bbox

                adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 5)

                if not show_detections:
                    track_cls = track.cls
                    cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0],
                                (0, 255, 0),
                                1)
                    cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)

            id_get_lost = [track.track_id for track in tracker.tracks if track.time_since_update >= 19
                           and track.age >= 19]
            id_inside_tracked = [track.track_id for track in tracker.tracks if track.age > 50]
            for val in counter.people_init.keys():
                # check bbox also
                vector_person = (counter.cur_bbox[val][0] - counter.people_bbox[val][0],
                                 counter.cur_bbox[val][1] - counter.people_bbox[val][1])

                if val in id_get_lost and counter.people_init[val] != -1:
                    iou_door = bb_intersection_over_union(counter.cur_bbox[val], door_array)

                    if counter.people_init[val] == 1 and iou_door <= 0.45 and vector_person[
                        1] > 50:  # and counter.people_bbox[val][3] > border_door \

                        counter.get_in()
                    elif counter.people_init[val] == 2 and iou_door > 0.03 and vector_person[
                        1] < -50:  # and counter.people_bbox[val][3] < border_door\

                        counter.get_out()
                    counter.people_init[val] = -1

                    print(find_centroid(counter.cur_bbox[val]))
                    print('\n', find_centroid(counter.people_bbox[val]))
                    print('\n', vector_person)
                    imaggg = cv2.line(frame, find_centroid(counter.cur_bbox[val]),
                                      find_centroid(counter.people_bbox[val]),
                                      (254, 0, 0), 7)
                    # cv2.imshow('frame', imaggg)
                    # cv2.waitKey(0)

                    del val
                elif val in id_inside_tracked and counter.people_init[val] == 1 \
                        and bb_intersection_over_union(counter.cur_bbox[val], door_array) <= 0.25 \
                        and vector_person[1] > 0:  # and \
                    # counter.people_bbox[val][3] > border_door:
                    counter.get_in()

                    counter.people_init[val] = -1
                    print(find_centroid(counter.cur_bbox[val]))
                    print('\n', find_centroid(counter.people_bbox[val]))
                    print('\n', vector_person)
                    imaggg = cv2.line(frame, find_centroid(counter.cur_bbox[val]),
                                      find_centroid(counter.people_bbox[val]),
                                      (0, 0, 255), 7)
                    # cv2.imshow('frame', imaggg)
                    # cv2.waitKey(0)

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

    mean_error = np.mean(error_values)
    print("mean error for {} videos: {}".format(len(files), mean_error))


if __name__ == '__main__':
    main(YOLO())
