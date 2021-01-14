#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
import tensorflow as tf
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
from rectangles import find_centroid, Rectangle, rect_square

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

warnings.filterwarnings('ignore')
rect_endpoint_tmp = []
rect_bbox = []
bbox_list_rois = []
drawing = False


class CountTruth:
    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside


def get_truth(video_name):
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
        return self.counter_in + self.counter_out

def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


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
    check_gpu()
    files = sorted(os.listdir('data_files/videos'))
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

        all_doors = read_door_info('data_files/doors_info.csv')
        door_array = all_doors[video_name]
        rect_door = Rectangle(door_array[0], door_array[1], door_array[2], door_array[3])

        border_door = door_array[3]
        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if not ret:
                y1 = (counter.counter_in - truth.inside) ** 2
                y2 = (counter.counter_out - truth.outside) ** 2
                total_count = counter.return_total_count()
                true_total = truth.inside + truth.outside
                if true_total != 0:
                    err = abs(total_count - true_total) / true_total
                else:
                    err = abs(total_count - true_total)
                mse = (y1 + y2) / 2
                log_res = "in video: {}\n predicted / true\n counter in: {} / {}\n counter out: {} / {}\n" \
                          " total: {} / {}\n error: {}\n mse error: {}\n______________\n".format(video_name,
                                                                                                 counter.counter_in,
                                                                                                 truth.inside,
                                                                                                 counter.counter_out,
                                                                                                 truth.outside,
                                                                                                 total_count,
                                                                                                 true_total, err, mse)
                with open('log_results.txt', 'a') as log:
                    log.write(log_res)
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
                    rect_head = Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    rect_door = Rectangle( int(door_array[0]), int(door_array[1]), int(door_array[2]), int(door_array[3]) )
                    intersection = rect_head & rect_door

                    if intersection:
                        squares_coeff = rect_square(*intersection)/ rect_square(*rect_head)
                        cv2.putText(frame, score + " inter: " + str(round(squares_coeff, 3)), (int(bbox[0]), int(bbox[3])), 0,
                                1e-3 * frame.shape[0], (0, 100, 255), 5)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                # first appearence of object with id=track.id

                if track.track_id not in counter.people_init or counter.people_init[track.track_id] == 0:
                    counter.obj_initialized(track.track_id)
                    rect_head = Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])

                    intersection = rect_head & rect_door
                    if intersection:

                        intersection_square = rect_square(*intersection)
                        head_square = rect_square(*rect_head)
                        #     was initialized in door, probably going in
                        if (intersection_square/ head_square ) >= 0.8:
                            counter.people_init[track.track_id] = 2
                            #     initialized in the bus, mb going out
                        elif (intersection_square/ head_square ) <= 0.4 or bbox[3] > border_door:
                            counter.people_init[track.track_id] = 1
                    # res is None, means that object is not in door contour
                    else:
                        counter.people_init[track.track_id] = 1
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

            id_get_lost = [track.track_id for track in tracker.tracks if track.time_since_update >= 25
                           and track.age >= 29]
            id_inside_tracked = [track.track_id for track in tracker.tracks if track.age > 60]
            for val in counter.people_init.keys():
                # check bbox also
                inter_square = 0
                cur_square = 0
                ratio = 0
                cur_c = find_centroid(counter.cur_bbox[val])
                init_c = find_centroid(counter.people_bbox[val])
                vector_person = (cur_c[0] - init_c[0],
                                 cur_c[1] - init_c[1])

                if val in id_get_lost and counter.people_init[val] != -1:
                    rect_сur = Rectangle(counter.cur_bbox[val][0], counter.cur_bbox[val][1], counter.cur_bbox[val][2], counter.cur_bbox[val][3])
                    inter = rect_сur & rect_door
                    if inter:

                        inter_square = rect_square(*inter)
                        cur_square = rect_square(*rect_сur)
                        try:
                            ratio = inter_square / cur_square
                        except ZeroDivisionError:
                            ratio = 0
                    # if vector_person < 0 then current coord is less than initialized, it means that man is going
                    # in the exit direction
                    if vector_person[1] > 70 and counter.people_init[val] == 2 \
                            and ratio < 0.9:  # and counter.people_bbox[val][3] > border_door \
                        counter.get_in()

                    elif vector_person[1] < -70 and counter.people_init[val] == 1 \
                            and ratio >= 0.6:
                        counter.get_out()

                    counter.people_init[val] = -1
                    # print(f"person left frame")
                    # print(f"current centroid - init : {cur_c} - {init_c}\n")
                    # print(f"vector: {vector_person}\n")
                    del val

                # elif val in id_inside_tracked and val not in id_get_lost and counter.people_init[val] == 1 \
                #         and bb_intersection_over_union(counter.cur_bbox[val], door_array) <= 0.3 \
                #         and vector_person[1] > 0:  # and \
                #     # counter.people_bbox[val][3] > border_door:
                #     counter.get_in()
                #
                #     counter.people_init[val] = -1
                #     print(f"person is tracked for a long time")
                #     print(f"current centroid - init : {cur_c} - {init_c}\n")
                #     print(f"vector: {vector_person}\n")
                #     imaggg = cv2.line(frame, find_centroid(counter.cur_bbox[val]),
                #                       find_centroid(counter.people_bbox[val]),
                #                       (0, 0, 255), 7)

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
                # print("FPS = %f" % fps)

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
        del door_array

    mean_error = np.mean(error_values)
    print("mean error for {} videos: {}".format(len(files), mean_error))


if __name__ == '__main__':
    main(YOLO())
