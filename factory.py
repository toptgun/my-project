#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector
import openvino as ov

FORCE_STOP = False


def thread_cam1(q):
    #  MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")
    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model(
        "/home/iklop237/github-training/my-project/smart-factory/EfficientNetB0/openvino/openvino.xml"
    )
    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    flag = True

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))
        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))
        # abnormal detect
        input_tensor = np.expand_dims(detected, 0)
        if flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)
            ppp.input().tensor().set_shape(input_tensor.shape).set_element_type(
                ov.Type.u8
            ).set_layout(ov.Layout("NHWC"))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout("NCHW"))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            flag = False
        
        # Inference OpenVINO
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        print(f"{probs}")
        # TODO: Calculate ratios
        x_ratio = probs[0] * 100  # 비율을 백분율로 변환
        circle_ratio = probs[1] * 100  # 비율을 백분율로 변환
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")
        # TODO: in queue for moving the actuator 1
        if x_ratio > 50:
            q.put(("PUSH", 1))
    cap.release()
    q.put(("DONE", None))
    exit()


def thread_cam2(q):
    # MotionDetector
    det = MotionDetector()
    det.load_preset("resources/motion.cfg", "default")
    # ColorDetector
    color = ColorDetector()
    color.load_preset("resources/color.cfg", "default")
    # HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        #  HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))
        #  Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue
        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))
        # Detect color
        predict = color.detect(detected)
        if not predict:
            continue
        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")
        # Enqueue to handle actuator 2
        if name == "blue":
            q.put(("PUSH", 2))

    cap.release()
    q.put(("DONE", None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP, ctrl

    parser = ArgumentParser(prog="python3 factory.py", description="Factory tool")

    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
    args = parser.parse_args()

    # Create a Queue
    q = Queue()
    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data

            try:
                event = q.get_nowait()
            except Empty:
                continue

            name, data = event
            if name.startswith("VIDEO:"):
                imshow(name[6:], data)
            elif name == "PUSH":
                ctrl.push_actuator(data)
            elif name == "DONE":
                FORCE_STOP = True
                ctrl.push_actuator(data)

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.

            q.task_done()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
