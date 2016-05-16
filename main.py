from threading import Thread
from time import time
import cv2
import argparse
import socket
import io
import numpy as np

_DEFAULTS = {
    'server': '127.0.0.1',
    'port': 8000,
    'timeout': 10}

_CONFIG = {}

_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

last_detected = None

def main():
    parse_arguments()
    main_loop()
    shut_down()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--display', action='store_true', default=False,
        help='Display the camera stream')
    parser.add_argument(
        '--highlight', action='store_true', default=False,
        help='Highlight detected incidents in the captured frame')
    parser.add_argument(
        '--noserver', action='store_false', default=True,
        help='Disable logging to the server')
    parser.add_argument(
        '--console', action='store_true', default=False,
        help='Enable console logging')
    parser.add_argument(
        '--faces', action='store_true', default=False,
        help='Enable faces mode (default mode detects people)')
    parser.add_argument(
        '--threading', action='store_true', default=False,
        help='Handle output in separate thread (experimental)')
    parser.add_argument(
        '-s', nargs='?', type=str, default=_DEFAULTS['server'],
        help='Set the hostname of the server running the webapp')
    parser.add_argument(
        '-p', nargs='?', type=int, default=_DEFAULTS['port'],
        help='Set the port number that the webapp is listening on')
    parser.add_argument(
        '-t', nargs='?', type=int, default=_DEFAULTS['timeout'],
        help='Set the timeout between incidents')
    setup(parser.parse_args())


def setup(args):
    _CONFIG['display'] = args.display
    _CONFIG['highlight'] = args.highlight
    _CONFIG['webapp'] = args.noserver
    _CONFIG['console'] = args.console
    _CONFIG['faces'] = args.faces
    _CONFIG['threading'] = args.threading
    _CONFIG['server'] = args.s
    _CONFIG['port'] = args.p
    _CONFIG['timeout'] = args.t


def main_loop():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    while True:
        process_camera_stream(camera, face_cascade, hog)
        if exit_pressed():
            break
    camera.release()


def process_camera_stream(camera, face_cascade, hog):
    return_code, frame = camera.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if check_timer():
        if _CONFIG['faces']:
            faces = detect_faces(face_cascade, grey)
            handle_output(faces, frame)
        else:
            people = detect_people(hog, grey)
            handle_output(people, frame)
    display_camera_stream(frame)


def check_timer():
    global last_detected
    if last_detected is None:
        last_detected = time()
        return True
    current_time = time()
    if current_time - last_detected > _CONFIG['timeout']:
        last_detected = current_time
        return True
    return False



def detect_faces(classifier, frame):
    return classifier.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30))


def detect_people(hog, frame):
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05)
    return rects


def handle_output(rects, frame):
    if _CONFIG['threading']:
        thread = Thread(target=handle_output_thread, args=(rects, frame,))
        thread.start()
    else:
        handle_output_thread(rects, frame)


def handle_output_thread(rects, frame):
    for rect in rects:
        highlight_image(frame, *rect)
        log_to_console(*rect)
    if len(rects) > 0:
        log_to_server(frame)


def highlight_image(frame, x, y, w, h):
    if _CONFIG['highlight']:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def log_to_server(frame):
    if _CONFIG['webapp']:
        retval, jpg = cv2.imencode('.jpg', frame)
        stream = io.BytesIO(jpg)
        s = socket.socket()
        s.connect((_CONFIG['server'], _CONFIG['port']))
        next = stream.read()
        while next:
            s.send(next)
            next = stream.read()
        s.shutdown(socket.SHUT_RDWR)
        s.close()


def log_to_console(x, y, w, h):
    if _CONFIG['console']:
        print 'Detected at (%d, %d), w: %d h: %d' % (x, y, w, h)


def display_camera_stream(frame):
    if _CONFIG['display']:
        cv2.imshow('S-Pi', frame)


def exit_pressed():
    return cv2.waitKey(1) & 0xFF == ord('q')


def shut_down():
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
