import cv2
import argparse
import numpy as np

_CONFIG = {
    'display': False,
    'highlight': False,
    'webapp': False,
    'console': False
}

_CASCADE_PATH = 'haarcascade_frontalface_default.xml'


def main():
    parse_arguments()
    main_loop()
    shut_down()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--highlight', action='store_true')
    parser.add_argument('--webapp', action='store_true')
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--faces', action='store_true')
    setup(parser.parse_args())


def setup(args):
    _CONFIG['display'] = args.display
    _CONFIG['highlight'] = args.highlight and args.display
    _CONFIG['webapp'] = args.webapp
    _CONFIG['console'] = args.console
    _CONFIG['faces'] = args.faces


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
    if _CONFIG['faces']:
        faces = detect_faces(face_cascade, grey)
        handle_output(faces, frame)
    else:
        people = detect_people(hog, grey)
        handle_output(people, frame)
    display_camera_stream(frame)


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
    for rect in rects:
        highlight_image(frame, *rect)
        log_to_server(frame, *rect)
        log_to_console(*rect)


def highlight_image(frame, x, y, w, h):
    if _CONFIG['highlight']:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def log_to_server(frame, x, y, w, h):
    pass


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
