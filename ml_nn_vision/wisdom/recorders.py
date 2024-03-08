import time
from contextlib import AbstractContextManager

import cv2
import mediapipe as mp
import numpy as np

from .common import CameraFeedProcessor, RealSenseFeedProcessor
from .settings import TRAINING_DATA_DIR
from .utils import get_bounding_box, landmark_to_ratio


class HandGestureRecorderLive(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, label):
        self.label = label

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

        self.training_data_dir = TRAINING_DATA_DIR / self.__class__.__name__ / self.label

        self.camera_feed_proc = RealSenseFeedProcessor()

    def _lazy_initialization(self):
        # Lazy initializations.
        if not self.training_data_dir.exists():
            self.training_data_dir.mkdir(parents=True)

    def record_landmark(self, landmark_data):
        timestamp = int(time.time())
        filename = f"{self.__class__.__name__}__{self.label}__{timestamp}.npy"
        np.save(self.training_data_dir / filename, np.array(landmark_data).flatten())

    def run(self):
        def process_frame_hook(frame, frame_rgb, frame_shape):
            h, w, c = frame_shape

            # Process the image with MediaPipe
            results = self.hands.process(frame_rgb)

            # This will only contain the final landmark data if there are more.
            processed_landmark_data = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    bbox = get_bounding_box((w, h), hand_landmarks)

                    # Draw the landmarks and a rectangle around it.
                    self.camera_feed_proc.draw_rectangle(frame, bbox)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    processed_landmark_data = []

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)

                        # Draw landmarks on image
                        # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                        # Convert landmark to ratio
                        ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                        processed_landmark_data.append((ratio_x, ratio_y))

                    self.record_landmark(processed_landmark_data)

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)


class BodyGestureRecorderLive(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, label):
        self.label = label

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.mp_pose = mp.solutions.pose

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65)

        self.training_data_dir = TRAINING_DATA_DIR / self.__class__.__name__ / self.label

        self.camera_feed_proc = RealSenseFeedProcessor()

    def _lazy_initialization(self):
        # Lazy initializations.
        if not self.training_data_dir.exists():
            self.training_data_dir.mkdir(parents=True)

    def record_landmark(self, landmark_data):
        timestamp = int(time.time())
        filename = f"{self.__class__.__name__}__{self.label}__{timestamp}.npy"
        np.save(self.training_data_dir / filename, np.array(landmark_data).flatten())

    def run(self):
        def key_event_hook(e):
            ...

        def process_frame_hook(frame, frame_rgb, frame_shape):
            h, w, c = frame_shape

            # Process the image with MediaPipe
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                body_landmarks = results.pose_landmarks
                # for body_landmarks in results.pose_landmarks:
                bbox = get_bounding_box((w, h), body_landmarks)

                # Draw the landmarks and a rectangle around it.
                self.camera_feed_proc.draw_rectangle(frame, bbox)
                self.mp_drawing.draw_landmarks(frame, body_landmarks, self.mp_pose.POSE_CONNECTIONS)

                processed_landmark_data = []

                for idx, landmark in enumerate(body_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    # Draw landmarks on image
                    # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                    # Convert landmark to ratio
                    ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                    processed_landmark_data.append((ratio_x, ratio_y))

                self.record_landmark(processed_landmark_data)

                #     for landmark in landmarks.landmark:
                #         x = int(landmark.x * image.shape[1])
                #         y = int(landmark.y * image.shape[0])
                #         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                # for i, landmark in enumerate(results.pose_landmarks.landmark):
                #         # Convert the landmark position to pixel coordinates
                #     landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                #                                                               frame.shape[1], frame.shape[0])
                #     if landmark_px:
                #         cv2.circle(frame, landmark_px, 5, (255, 0, 0), -1)  # Draw landmark
                #
                # # Alternatively, to draw connections, use the draw_landmarks function with customized connections
                # self.mp_drawing.draw_landmarks(
                #     frame_rgb,
                #     results.pose_landmarks,
                #     self.mp_pose.POSE_CONNECTIONS,
                #     landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         bbox = get_bounding_box((w, h), hand_landmarks)
            #
            #         # Draw the landmarks and a rectangle around it.
            #         self.camera_feed_proc.draw_rectangle(frame, bbox)
            #         self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            #
            #         processed_landmark_data = []
            #
            #         for idx, landmark in enumerate(hand_landmarks.landmark):
            #             x, y = int(landmark.x * w), int(landmark.y * h)
            #
            #             # Draw landmarks on image
            #             # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            #
            #             # Convert landmark to ratio
            #             ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
            #             processed_landmark_data.append((ratio_x, ratio_y))
            #
            #         self.record_landmark(processed_landmark_data)

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
