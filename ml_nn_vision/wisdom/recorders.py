import time
from contextlib import AbstractContextManager

import cv2
import mediapipe as mp
import numpy as np

from .settings import TRAINING_DATA_DIR
from .utils import get_bounding_box, landmark_to_ratio


class HandGestureRecorder(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def __init__(self, label):
        self.label = label

        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

        self.training_data_dir = TRAINING_DATA_DIR / self.__class__.__name__ / self.label

    def _lazy_initialization(self):
        # Lazy initializations.
        if not self.training_data_dir.exists():
            self.training_data_dir.mkdir(parents=True)

    def record_landmark(self, landmark_data):
        timestamp = int(time.time())
        filename = f"{self.__class__.__name__}__{self.label}__{timestamp}.npy"
        np.save(self.training_data_dir / filename, np.array(landmark_data).flatten())

    def run(self):
        while self.cap.isOpened():
            # Read a frame from the webcam
            ret, frame = self.cap.read()

            if not ret:
                break

            h, w, c = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results = self.hands.process(frame_rgb)

            # This will only contain the final landmark data if there are more.
            processed_landmark_data = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    bbox = get_bounding_box((w, h), hand_landmarks)

                    # Draw the landmarks and a rectangle around it.
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    processed_landmark_data = []

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)

                        # Draw landmarks on image
                        # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

                        # Convert landmark to ratio
                        ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                        processed_landmark_data.append((ratio_x, ratio_y))

            cv2.imshow("Hand Landmarks", frame)

            # Wait for a key press (1-millisecond delay).
            key = cv2.waitKey(1) & 0xFF

            # Break the loop when 'q' is pressed.
            if key == ord("q"):
                break

            # Record the frame when 'r' is pressed.
            if key == ord("r"):
                if processed_landmark_data is not None:
                    self.record_landmark(processed_landmark_data)
