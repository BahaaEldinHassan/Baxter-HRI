import time
from contextlib import AbstractContextManager

import cv2
import mediapipe as mp
import numpy as np

from .settings import TRAINING_DATA_DIR


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

    def record_frame(self, data):
        timestamp = int(time.time())
        filename = f"{self.__class__.__name__}__{self.label}__{timestamp}.npy"
        np.save(self.training_data_dir / filename, np.array(data))

    def run(self):
        while self.cap.isOpened():
            # Read a frame from the webcam
            ret, frame = self.cap.read()

            if not ret:
                break

            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results = self.hands.process(frame_rgb)
            landmarks_data = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Convert landmarks to a 2D array
                    landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                    # Render landmarks on the image
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Hand Landmarks", frame)

            # Wait for a key press (1-millisecond delay).
            key = cv2.waitKey(1) & 0xFF

            # Break the loop when 'q' is pressed.
            if key == ord("q"):
                break

            # Record the frame when 'r' is pressed.
            if key == ord("r"):
                if landmarks_data is not None:
                    self.record_frame(landmarks_data)
