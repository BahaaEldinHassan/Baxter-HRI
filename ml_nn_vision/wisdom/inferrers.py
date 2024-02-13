import typing
from contextlib import AbstractContextManager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch.nn.functional as F

from savant.networks.cnn import CNNRunner
from wisdom.utils import get_bounding_box, landmark_to_numpy, landmark_to_ratio


class HandGestureInferrer(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._initialize()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def __init__(self):
        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

        self.network: typing.Optional[CNNRunner] = None

    def _initialize_network(self):
        runner = CNNRunner()

        checkpoint_filename = f"{runner.__class__.__name__}_checkpoint.ckpt"

        if not (Path(checkpoint_filename).exists() and Path(checkpoint_filename).is_file()):
            raise RuntimeError(f"Checkpoint file not found: {checkpoint_filename}")

        runner.load_state(checkpoint_filename)

        # runner.predict(final_test_set)
        self.network = runner

    def _initialize(self):
        self._initialize_network()

    def display_label_text(self, frame, bbox, label, font_scale=0.5, thickness=1):
        # Unpack the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)

        # Calculate position for label text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size[0]
        text_baseline = text_size[1]
        text_x = x_min
        text_y = y_min - text_baseline

        # Draw label text
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    def run(self):
        while self.cap.isOpened():
            # Read a frame from the webcam
            ret, frame = self.cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            h, w, c = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    bbox = get_bounding_box((w, h), hand_landmarks)

                    # Draw the landmarks and a rectangle around it.
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    processed_landmark_data = []

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)

                        # Convert landmark to ratio
                        ratio_x, ratio_y = landmark_to_ratio((x, y), bbox)
                        processed_landmark_data.append((ratio_x, ratio_y))

                    # Infer (predict gesture).
                    if processed_landmark_data:
                        logits = self.network.predict(np.array([processed_landmark_data]))
                        probabilities = F.softmax(logits, dim=1)
                        # This dict is sorted.
                        label_probabilities = self.network.generate_label_probabilities(probabilities)
                        label_with_highest_probability = next(iter(label_probabilities))

                        self.display_label_text(
                            frame,
                            bbox,
                            f"{label_with_highest_probability} ({label_probabilities[label_with_highest_probability]})",
                        )

            cv2.imshow("Hand Landmarks", frame)

            # Wait for a key press (1-millisecond delay).
            key = cv2.waitKey(1) & 0xFF

            # Break the loop when 'q' is pressed.
            if key == ord("q"):
                break
