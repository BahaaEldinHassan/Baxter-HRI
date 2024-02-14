import typing
from contextlib import AbstractContextManager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch.nn.functional as F
from savant.networks.cnn import CNNRunner
from wisdom.common import CameraFeedProcessor
from wisdom.utils import get_bounding_box, landmark_to_numpy, landmark_to_ratio


class HandGestureInferrer(AbstractContextManager):
    def __enter__(self):
        # self.run()
        self._initialize()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, **kwargs):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

        self.network: typing.Optional[CNNRunner] = None

        self.camera_feed_proc = CameraFeedProcessor()

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

    def run(self):
        def process_frame_hook(frame, frame_rgb, frame_shape):
            h, w, c = frame_shape

            # Process the image with MediaPipe
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    bbox = get_bounding_box((w, h), hand_landmarks)

                    self.camera_feed_proc.draw_rectangle(frame, bbox)
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

                        self.camera_feed_proc.draw_text_around_bounding_box(
                            frame,
                            bbox,
                            f"{label_with_highest_probability}",
                        )

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
