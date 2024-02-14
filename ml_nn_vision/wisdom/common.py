import time

import cv2


class CameraFeedProcessor:
    def __init__(self, **kwargs):
        # OpenCV Video Capture
        device_index = kwargs.get("device", 0)
        self.cap = cv2.VideoCapture(device_index)

        desired_fps = kwargs.get("fps", 10)
        # Wait for a certain amount of time to achieve the desired FPS
        # Calculate the delay based on the desired FPS
        self.delay_ms = int(1000 / desired_fps)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def draw_text_around_bounding_box(
        frame,
        bbox,
        text,
        fg_color=(0, 0, 0),
        bg_color=(0, 0, 0),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.5,
        thickness=2,
        border=False,
    ):
        # Unpack the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Calculate position for label text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size[0]
        text_box_padding = 10
        text_baseline = text_size[1]
        text_x = x_min + text_box_padding
        text_y = y_min + text_baseline + text_box_padding + 2

        if border:
            cv2.rectangle(
                frame, (x_min, y_min), (x_max, text_y + text_box_padding), color=fg_color, thickness=thickness
            )

        # Draw label text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=fg_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def draw_rectangle(frame, bbox, color=(0, 0, 0), thickness=2):
        # Draw the landmarks and a rectangle around it.
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=thickness)

    def run(self, key_event_hook=None, process_frame_hook=None):
        while self.cap.isOpened():
            start_time = time.time()

            # Read a frame from the camera.
            ret, frame = self.cap.read()

            if not ret:
                break

            # Flip the frame for mirror effect.
            frame = cv2.flip(frame, 1)

            h, w, c = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if process_frame_hook:
                process_frame_hook(frame, frame_rgb, (h, w, c))

            # Draw FPS text
            fps_text = "FPS: {:.2f}".format(1.0 / (time.time() - start_time))
            self.draw_text_around_bounding_box(frame, (0, 0, w, h), fps_text)

            cv2.imshow("Camera feed", frame)

            # Wait for a key press (self.delay_ms-millisecond delay).
            key = cv2.waitKey(self.delay_ms) & 0xFF

            # Break the loop when 'q' is pressed.
            if key == ord("q"):
                break

            if key_event_hook:
                key_event_hook(key)
