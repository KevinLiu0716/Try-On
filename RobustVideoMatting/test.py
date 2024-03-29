import cv2
import torch
import time
from model import MattingNetwork
from PIL import Image
from torchvision import transforms
from threading import Thread, Lock

import os
import subprocess

import zmq 
from datetime import datetime




test_script_path = os.path.join(os.getcwd(), '../DM-VTON/test.py')
# command = [
#         'python', test_script_path,
#         '--project', '../DM-VTON/react-flask/results',
#         '--name', 'test',
#         '--device', '0',
#         '--align_corners',
#         '--batch_size', '1',
#         '--workers', '12',
#         '--dataroot', '../DM-VTON/dataset/testdata',
#         '--pf_warp_checkpoint', '../DM-VTON/checkpoints/dmvton_pf_warp.pt',
#         '--pf_gen_checkpoint', '../DM-VTON/checkpoints/dmvton_pf_gen.pt'
#     ]
command = [
        'python', test_script_path,
        '--project', 'tryon',
        '--name', 'test',
        '--device', '0',
        '--align_corners',
        '--batch_size', '1',
        '--workers', '12',
        '--dataroot', 'screenshot',
        '--pf_warp_checkpoint', '../DM-VTON/checkpoints/dmvton_pf_warp.pt',
        '--pf_gen_checkpoint', '../DM-VTON/checkpoints/dmvton_pf_gen.pt'
    ]
process = subprocess.Popen(command)

print("sssssssssssssss")




# ----------- Utility classes -------------


class Camera:
    """
    A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
    Use .read() in a tight loop to get the newest frame.
    """

    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


class FPSTracker:
    """
    An FPS tracker that computes exponentialy moving average FPS.
    """

    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (
                    1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


class Displayer:
    """
    Wrapper for playing a stream with cv2.imshow().
    It also tracks FPS and optionally overlays info onto the stream.
    """

    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)

    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            # message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            message = f"{int(fps_estimate)} fps"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


def cv2_frame_to_cuda(frame):
    """
    convert cv2 frame to tensor.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    loader = transforms.ToTensor()
    return loader(Image.fromarray(frame)).to(device, dtype, non_blocking=True).unsqueeze(0)


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


# --------------- Main ---------------

if __name__ == '__main__':

    width, height = (1280, 720)  # the show windows size.
    output_background = 'white'  # Options: ["green", "white", "image"].
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load mobilenetv3 model
    model = MattingNetwork('mobilenetv3')
    model = model.to(device, dtype, non_blocking=True).eval()
    model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
    # model = torch.jit.script(model)
    # model = torch.jit.freeze(model)

    cam = Camera(width=width, height=height)
    dsp = Displayer('VideoMatting', cam.width, cam.height, show_info=True)

    bgr = None
    if output_background == 'white':
        bgr = torch.tensor([255, 255, 255], device=device, dtype=dtype).div(255).view(3, 1, 1)  # white background
    elif output_background == 'green':
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(3, 1, 1)  # green background

    with torch.no_grad():
            
            # Create a ZMQ context and socket
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.bind("tcp://*:5001")
            
            
            time.sleep(2)
            # Save each frame
            frame_count = 0
            while True:
                # matting
                frame = cam.read()
                src = cv2_frame_to_cuda(frame)
                rec = [None] * 4
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                if bgr is None:
                    h, w = src.shape[2:]
                    # print(h, w)
                    transform = transforms.Compose([
                        transforms.Resize(size=(h, w)),
                        transforms.ToTensor()
                    ])
                    img = Image.open("work/background/background3.jpg")
                    bgr = transform(img).to(device, dtype, non_blocking=True)

                com = fgr * pha + bgr * (1 - pha)
                com = com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
                com = cv2.cvtColor(com, cv2.COLOR_RGB2BGR)

                # Resize and crop
                # com_resized = cv2.resize(com, (192, 256))
                # h, w = com_resized.shape[:2]
                # start_h = (h - 256) // 2
                # start_w = (w - 192) // 2
                # com_cropped = com_resized[start_h:start_h+256, start_w:start_w+192]

                # Crop
                com_cropped = com[40:440, 170:470]  # Crop to 300x400 (480/2-200 : 480/2+200, 640/2-235 : 640/2+235)

                # Resize to 192x256
                com_resized = cv2.resize(com_cropped, (192, 256))  # Resize to 192x256

                # Save frame
                frame_count += 1
                cv2.imwrite(os.path.join('screenshot/test_img/', f"frame_{frame_count}.jpg"), com_resized)
                

                # if frame_count % 30 == 0:
                    # Send message
                message = f"Frame {frame_count} processed and saved."
                socket.send_string(f"frame_{frame_count}.jpg")

                key = dsp.step(com)
                time.sleep(0.01)
                # if frame_count == 0:
                #     time.sleep(5)
                if key == ord('b'):
                    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                    # Terminate the subprocess when 'b' key is pressed
                    process.terminate()
                    process.wait()  # Wait for the process to terminate
                    break
                elif key == ord('q'):
                    print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
                    process.terminate()
                    process.wait()  # Wait for the process to terminate
                    exit()

