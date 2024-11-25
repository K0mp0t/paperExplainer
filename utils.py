import numpy as np
import cv2
import base64

def img2b64(img: np.ndarray[np.uint8]) -> str:
    retval, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def b642img(b64: str) -> np.ndarray[np.uint8]:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
