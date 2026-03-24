"""测试运行入口 - mock 掉系统级依赖后运行 test_auto_annotator。"""

import sys
import types
import os

# Mock 掉不需要真正加载的外部库
np_mock = types.ModuleType('numpy')
np_mock.uint8 = 0

class FakeNdarray:
    def __init__(self, shape=(100, 100, 3), dtype=None):
        self.shape = shape

np_mock.ndarray = FakeNdarray
np_mock.zeros = lambda shape, dtype=None: FakeNdarray(shape, dtype)
sys.modules['numpy'] = np_mock

cv2_mock = types.ModuleType('cv2')

class FakeVideoCapture:
    def __init__(self, path):
        pass
    def isOpened(self):
        return True
    def get(self, prop):
        return 30.0
    def read(self):
        return False, None
    def release(self):
        pass

cv2_mock.VideoCapture = FakeVideoCapture
cv2_mock.CAP_PROP_FRAME_COUNT = 7
cv2_mock.CAP_PROP_FPS = 5
cv2_mock.IMWRITE_JPEG_QUALITY = 1
cv2_mock.imencode = lambda fmt, frame, params=None: (True, type('Buf', (), {'tobytes': lambda self: b'\xff\xd8fake'})())
sys.modules['cv2'] = cv2_mock

ul_mock = types.ModuleType('ultralytics')
ul_mock.YOLO = lambda *a: None
sys.modules['ultralytics'] = ul_mock

pyside = types.ModuleType('PySide6')
pyside_core = types.ModuleType('PySide6.QtCore')
pyside_core.QThread = type('QThread', (), {'__init__': lambda s, *a, **k: None})
pyside_core.Signal = lambda *a, **k: None
sys.modules['PySide6'] = pyside
sys.modules['PySide6.QtCore'] = pyside_core

pynput = types.ModuleType('pynput')
sys.modules['pynput'] = pynput
sys.modules['pynput.keyboard'] = types.ModuleType('pynput.keyboard')
sys.modules['pynput.mouse'] = types.ModuleType('pynput.mouse')

# 项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from tests.test_auto_annotator import (
    TestParseAction,
    TestQueryLLM,
    TestToolCalling,
    TestSceneSummary,
    TestEndToEnd,
    TestClaudeImageConversion,
)

if __name__ == '__main__':
    unittest.main(verbosity=2)
