# 카메라 왜곡 보정

from src.calibration.load_calibration_info import CalibrationInfoLoader

def load_instrinsic_txt(path):
    """
    Deprecated: CalibrationInfoLoader.load_camera_calibration() 사용 권장
    """
    loader = CalibrationInfoLoader()
    return loader.load_camera_calibration(path)
