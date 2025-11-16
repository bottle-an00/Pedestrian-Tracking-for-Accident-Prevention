## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    # 모든 코드가 스크립트 파일 하나에 있으므로, 별도의 패키지 설정이 필요 없습니다.
    scripts=['scripts/yolo_detector.py'])

setup(**d)