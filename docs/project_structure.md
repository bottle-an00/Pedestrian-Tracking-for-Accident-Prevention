# Project Sturcture

<br>

**Pedestrian-Tracking-for-Accident-Prevention** ** 
전체 디렉터리 구조 및 각 폴더/파일 설명

<br>

```text
Pedestrian-Tracking-for-Accident-Prevention/
├── configs/                                # 시스템 설정 YAML 모음 (모듈별 파라미터)
│   ├── bev/
│   │   └── homography.yaml                 # Pixel→BEV 변환 행렬 및 BEV 좌표 설정
│   ├── detector/
│   │   └── yolo_v8_ped.yaml                # YOLO 검출기 설정 (모델 경로, threshold 등)
│   ├── prediction/
│   │   └── ekf.yaml                        # EKF 예측 설정 (noise, dt 등)
│   ├── system.yaml                          # 전체 시스템 설정 (경로, 활성화 모듈 등)
│   └── tracker/
│       └── deepsort.yaml                   # DeepSORT 설정 (encoder, metric 등)
│
├── data/                                   # 데이터셋 저장 공간 (Git 관리 제외 권장)
│   ├── processed/                          # 전처리/BEV 변환된 데이터
│   └── raw/                                # 원본 데이터 (영상, 센서 로그 등)
│
├── docker/                                 # Docker 기반 개발/실행 환경
│   ├── docker-compose.yml                  # 서비스 실행/볼륨/환경 설정
│   └── Dockerfile                          # 컨테이너 이미지 정의
│
├── docs/                                   # 문서 디렉터리 (설계, 실험, API 등)
│   ├── api_design.md                       # 모듈 간 API·클래스 설계
│   ├── architecture.md                     # 전체 시스템 아키텍처 및 파이프라인 구조
│   ├── dataset.md                          # 데이터셋 설명, 캘리브레이션 정보
│   ├── experiments.md                      # 실험/튜닝 결과 로그 문서화
│   └── project_structure.md                # (현재 문서) 프로젝트 구조 설명
│
├── logs/                                   # 로그 및 실행 결과
│   ├── metrics/                            # 예측/추적 성능 메트릭
│   └── runs/                               # 실행 결과 영상/이미지 저장
│
├── models/                                 # YOLO/DeepSORT 등 모델 가중치 폴더
│   ├── tracker/                            # DeepSORT용 ReID 모델
│   └── yolo/                               # YOLO 모델 파일(.pt)
│
├── README.md                               # 프로젝트 개요 및 사용 방법
├── requirements.txt                        # Python 의존성 목록
│
├── scripts/                                # 실행 스크립트 (Pipeline 데모/테스트 용)
│   ├── export_video.py                     # 결과 영상 내보내기
│   ├── run_detector.py                     # YOLO 검출만 실행해보는 스크립트
│   ├── run_pipeline.py                     # 전체 파이프라인 실행 엔트리포인트
│   ├── run_tracker.py                      # YOLO + DeepSORT 추적 테스트
│   └── tools/
│       ├── calib_viewer.py                 # 캘리브레이션 / 좌표계 시각화 도구
│       └── visualize_bev.py                # BEV 변환 결과 확인용 유틸
│
└── src/                                    # 핵심 소스 코드 (라이브러리 모듈)
    ├── app/
    │   ├── __init__.py
    │   └── main.py                         # 실제 전체 파이프라인을 orchestration하는 엔트리포인트
    │
    ├── bev/
    │   ├── bev_transformer.py              # Pixel→BEV 좌표 변환 기능 모듈
    │   └── __init__.py
    │
    ├── calibration/
    │   ├── camera_params.py                # intrinsic/extrinsic 로드·관리
    │   ├── homography.py                   # Homography 계산, BEV 변환 행렬 생성
    │   └── __init__.py
    │
    ├── core/
    │   ├── config.py                       # YAML 로더, 전역 설정 관리자
    │   ├── __init__.py
    │   ├── timing.py                       # FPS 측정, 시간 유틸
    │   └── types.py                        # Detections, Tracks 등 dataclass 정의
    │
    ├── detection/
    │   ├── __init__.py
    │   ├── postprocess.py                  # NMS, foot point 계산 등 후처리
    │   └── yolo_detector.py                # YOLO 추론 래퍼
    │
    ├── io/
    │   ├── __init__.py
    │   ├── sensor_loader.py                # GNSS/IMU 로더
    │   ├── sync.py                         # 센서·프레임 타임싱크 기능
    │   └── video_reader.py                 # 이미지/영상 로더
    │
    ├── preprocessing/
    │   ├── image_ops.py                    # normalize/resize 등 공통 전처리
    │   ├── __init__.py
    │   └── undistort.py                    # 카메라 왜곡 보정
    │
    ├── tracking/
    │   ├── deepsort_tracker.py             # DeepSORT 기반 Multi-Object Tracking
    │   ├── __init__.py
    │   └── track_buffer.py                 # 객체별 궤적 history buffer
    │
    ├── trajectory/
    │   ├── ego_motion.py                   # GNSS 기반 ego-motion 보정
    │   ├── ekf_predictor.py                # EKF 기반 다중-horizon 궤적 예측
    │   ├── __init__.py
    │   └── trajectory_buffer.py            # BEV 기반 궤적 저장 버퍼
    │
    └── visualization/
        ├── __init__.py
        ├── overlay_2d.py                   # 2D bbox/ID 시각화
        └── overlay_bev.py                  # BEV 상 trajectory/예측 결과 시각화

```

