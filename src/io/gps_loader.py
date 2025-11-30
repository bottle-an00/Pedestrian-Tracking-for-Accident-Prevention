# GNSS/IMU 로더
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple, List

import csv

@dataclass
class GpsImuSample:
    time: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    heading: float
    vel_x: float
    vel_y: float
    vel_z: float
    acc_body_x: float
    acc_body_y: float
    acc_body_z: float
    angle_rate_x: float
    angle_rate_y: float
    angle_rate_z: float

GPS_EXT = {".txt"}

class GpsImuLoader:

    def __init__(self):
        pass

    def iter_gps_imu_path(self, root_dir: str | Path) -> Iterable[Path]:
        root = Path(root_dir)
    
        if not root.exists():
            raise FileNotFoundError(f"[iter_gps_imu_paths] Directory not found: {root}")
    
        gps_files = sorted(
            [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in GPS_EXT],
            key=lambda p: int(p.stem)   # ← 숫자 기반 정렬
        )
    
        for p in gps_files:
            yield p

    def list_gps_imu_paths(self, root_dir: str | Path) -> List[Path]:
        return list(self.iter_gps_imu_path(root_dir))

    def load_data(self, path: str | Path) -> GpsImuSample:

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[GpsImuLoader] File not found: {path}")

        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = None

            for line in reader:
                if not line or all(not cell.strip() for cell in line):
                    continue
                header = [cell.strip() for cell in line ]
                break
            if header is None:
                raise ValueError(f"[Load GPS IMU data] Header not found in file: {path}")

            header = [h.lstrip("@").strip() for h in header]
            col_idx = {name: i for i, name in enumerate(header)}

            for line in reader:
                if not line or all(not cell.strip() for cell in line):
                    continue

                if len(line) < len(header):
                    line = line + ["0"] * (len(header) - len(line))

                def fv(name: str) -> float:
                    return float(line[col_idx[name]])

                try:
                    sample = GpsImuSample(
                        time=fv("time"),
                        x=fv("X"),
                        y=fv("Y"),
                        z=fv("Z"),
                        roll=fv("Roll"),
                        pitch=fv("Pitch"),
                        heading=fv("Heading"),
                        vel_x=fv("VelX"),
                        vel_y=fv("VelY"),
                        vel_z=fv("VelZ"),
                        acc_body_x=fv("AccBodyX"),
                        acc_body_y=fv("AccBodyY"),
                        acc_body_z=fv("AccBodyZ"),
                        angle_rate_x=fv("AngleRateX"),
                        angle_rate_y=fv("AngleRateY"),
                        angle_rate_z=fv("AngleRateZ"),
                    )
                except Exception as e:
                    print(f"[load_gps_imu_txt] Skip invalid line {line}: {e}")
                    continue

        return sample

    def iter_data(self,
                  root_dir: str | Path
    ) -> Iterator[Tuple[Path, List[GpsImuSample]]]:

        for gps_imu_path in self.iter_gps_imu_path(root_dir):
            try:
                data = self.load_data(gps_imu_path)
            except Exception as e:
                print(f"[iter_data] Failed to read {gps_imu_path}: {e}")
                continue

            yield gps_imu_path, data