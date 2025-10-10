
import sys
import os
import datetime
from collections import defaultdict
import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import xlsxwriter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import re
import yaml

###############################################################################
# 1) ViewInfo: 폴더 파싱 정보 및 화면 공통 정보 설정/관리
###############################################################################
class ViewInfo:
    def __init__(self):
        self.parsed_data = []
        self.grouped_data = []  # 그룹화된 데이터를 저장하는 리스트
        self.recipe_stats = {}  # 레시피별 통계 정보를 저장하는 딕셔너리
        self.chip_range = ((9,9), (11,11))  # 분석 범위 (시작 chip, 끝 chip)
        
        # 설정 파일 로드
        self.subgroups_config = self.load_subgroups_config()

    def find_cond_file(self, folder_path, image_name):
        """cond.txt 파일 찾기"""
        cnd_folder = os.path.join(folder_path, f"{image_name}_cnd")
        cond_path = os.path.join(cnd_folder, "cond.txt")

        # 경로가 존재하는지 확인
        if os.path.exists(cond_path):
            return cond_path
        # else:
        #     # print(f"Warning: Cond file not found at {cond_path}")
        #     return ""  # Return empty string if no valid cond.txt is found

        for item_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, item_name)
            if os.path.isdir(full_path) and image_name in item_name:
                # {image_name}_cnd 폴더는 이미 확인했으므로 건너뜁니다.
                if item_name == f"{image_name}_cnd":
                    continue
                
                cond_path = os.path.join(full_path, "cond.txt")
                if os.path.exists(cond_path):
                    return cond_path

    def is_cond_file_valid(self, content):
        """cond.txt 파일이 유효한지 검사"""
        # 내용이 5개 미만인 경우 유효하지 않다고 판단
        if len(content) < 5:
            return False
        return True

    def extract_cursor_inf(self, content):
        """cond.txt에서 !Cursor_inf를 추출"""
        cursor_inf01 = ""
        for line in content:
            if line.startswith("!Cursor_inf"):
                cursor_values = line.split(',')
                if len(cursor_values) > 15:
                    cursor_inf01 = cursor_values[14].strip().replace('"', '')
        return cursor_inf01

    def extract_datetime(self, content):
        """cond.txt에서 Date_&_Time의 시:분:초 부분을 추출"""
        time_str = ""
        pattern = r'\w+\s\w+\s\d+\s(\d+:\d+:\d+)\s\d+'
        
        for line in content:
            line = line.strip()
            if line.startswith("Date_&_Time"):
                match = re.search(pattern, line)
                if match:
                    time_str = match.group(1)  # 시:분:초 부분을 추출
                    break
                    
        return time_str

    def parse_cond_file(self, content):
        """cond.txt에서 Chip_number, Wafer_coordinate를 추출"""
        chip_num = ""
        wafer_coord = ""
        for line in content:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, val = parts
                if key == "Chip_number":
                    chip_num = val.strip()
                elif key == "Wafer_coordinate":
                    wafer_coord = val.strip()

        return chip_num, wafer_coord

    def find_cursor_info(self, folder_path, image_name):
        """새로운 폴더 구조에서 cursor_inf01과 시간 정보를 cond.txt에서 추출"""
        chip_number = "no cond.txt"
        wafer_coord = "-"
        cursor_inf01 = "-"
        time_str = "-"

        # cond.txt 파일 찾기
        cond_path = self.find_cond_file(folder_path, image_name)
        if cond_path:  # cond.txt가 존재하면
            with open(cond_path, 'r', encoding='utf-8') as cf:
                content = cf.readlines()
                if not self.is_cond_file_valid(content):
                    cursor_inf01 = "cond.txt blank"  # 내용이 부족하면 "cond.txt blankt"로 설정
                else:
                    cursor_inf01 = self.extract_cursor_inf(content)
                    chip_number, wafer_coord = self.parse_cond_file(content)  # Chip_number와 Wafer_coordinate를 가져오기
                    time_str = self.extract_datetime(content)  # Date_&_Time 시:분:초 추출

        return cursor_inf01, chip_number, wafer_coord, time_str

    def load_folder_and_parse(self, folder_path):
        """폴더 경로로부터 데이터를 파싱하고 self.parsed_data에 저장"""
        from_path = folder_path

        all_files = os.listdir(from_path)
        jpg_files = sorted([f for f in all_files 
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                            and os.path.isfile(os.path.join(from_path, f))])

        self.parsed_data = []

        for jpg_file in jpg_files:
            image_path = os.path.join(from_path, jpg_file)
            
            # 이미지 파일명에서 확장자를 제거하여 기본 이름 추출
            image_name_without_ext = os.path.splitext(jpg_file)[0]
            
            cursor_inf, chip_num, wafer_coord, time_str = self.find_cursor_info(from_path, image_name_without_ext)  # 시간정보 추가

            # 이미지 크기가 0KB인 경우 ""
            if os.path.getsize(image_path) == 0:
                thumbnail = ""
            else:
                thumbnail = image_path

            # Thumbnail '1': 동일 chip_number 다른 이미지 가능
            self.parsed_data.append({
                "Image Name": jpg_file,
                "Chip_number": chip_num,
                "Wafer_coordinate": wafer_coord,
                "Cursor_inf01": cursor_inf,
                "time_str": time_str,  # 시:분:초 정보 추가
                "Thumbnail 1": thumbnail,
            })
        
        # 시간 차이 계산 (초 단위) 및 데이터 추가
        self.calculate_time_differences()
        
        # 데이터를 로드한 후 그룹화 처리
        self.process_groups()

    def calculate_time_differences(self):
        """인접한 이미지 간 시간 차이(초) 계산"""
        if not self.parsed_data:
            return
            
        # 첫 번째 항목은 시간 차이를 계산할 이전 항목이 없으므로 "-"로 설정
        self.parsed_data[0]["diff_sec"] = "-"
        
        for i in range(1, len(self.parsed_data)):
            curr_time_str = self.parsed_data[i]["time_str"]
            prev_time_str = self.parsed_data[i-1]["time_str"]
            
            # 시간 정보가 유효한 경우에만 차이 계산
            if curr_time_str != "-" and prev_time_str != "-":
                try:
                    # 시:분:초 형식의 문자열을 datetime 객체로 변환
                    curr_time = datetime.datetime.strptime(curr_time_str, "%H:%M:%S")
                    prev_time = datetime.datetime.strptime(prev_time_str, "%H:%M:%S")
                    
                    # 시간 차이 계산 (초 단위)
                    diff = (curr_time - prev_time).total_seconds()
                    self.parsed_data[i]["diff_sec"] = str(int(diff))
                except (ValueError, TypeError):
                    self.parsed_data[i]["diff_sec"] = "error"
            else:
                self.parsed_data[i]["diff_sec"] = "-"
        
    def process_groups(self):
        """
        Chip_number가 동일한 항목을 그룹화하고 통계 처리
        - 각 그룹에 대해 cursor_inf 평균 및 표준편차 계산
        - jpg_file을 파일명의 숫자 기준으로 내림차순 정렬
        - Wafer_coordinate 값이 이전 파일과 유사하면 제외
        """
        # 초기화
        self.grouped_data = []
        
        # Chip_number로 그룹화
        groups = defaultdict(list)
        for item in self.parsed_data:
            chip_number = item["Chip_number"]
            groups[chip_number].append(item)
        
        # 각 그룹 처리
        for chip_number, items in groups.items():
            # 중복 제거: Wafer_coordinate 기준으로 5/100000 차이 이내면 동일 지점으로 간주
            unique_coord_items = {}
            for item in items:
                # Wafer_coordinate 파싱 (comma로 구분하고 단위 제거)
                wafer_coord = item["Wafer_coordinate"]
                coord_values = []
                
                if ',' in wafer_coord:
                    x_str, y_str = wafer_coord.split(',', 1)
                    
                    # 단위 제거 및 숫자 변환
                    for coord_str in [x_str, y_str]:
                        coord_str = coord_str.strip()
                        if ' um' in coord_str:
                            coord_values.append(float(coord_str.replace(' um', '')))
                        elif ' nm' in coord_str:
                            # nm를 um으로 변환 (1000분의 1)
                            coord_values.append(float(coord_str.replace(' nm', '')) / 1000)
                        else:
                            try:
                                coord_values.append(float(coord_str))
                            except ValueError:
                                coord_values.append(0)
                
                # 좌표 값이 없거나 cursor_inf가 없는 경우 처리하지 않음
                if len(coord_values) != 2 or item["Cursor_inf01"] == "-" or item["Cursor_inf01"] == "":
                    continue
                
                # 기존 좌표와 비교하여 가까운 포인트 확인
                is_duplicate = False
                duplicate_key = None
                
                for key, existing_item in unique_coord_items.items():
                    existing_coords = key.split(',')
                    existing_x = float(existing_coords[0])
                    existing_y = float(existing_coords[1])
                    
                    # 측정 위치 차이가 근소하면 중복 촬영으로 판단하여 제외
                    if (abs(existing_x - coord_values[0]) <= 1/10000*coord_values[0] and 
                        abs(existing_y - coord_values[1]) <= 2/10000*coord_values[1]):
                        is_duplicate = True
                        duplicate_key = key
                        break
                
                # 이미지 번호 추출
                img_number = self._extract_image_number(item["Image Name"])
                
                if is_duplicate:
                    # 이미지 번호가 더 큰 경우에만 기존 항목 대체
                    existing_img_number = self._extract_image_number(unique_coord_items[duplicate_key]["Image Name"])
                    if img_number > existing_img_number:
                        unique_coord_items[duplicate_key] = item
                else:
                    # 새로운 좌표로 등록
                    new_key = f"{coord_values[0]},{coord_values[1]}"
                    unique_coord_items[new_key] = item
            
            unique_items = list(unique_coord_items.values())
            
            # 이미지 파일명에서 숫자 추출 후 내림차순 정렬
            unique_items.sort(key=lambda x: self._extract_image_number(x["Image Name"]), reverse=True)
            
            # cursor_inf 값에 대한 통계 계산 (숫자로 변환 가능한 값만)
            cursor_values = []
            for item in unique_items:
                cursor_inf = item["Cursor_inf01"]
                if isinstance(cursor_inf, str):
                    # Remove "um" or "nm" and trim
                    cursor_inf = cursor_inf.strip()
                    
                    # Check if it has units and process accordingly
                    if "um" in cursor_inf.lower():
                        # Remove "um" and convert to float, multiply by 1000 to convert to nm
                        try:
                            value = float(cursor_inf.lower().replace("um", "").strip()) * 1000
                            cursor_values.append(value)
                        except (ValueError, TypeError):
                            pass
                    elif "nm" in cursor_inf.lower():
                        # Remove "nm" and convert to float
                        try:
                            value = float(cursor_inf.lower().replace("nm", "").strip())
                            cursor_values.append(value)
                        except (ValueError, TypeError):
                            pass
                    else:
                        # Try to convert directly if no units are specified
                        try:
                            value = float(cursor_inf)
                            cursor_values.append(value)
                        except (ValueError, TypeError):
                            pass
            
            # 평균 및 표준편차 계산
            cd_avg = np.mean(cursor_values) if cursor_values else float('nan')
            cd_std = np.std(cursor_values) if cursor_values else float('nan')
            
            # 그룹 정보 저장
            group_info = {
                "Chip_number": chip_number,
                "Items": unique_items,
                "CD_avg": cd_avg,
                "CD_std": cd_std
            }
            
            self.grouped_data.append(group_info)
        
        # Chip_number를 기준으로 정렬
        self.grouped_data.sort(key=lambda x: x["Chip_number"])

    def _extract_image_number(self, image_name):
        """이미지 파일명에서 4자리 숫자 추출"""
        match = re.search(r'\d{4}', image_name)
        if match:
            return int(match.group())
        return 0

    def get_cursor_value(self, cursor_inf):
        """cursor_inf 문자열에서 숫자 값을 추출하고 nm 단위로 반환"""
        if not isinstance(cursor_inf, str):
            return 0
            
        cursor_inf = cursor_inf.strip()
        
        # 단위에 따른 처리
        try:
            if "um" in cursor_inf.lower():
                # "um" 제거 후 float로 변환, 1000을 곱해 nm 단위로 변환
                value = float(cursor_inf.lower().replace("um", "").strip()) * 1000
                return value
            elif "nm" in cursor_inf.lower():
                # "nm" 제거 후 float로 변환
                value = float(cursor_inf.lower().replace("nm", "").strip())
                return value
            else:
                # 단위가 없는 경우 직접 변환 시도
                return float(cursor_inf)
        except (ValueError, TypeError):
            return 0  # 변환할 수 없는 경우 0 반환

    def create_subgroups(self, group_items):
        """cursor_inf 값에 따른 하위 그룹 분류 (설정 파일 기반)"""
        # 동적으로 subgroups 딕셔너리 생성
        subgroups = {config["name"]: [] for config in self.subgroups_config}
        
        for item in group_items:
            cursor_inf = item["Cursor_inf01"]
            value = self.get_cursor_value(cursor_inf)
            
            if value > 0:  # 유효한 값만 처리
                # 설정 파일 기반으로 범위 확인
                added_to_group = False
                for config in self.subgroups_config:
                    if config["min"] is not None and config["max"] is not None:
                        if config["min"] <= value <= config["max"]:
                            subgroups[config["name"]].append(item)
                            added_to_group = True
                            break
                
                # 어떤 범위에도 해당하지 않으면 'etc' 그룹에 추가
                if not added_to_group:
                    etc_group = next((config["name"] for config in self.subgroups_config 
                                    if config["min"] is None and config["max"] is None), "etc")
                    if etc_group in subgroups:
                        subgroups[etc_group].append(item)
            else:
                # 값을 파싱할 수 없는 경우 'etc' 그룹에 추가
                etc_group = next((config["name"] for config in self.subgroups_config 
                                if config["min"] is None and config["max"] is None), "etc")
                if etc_group in subgroups:
                    subgroups[etc_group].append(item)
        
        # 각 서브그룹 내에서 이미지 이름으로 정렬 및 통계 계산
        result = {}
        for subgroup_name, items in subgroups.items():
            if not items:
                continue  # 항목이 없는 하위 그룹은 건너뜀
            
            # 이미지 이름 기준으로 오름차순 정렬
            items.sort(key=lambda x: x["Image Name"])
            
            # 하위 그룹의 cursor_inf 값들에 대한 평균 및 표준편차 계산
            cursor_values = []
            for item in items:
                value = self.get_cursor_value(item["Cursor_inf01"])
                if value > 0:  # 유효한 값만 포함
                    cursor_values.append(value)
            
            # 평균 및 표준편차 계산
            if cursor_values:
                avg_cd = np.mean(cursor_values)
                std_cd = np.std(cursor_values)
                stats = {
                    "avg_cd": avg_cd,
                    "std_cd": std_cd
                }
            else:
                stats = {
                    "avg_cd": 0,
                    "std_cd": 0
                }
            
            result[subgroup_name] = {
                "items": items,
                "stats": stats
            }
            
        return result

    def analyze_recipes(self):
        """
        레시피별 CD값 평균 분석
        - 각 subgroup의 동일 위치 이미지를 같은 레시피로 간주
        - 특정 chip_number 범위 내의 이미지만 분석 (9,9 ~ 12,12)
        """
        # 초기화
        self.recipe_stats = {}
        
        # 분석할 범위 정의 - chip_number (9,9) ~ (12,12)
        min_x, min_y = self.chip_range[0]
        max_x, max_y = self.chip_range[1]
        
        # 레시피 이름 패턴 생성 (subgroup_idx, image_idx)
        subgroups = [config["name"] for config in self.subgroups_config]
        
        # 레시피별 데이터 수집
        for group in self.grouped_data:
            # Chip_number가 범위 내에 있는지 확인
            chip_str = group['Chip_number']
            if "," in chip_str:
                try:
                    x_str, y_str = chip_str.split(',')
                    chip_x, chip_y = float(x_str.strip()), float(y_str.strip())
                    
                    # 범위 밖이면 건너뜀
                    if not (min_x <= chip_x <= max_x and min_y <= chip_y <= max_y):
                        continue
                except:
                    continue
            else:
                continue  # 올바른 좌표 형식이 아니면 건너뜀
            
            # 현재 칩에 대한 서브그룹 데이터 생성
            subgroup_data = self.create_subgroups(group['Items'])
            
            # 각 서브그룹에 대해 처리
            for subgroup_idx, subgroup_name in enumerate(subgroups):
                if subgroup_name not in subgroup_data:
                    continue
                    
                # 현재 서브그룹의 아이템들
                items = subgroup_data[subgroup_name]['items']
                
                # 레시피별 이미지 처리 (이미지 인덱스 기준)
                for img_idx, item in enumerate(items):
                    if img_idx >= 5:  # 최대 5개 이미지만 처리
                        break
                        
                    # 레시피 키 생성 (subgroup_idx, img_idx)
                    recipe_key = f"recipe_{subgroup_idx+1}_{img_idx+1}"
                    
                    # CD 값 추출
                    cd_value = self.get_cursor_value(item["Cursor_inf01"])
                    if cd_value <= 0:  # 유효하지 않은 값은 건너뜀
                        continue
                        
                    # 레시피 통계에 값 추가
                    if recipe_key not in self.recipe_stats:
                        self.recipe_stats[recipe_key] = {
                            'values': [],
                            'subgroup_name': subgroup_name,
                            'image_index': img_idx + 1
                        }
                    
                    self.recipe_stats[recipe_key]['values'].append(cd_value)
        
        # 각 레시피에 대한 평균 및 표준편차 계산
        for recipe_key, data in self.recipe_stats.items():
            values = data['values']
            if values:
                data['avg'] = round(np.mean(values), 2)
                # 표본 표준편차 계산 (ddof=1). 일부 지점 데이터로 전체 공정/공간/시간대를 추정
                data['std'] = round(np.std(values, ddof=1), 2)
                data['count'] = len(values)
            else:
                data['avg'] = 0
                data['std'] = 0
                data['count'] = 0
        
        return self.recipe_stats

    def load_subgroups_config(self):
        """subgroups 설정 파일 로드"""
        # PyInstaller 실행파일일 때와 스크립트 실행시 모두 대응
        if getattr(sys, 'frozen', False):
            # PyInstaller로 생성된 실행파일인 경우
            application_path = os.path.dirname(sys.executable)
        else:
            # 일반 Python 스크립트로 실행되는 경우
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        config_path = os.path.join(application_path, 'subgroups_config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config['subgroups']
        except FileNotFoundError:
            # 기본 설정 사용
            print(f"Warning: Config file not found at {config_path}, using default settings")
            return [
                {"name": "80~250 nm", "min": 80, "max": 250},
                {"name": "280~400 nm", "min": 280, "max": 400},
                {"name": "440~580 nm", "min": 440, "max": 580},
                {"name": "850~1200 nm", "min": 850, "max": 1200},
                {"name": "etc", "min": None, "max": None}
            ]
        except Exception as e:
            print(f"Error loading config: {e}, using default settings")
            return [
                {"name": "80~250 nm", "min": 80, "max": 250},
                {"name": "280~400 nm", "min": 280, "max": 400},
                {"name": "440~580 nm", "min": 440, "max": 580},
                {"name": "850~1200 nm", "min": 850, "max": 1200},
                {"name": "etc", "min": None, "max": None}
            ]

###############################################################################
# (추가) MarkerInfoDialog: 마커 클릭 시 점의 상세 정보 + 이미지 표시
###############################################################################
class MarkerInfoDialog(QtWidgets.QDialog):
    def __init__(self, data_item, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Marker Info")
        self.resize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        info_text = (f"Image Name: {data_item['Image Name']}\n"
                     f"Chip_number: {data_item['Chip_number']}\n"
                     f"Wafer_coordinate: {data_item['Wafer_coordinate']}\n"
                     f"Cursor_inf01: {data_item['Cursor_inf01']}\n"
                     f"Time: {data_item.get('time_str', '-')}\n")
        info_label = QtWidgets.QLabel(info_text)
        layout.addWidget(info_label)

        image_path = data_item["Thumbnail 1"]
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label)

        if image_path and os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            self._original_pixmap = pixmap
            self._scale_factor = 1.0
            self._update_image()
            self.image_label.installEventFilter(self)
        else:
            self.image_label.setText("이미지를 찾을 수 없습니다.")

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Wheel and source is self.image_label:
            delta = event.angleDelta().y() / 120
            scale_step = 1.1
            if delta > 0:
                self._scale_factor *= scale_step
            else:
                self._scale_factor /= scale_step
            self._scale_factor = max(0.1, min(self._scale_factor, 5.0))
            self._update_image()
            return True
        return super().eventFilter(source, event)

    def _update_image(self):
        if hasattr(self, "_original_pixmap") and self._original_pixmap:
            scaled = self._original_pixmap.scaled(
                self._original_pixmap.size() * self._scale_factor,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


###############################################################################
# 2) PreviewDialog: 썸네일 더블클릭 시 큰 이미지를 보여주는 팝업
###############################################################################
class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, image_path, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 900)

        vlayout = QtWidgets.QVBoxLayout(self)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        vlayout.addWidget(self.image_label)

        pixmap = QtGui.QPixmap(image_path)
        self._original_pixmap = pixmap
        self._scale_factor = 1.0
        self._update_image()

        self.image_label.installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Wheel and source is self.image_label:
            delta = event.angleDelta().y() / 120
            scale_step = 1.1
            if delta > 0:
                self._scale_factor *= scale_step
            elif delta < 0:
                self._scale_factor /= scale_step
            self._scale_factor = max(0.1, min(self._scale_factor, 5.0))
            self._update_image()
            return True
        return super().eventFilter(source, event)

    def _update_image(self):
        if self._original_pixmap:
            scaled = self._original_pixmap.scaled(
                self._original_pixmap.size() * self._scale_factor,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


###############################################################################
# 3) CdSemWaferViewer: matplotlib 활용 "Wafer View" 모드
###############################################################################
class CdSemWaferViewer(QtWidgets.QWidget):
    """
    폴더 파싱된 결과(ViewInfo.parsed_data)를 활용하여
    원형 영역에 Chip_number 위치를 표시하고,
    호버/클릭 시 이미지를 썸네일로 보여주는 예시
    """
    chip_selected = QtCore.pyqtSignal(str)  # 칩 선택 시 발생하는 시그널 추가
    
    def __init__(self, view_info, parent=None):
        super().__init__(parent)
        self.view_info = view_info
        self.setWindowTitle("CD-SEM Wafer Viewer")
        self.resize(1000, 800)

        # 메인 레이아웃
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)
        
        # 상단 레이아웃(종료 버튼 배치)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addStretch(1)  # 남은 공간을 최대화하여 버튼이 오른쪽 정렬되도록 설정
        close_button = QtWidgets.QPushButton("종료")
        close_button.clicked.connect(self.close)  # 버튼 클릭 시 창 닫기
        top_layout.addWidget(close_button)

        # 상단 레이아웃을 메인 레이아웃에 추가
        layout.addLayout(top_layout)

        # matplotlib Figure
        self.canvas = WaferCanvas(self, self.view_info)
        layout.addWidget(self.canvas)

        # 가이드 라벨
        guide_label = QtWidgets.QLabel("마우스 호버: 썸네일 표시,  좌표 클릭: Chip Group View로 이동")
        guide_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(guide_label)

        # 레시피 통계 표시 영역 추가
        self.recipe_stats_widget = QtWidgets.QTableWidget()
        self.recipe_stats_widget.setMinimumHeight(200)
        self.recipe_stats_widget.setColumnCount(5)
        self.recipe_stats_widget.setHorizontalHeaderLabels([
            "Recipe", "Subgroup", "Avg CD (nm)", "Std Dev.P", "Sample Count"
        ])
        
        # 테이블 스타일 설정
        self.recipe_stats_widget.setColumnWidth(0, 100)
        self.recipe_stats_widget.setColumnWidth(1, 150)
        self.recipe_stats_widget.setColumnWidth(2, 100)
        self.recipe_stats_widget.setColumnWidth(3, 100)
        self.recipe_stats_widget.setColumnWidth(4, 100)
        
        # 영역 설명 라벨 추가
        range_info = self.view_info.chip_range
        range_label = QtWidgets.QLabel(
            f"레시피별 CD 평균 (Chip 범위: ({range_info[0][0]},{range_info[0][1]}) ~ ({range_info[1][0]},{range_info[1][1]}))"
        )
        range_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(range_label)
        layout.addWidget(self.recipe_stats_widget)
        
        # 레시피 통계 표시
        self.update_recipe_stats()

    def update_recipe_stats(self):
        """레시피별 통계 테이블 업데이트"""
        # 레시피 통계 계산
        recipe_stats = self.view_info.analyze_recipes()
        
        # 테이블 설정
        self.recipe_stats_widget.setRowCount(len(recipe_stats))
        
        # 데이터 채우기
        for row, (recipe_key, data) in enumerate(sorted(recipe_stats.items())):
            # 레시피 이름
            self.recipe_stats_widget.setItem(row, 0, QtWidgets.QTableWidgetItem(recipe_key))
            
            # 서브그룹 이름
            self.recipe_stats_widget.setItem(row, 1, QtWidgets.QTableWidgetItem(data['subgroup_name']))
            
            # 평균 CD
            avg_item = QtWidgets.QTableWidgetItem(f"{data['avg']:.2f}")
            self.recipe_stats_widget.setItem(row, 2, avg_item)
            
            # 표준편차
            std_item = QtWidgets.QTableWidgetItem(f"{data['std']:.2f}")
            self.recipe_stats_widget.setItem(row, 3, std_item)
            
            # 샘플 수
            count_item = QtWidgets.QTableWidgetItem(str(data['count']))
            self.recipe_stats_widget.setItem(row, 4, count_item)


class WaferCanvas(FigureCanvas):
    """matplotlib FigureCanvas 에서 원형 Wafer와 Chip 좌표를 표시하는 예시"""
    def __init__(self, parent, view_info, width=5, height=5, dpi=100):
        self.view_info = view_info
        self.fig = Figure(figsize=(width, height), dpi=dpi) #그래프 크기
        super().__init__(self.fig)
        self.setParent(parent)
        self.parent_widget = parent

        

        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='datalim')
        self._create_wafer_plot()
        
        # PyQt 이벤트 필터 등록
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_press_event', self.on_click)

        # 호버/클릭 시 보여줄 임시용 팝업(라벨)
        self.hover_label = QtWidgets.QLabel(parent)
        self.hover_label.setStyleSheet("background-color: white; border: 1px solid black;")
        self.hover_label.hide()

        self.points = []  # (x, y, index)
        self._init_points()

        # 선택된 지점 저장 변수 추가
        self.selected_point = None
        self.selected_marker = None
        
        self._set_axis_limits()  # x, y 최솟값, 최댓값 고려
        self._set_grid()  # 격자 설정

    def _create_wafer_plot(self):
        """wafer 테두리"""
        self.ax.clear()
        
        data = self.view_info.parsed_data
        
        max_x, max_y = 0.0, 0.0
        
        for item in data:
            chip_str = item["Chip_number"]
            
            if "," in chip_str:
                try:
                    x_str, y_str = chip_str.split(',')
                    x, y = float(x_str.strip()), float(y_str.strip())
                except:
                    x, y = 0.0, 0.0
            else:
                x, y = 0.0, 0.0  # 예외 처리
                
            if x> max_x:
                max_x = x
            if y> max_y:
                max_y = y
                
        self.ax.set_aspect("equal", adjustable="box")
        
        # 데이터 영역이 모두 보이도록 x, y 범위를 약간 넉넉하게 설정
        self.ax.set_xlim(0, max_x + 10)
        self.ax.set_ylim(0, max_y + 10)
        
        # chip_range에 해당하는 사각형 영역 표시
        min_chip_x, min_chip_y = self.view_info.chip_range[0]
        max_chip_x, max_chip_y = self.view_info.chip_range[1]
        
        # 사각형 너비와 높이 계산
        width = max_chip_x - min_chip_x + 1  # +1을 하여 끝 점도 포함
        height = max_chip_y - min_chip_y + 1  # +1을 하여 끝 점도 포함
        
        # 사각형 테두리는 빨간색, 내부는 투명하게 그리기
        rect = plt.Rectangle(
            (min_chip_x - 0.5, min_chip_y - 0.5),  # 좌측 하단 좌표 (0.5를 빼서 셀 경계에 맞춤)
            width, height,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        self.ax.add_patch(rect)

        self.ax.set_title("Wafer View")

    def _set_grid(self):
        """격자 설정"""
        # x, y 범위를 정수로 설정하고, 격자를 표시
        self.ax.set_xticks(range(int(self.ax.get_xlim()[0]), int(self.ax.get_xlim()[1]) + 1, 1))
        self.ax.set_yticks(range(int(self.ax.get_ylim()[0]), int(self.ax.get_ylim()[1]) + 1, 1))
        self.ax.grid(True)  # 격자 표시

        self.ax.grid(True, color='#979797', alpha=0.2)  # 격자 선 색상 변경

    def _init_points(self):
        """
        parsed_data 의 Chip_number를 x, y로 사용.
         - 형식 "123,456" 가정
         - Image Name의 첫 글자: 'E' -> 빨간색, 'S' -> 파란색, 그 외(예: 'X') -> 기본색
         - marker='x', markersize=6
        """
        data = self.view_info.parsed_data
        import matplotlib.pyplot as plt

        for i, item in enumerate(data):
            chip_str = item["Chip_number"]
            if "," in chip_str:
                try:
                    x_str, y_str = chip_str.split(',')
                    x, y = float(x_str.strip()), float(y_str.strip())
                except:
                    x, y = 0.0, 0.0
            else:
                x, y = 0.0, 0.0  # 예외 처리

            self.points.append((x, y, i))

            # Image Name의 첫 글자에 따라 다른 색상
            first_char = item["Image Name"][0] if item["Image Name"] else ""
            # print(item["Image Name"], re.search(r"UM\d", item["Image Name"]))
            if re.search(r"UM\d", item["Image Name"]):  # 'UM' 뒤에 숫자가 있으면 빨간색
                color = '#bc2c68'
            elif first_char == 'E':
                color = '#bc2c68' # 선홍색
            elif first_char == 'S':
                color = 'blue'
            else:
                color = 'black'  # 기본색

            # print(color)
            # 작은 'x' 표, markersize=6
            self.ax.plot(x, y, marker='x', color=color, markersize=6, linewidth=2, markeredgewidth=2, alpha=0.4)

        self.draw()

    def _set_axis_limits(self):
        """points 에 있는 x, y 의 min, max 기준으로 xlim, ylim 설정"""
        if not self.points:
            return

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 모든 좌표가 0인 경우를 대비
        if min_x == max_x:
            min_x -= 1
            max_x += 1
        if min_y == max_y:
            min_y -= 1
            max_y += 1

        # 약간의 여유(margin)
        margin_x = (max_x - min_x) * 0.1
        margin_y = (max_y - min_y) * 0.1

        self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
        self.ax.set_ylim(min_y - margin_y)
        self.draw()

    def on_mouse_move(self, event):
        """마우스 위치에 따라 가까운 점(Chip) 썸네일을 표시 (호버)"""
        if not event.inaxes:
            self.hover_label.hide()
            return

        # 픽셀 거리 기준으로 가까운 점 찾기
        tolerance = 10  # 픽셀 단위
        xdata, ydata = event.xdata, event.ydata
        closest_idx = None
        min_dist = float('inf')

        for (px, py, idx) in self.points:
            # 이벤트 좌표와 데이터 점 좌표 간의 스크린 좌표 거리를 확인
            pt_screen = self.ax.transData.transform((px, py))
            evt_screen = (event.x, event.y)
            dist = ((pt_screen[0] - evt_screen[0])**2 + (pt_screen[1] - evt_screen[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        if min_dist < tolerance and closest_idx is not None:
            # 호버 라벨 표시
            data_item = self.view_info.parsed_data[closest_idx]
            thumb_path = data_item["Thumbnail 1"]
            name = data_item["Image Name"]
            if os.path.exists(thumb_path):
                pixmap = QtGui.QPixmap(thumb_path).scaled(
                    100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                self.hover_label.setPixmap(pixmap)
                self.hover_label.setToolTip(name)
            else:
                self.hover_label.setText(f"{name}\n[Thumbnail 없음]")
            
            self.hover_label.adjustSize()
            
            # 라벨 위치 이동 (살짝 오른쪽/위로 배치)
            # -------------------------------------------------------
            parent_height = self.hover_label.parentWidget().height() 
            label_x = event.x + 20
            label_y = parent_height-530-event.y  # '살짝 위'
            # -------------------------------------------------------
            # label_y= event.y - 20
            # --------모니터에 따라 라벨위치 달라지면 수정 필요----------
            

            # 라벨 위치 이동
            self.hover_label.move(label_x, label_y)
            self.hover_label.show()
        else:
            self.hover_label.hide()

    def on_click(self, event):
        """좌표 클릭 시 세부정보 팝업(텍스트 + 원본 이미지)"""
        if not event.inaxes:
            return

        xdata, ydata = event.xdata, event.ydata
        # 가까운 점 찾기
        closest_idx = None
        min_dist = float('inf')
        for (px, py, idx) in self.points:
            dist = math.hypot(px - xdata, py - ydata)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        # 어느 정도 거리 이하인 경우만 유효
        if closest_idx is not None and min_dist < 0.5:
            data_item = self.view_info.parsed_data[closest_idx]
            # 클릭한 위치의 Chip_number 시그널 발생
            self.parent_widget.chip_selected.emit(data_item["Chip_number"])
            
            # 이전 선택 마커 제거
            if self.selected_marker:
                self.selected_marker.remove()
                self.selected_marker = None
            
            # 선택된 위치 저장 및 동그라미 표시
            px, py, _ = self.points[closest_idx]
            self.selected_point = (px, py)
            self.selected_marker = self.ax.plot(
                px, py, 
                marker='o', 
                color='red', 
                markersize=8, 
                alpha=0.8,
                markerfacecolor='none',
                markeredgewidth=2
            )[0]
            
            # 그래프 업데이트
            self.draw()

            # MarkerInfoDialog 팝업 부분을 주석 처리
            """
            # MarkerInfoDialog 팝업
            dialog = MarkerInfoDialog(data_item, parent=self.parent_widget)
            dialog.exec_()
            """

    
###############################################################################
# 4) ChipGroupViewer: chip_number별 그룹화된 이미지 표시
###############################################################################
class ChipGroupViewer(QtWidgets.QScrollArea):
    """
    chip_number가 동일한 항목끼리 그룹화하여 이미지, cursor_inf, 통계를 표시하는 뷰어
    """
    def __init__(self, view_info, parent=None):
        super().__init__(parent)
        self.view_info = view_info
        self.setWidgetResizable(True)
        
        # 스크롤 가능한 내부 위젯
        container = QtWidgets.QWidget()
        self.setWidget(container)
        
        # 메인 레이아웃
        self.main_layout = QtWidgets.QVBoxLayout(container)
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)
        
        # 초기 렌더링
        self.render_groups()
        
        # 그룹 헤더 위젯을 저장할 딕셔너리
        self.group_headers = {}
        
    def render_groups(self):
        """그룹화된 데이터를 UI에 렌더링"""
        # 기존 위젯 제거
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 그룹 헤더 딕셔너리 초기화
        self.group_headers = {}
        
        # 각 chip_number 그룹 렌더링
        for group in self.view_info.grouped_data:
            # 그룹 헤더 (Chip_number 및 통계)
            group_header = QtWidgets.QLabel(
                f"<b>Chip #{group['Chip_number']}</b> - Avg CD: {group['CD_avg']:.2f} ± {group['CD_std']:.2f} nm"
            )
            group_header.setStyleSheet("font-size: 14px; padding: 5px;")
            self.main_layout.addWidget(group_header)
            
            # 그룹 헤더를 딕셔너리에 저장
            self.group_headers[group['Chip_number']] = group_header
            
            # ViewInfo에서 서브그룹 데이터 가져오기
            subgroups = self.view_info.create_subgroups(group['Items'])
            
            # 각 하위 그룹 렌더링
            for subgroup_name, subgroup_data in subgroups.items():
                items = subgroup_data["items"]
                stats = subgroup_data["stats"]
                
                # 통계 텍스트 구성
                stat_text = f" - Avg CD: {stats['avg_cd']:.2f} ± {stats['std_cd']:.2f} nm ({len(items)} items)"
                
                # 하위 그룹 헤더 (통계 정보 포함)
                subgroup_header = QtWidgets.QLabel(f"<b>{subgroup_name}</b>{stat_text}")
                subgroup_header.setStyleSheet("font-size: 12px; padding: 3px; color: #2c3e50; background-color: #ecf0f1;")
                self.main_layout.addWidget(subgroup_header)
                
                # 이미지 컨테이너
                image_container = QtWidgets.QWidget()
                image_layout = QtWidgets.QHBoxLayout(image_container)
                image_layout.setSpacing(20)  # 이미지 간 가로 간격
                
                # 각 이미지 항목 추가
                for item in items:
                    img_path = item["Thumbnail 1"]
                    if img_path and os.path.exists(img_path):
                        # 이미지 컨테이너 생성
                        img_widget = QtWidgets.QWidget()
                        img_layout = QtWidgets.QVBoxLayout(img_widget)
                        img_layout.setContentsMargins(0, 0, 0, 0)
                        
                        # 커서 값 라벨 (위쪽 흰색 공간)
                        cursor_label = QtWidgets.QLabel(f"{item['Cursor_inf01']}")
                        cursor_label.setAlignment(QtCore.Qt.AlignCenter)
                        cursor_label.setFixedHeight(50)
                        cursor_label.setStyleSheet("background-color: white; color: black;")
                        img_layout.addWidget(cursor_label)
                        
                        # 이미지 라벨
                        image_label = QtWidgets.QLabel()
                        pixmap = QtGui.QPixmap(img_path).scaled(
                            200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                        )
                        image_label.setPixmap(pixmap)
                        image_label.setFixedSize(200, 200)
                        
                        image_label.setAlignment(QtCore.Qt.AlignCenter)
                        image_label.setToolTip(item["Image Name"])
                        img_layout.addWidget(image_label)
                        
                        # 이미지명 라벨 (아래쪽)
                        name_label = QtWidgets.QLabel(item["Image Name"])
                        name_label.setAlignment(QtCore.Qt.AlignCenter)
                        name_label.setWordWrap(True)
                        img_layout.addWidget(name_label)
                        
                        # 이미지 클릭 이벤트 연결
                        image_label.mousePressEvent = lambda event, path=img_path, name=item["Image Name"]: self.show_preview(path, name)
                        image_label.setCursor(QtCore.Qt.PointingHandCursor)
                        
                        # 레이아웃에 추가
                        image_layout.addWidget(img_widget)
                
                # 스크롤 영역에 이미지 컨테이너 추가
                self.main_layout.addWidget(image_container)
            
            # 구분선 추가
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.main_layout.addWidget(line)       
    
    # ChipGroupViewer 클래스의 show_preview 메서드를 수정
    def show_preview(self, image_path, title):
        """이미지 클릭 시 MarkerInfoDialog를 보여주는 다이얼로그"""
        # 클릭한 이미지에 해당하는 데이터 항목 찾기
        for group in self.view_info.grouped_data:
            for item in group['Items']:
                if item["Thumbnail 1"] == image_path:
                    # MarkerInfoDialog 팝업
                    dialog = MarkerInfoDialog(item, self)
                    dialog.exec_()
                    return
                    
        # 해당 이미지를 찾지 못한 경우 기존 PreviewDialog로 폴백
        dialog = PreviewDialog(image_path, title, self)
        dialog.exec_()
        
    def scroll_to_chip(self, chip_number):
        """특정 chip_number로 스크롤"""
        if chip_number in self.group_headers:
            header_widget = self.group_headers[chip_number]
            self.ensureWidgetVisible(header_widget, 0, 0)
            # 가로 스크롤은 제일 왼쪽으로 설정
            self.horizontalScrollBar().setValue(0)

###############################################################################
# 5) CdSemTableViewer: 메인 뷰어 
###############################################################################
class CdSemTableViewer(QtWidgets.QMainWindow):
    def __init__(self, view_info, parent=None):
        super().__init__(parent)
        self.view_info = view_info  # ViewInfo 객체 주입
        self.setWindowTitle("CD-SEM Viewer v2.4(250605)")
        self.resize(1400, 800)

        self._create_widgets()
        self.setAcceptDrops(True)  # 드래그 & 드롭 허용
        
        # 경로 표시 라벨 추가
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        self.statusBar().addPermanentWidget(self.path_label, 1)  # 상태 바에 라벨 추가

    def _create_widgets(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        export_action = QtWidgets.QAction("Export to Excel", self)
        export_action.triggered.connect(self.export_to_excel)
        file_menu.addAction(export_action)

        view_menu = menu_bar.addMenu("View")

        table_view_action = QtWidgets.QAction("Table View", self)
        table_view_action.triggered.connect(self.show_table_view)
        view_menu.addAction(table_view_action)

        wafer_view_action = QtWidgets.QAction("Wafer View", self)
        wafer_view_action.triggered.connect(self.show_wafer_view)
        view_menu.addAction(wafer_view_action)
        
        # Chip Group View 메뉴 추가
        chip_group_view_action = QtWidgets.QAction("Chip Group View", self)
        chip_group_view_action.triggered.connect(self.show_chip_group_view)
        view_menu.addAction(chip_group_view_action)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(central_widget)

        self.info_label = QtWidgets.QLabel("폴더를 드래그 & 드롭 하세요.")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.info_label)

        # 스택 위젯 생성 (여러 뷰 전환용)
        self.stack_widget = QtWidgets.QStackedWidget()
        self.main_layout.addWidget(self.stack_widget)
        
        # 테이블 뷰 생성 (컬럼 변경: CD_avg, CD_std 제거, date, diff_sec 추가)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        headers = ["Image Name", "Chip_number", "Wafer_coordinate", 
                   "Cursor_inf01", "date", "diff_sec", "Thumbnail 1"]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setColumnWidth(0, 300)   
        self.table.setColumnWidth(1, 120)   
        self.table.setColumnWidth(2, 180)   
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 120)    # date 컬럼
        self.table.setColumnWidth(5, 80)     # diff_sec 컬럼
        self.table.setColumnWidth(6, 200)   
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.stack_widget.addWidget(self.table)
        
        # Chip Group 뷰 생성 및 스택에 추가
        self.chip_group_viewer = ChipGroupViewer(self.view_info)
        self.stack_widget.addWidget(self.chip_group_viewer)
        
        # Wafer 뷰어 참조 저장 변수
        self.wafer_viewer = None

        self.statusBar().showMessage("")

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if os.path.isdir(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                folder_path = url.toLocalFile()
                if os.path.isdir(folder_path):
                    self.view_info.load_folder_and_parse(folder_path)
                    self.update_table()
                    self.update_chip_group_view()
                    self.info_label.setText(f"총 {len(self.view_info.parsed_data)}개 이미지 로딩 완료!")
                    
                    # 드롭된 폴더 경로를 UI 하단에 표시
                    self.path_label.setText(folder_path)
                    
                    self.show_chip_group_view()  # 기본 뷰로 Chip Group View 표시

    def update_table(self):
        """테이블 뷰 데이터 업데이트 (CD_avg, CD_std 제거, date, diff_sec 추가)"""
        self.table.setRowCount(0)

        for row_index, data in enumerate(self.view_info.parsed_data):
            self.table.insertRow(row_index)
            self.table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(data["Image Name"]))
            self.table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(data["Chip_number"]))
            self.table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(data["Wafer_coordinate"]))
            self.table.setItem(row_index, 3, QtWidgets.QTableWidgetItem(data["Cursor_inf01"]))
            
            # date 컬럼 (시:분:초) 추가
            self.table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(data.get("time_str", "-")))
            
            # diff_sec 컬럼 (이전 이미지와의 초 차이) 추가
            self.table.setItem(row_index, 5, QtWidgets.QTableWidgetItem(data.get("diff_sec", "-")))

            # 썸네일 표시
            thumbnail_item1 = QtWidgets.QTableWidgetItem()
            if data["Thumbnail 1"]:
                pixmap1 = QtGui.QPixmap(data["Thumbnail 1"]).scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                thumbnail_item1.setData(QtCore.Qt.DecorationRole, pixmap1)
            else:
                thumbnail_item1.setText("no img")
            self.table.setItem(row_index, 6, thumbnail_item1)

            self.table.setRowHeight(row_index, 100)

    def update_chip_group_view(self):
        """Chip Group 뷰 데이터 업데이트"""
        self.chip_group_viewer.render_groups()

    def show_table_view(self):
        """테이블 뷰 표시"""
        self.stack_widget.setCurrentWidget(self.table)

    def show_wafer_view(self):
        """웨이퍼 뷰 표시 (팝업)"""
        # 별도 창으로 Wafer View 표시
        if self.wafer_viewer is None or not self.wafer_viewer.isVisible():
            self.wafer_viewer = CdSemWaferViewer(self.view_info)
            # Wafer View에서 Chip 선택 시 Chip Group View로 이동하도록 시그널 연결
            self.wafer_viewer.chip_selected.connect(self.on_chip_selected)
            self.wafer_viewer.show()
    
    def show_chip_group_view(self):
        """Chip Group 뷰 표시"""
        self.stack_widget.setCurrentWidget(self.chip_group_viewer)
    
    def on_chip_selected(self, chip_number):
        """Wafer View에서 Chip 선택 시 호출되는 슬롯"""
        # Chip Group View로 전환
        self.show_chip_group_view()
        # 선택된 Chip으로 스크롤
        self.chip_group_viewer.scroll_to_chip(chip_number)

    def export_to_excel(self):
        """테이블 데이터를 엑셀로 내보내기"""
        """테이블 데이터를 엑셀로 내보내기"""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx)", options=options)
        if not file_path:
            return

        # NaN/INF 값을 처리하기 위한 옵션 추가
        workbook = xlsxwriter.Workbook(file_path, {'nan_inf_to_errors': True})

        # Sheet1: 테이블 뷰 데이터 (기존과 동일)
        worksheet1 = workbook.add_worksheet("Table Data")
        headers = ["Image Name", "Chip_number", "Wafer_coordinate", "Cursor_inf01", "date", "diff_sec"]
        for col, header in enumerate(headers):
            worksheet1.write(0, col, header)

        for row_index, data in enumerate(self.view_info.parsed_data):
            worksheet1.write(row_index + 1, 0, data["Image Name"])
            worksheet1.write(row_index + 1, 1, data["Chip_number"])
            worksheet1.write(row_index + 1, 2, data["Wafer_coordinate"])
            worksheet1.write(row_index + 1, 3, data["Cursor_inf01"])
            worksheet1.write(row_index + 1, 4, data.get("time_str", "-"))
            worksheet1.write(row_index + 1, 5, data.get("diff_sec", "-"))
            if data["Thumbnail 1"]:
                worksheet1.write_url(row_index + 1, 6, f'file:///{data["Thumbnail 1"]}', string="View Image")

        # Sheet2: 레시피별 CD 통계 (피벗 테이블 형식 - img_idx 기준 열)
        worksheet2 = workbook.add_worksheet("Recipe Statistics Pivot")

        # 레시피 통계 계산
        recipe_stats = self.view_info.analyze_recipes()

        # 피벗 데이터 구조 생성: {subgroup: {img_idx: {'avg': ..., 'std': ...}}}
        pivot_data = defaultdict(lambda: defaultdict(lambda: {'avg': float('nan'), 'std': float('nan')})) # 기본값 설정
        all_img_indices = set()
        subgroup_order = ["80~250 nm", "280~400 nm", "440~580 nm", "850~1200 nm", "etc"] # 행 순서 정의

        for recipe_key, data in recipe_stats.items():
            match = re.match(r"recipe_\d+_(\d+)", recipe_key) # img_idx만 추출
            if match:
                img_idx = int(match.group(1)) # 두 번째 그룹 (이미지 인덱스)
                subgroup_name = data.get('subgroup_name', 'Unknown')

                all_img_indices.add(img_idx)
                pivot_data[subgroup_name][img_idx] = { # 키로 img_idx 사용
                    'avg': data.get('avg', float('nan')),
                    'std': data.get('std', float('nan'))
                }

        # 이미지 인덱스 정렬 (숫자 기준: 1, 2, 3, ...)
        sorted_img_indices = sorted(list(all_img_indices))

        # 헤더 작성
        header_format = workbook.add_format({'bold': True, 'align': 'center'})
        worksheet2.write(0, 0, "Subgroup", header_format)
        col_idx = 1
        for img_idx in sorted_img_indices:
            worksheet2.write(0, col_idx, f"{img_idx} (Avg)", header_format) # 헤더에 img_idx만 사용
            worksheet2.write(0, col_idx + 1, f"{img_idx} (Std)", header_format) # 헤더에 img_idx만 사용
            col_idx += 2

        # 데이터 작성
        row_idx = 1
        subgroup_order = [config["name"] for config in self.view_info.subgroups_config]
        for subgroup_name in subgroup_order:
            # 데이터가 없는 서브그룹도 행은 표시 (선택적)
            worksheet2.write(row_idx, 0, subgroup_name)
            col_idx = 1
            for img_idx in sorted_img_indices:
                stats = pivot_data[subgroup_name][img_idx] # img_idx로 데이터 조회

                # Avg CD 값 처리
                avg_cd = stats['avg']
                if math.isnan(avg_cd) or math.isinf(avg_cd):
                    worksheet2.write(row_idx, col_idx, "N/A")
                else:
                    worksheet2.write(row_idx, col_idx, avg_cd)

                # Std Dev 값 처리
                std_dev = stats['std']
                if math.isnan(std_dev) or math.isinf(std_dev):
                    worksheet2.write(row_idx, col_idx + 1, "N/A")
                else:
                    worksheet2.write(row_idx, col_idx + 1, std_dev)

                col_idx += 2
            row_idx += 1

        # 열 너비 자동 조정 (선택 사항)
        worksheet2.autofit()

        workbook.close()
        QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {file_path}")
    
    def on_cell_double_clicked(self, row, col):
        item_name = self.table.item(row, 0)
        if not item_name:
            return
        data_item = self.view_info.parsed_data[row]
        if col == 6:  # Thumbnail 1
            image_path = data_item.get("Thumbnail 1")
        else:
            image_path = None

        title = item_name.text()[:9] if item_name.text() else "이미지 미리보기"
        if image_path and os.path.exists(image_path):
            dlg = PreviewDialog(image_path, title, self)
            dlg.exec_()
        else:
            QtWidgets.QMessageBox.warning(self, "에러", "이미지 파일을 찾을 수 없습니다.")

###############################################################################
# 6) main 함수
###############################################################################
def main():
    app = QtWidgets.QApplication(sys.argv)

    view_info = ViewInfo()

    viewer = CdSemTableViewer(view_info)
    viewer.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
