# Trench Bucket Scanner - 사용 가이드

SEM 이미지에서 trench bucket 경계(edge)를 감지하고 좌표를 추출하는 도구입니다.

---

## 개요

**최종 목적: Boundary 좌표 추출**

알고리즘은 크게 두 가지 카테고리로 분류됩니다:

| 카테고리 | 알고리즘 | Training 필요 |
|---------|---------|--------------|
| **Edge Detection** | Ray, Canny, Adaptive, Sobel | ❌ 불필요 |
| **Shape Learning** | Statistical, Template, Neural | ✅ 필요 |

---

## 기본 사용법

### 1단계: 알고리즘 선택 (최상단)

페이지 상단의 **Algorithm** 드롭다운에서 원하는 알고리즘을 선택합니다.

- **Edge Detection (No Training)**: 이미지 처리만으로 boundary 추출
- **Shape Learning (Training Required)**: 학습 데이터로 모델 생성 후 Detection

### Edge Detection 워크플로우
```
Load Images → Set Start Point → Detect → Export
```

### Shape Learning 워크플로우
```
Training: Load Images → Set Points → Extract Boundaries → Build Models → (Save)
Detection: Load Images → Set Points → Detect → Export
```

---

## Edge Detection Algorithms (Training 불필요)

### Ray (Threshold) - 기본 방식
시작점에서 ray를 쏴서 밝기가 threshold를 넘는 첫 지점을 edge로 판정.

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| **Threshold Factor** | 1.0 | 0.5-2.0 | global mean × factor = threshold |

**값 변경 시 경향:**
- **↑ 증가**: 더 밝은 픽셀만 edge로 인식 → edge가 바깥쪽으로 이동
- **↓ 감소**: 약간 밝은 픽셀도 edge로 인식 → edge가 안쪽으로 이동

**주의**: 시작점이 밝은 영역에 있으면 즉시 edge 감지되어 모든 점이 시작점 근처에 몰림

---

### Canny Edge Detection
Gaussian blur → Sobel gradient → Non-maximum suppression → Hysteresis thresholding

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| **Gaussian Sigma** | 1.4 | 0.5-5.0 | 블러 강도 (노이즈 제거) |
| **Low Threshold** | 20 | 1-100 | 약한 edge 기준 |
| **High Threshold** | 50 | 10-200 | 강한 edge 기준 |

**값 변경 시 경향:**

**Gaussian Sigma:**
- **↑ 증가**: 더 많은 블러 → 노이즈 감소, 세밀한 edge 손실
- **↓ 감소**: 적은 블러 → 노이즈 민감, 세밀한 edge 유지

**Low Threshold:**
- **↑ 증가**: 약한 edge 무시 → 깨끗하지만 불연속적인 edge
- **↓ 감소**: 약한 edge 포함 → 연속적이지만 노이즈 포함

**High Threshold:**
- **↑ 증가**: 강한 edge만 감지 → edge 개수 감소, 놓치는 edge 발생
- **↓ 감소**: 더 많은 edge 감지 → edge 개수 증가, 노이즈 edge 포함

**권장 설정:**
- 노이즈가 많은 이미지: Sigma ↑ (2.0-3.0)
- edge가 안 잡히면: High Threshold ↓ (30-40)
- 일반적인 SEM: Sigma 1.4, Low 20, High 50

---

### Adaptive Threshold
지역 평균을 기준으로 binary threshold 적용 후 dark→bright 전환점 감지

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| **Block Size** | 31 | 3-101 (홀수) | 지역 평균 계산 영역 크기 |
| **Constant C** | 10 | -50 ~ 50 | threshold = local_mean - C |

**값 변경 시 경향:**

**Block Size:**
- **↑ 증가**: 더 넓은 영역의 평균 → 전체적인 밝기 변화에 둔감, 안정적
- **↓ 감소**: 좁은 영역의 평균 → 지역적 밝기 변화에 민감, 노이즈 영향

**Constant C:**
- **↑ 증가**: threshold 낮아짐 → 더 어두운 픽셀도 "밝음"으로 판정 → edge가 안쪽으로
- **↓ 감소**: threshold 높아짐 → 더 밝은 픽셀만 "밝음"으로 판정 → edge가 바깥쪽으로
- **음수 값**: 매우 밝은 픽셀만 감지

**권장 설정:**
- 밝기 변화가 큰 이미지: Block Size ↑ (41-51)
- edge가 흐리면: C ↓ (5-10)
- 일반적인 SEM: Block 31, C 10

---

### Sobel Gradient
Ray를 따라 gradient magnitude를 계산하고 peak 위치를 edge로 판정

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| **Gradient Threshold** | 30 | 5-100 | 최소 gradient 강도 |

**값 변경 시 경향:**
- **↑ 증가**: 강한 gradient만 edge로 인식 → 뚜렷한 edge만 감지
- **↓ 감소**: 약한 gradient도 edge로 인식 → 흐린 edge도 감지, 노이즈 영향

**특징:**
- Ray 방식과 달리 **절대 밝기가 아닌 변화량**으로 감지
- 시작점 밝기에 덜 민감
- Ray를 따라 **가장 큰 gradient peak**를 찾음

**권장 설정:**
- edge가 뚜렷한 이미지: 30-50
- edge가 흐린 이미지: 15-25

---

## Shape Learning Algorithms (Training 필요)

### Statistical Model
학습 데이터에서 각 각도별 반경(radius)의 평균과 표준편차를 계산합니다.
Detection 시 평균 ± (sigma × std) 범위를 벗어나는 점을 필터링합니다.

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| **Sigma Threshold** | 2.0 | 허용 표준편차 범위 (값이 클수록 더 많은 점 허용) |

### Template Matching
학습 데이터의 boundary 형상을 템플릿으로 저장합니다.
Detection 시 NCC(Normalized Cross-Correlation)로 유사도를 계산합니다.

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| **Match Threshold** | 0.7 | 최소 매칭 점수 (0-1) |

### Neural Network
TensorFlow.js를 사용한 딥러닝 모델입니다. (실험적)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| **Epochs** | 10 | 학습 반복 횟수 |
| **Batch Size** | 4 | 배치 크기 |

---

## Direction Settings

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| **Direction Angle** | 180° | 메인 스캔 방향 (0°=오른쪽, 90°=아래, 180°=왼쪽) |
| **Scan Range** | 90° | Direction ± Range 범위에서 ray 발사 |
| **Angle Step** | 1° | Ray 간격 (작을수록 촘촘함) |

---

## Noise Filtering

### FFT Filter
주파수 도메인에서 고주파 노이즈 제거

| 파라미터 | 설명 |
|---------|------|
| **FFT Cutoff** | 0.01-0.5 (낮을수록 더 smooth) |

### Consistency Filter
급격한 변화 (이상치) 보간

| 파라미터 | 설명 |
|---------|------|
| **Max Derivative** | 허용 최대 변화율 (1-20) |

### RANSAC Filter
다항식 피팅으로 outlier 제거

| 파라미터 | 설명 |
|---------|------|
| **Iterations** | 반복 횟수 (10-1000) |
| **Threshold** | inlier 거리 기준 (0.5-10 px) |
| **Poly Degree** | 다항식 차수 (2-6) |

---

## Troubleshooting

### 모든 점이 시작점 근처에 몰려있음
**원인**: 시작점이 밝은 영역에 있거나 threshold가 너무 낮음

**해결:**
1. **시작점 이동**: trench 내부 어두운 영역으로 클릭 위치 변경
2. **알고리즘 변경**: Ray → Canny 또는 Sobel로 변경
3. **파라미터 조정**:
   - Ray: Threshold Factor ↑
   - Canny: High Threshold ↓
   - Adaptive: C ↓
   - Sobel: Gradient Threshold ↓

### Edge가 불연속적/끊김
**해결:**
- Canny: Low Threshold ↓, High Threshold ↓
- Noise Filter: Consistency 또는 RANSAC 적용

### 노이즈가 많이 검출됨
**해결:**
- Canny: Sigma ↑, High Threshold ↑
- Adaptive: Block Size ↑
- Sobel: Gradient Threshold ↑
- Noise Filter: FFT (Cutoff 낮춤) 또는 RANSAC 적용

### Edge가 실제보다 안쪽/바깥쪽
**안쪽으로 조정:**
- Ray: Threshold Factor ↓
- Adaptive: C ↑

**바깥쪽으로 조정:**
- Ray: Threshold Factor ↑
- Adaptive: C ↓

---

## 알고리즘 비교

### Edge Detection

| 방식 | 장점 | 단점 | 권장 상황 |
|------|------|------|----------|
| **Ray** | 빠름, 단순 | 시작점 밝기에 민감 | 시작점이 확실히 어두운 경우 |
| **Canny** | 정확, 노이즈 강건 | 상대적으로 느림 | 일반적인 SEM 이미지 |
| **Adaptive** | 지역 밝기 보정 | 파라미터 튜닝 필요 | 밝기 불균일한 이미지 |
| **Sobel** | 밝기 무관, 변화량 기반 | peak 선택 문제 | Ray가 안 되는 경우 |

### Shape Learning

| 방식 | 장점 | 단점 | 권장 상황 |
|------|------|------|----------|
| **Statistical** | 빠름, 이상치 필터링 | 정규분포 가정 | 일정한 형상의 trench |
| **Template** | 형상 매칭 | 템플릿 의존적 | 기준 형상이 있는 경우 |
| **Neural** | 복잡한 패턴 학습 | 많은 학습 데이터 필요 | 대량 데이터 분석 |

---

## 파일 구조

```
utils/
├── trench-bucket-scanner.html      # 메인 도구
├── index.html                       # Batch Measurement Tool
├── data/                            # 샘플 SEM 이미지
└── README-trench-bucket-scanner.md  # 이 문서
```
