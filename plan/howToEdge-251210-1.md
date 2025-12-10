## Q
trench 형상 흰색 안쪽 테두리를 파악하기 위해 방법론 설계 및 코드 적용
모드: 기존 ray 및 추가한 방법들을 선택 가능
이미지 내에서 측정 시작점을 선택하고 그 아래에 있는 파인 부분만 형상을 파악. 테두리를 파악할 경우 의미상 일관되어야 
합니다(흰색 테두리가 두꺼울 경우 안쪽으로 일정하게). 버킷 모양 테두리를 파악해야 합니다. 안에 있는 주름은 신경쓰지 않고요
대상 이미지: data/3 경로 내 *.jpg 

### require
1. 테두리 기준: 어두운 영역이 끝나고 흰색 테두리가 시작되는 지점 (바깥쪽 경계)
  2. 알고리즘: Ray + Canny + Adaptive Threshold 세 가지 모드 모두 제공
  3. 측정 범위: 기존처럼 각도 범위(angleStart~angleEnd)로 하향 방향만 측정

1. 테두리 기준: 어두운 영역이 끝나고 흰색 테두리가 시작되는 지점 (바깥쪽 경계)
  2. 알고리즘: Ray + Canny + Adaptive Threshold 세 가지 모드 모두 제공
  3. 측정 범위: 기존처럼 각도 범위(angleStart~angleEnd)로 하향 방향만 측정

· 엣지 맵(검출된 경계선) 시각화 오버레이 기능이 필요하신가요? → 필요함
     · 측정 결과 스무딩(주름 제거용 median filter) 기능이 필요하신가요? → 둘다 선택 가능

## A
계획 요약

  목표: SEM 이미지에서 트렌치 흰색 테두리 검출을 위한 3가지 방법론 구현

  구현할 기능:

  1. Detection Mode 선택: Ray / Canny / Adaptive 세 가지 모드
  2. 검출 기준: 어두운 영역 → 흰색 테두리 전환점 (dark-to-bright transition)
  3. 엣지 맵 시각화 오버레이: 체크박스로 켜고 끌 수 있는 기능
  4. 결과 스무딩: Median filter로 주름 제거 (선택 가능)

  주요 알고리즘:

  - Ray: 기존 방식 유지 (intensity > threshold 검출)
  - Canny: Gaussian blur → Sobel gradient → Non-max suppression → Hysteresis
  - Adaptive: Integral image → Local mean → Binary threshold → Edge detection

  예상 작업량:

  - 약 600 라인의 JavaScript/HTML 코드 추가
  - 대상 파일: /home/codespace/utils/index.html

  계획이 요구사항에 맞는지 확인해 주세요. 승인하시면 구현을 시작하겠습니다.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Trench 형상 흰색 안쪽 테두리 검출 - 구현 계획

 목표

 SEM 이미지에서 트렌치(etch profile)의 흰색 테두리를 검출하는 다중 방법론 구현
 - 기존 Ray 방식 유지
 - Canny Edge Detection 추가
 - Adaptive Threshold 추가
 - 세 가지 모드를 UI에서 선택 가능하게 구현

 대상 파일

 - /home/codespace/utils/index.html (단일 파일 웹 앱)

 검출 기준

 - 테두리 정의: 어두운 영역이 끝나고 흰색 테두리가 시작되는 지점 (dark-to-bright transition)
 - 측정 범위: angleStart~angleEnd 각도 범위로 하향 방향만 측정
 - 내부 주름 무시: 트렌치 내부의 세로 줄무늬는 검출하지 않음

 ---
 1. UI 변경사항

 1.1 모드 선택 UI (Settings Panel에 추가)

 <div class="setting-row">
     <label>Detection Mode:</label>
     <select id="detectionMode">
         <option value="ray">Ray (Threshold)</option>
         <option value="canny">Canny Edge Detection</option>
         <option value="adaptive">Adaptive Threshold</option>
     </select>
 </div>

 1.2 Canny 전용 설정 (조건부 표시)

 <div id="cannySettings" style="display: none;">
     <div class="setting-row">
         <label>Gaussian Sigma:</label>
         <input type="number" id="gaussianSigma" value="1.4" min="0.5" max="5.0" step="0.1">
     </div>
     <div class="setting-row">
         <label>Low Threshold:</label>
         <input type="number" id="cannyLowThreshold" value="20" min="1" max="100">
     </div>
     <div class="setting-row">
         <label>High Threshold:</label>
         <input type="number" id="cannyHighThreshold" value="50" min="1" max="255">
     </div>
 </div>

 1.3 Adaptive 전용 설정 (조건부 표시)

 <div id="adaptiveSettings" style="display: none;">
     <div class="setting-row">
         <label>Block Size (px):</label>
         <input type="number" id="adaptiveBlockSize" value="31" min="3" max="101" step="2">
     </div>
     <div class="setting-row">
         <label>Threshold Constant (C):</label>
         <input type="number" id="adaptiveC" value="10" min="-50" max="50" step="1">
     </div>
 </div>

 1.4 모드 전환 이벤트

 - 모드 변경 시 해당 설정 패널만 표시
 - 기존 측정 결과가 있으면 자동 재측정

 ---
 2. Canny Edge Detection 구현

 2.1 핵심 함수들

 Gaussian Blur (노이즈 제거)

 function generateGaussianKernel(sigma, size)
 function applyGaussianBlur(grayData, width, height, sigma)

 Gradient 계산 (Sobel 연산자)

 function computeGradients(blurredData, width, height)
 // Returns: { gradientMagnitude, gradientDirection }

 Non-Maximum Suppression (엣지 얇게)

 function nonMaximumSuppression(gradientMagnitude, gradientDirection, width, height)

 Hysteresis Thresholding (이중 임계값)

 function hysteresisThreshold(suppressedData, width, height, lowThreshold, highThreshold)
 // Returns: Uint8Array (255 = edge, 0 = no edge)

 완전한 Canny 파이프라인

 function performCannyEdgeDetection(grayData, width, height, sigma, lowThreshold, highThreshold)

 2.2 Ray 기반 엣지 선택 (Dark-to-Bright 검증)

 function findFirstEdgeAlongRay(edgeMap, grayData, width, height, startX, startY, angle, maxRadius)
 - 각 레이를 따라 첫 번째 엣지 픽셀 찾기
 - Dark-to-Bright transition 검증: 엣지 전후 intensity 비교하여 어두운→밝은 전환만 선택
 - 내부 주름(밝은→어두운 전환) 무시

 2.3 메인 Canny 검출 함수

 function performCannyDetection(img, angleStart, angleEnd, angleStep, sigma, lowThreshold, highThreshold)

 ---
 3. Adaptive Threshold 구현

 3.1 Integral Image (성능 최적화)

 function computeIntegralImage(grayData, width, height)
 function getLocalMean(integral, width, height, x, y, blockRadius)
 - O(1) 시간에 지역 평균 계산 가능

 3.2 Adaptive Binary 생성

 function computeAdaptiveBinary(grayData, integral, width, height, blockSize, C)
 - threshold = localMean - C
 - pixel > threshold ? 255 : 0

 3.3 Inner Edge 검출

 function findInnerEdge(binary, grayData, width, height, startX, startY, angle)
 - Binary 이미지에서 dark(0) → white(255) 전환점 찾기
 - Sub-pixel 정밀도를 위해 bilinearSample 활용

 3.4 메인 Adaptive 검출 함수

 function performAdaptiveDetection(img, angleStart, angleEnd, angleStep, blockSize, thresholdC)

 ---
 4. measureAll() 함수 수정

 async function measureAll() {
     const detectionMode = document.getElementById('detectionMode').value;

     for (const img of state.images) {
         if (!img.startPoint) continue;

         switch (detectionMode) {
             case 'ray':
                 img.results = performRaycast(img, angleStart, angleEnd, angleStep, rayThreshold);
                 break;
             case 'canny':
                 img.results = performCannyDetection(img, angleStart, angleEnd, angleStep,
                     gaussianSigma, cannyLowThreshold, cannyHighThreshold);
                 break;
             case 'adaptive':
                 img.results = performAdaptiveDetection(img, angleStart, angleEnd, angleStep,
                     adaptiveBlockSize, adaptiveC);
                 break;
         }

         redrawCanvas(i);
     }
 }

 ---
 5. 출력 포맷 (통일)

 모든 방식이 동일한 결과 포맷 사용 (기존 export 기능 호환):
 {
     angle_deg: number,    // 각도 (도)
     r_pixels: number,     // 시작점으로부터 거리 (픽셀)
     x: number,            // X 좌표
     y: number,            // Y 좌표
     value: number         // 해당 지점 intensity
 }

 Export 시 모드 정보 추가

 {
     exportDate: "...",
     detectionMode: "canny",  // 추가
     settings: {
         angleStart, angleEnd, angleStep,
         modeSpecific: { ... }  // 모드별 파라미터
     },
     images: [...]
 }

 ---
 6. 추가 기능

 6.1 엣지 맵 시각화 오버레이

 검출된 엣지를 색상 오버레이로 표시하는 토글 옵션 추가

 <div class="setting-row">
     <label><input type="checkbox" id="showEdgeOverlay"> Show Edge Map Overlay</label>
 </div>

 function drawEdgeOverlay(ctx, edgeMap, width, height, alpha = 0.5) {
     const imageData = ctx.getImageData(0, 0, width, height);
     const data = imageData.data;

     for (let i = 0; i < edgeMap.length; i++) {
         if (edgeMap[i] === 255) {
             // 녹색 오버레이로 엣지 표시
             data[i * 4 + 1] = Math.min(255, data[i * 4 + 1] + 200 * alpha); // G
         }
     }
     ctx.putImageData(imageData, 0, 0);
 }

 6.2 결과 스무딩 (Median Filter)

 주름으로 인한 outlier 제거용 선택적 스무딩

 <div class="setting-row">
     <label><input type="checkbox" id="enableSmoothing"> Enable Result Smoothing</label>
 </div>
 <div class="setting-row" id="smoothingWindowRow" style="display: none;">
     <label>Smoothing Window:</label>
     <input type="number" id="smoothingWindow" value="5" min="3" max="21" step="2">
 </div>

 function smoothContourResults(results, windowSize = 5) {
     if (results.length < windowSize) return results;

     const smoothed = [];
     const halfWindow = Math.floor(windowSize / 2);

     for (let i = 0; i < results.length; i++) {
         const windowVals = [];
         for (let j = -halfWindow; j <= halfWindow; j++) {
             const idx = Math.max(0, Math.min(results.length - 1, i + j));
             windowVals.push(results[idx].r_pixels);
         }

         windowVals.sort((a, b) => a - b);
         const medianR = windowVals[Math.floor(windowVals.length / 2)];

         const angle = results[i].angle_deg;
         const rad = angle * Math.PI / 180;
         const startX = /* from context */;
         const startY = /* from context */;

         smoothed.push({
             angle_deg: angle,
             r_pixels: medianR,
             x: startX + medianR * Math.cos(rad),
             y: startY + medianR * Math.sin(rad),
             value: results[i].value
         });
     }
     return smoothed;
 }

 ---
 7. 기본 파라미터 값

 | 모드       | 파라미터          | 기본값  | 설명                     |
 |----------|---------------|------|------------------------|
 | Ray      | threshold     | 1.00 | globalMean × threshold |
 | Canny    | gaussianSigma | 1.4  | 노이즈 제거 강도              |
 | Canny    | lowThreshold  | 20   | 약한 엣지 임계값              |
 | Canny    | highThreshold | 50   | 강한 엣지 임계값              |
 | Adaptive | blockSize     | 31   | 지역 평균 계산 블록 크기 (홀수)    |
 | Adaptive | C             | 10   | 임계값 조정 상수              |

 ---
 7. 구현 순서

 Step 1: UI 추가

 1. Detection Mode 드롭다운 추가
 2. Canny 설정 섹션 추가
 3. Adaptive 설정 섹션 추가
 4. Edge Overlay 체크박스 추가
 5. Smoothing 체크박스 및 window 설정 추가
 6. 모드 전환 이벤트 핸들러

 Step 2: Canny Edge Detection

 1. Gaussian blur 함수
 2. Sobel gradient 함수
 3. Non-maximum suppression 함수
 4. Hysteresis thresholding 함수
 5. Ray 기반 엣지 선택 함수 (transition 검증 포함)
 6. performCannyDetection() 메인 함수

 Step 3: Adaptive Threshold

 1. Integral image 함수
 2. Local mean 함수
 3. Adaptive binary 함수
 4. Inner edge 검출 함수
 5. performAdaptiveDetection() 메인 함수

 Step 4: 추가 기능

 1. drawEdgeOverlay() - 엣지 맵 시각화
 2. smoothContourResults() - Median filter 스무딩

 Step 5: 통합

 1. measureAll() 함수에 모드 분기 추가
 2. redrawCanvas()에 edge overlay 추가
 3. Export 함수에 모드 정보 추가
 4. 설정 변경 시 자동 재측정 연결

 ---
 8. 예상 코드 추가량

 | 구성요소          | 예상 라인 수    |
 |---------------|------------|
 | UI HTML       | ~70 lines  |
 | Canny 알고리즘    | ~250 lines |
 | Adaptive 알고리즘 | ~150 lines |
 | 시각화 & 스무딩     | ~60 lines  |
 | 통합 및 이벤트      | ~70 lines  |
 | 합계            | ~600 lines |

 ---
 9. 수정 위치 (index.html)

 | 위치            | 수정 내용                           |
 |---------------|---------------------------------|
 | Line ~248     | Detection Mode 드롭다운 및 모드별 설정 추가 |
 | Line ~274     | 모드 전환 이벤트 핸들러 추가                |
 | Line ~579-618 | measureAll() 함수 모드 분기 추가        |
 | Line ~677 이후  | 새 알고리즘 함수들 추가                   |
 | Line ~756-886 | exportAll()에 모드 정보 추가           |

 ## Result
 구현 완료 요약

  /home/codespace/utils/index.html 파일에 다음 기능이 추가되었습니다:

  1. UI 변경사항

  - Detection Mode 드롭다운: Ray / Canny / Adaptive 세 가지 모드 선택
  - 모드별 설정 패널: 각 모드에 맞는 파라미터 입력 필드
  - Edge Map Overlay 체크박스: 검출된 엣지를 녹색 오버레이로 시각화
  - Result Smoothing 체크박스: Median filter로 주름 제거 옵션

  2. 알고리즘 구현

  - Canny Edge Detection (~180줄):
    - Gaussian blur (generateGaussianKernel, applyGaussianBlur)
    - Sobel gradient 계산 (computeGradients)
    - Non-maximum suppression (nonMaximumSuppression)
    - Hysteresis thresholding (hysteresisThreshold)
    - Dark-to-bright transition 검증 (findFirstEdgeAlongRay)
  - Adaptive Threshold (~80줄):
    - Integral image (computeIntegralImage) - O(1) local mean 계산
    - Adaptive binary 생성 (computeAdaptiveBinary)
    - Inner edge 검출 (findInnerEdge) - dark→white 전환점
  - Smoothing (~30줄):
    - Median filter (smoothContourResults)

  3. 통합 기능

  - measureAll(): 모드별 분기 처리, 스무딩 적용
  - redrawCanvas(): 엣지 맵 오버레이 표시
  - exportAll(): JSON/XLSX에 detection mode 정보 포함

  사용법

  1. 이미지 업로드
  2. Detection Mode 선택 (Ray / Canny / Adaptive)
  3. 모드별 파라미터 조정
  4. 시작점 클릭 → Propagate → Measure All
  5. Show Edge Map Overlay 체크하여 검출 결과 시각화
  6. Export All로 결과 저장