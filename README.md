---
title: WhatBreath
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# WhatBreath — 폐음 분석 서비스

청진기 없이 스마트폰 마이크만으로 호흡음을 녹음해 정상/비정상을 1차 스크리닝하는 웹앱입니다.
EfficientNet-B0가 mel-spectrogram을 분석하고, Groq가 결과를 사용자가 읽기 쉬운 조언으로 출력합니다.

**폐음 데이터셋 URL: https://www.kaggle.com/datasets/nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database**
**ESC-50 데이터셋 URL: https://github.com/karolpiczak/ESC-50**

> ⚠️ 이 서비스는 의학적 진단을 대체하지 않는 참고용 스크리닝 도구예요.

---

## 1. 프로젝트 구조

```
WhatBreath/
├── .gitignore
├── README.md                 # 설명해주는 파일
├── app.py                    # Flask 추론 서버 (모델 로드 + API + 프론트 서빙)
├── requirements.txt          # Python 패키지 목록
├── .env                       # API 키 등 환경변수 (직접 생성, Git에 올리지 않음)
├── best_model.pth             # 학습된 EfficientNet-B0 가중치 (직접 다운로드)
│                                 # Colab에서는 best_model_v4.pth였지만, 가독성을 위해
│                                 # 다운로드 후 파일명을 best_model.pth로 변경했어요.
│                                 # (원래 best_checkpoints 폴더 안의 여러 버전 중 v4만 다운로드)
├── index.html                 # 웹사이트 메인 페이지
├── style.css                  # 스타일시트
├── script.js                  # 프론트엔드 로직 (녹음, 분석 요청, 결과 표시)
├── assets/
│   └── mic-position.jpg       # 청진 위치 안내 이미지
├── 폐음 분석 코드/                 # Colab에서 실행한 데이터 처리·학습 노트북
│   ├── 1.domain_transfer.ipynb
│   ├── 2.lung_preprocess.ipynb
│   ├── 3.split_dataset.ipynb
│   └── 4.efficientnet_train.ipynb
│   └── 99.lung_sound_time.ipynb
├── data/                      # 원본 및 전처리 데이터
│   ├── A.zip                  # 스마트폰 도메인으로 바꾼 청진기 데이터
│   ├── ESC_audio.zip          # ESC-50 소음 원본 데이터
│   ├── esc50.csv              # ESC-50 공식 라벨 파일
│   ├── esc_noise_log_final.csv# 어떤 청진기 파일에 어떤 ESC-50 노이즈가 적용됐는지 기록
│   ├── ICBHI_dataset.zip       # ICBHI 원본 데이터
│   ├── smartphone_converted.zip# 청진기 소리를 스마트폰 도메인으로 변환한 데이터
│   ├── labels.json
│   └── split_labels.json
├── result_image/               # 결과 시각화 이미지
│   ├── confusion_matrix.png
│   ├── eq_comparison_final.png
│   ├── eq_curve_final.png
│   ├── loss_accuracy.png
│   ├── roc_curve.png
│   └── split_distribution_final.png
└── docs/                       # 보고서 / 발표자료
    ├── 보고서.docx
    └── 발표자료.pdf
```

---

## 2. 전체 파이프라인 개요

```
[1] 청진기 원본 데이터 (ICBHI, 920개 wav)
        │  도메인 변환 (EQ 필터 + 가우시안 노이즈 + ESC-50 환경음)
        ▼
[2] 스마트폰 도메인 wav (smartphone_converted/)
        │  5초 단위 분할 + mel-spectrogram 변환 + 정상/비정상 라벨링
        ▼
[3] 학습용 데이터셋 (npy, 약 3,901개) → Train/Val/Test 분할 (70/15/15)
        │  EfficientNet-B0 학습 (50 epoch, drop_rate=0.4)
        ▼
[4] best_model.pth (Test Accuracy 73.3%)
        │  Flask 서버에 로드
        ▼
[5] 사용자가 웹사이트에서 실시간 녹음 (15초 → 5초씩 3구간)
        │  서버가 각 구간을 동일한 mel-spectrogram 변환으로 추론
        │  3구간 중 2구간 이상 비정상이면 "비정상", 1구간만이면 "주의" 표시
        ▼
[6] Groq(Llama)가 결과를 자연어 조언으로 생성 → 화면 표시
```

**중요:** 1번 단계(EQ/노이즈)는 학습 데이터를 만들 때만 적용해요. 5번 단계(실제 사용자 녹음)는
이미 진짜 스마트폰 도메인이므로 EQ/노이즈를 추가로 적용하지 않고, mel-spectrogram 변환만 동일하게 거쳐요.

---

## 3. 데이터 준비 및 모델 학습 (Google Colab)

### 3.1 필요한 것
- Google Colab 계정 (GPU 런타임 사용)
- Google Drive에 ICBHI 데이터셋(`stethoscope/` 폴더, wav+txt 쌍)
- ESC-50 환경음 데이터셋 (`ESC-50-master/`)
- 본인 스마트폰으로 녹음한 환경음 (`smartphone_wav/`, 1시간 이상)

### 3.2 Drive 폴더 구조 (학습 전 미리 준비)

```
MyDrive/lung_sound/
├── stethoscope/           # ICBHI 원본 wav + txt
├── ESC-50-master/         # ESC-50 환경음 데이터셋
├── smartphone_wav/        # 직접 녹음한 스마트폰 환경음
└── (이하 노트북 실행 시 자동 생성)
    ├── smartphone_converted/
    ├── processed/
    ├── checkpoints/
    └── best_checkpoints/
```

### 3.3 실행 순서

| 순서 | 노트북 | 하는 일 | 비고 |
|---|---|---|---|
| 1 | `domain_transfer.ipynb` | 청진기 wav에 EQ 필터(PchipInterpolator) + 가우시안 노이즈(SNR 15dB 고정) + ESC-50 환경음(SNR 15~20dB, 50% 확률)을 적용해 스마트폰 도메인으로 변환 | `smartphone_converted/`에 저장 |
| 2 | `lung_preprocess.ipynb` | 변환된 wav를 5초 단위로 자르고, txt 라벨 중 겹침 시간이 더 긴 라벨을 채택해 정상(0)/비정상(1)으로 매핑한 뒤 mel-spectrogram(128 mel, hop 512) 변환 | `processed/A/`, `labels.json` |
| 3 | `split_dataset.ipynb` | 클래스 비율을 유지하며 Train 70% / Val 15% / Test 15%로 분할 | `split_labels.json` |
| 4 | `efficientnet_train.ipynb` | EfficientNet-B0(`drop_rate=0.4`, `lr=2e-4`)를 50 epoch 학습, WeightedRandomSampler + SpecAugment + Label Smoothing(0.05) 적용, Val Accuracy 최고 시점 가중치 저장 | `best_checkpoints/best_model_v4.pth` |

각 노트북 상단의 경로 변수(`INPUT_DIR`, `OUTPUT_DIR` 등)는 본인 Drive 구조에 맞게 확인 후 실행하세요.

### 3.4 학습 결과

- 최종 체크포인트: epoch 48, Val Accuracy 73.63%
- Test Accuracy: **73.3%** (정상 precision/recall 0.73, 비정상 precision/recall 0.73로 균형)
- ICBHI Score(자체 70:15:15 split 기준): 0.7325 / ROC AUC: 0.8028
- 단순 정확도가 가장 높은 모델 대신, 정상/비정상 클래스 간 recall이 균형을 이루는 모델을 최종으로 채택함
  (의료 스크리닝 특성상 한쪽 클래스 탐지 누락이 더 큰 위험으로 이어질 수 있다는 판단)

### 3.5 모델 가중치 다운로드

학습 완료 후 Colab에서 아래 셀을 실행해 가중치를 로컬로 받으세요.

```python
from google.colab import files
files.download('/content/drive/MyDrive/lung_sound/best_checkpoints/best_model_v4.pth')
```

다운로드한 파일을 `best_model.pth`로 이름을 바꾼 뒤 `app.py`와 같은 폴더에 둡니다.

---

## 4. 로컬 환경 설정 (웹사이트 + 추론 서버)

### 4.1 Python 버전

PyTorch 호환성을 위해 **Python 3.10**을 권장해요. (3.13/3.14는 호환 문제가 발생할 수 있어요)

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 4.2 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt` 내용:
```
flask
flask-cors
python-dotenv
torch
torchvision
timm
librosa
numpy
requests
```

> webm 형식으로 녹음된 마이크 입력을 서버가 읽으려면 **ffmpeg**이 시스템에 설치되어 있어야 해요.
> (`winget install ffmpeg` 또는 공식 빌드 설치 후 터미널 재시작)

### 4.3 환경변수 설정 (.env 파일 생성)

프로젝트 루트에 `.env` 파일을 만들고 아래 내용을 채워주세요.

```dotenv
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
BREATHLINE_CKPT_PATH=./best_model.pth
```

- `GROQ_API_KEY`: console.groq.com 에서 발급받은 키
- `GROQ_MODEL`: 사용할 모델명. 정확한 모델명은 Groq 콘솔(console.groq.com/docs/models)에서 확인하는 것이 가장 정확해요.
- `BREATHLINE_CKPT_PATH`: 모델 가중치 파일 경로 (기본값을 쓸 거면 생략 가능)

> 이 키가 없어도 정상/비정상 분류 기능 자체는 동작하고, AI 조언 문구만 비어있게 돼요.

### 4.4 서버 실행

```bash
python app.py
```

정상적으로 뜨면 터미널에 아래와 같이 표시돼요.

```
모델 로드 완료. 체크포인트 epoch=48, val_acc=73.63...
* Running on http://127.0.0.1:5000
```

### 4.5 웹사이트 접속

file:///C:/Users/JY%20Han/Downloads/WhatBreath%20%E2%80%94%20%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0%20%ED%8F%90%EC%9D%8C%20%EB%B6%84%EC%84%9D%20(2).html

이 사이트로 들어가면 됩니다.

---

## 5. 사용 방법

1. **녹음·분석** 탭으로 이동
2. **녹음 시작** 클릭 → 마이크 권한 허용 → 흉부(등 위쪽, 어깨뼈 아래)에 스마트폰을 대고 15초간 평소 호흡
3. (또는 **파일 업로드**로 15초 이상 길이의 오디오 파일 업로드)
4. **분석 시작하기** 클릭 → 5초씩 3구간으로 나뉘어 각각 분석
5. 결과 확인:
   - **정상**: 3구간 중 0~1구간만 비정상
   - **주의 필요**: 3구간 중 1구간에서만 이상 신호 감지 (종합은 정상이나 참고 안내)
   - **비정상**: 3구간 중 2구간 이상에서 이상 신호 감지
   - **노이즈 감지**: 전체 구간의 평균 신뢰도가 낮으면 재녹음 안내
6. 구간별 카드를 클릭하면 해당 구간의 mel-spectrogram을 확인할 수 있어요

---

## 6. 핵심 설계 결정과 근거

### 6.1 도메인 변환: CycleGAN → 신호처리 기반 EQ 필터

처음에는 CycleGAN으로 청진기→스마트폰 도메인 변환을 시도했으나, mode collapse(변환 결과가
다양성을 잃고 특정 패턴으로 수렴하는 현상)가 발생해 분류 정확도가 오히려 낮아지는 문제가 있었어요.
이를 신호처리 기반 EQ 필터(PchipInterpolator로 주파수별 gain 보정) + 가우시안 노이즈 + ESC-50
환경음 합성으로 대체한 결과, 학습이 안정적으로 진행되고 정확도도 개선됐어요.

### 6.2 종합 판정 규칙: OR → 중앙값 → 다수결(majority vote)

| 단계 | 방식 | 문제점 |
|---|---|---|
| 1차 | 30초 / 6구간, 한 구간이라도 비정상이면 전체 비정상(OR) | 정상 recall 0.73 기준, 진짜 정상인 사용자가 오판될 확률이 1 - 0.73⁶ ≈ 84.8%까지 치솟음 |
| 2차 | 25초 / 5구간, 구간별 확률의 중앙값 기준 판정 | 5구간 중 1구간에서만 이상이 감지돼도 중앙값이 이를 거의 무시함 (실측: 정상 확률 99.6%로 산출되어 짧은 이상 신호가 묻힘) |
| 최종 | 15초 / 3구간, 2구간 이상 비정상이면 전체 비정상(다수결) + 1구간만 비정상이면 "주의" 별도 표시 | 거짓양성 확률을 약 18%로 낮추면서도, 짧게 나타나는 이상 신호를 완전히 묵살하지 않음 |

폐음(수포음, 천명음)은 평균 2.7초 길이의 단일 호흡 사이클 내에서 짧게 나타나는 경우가 많아,
구간 수와 판정 규칙을 이 특성에 맞게 조정했어요.

### 6.3 추론 시 EQ/노이즈 미적용

학습 데이터는 청진기 원본을 스마트폰 도메인처럼 "위장"시키기 위해 EQ/노이즈를 적용했지만,
실제 서비스 사용자는 이미 진짜 스마트폰으로 녹음하므로 추가 변환이 필요 없어요. 오히려 이중으로
적용하면 학습 때 본 적 없는 입력 분포가 되어 성능이 떨어질 수 있어 추론 시에는 mel-spectrogram
변환만 동일하게 적용해요.

---

## 7. 한계 및 향후 개선 방향

- 학습 데이터가 3,901개로 비교적 적고, 노이즈 도메인이 단일 화자·단일 스마트폰 환경에 한정됨
- EQ 필터의 보정 수치(-10~+5dB)는 특정 논문의 수치를 직접 인용한 것이 아니라, 마이크 하드웨어의
  일반적인 주파수 응답 특성(Widder & Morcelli, 2014, "Basic principles of MEMS microphones", EDN)과
  폐음의 인정된 주파수 대역(TTAC, n.d., "Electronic stethoscopes – Technology overview")을 참고해
  경험적으로 설계한 값임
- 비정상 클래스 recall이 정상 클래스보다 낮게 나오는 경향이 있어, 이상 신호를 놓치는 false negative를
  줄이기 위한 추가 튜닝(결정 임계값 조정, 데이터 증강 등)이 필요함
- 4클래스(정상/수포음/천명음/복합음) 분류를 시도했으나 데이터 부족으로 정확도 36%에 그쳐 채택하지
  못했으며, 추가 데이터 확보를 통한 세부 분류 고도화가 필요함
- 현재는 신체 부위(등/가슴 앞뒤 등)를 구분하지 않고 단일 위치 녹음만 지원함

---

## 8. 의료 면책 조항

이 서비스는 의학적 진단을 제공하지 않으며, 참고용 스크리닝 결과만을 제공합니다.
이상 신호가 감지되거나 호흡 관련 증상이 있는 경우 반드시 의료 전문가와 상담하세요.

---

## 9. 참고 문헌 및 데이터 출처

- ICBHI 2017 Respiratory Sound Database.
- Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.
- Baptista, B., Pais-Cunha, I., Amaral, R., Vieira-Marques, P., Valente, J., Almeida, R., Costa-Santos, C., Azevedo, I., Fonseca, J. A., Ferreira-Magalhães, M., & Jácome, C. (2026). Lung auscultation using smartphone built-in microphone versus digital stethoscope: a comparative early feasibility study. Minerva Pediatrics, 78(2), 155-163.
- Shim, V. J., Shim, H., & Roh, S. (2024). EfficientNet-B0 outperforms other CNNs in image-based five-class embryo grading: a comparative analysis. Journal of Animal Reproduction and Biotechnology, 39(4), 267-277.
- Petmezas, G., Cheimariotis, G.-A., Stefanopoulos, L., Rocha, B., Paiva, R. P., Katsaggelos, A. K., & Maglaveras, N. (2022). Automated lung sound classification using a hybrid CNN-LSTM network and focal loss function. Sensors, 22(3), 1232.
- Widder, J., & Morcelli, A. (2014, May 14). Basic principles of MEMS microphones. EDN.
- Telehealth Technology Assessment Resource Center (TTAC). (n.d.). Electronic stethoscopes – Technology overview. Alaska Native Tribal Health Consortium.
- Rocha, B. M., et al. (2019). An open access database for the evaluation of respiratory sound classification algorithms. Physiological Measurement, 40(3), 035001.
