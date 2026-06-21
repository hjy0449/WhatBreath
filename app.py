"""
WhatBreath 추론 서버 (Flask)
- 사용자가 녹음한 wav(5초 구간)를 받아 정상/비정상 판정 (EfficientNet-B0)
- 그 확률 결과를 Groq(Llama 모델)에게 보내 사용자에게 보여줄 조언 문구 생성
- 학습 때와 동일한 mel-spectrogram 변환만 적용 (EQ/노이즈는 추론 시 미적용)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
import librosa
import timm
import io
import tempfile
import os
import json
import requests

load_dotenv()  # .env 파일에 있는 환경변수를 자동으로 불러옴

# ============================================================
# 설정 (학습 때와 반드시 동일해야 하는 값들)
# ============================================================
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
SEGMENT_SECONDS = 5
TOTAL_SECONDS = 15  # 프론트엔드와 동일하게 맞춤 (script.js의 TOTAL_SECONDS와 일치해야 함) - 5초 x 3구간
N_CLASSES = 2
DROP_RATE = 0.4
LABEL_NAMES = ['정상', '비정상']

CKPT_PATH = os.environ.get(
    'BREATHLINE_CKPT_PATH',
    './best_model.pth'  # 실행 환경에 맞게 경로 수정
)

# ---- Groq (Llama 모델 호출용) ----
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY가 설정되지 않았어요. /analyze-full 호출 시 advice는 빈 값으로 반환돼요.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 프론트엔드(index.html, style.css, script.js, assets/...)를
# app.py와 같은 폴더에서 직접 서빙해요.
# 이러면 python -m http.server를 따로 띄울 필요 없이
# 이 서버 하나만 켜면 http://127.0.0.1:5000/ 에서 바로 웹사이트가 열려요.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)  # 다른 origin(예: 별도 포트로 띄운 프론트)에서도 호출 가능하도록 허용


@app.route('/')
def serve_index():
    """루트 접속 시 index.html을 보여줘요. (이전엔 여기서 404가 떴어요)"""
    return app.send_static_file('index.html')

# ============================================================
# 모델 로드 (서버 시작 시 1회만)
# ============================================================
def load_model():
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=False,  # 추론 시에는 학습된 가중치만 쓰면 되므로 False
        num_classes=N_CLASSES,
        drop_rate=DROP_RATE,
    )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f"모델 로드 완료. 체크포인트 epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')}")
    return model

model = load_model()


# ============================================================
# 전처리 (학습 때 wav_to_mel과 동일한 로직, EQ/노이즈는 적용 안 함)
# ============================================================
def wav_to_mel(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db.astype(np.float32)


def preprocess_segment(file_bytes):
    """업로드된 5초 wav(bytes) -> 모델 입력 텐서"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SR)

        # 5초 길이로 고정 (짧으면 패딩, 길면 자르기)
        target_len = int(SR * SEGMENT_SECONDS)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = wav_to_mel(y)
        mel_3ch = np.stack([mel, mel, mel], axis=0)  # (3, H, W)
        tensor = torch.FloatTensor(mel_3ch).unsqueeze(0)  # (1, 3, H, W)
        return tensor
    finally:
        os.unlink(tmp_path)


# ============================================================
# 추론
# ============================================================
def predict_segment(tensor):
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    abnormal_prob = float(probs[1])
    normal_prob = float(probs[0])
    is_abnormal = abnormal_prob > normal_prob

    return {
        'abnormal': bool(is_abnormal),
        'label': LABEL_NAMES[1] if is_abnormal else LABEL_NAMES[0],
        'normal_prob': round(normal_prob * 100, 2),
        'abnormal_prob': round(abnormal_prob * 100, 2),
        'confidence': round(max(normal_prob, abnormal_prob) * 100, 2),
    }


# ============================================================
# Groq(LLM): 확률 결과 -> 사용자용 조언 문구 생성
# ============================================================
ADVICE_PROMPT = """\
당신은 의료 스크리닝 보조 도구의 설명 담당 AI입니다.
아래는 사용자의 호흡음을 AI 모델(EfficientNet-B0)이 분석한 결과입니다.

[분석 결과]
- 정상 확률(구간 평균): {avg_normal}%
- 비정상(이상음 감지) 확률(구간 평균): {avg_abnormal}%
- 총 {total_segments}개 구간 중 {abnormal_segments}개 구간에서 이상 신호 감지
- 종합 판정: {overall_label} (과반수 기준)

[지침]
0. 무조건 한국어만 사용해야 합니다. 다른 나라의 말은 절대 사용하면 안됩니다.
1. 이 결과는 의학적 진단이 아니라 참고용 스크리닝 결과임을 반드시 먼저 명시하세요.
2. 위에 주어진 확률 수치(%)를 반드시 문장에 직접 포함해서 설명하세요. "매우 높습니다" 같은 추상적 표현 대신 실제 숫자를 써주세요.
3. 절대로 특정 질병명(천식, 폐렴, 기관지염 등)을 단정적으로 언급하지 마세요.
4. 비정상으로 판정된 경우(과반수 구간에서 이상 감지), 호흡기내과 진료를 권유하세요.
5. 비정상 확률이 낮더라도 기침, 호흡곤란 등 증상이 있다면 결과와 무관하게 진료를 받으라고 안내하세요.
6. 말투는 반드시 친근한 "해요체"를 사용하세요. "~입니다/~습니다" 같은 격식체나 딱딱한 문어체는 쓰지 마세요.
   좋은 예: "비정상 확률이 78%로 꽤 높게 나왔어요. 호흡기내과에 가서 한번 확인해보시는 게 좋을 것 같아요."
   나쁜 예: "비정상 확률이 높은 것으로 나타났습니다. 진료를 받으시기 바랍니다."
7. 따뜻하고 차분한 어조로, 3~4문장 이내로 간결하게 답하세요.
8. 의학 전문용어 대신 일반인이 이해할 수 있는 표현을 쓰세요.
9. 마크다운(별표, 헤더 등) 없이 자연스러운 줄글로만 작성하세요.

사용자에게 전달할 조언을 작성해주세요.
"""

CAUTION_PROMPT = """\
당신은 의료 스크리닝 보조 도구의 설명 담당 AI입니다.
아래는 사용자의 호흡음을 AI 모델(EfficientNet-B0)이 분석한 결과입니다.

[분석 결과]
- 총 {total_segments}개 구간 중 {abnormal_segments}개 구간에서만 이상 신호가 짧게 감지됐습니다.
- 과반수 기준으로는 "정상"이지만, 1개 구간에서 이상 신호가 있었던 만큼 완전히 무시하기는 어렵습니다.
- 정상 확률(구간 평균): {avg_normal}%, 비정상 확률(구간 평균): {avg_abnormal}%

[지침]
1. 이 결과는 의학적 진단이 아니라 참고용 스크리닝 결과임을 반드시 먼저 명시하세요.
2. 전반적으로는 정상 범위지만, 짧게 한 번 이상 신호가 감지됐다는 점을 부드럽게 설명하세요. 위 확률 수치(%)도 자연스럽게 문장에 포함하세요.
3. 절대로 특정 질병명을 단정적으로 언급하지 마세요.
4. 기침, 가래, 호흡곤란 등 증상이 있다면 진료를 받아보길 권유하고, 증상이 없다면 경과를 지켜봐도 괜찮다고 안내하세요.
5. 말투는 반드시 친근한 "해요체"를 사용하세요. "~입니다/~습니다" 같은 격식체나 딱딱한 문어체는 쓰지 마세요.
6. 따뜻하고 차분한 어조로, 3~4문장 이내로 간결하게 답하세요.
7. 마크다운 없이 자연스러운 줄글로만 작성하세요.

사용자에게 전달할 조언을 작성해주세요.
"""

NOISY_PROMPT = """\
당신은 의료 스크리닝 보조 도구의 설명 담당 AI입니다.
사용자가 호흡음을 녹음했지만, AI 모델이 전반적으로 낮은 확신도를 보였습니다.
(정상 확률 평균 {avg_normal}%, 비정상 확률 평균 {avg_abnormal}%)

이는 주변 소음이나 녹음 환경 문제로 신호 품질이 낮았을 가능성이 높습니다.

[지침]
1. 분석이 어려웠던 이유를 간단히 설명하세요.
2. 조용한 곳에서 스마트폰을 흉부에 밀착시켜 다시 녹음해보라고 안내하세요.
3. 말투는 반드시 친근한 "해요체"를 사용하세요. "~입니다/~습니다" 같은 격식체나 딱딱한 문어체는 쓰지 마세요.
4. 2~3문장 이내로 간결하고 친절하게 작성하세요.
5. 마크다운 없이 자연스러운 줄글로만 작성하세요.

사용자에게 전달할 안내 문구를 작성해주세요.
"""


def call_groq(system_prompt, user_message="조언을 작성해주세요."):
    """Groq를 통해 LLM 호출 (OpenAI 호환 chat completions 형식)"""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if not response.ok:
        # Groq가 에러 본문에 구체적인 이유(모델명 오류, 키 오류 등)를 담아주므로 그대로 노출
        raise RuntimeError(f"Groq API 응답 오류 {response.status_code}: {response.text[:300]}")

    data = response.json()
    return data['choices'][0]['message']['content'].strip()


def generate_advice(summary):
    """
    summary: {
        'status': 'ok' | 'caution' | 'noisy',
        'overall_label': str | None,
        'avg_normal_prob': float,
        'avg_abnormal_prob': float,
        'abnormal_segment_count': int,
        'total_segment_count': int,
    }
    """
    if not GROQ_API_KEY:
        return "(Groq API 키가 설정되지 않아 조언을 생성할 수 없어요. GROQ_API_KEY 환경변수를 설정해주세요.)"

    try:
        if summary['status'] == 'noisy':
            prompt = NOISY_PROMPT.format(
                avg_normal=summary['avg_normal_prob'],
                avg_abnormal=summary['avg_abnormal_prob'],
            )
        elif summary['status'] == 'caution':
            prompt = CAUTION_PROMPT.format(
                avg_normal=summary['avg_normal_prob'],
                avg_abnormal=summary['avg_abnormal_prob'],
                total_segments=summary['total_segment_count'],
                abnormal_segments=summary['abnormal_segment_count'],
            )
        else:
            prompt = ADVICE_PROMPT.format(
                avg_normal=summary['avg_normal_prob'],
                avg_abnormal=summary['avg_abnormal_prob'],
                total_segments=summary['total_segment_count'],
                abnormal_segments=summary['abnormal_segment_count'],
                overall_label=summary['overall_label'],
            )

        return call_groq(system_prompt=prompt, user_message="위 분석 결과를 바탕으로 사용자에게 전달할 조언을 작성해주세요.")
    except Exception as e:
        print(f"Groq 호출 실패 (모델: {GROQ_MODEL}): {repr(e)}")
        return "조언을 생성하는 중 오류가 발생했어요. 잠시 후 다시 시도해주세요."


# ============================================================
# API 엔드포인트
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/analyze-segment', methods=['POST'])
def analyze_segment():
    """
    단일 5초 구간 분석
    multipart/form-data: file (wav), segment_index (int, optional)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'file이 필요해요'}), 400

    file = request.files['file']
    segment_index = request.form.get('segment_index', 0)

    try:
        file_bytes = file.read()
        tensor = preprocess_segment(file_bytes)
        result = predict_segment(tensor)
        result['segment_index'] = int(segment_index)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-full', methods=['POST'])
def analyze_full():
    """
    15초 녹음 전체를 받아 서버에서 5초씩 3구간으로 분할 후 일괄 분석.

    종합 판정 기준 (다수결, majority vote):
    - 3구간 중 2구간 이상이 "비정상"이면 -> 전체 "비정상"
    - 3구간 중 1구간만 "비정상"이면 -> 종합 판정은 "정상"이되 'caution' 상태로 표시
      (짧게 한 번 나타나는 이상음을 완전히 묵살하지 않기 위함)
    - 3구간 모두 정상이면 -> 전체 "정상"
    - 노이즈 판정(모델이 양쪽 다 50% 미만 확신)은 다수결과 별개로,
      전체 구간의 평균 신뢰도가 낮을 때 적용
    multipart/form-data: file (wav, 15초)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'file이 필요해요'}), 400

    file = request.files['file']

    try:
        file_bytes = file.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=SR)
        os.unlink(tmp_path)

        recorded_seconds = len(y) / SR
        if recorded_seconds < TOTAL_SECONDS:
            return jsonify({
                'error': 'too_short',
                'message': f'녹음이 {recorded_seconds:.1f}초밖에 안 돼요. {TOTAL_SECONDS}초 이상 녹음해주세요.',
                'recorded_seconds': round(recorded_seconds, 1),
                'required_seconds': TOTAL_SECONDS,
            }), 400

        segment_len = int(SR * SEGMENT_SECONDS)
        max_segments = TOTAL_SECONDS // SEGMENT_SECONDS  # 15초보다 길어도 앞부분 3구간만 사용
        n_segments = min(len(y) // segment_len, max_segments)

        segments_result = []
        for i in range(n_segments):
            chunk = y[i * segment_len: (i + 1) * segment_len]
            mel = wav_to_mel(chunk)
            mel_3ch = np.stack([mel, mel, mel], axis=0)
            tensor = torch.FloatTensor(mel_3ch).unsqueeze(0)

            result = predict_segment(tensor)
            result['segment_index'] = i
            result['start_sec'] = i * SEGMENT_SECONDS
            result['end_sec'] = (i + 1) * SEGMENT_SECONDS
            segments_result.append(result)

        # ---- 다수결(majority vote) 기반 종합 판정 ----
        abnormal_segments = [s for s in segments_result if s['abnormal']]
        abnormal_count = len(abnormal_segments)
        total_count = len(segments_result)

        # 노이즈 판정: 모든 구간의 confidence(더 높은 쪽 확률)가 평균적으로 낮으면
        # 모델이 전반적으로 확신을 못하는 상태로 보고 별도 안내
        avg_confidence = sum(s['confidence'] for s in segments_result) / total_count

        majority_threshold = (total_count // 2) + 1  # 3구간 기준 2 이상

        if avg_confidence < 50:
            status = 'noisy'
            overall_label = None
            overall_abnormal = None
        elif abnormal_count >= majority_threshold:
            status = 'ok'
            overall_label = LABEL_NAMES[1]
            overall_abnormal = True
        elif abnormal_count >= 1:
            # 1구간(과반 미만)만 비정상: 종합은 정상이지만 주의가 필요한 상태
            status = 'caution'
            overall_label = LABEL_NAMES[0]
            overall_abnormal = False
        else:
            status = 'ok'
            overall_label = LABEL_NAMES[0]
            overall_abnormal = False

        # 참고용 평균 확률 (화면에 정상/비정상 정도를 보여줄 때 사용)
        avg_normal = float(np.mean([s['normal_prob'] for s in segments_result]))
        avg_abnormal = float(np.mean([s['abnormal_prob'] for s in segments_result]))

        response = {
            'status': status,  # 'ok' | 'caution' | 'noisy'
            'overall_label': overall_label,
            'overall_abnormal': overall_abnormal,
            'avg_normal_prob': round(avg_normal, 2),
            'avg_abnormal_prob': round(avg_abnormal, 2),
            'abnormal_segment_count': abnormal_count,
            'total_segment_count': total_count,
            'majority_threshold': majority_threshold,
            'segments': segments_result,
        }

        if status == 'noisy':
            response['message'] = '주변이 너무 시끄럽습니다. 조용한 곳으로 이동 후 다시 녹음해주세요.'
        elif status == 'caution':
            response['message'] = f'{total_count}구간 중 1구간에서 이상 신호가 짧게 감지됐어요. 종합적으로는 정상 범위지만 참고해주세요.'

        # Groq에게 확률 결과를 보내 사용자용 조언 문구 생성
        response['advice'] = generate_advice(response)


        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)