// ============================================================
// Mel-spectrogram 합성 데이터 + 색상 매핑 (데모용, 실제 모델 미연결)
// ============================================================
function generateMelData(seed, abnormal) {
  const rows = 48;
  const cols = 96;
  const data = [];
  let s = seed;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const freqFalloff = Math.exp(-r / 22);
      const timeWave = Math.sin(c / 6 + r / 9) * 0.3 + Math.sin(c / 13) * 0.2;
      let v = freqFalloff * (0.45 + timeWave) + rand() * 0.18;
      if (abnormal && r > 18 && r < 34 && Math.sin(c / 4) > 0.6) {
        v += 0.4 * Math.exp(-Math.pow((c % 24) - 12, 2) / 20);
      }
      row.push(Math.max(0, Math.min(1, v)));
    }
    data.push(row);
  }
  return data;
}

function melToColor(v) {
  const stops = [
    [10, 20, 40],
    [20, 60, 130],
    [30, 110, 200],
    [70, 170, 220],
    [140, 220, 200],
    [230, 240, 170],
  ];
  const pos = v * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = stops[i];
  const b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

function drawMelSpectrogram(canvas, data) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  const rows = data.length;
  const cols = data[0].length;
  const cw = w / cols;
  const ch = h / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = melToColor(data[rows - 1 - r][c]);
      ctx.fillRect(c * cw, r * ch, cw + 0.6, ch + 0.6);
    }
  }
}

// ============================================================
// 네비게이션
// ============================================================
const pages = {
  home: document.getElementById('page-home'),
  record: document.getElementById('page-record'),
};
const tabs = document.querySelectorAll('.tab');

function goToPage(name) {
  Object.keys(pages).forEach((key) => {
    pages[key].style.display = key === name ? '' : 'none';
  });
  tabs.forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.nav === name);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (name === 'record') resetRecordPage();
}

document.querySelectorAll('[data-nav]').forEach((el) => {
  el.addEventListener('click', () => goToPage(el.dataset.nav));
});

// ============================================================
// 히어로 비주얼: 파형 <-> mel-spectrogram 모핑
// ============================================================
(function initHeroVisual() {
  const canvas = document.getElementById('heroMelCanvas');
  const label = document.getElementById('heroMelLabel');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const melData = generateMelData(7, false);
  let t = 0;
  let progress = 0;
  let dir = 1;

  function draw() {
    t += 0.05;
    progress += dir * 0.006;
    if (progress >= 1) { progress = 1; dir = -1; }
    if (progress <= 0) { progress = 0; dir = 1; }
    label.textContent = progress > 0.5 ? 'Mel-Spectrogram 변환' : '원본 음향 신호';

    ctx.clearRect(0, 0, w, h);

    // waveform layer
    const bars = 48;
    const barW = w / bars;
    for (let i = 0; i < bars; i++) {
      const phase = i * 0.35;
      const env = 0.3 + 0.6 * Math.abs(Math.sin(t * 1.4 + phase) * 0.6 + Math.sin(t * 2.3 + phase * 1.4) * 0.3);
      const barH = env * h * (1 - progress);
      const x = i * barW + barW * 0.22;
      const y = (h - barH) / 2;
      ctx.fillStyle = `rgba(255,255,255,${0.85 * (1 - progress)})`;
      ctx.fillRect(x, y, barW * 0.56, barH);
    }

    // mel layer
    if (progress > 0) {
      const rows = melData.length;
      const cols = melData[0].length;
      const cw = w / cols;
      const ch = h / rows;
      ctx.globalAlpha = progress;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          ctx.fillStyle = melToColor(melData[rows - 1 - r][c]);
          ctx.fillRect(c * cw, r * ch, cw + 0.6, ch + 0.6);
        }
      }
      ctx.globalAlpha = 1;
    }

    requestAnimationFrame(draw);
  }
  draw();
})();

// ============================================================
// 녹음 페이지 상태 관리
// ============================================================
const stageIdle = document.getElementById('stage-idle');
const stageUploaded = document.getElementById('stage-uploaded');
const stageAnalyzing = document.getElementById('stage-analyzing');
const stageResult = document.getElementById('stage-result');
const idleControls = document.getElementById('idle-controls');
const recordingControls = document.getElementById('recording-controls');
const stepLabel = document.getElementById('stepLabel');
const recordTitle = document.getElementById('recordTitle');

let recordWaveActive = false;
let recordWaveRAF = null;
let recordTimer = null;
let recordSeconds = 0;
let currentFileName = '';

function resetRecordPage() {
  stageIdle.style.display = '';
  stageUploaded.style.display = 'none';
  stageAnalyzing.style.display = 'none';
  stageResult.style.display = 'none';
  idleControls.style.display = '';
  recordingControls.style.display = 'none';
  stepLabel.textContent = 'STEP 1 / 3';
  recordTitle.textContent = '호흡음 녹음 또는 업로드';
  recordWaveActive = false;
  recordSeconds = 0;
  clearInterval(recordTimer);
  document.querySelectorAll('.segment-dot').forEach((d) => d.classList.remove('filled'));
}

// ---- 녹음 파형 캔버스 ----
(function initRecordWave() {
  const canvas = document.getElementById('recordWaveCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let t = 0;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }
  resize();
  window.addEventListener('resize', resize);

  function draw() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    t += recordWaveActive ? 0.045 : 0.01;
    ctx.clearRect(0, 0, w, h);
    const bars = 64;
    const barW = w / bars;
    for (let i = 0; i < bars; i++) {
      const phase = i * 0.35;
      const env = recordWaveActive
        ? 0.35 + 0.65 * Math.abs(Math.sin(t * 1.8 + phase) * 0.6 + Math.sin(t * 3.1 + phase * 1.4) * 0.3 + Math.sin(t * 0.7 + phase * 0.5) * 0.2)
        : 0.08 + 0.04 * Math.sin(t * 2 + phase);
      const barH = env * h;
      const x = i * barW + barW * 0.22;
      const y = (h - barH) / 2;
      ctx.fillStyle = recordWaveActive ? '#1E5FCC' : '#C7D7EC';
      const bw = barW * 0.56;
      const radius = barW * 0.28;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x, y, bw, barH, radius);
      } else {
        ctx.rect(x, y, bw, barH);
      }
      ctx.fill();
    }
    recordWaveRAF = requestAnimationFrame(draw);
  }
  draw();
})();

// ---- 녹음 시작/중단 (실제 마이크 녹음, 15초, 5초 x 3구간) ----
const TOTAL_SECONDS = 15;           // 서버에 보고되는 공식 구간 길이 (3구간 × 5초)
const RECORD_BUFFER_SECONDS = 0.6;  // 인코딩 손실 보정용 여유시간
const RECORD_TARGET_SECONDS = TOTAL_SECONDS + RECORD_BUFFER_SECONDS;  // 실제 녹음 목표(15.6초)
const SEGMENT_SECONDS = 5;
const SEGMENT_COUNT = TOTAL_SECONDS / SEGMENT_SECONDS;

let mediaRecorder = null;
let recordedChunks = [];
let currentAudioBlob = null; // 서버로 보낼 실제 오디오 데이터
let stoppedEarly = false; // 15초가 차기 전에 사용자가 직접 중단했는지

(function initSegmentDots() {
  const wrap = document.getElementById('segmentDots');
  if (!wrap) return;
  for (let i = 0; i < SEGMENT_COUNT; i++) {
    const dot = document.createElement('div');
    dot.className = 'segment-dot';
    dot.dataset.index = i;
    wrap.appendChild(dot);
  }
})();

document.getElementById('btnStartRecord').addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];

    // 브라우저가 지원하는 형식으로 녹음 (보통 webm/opus)
    const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      currentAudioBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      stream.getTracks().forEach((track) => track.stop()); // 마이크 점유 해제
      finishRecording();
    };

    mediaRecorder.start();

    idleControls.style.display = 'none';
    recordingControls.style.display = '';
    recordWaveActive = true;
    recordSeconds = 0;
    document.getElementById('timerValue').textContent = '0.0';
    document.querySelectorAll('.segment-dot').forEach((d) => d.classList.remove('filled'));

    recordTimer = setInterval(() => {
      recordSeconds += 0.1;
      if (recordSeconds >= RECORD_TARGET_SECONDS) {
        document.getElementById('timerValue').textContent = RECORD_TARGET_SECONDS.toFixed(1);
        clearInterval(recordTimer);
        stoppedEarly = false; // 15초를 다 채우고 자동으로 멈춘 것
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        return;
      }
      document.getElementById('timerValue').textContent = recordSeconds.toFixed(1);
      const filledCount = Math.floor(recordSeconds / SEGMENT_SECONDS);
      document.querySelectorAll('.segment-dot').forEach((d, i) => {
        d.classList.toggle('filled', i < filledCount);
      });
    }, 100);
  } catch (err) {
    alert('마이크 접근 권한이 필요해요. 브라우저 설정에서 마이크 권한을 허용해주세요.');
    console.error(err);
  }
});

document.getElementById('btnStopRecord').addEventListener('click', () => {
  clearInterval(recordTimer);
  stoppedEarly = recordSeconds < TOTAL_SECONDS; // 25초 전에 사용자가 직접 중단한 경우
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop(); // onstop에서 finishRecording 호출됨
  } else {
    finishRecording();
  }
});

function finishRecording() {
  recordWaveActive = false;

  if (stoppedEarly) {
    // 25초를 채우지 못하고 중단한 경우: 분석 단계로 넘어가지 않고 다시 녹음하도록 안내
    alert(`녹음이 ${recordSeconds.toFixed(1)}초밖에 되지 않았어요. 정확한 분석을 위해 ${TOTAL_SECONDS}초를 채워서 녹음해주세요.`);
    currentAudioBlob = null;
    resetRecordPage();
    return;
  }

  const now = new Date();
  const hh = String(now.getHours()).padStart(2, '0');
  const mm = String(now.getMinutes()).padStart(2, '0');
  currentFileName = `녹음_${hh}${mm}.webm`;
  showUploadedStage();
}

// 업로드된 오디오 파일(Blob/File)의 길이를 초 단위로 읽어옴
function getAudioDuration(file) {
  return new Promise((resolve, reject) => {
    const audio = document.createElement('audio');
    const url = URL.createObjectURL(file);
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(audio.duration);
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('오디오 메타데이터를 읽을 수 없어요.'));
    };
    audio.src = url;
  });
}

// ---- 파일 업로드 ----
document.getElementById('btnUploadFile').addEventListener('click', () => {
  document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  // 업로드된 오디오 길이를 미리 확인해서 25초 미만이면 거절
  try {
    const duration = await getAudioDuration(file);
    if (duration < TOTAL_SECONDS) {
      alert(`업로드한 파일이 ${duration.toFixed(1)}초밖에 안 돼요. ${TOTAL_SECONDS}초 이상인 녹음 파일을 올려주세요.`);
      e.target.value = ''; // 같은 파일 다시 선택할 수 있도록 입력 초기화
      return;
    }
  } catch (err) {
    console.warn('오디오 길이를 확인하지 못했어요. 서버 쪽 검증에 맡겨요.', err);
  }

  currentFileName = file.name;
  currentAudioBlob = file; // File은 Blob의 하위 타입이라 그대로 사용 가능
  showUploadedStage();
});

function showUploadedStage() {
  stageIdle.style.display = 'none';
  stageUploaded.style.display = '';
  stepLabel.textContent = 'STEP 2 / 3';
  document.getElementById('uploadedFileName').textContent = currentFileName;
}

document.getElementById('btnReselect').addEventListener('click', () => {
  resetRecordPage();
});

// ---- 분석 실행: 실제 서버(/analyze-full)에 오디오를 보내고 결과를 받음 ----
// Flask가 프론트엔드도 같이 서빙하므로 같은 origin이면 빈 문자열(상대경로)이면 충분해요.
// 만약 index.html을 다른 포트(예: http.server)로 따로 띄운다면
// 'http://127.0.0.1:5000' 처럼 Flask 서버 주소를 명시해주세요.
const API_BASE_URL = '';

(function initSegmentProgressChips() {
  const wrap = document.getElementById('segmentProgress');
  if (!wrap) return;
  for (let i = 0; i < SEGMENT_COUNT; i++) {
    const chip = document.createElement('div');
    chip.className = 'segment-chip';
    chip.id = 'progChip' + i;
    chip.textContent = i + 1;
    wrap.appendChild(chip);
  }
})();

document.getElementById('btnAnalyze').addEventListener('click', async () => {
  if (!currentAudioBlob) {
    alert('분석할 오디오가 없어요. 다시 녹음하거나 파일을 선택해주세요.');
    return;
  }

  stageUploaded.style.display = 'none';
  stageAnalyzing.style.display = '';
  stepLabel.textContent = 'STEP 3 / 3';

  // 칩을 순서대로 "확인 중" 상태로만 보여주는 로딩 연출 (실제 진행률은 서버가 한 번에 처리)
  for (let i = 0; i < SEGMENT_COUNT; i++) {
    const chip = document.getElementById('progChip' + i);
    chip.className = 'segment-chip';
    chip.textContent = i + 1;
  }
  let chipTimer = 0;
  const chipInterval = setInterval(() => {
    if (chipTimer > 0) {
      const prevChip = document.getElementById('progChip' + (chipTimer - 1));
      prevChip.classList.remove('checking');
      prevChip.classList.add('done-normal');
      prevChip.textContent = '✓';
    }
    if (chipTimer < SEGMENT_COUNT) {
      document.getElementById('progChip' + chipTimer).classList.add('checking');
    }
    chipTimer += 1;
    if (chipTimer > SEGMENT_COUNT) clearInterval(chipInterval);
  }, 500);

  try {
    const formData = new FormData();
    formData.append('file', currentAudioBlob, currentFileName);

    const res = await fetch(`${API_BASE_URL}/analyze-full`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      if (errBody.error === 'too_short') {
        clearInterval(chipInterval);
        alert(errBody.message); // "녹음이 X초밖에 안 돼요. 25초 이상 녹음해주세요." 같은 메시지
        resetRecordPage();
        return;
      }
      throw new Error(errBody.error || `서버 오류 (${res.status})`);
    }

    const data = await res.json();
    clearInterval(chipInterval);
    finalizeAnalysis(data);
  } catch (err) {
    clearInterval(chipInterval);
    console.error(err);
    alert(`분석 중 오류가 발생했어요: ${err.message}\n\nFlask 서버(${API_BASE_URL})가 실행 중인지 확인해주세요.`);
    resetRecordPage();
  }
});

// 서버 응답(snake_case)을 화면에서 쓰는 형태로 변환
function finalizeAnalysis(serverData) {
  const segments = serverData.segments.map((s) => ({
    index: s.segment_index,
    startSec: s.start_sec,
    endSec: s.end_sec,
    abnormal: s.abnormal,
    confidence: s.confidence,
    normalProb: s.normal_prob,
    abnormalProb: s.abnormal_prob,
    mel: generateMelData(Math.floor(Math.random() * 1000) + s.segment_index * 17, s.abnormal), // mel 시각화는 데모 패턴 사용
  }));

  const abnormalSegments = segments.filter((s) => s.abnormal);
  const status = serverData.status; // 'ok' | 'caution' | 'noisy'
  const overallAbnormal = serverData.overall_abnormal;
  const avgNormal = serverData.avg_normal_prob;
  const avgAbnormal = serverData.avg_abnormal_prob;

  const result = {
    status,
    label: status === 'noisy' ? null : (overallAbnormal ? '비정상' : '정상'),
    abnormal: overallAbnormal,
    confidence: status === 'noisy' ? Math.max(avgNormal, avgAbnormal) : (overallAbnormal ? avgAbnormal : avgNormal),
    avgNormal,
    avgAbnormal,
    abnormalCount: serverData.abnormal_segment_count,
    totalCount: serverData.total_segment_count,
    majorityThreshold: serverData.majority_threshold,
    message: serverData.message || null, // caution/noisy 상태일 때 서버가 주는 안내문
    segments,
    mel: segments[0].mel,
    selectedSegmentIndex: abnormalSegments.length > 0 ? abnormalSegments[0].index : 0,
    timestamp: new Date(),
    fileName: currentFileName,
    advice: serverData.advice || '조언을 불러올 수 없어요.',
  };
  showResult(result);
}

function showResult(result) {
  stageAnalyzing.style.display = 'none';
  stageResult.style.display = '';
  recordTitle.textContent = '분석 결과';

  const banner = document.getElementById('resultBanner');
  const icon = document.getElementById('resultIcon');

  // 공통: 배너/아이콘 클래스 초기화
  banner.classList.remove('abnormal', 'noisy', 'caution');
  icon.classList.remove('abnormal', 'noisy', 'caution');

  // ---- 노이즈로 판정 불가한 경우 ----
  if (result.status === 'noisy') {
    banner.classList.add('noisy');
    icon.classList.add('noisy');
    icon.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 3 1 21h22L12 3z" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 9v5M12 17.5v.01" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>';

    document.getElementById('resultTitle').textContent = '주변이 너무 시끄러워요';
    document.getElementById('resultSub').textContent = `${result.fileName} · ${TOTAL_SECONDS}초 녹음 · ${result.totalCount}구간 분석 · ${result.timestamp.toLocaleString('ko-KR')}`;

    document.getElementById('metricNormal').textContent = result.avgNormal.toFixed(1) + '%';
    document.getElementById('metricAbnormal').textContent = result.avgAbnormal.toFixed(1) + '%';
    document.getElementById('resultBarGreen').style.width = result.avgNormal + '%';
    document.getElementById('resultBarRed').style.width = result.avgAbnormal + '%';

    document.getElementById('disclaimerText').textContent =
      result.message || '신뢰도가 낮아 신뢰할 만한 분석이 어려웠어요. 조용한 곳으로 이동한 뒤 다시 녹음해주세요.';
    document.getElementById('adviceText').textContent = result.advice || '조언을 불러올 수 없어요.';

    renderSegmentCards(result);
    return;
  }

  // ---- 1구간만 이상(과반 미만): 종합은 정상이지만 주의 표시 ----
  if (result.status === 'caution') {
    banner.classList.add('caution');
    icon.classList.add('caution');
    icon.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 3 1 21h22L12 3z" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 9v5M12 17.5v.01" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>';

    document.getElementById('resultTitle').textContent = `정상 범위지만 주의가 필요해요 (${result.abnormalCount}/${result.totalCount}구간 이상 감지)`;
    document.getElementById('resultSub').textContent = `${result.fileName} · ${TOTAL_SECONDS}초 녹음 · ${result.totalCount}구간 분석 · 과반수 기준 종합판정 · ${result.timestamp.toLocaleString('ko-KR')}`;

    document.getElementById('metricNormal').textContent = result.avgNormal.toFixed(1) + '%';
    document.getElementById('metricAbnormal').textContent = result.avgAbnormal.toFixed(1) + '%';
    document.getElementById('resultBarGreen').style.width = result.avgNormal + '%';
    document.getElementById('resultBarRed').style.width = result.avgAbnormal + '%';

    document.getElementById('disclaimerText').textContent =
      result.message || `${result.totalCount}구간 중 1구간에서 이상 신호가 짧게 감지됐어요. 종합적으로는 정상 범위지만 참고해주세요. 증상이 있다면 진료를 받아보세요.`;
    document.getElementById('adviceText').textContent = result.advice || '조언을 불러올 수 없어요.';

    renderSegmentCards(result);
    return;
  }

  // ---- 일반 정상/비정상 (과반수 기준) ----
  icon.classList.toggle('abnormal', result.abnormal);
  banner.classList.toggle('abnormal', result.abnormal);

  icon.innerHTML = result.abnormal
    ? '<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 3 1 21h22L12 3z" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 9v5M12 17.5v.01" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>'
    : '<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M5 13l4 4L19 7" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  document.getElementById('resultTitle').textContent = result.abnormal
    ? `이상 징후가 감지됐어요 (${result.abnormalCount}/${result.totalCount}구간 비정상 판정)`
    : '정상 범위로 분석됐어요';
  document.getElementById('resultSub').textContent = `${result.fileName} · ${TOTAL_SECONDS}초 녹음 · ${result.totalCount}구간 분석 · 과반수 기준 종합판정 · ${result.timestamp.toLocaleString('ko-KR')}`;

  document.getElementById('metricNormal').textContent = result.avgNormal.toFixed(1) + '%';
  document.getElementById('metricAbnormal').textContent = result.avgAbnormal.toFixed(1) + '%';
  document.getElementById('resultBarGreen').style.width = result.avgNormal + '%';
  document.getElementById('resultBarRed').style.width = result.avgAbnormal + '%';

  document.getElementById('disclaimerText').textContent = result.abnormal
    ? `이 결과는 의학적 진단이 아닌 참고용 스크리닝이에요. ${result.totalCount}구간 중 과반수에서 이상 신호가 감지됐어요. 증상이 있거나 결과가 우려된다면 호흡기내과 진료를 받아보세요.`
    : '이 결과는 의학적 진단이 아닌 참고용 스크리닝이에요. 증상이 있다면 결과와 무관하게 진료를 받아보세요.';
  document.getElementById('adviceText').textContent = result.advice || '조언을 불러올 수 없어요.';

  renderSegmentCards(result);
}

function renderSegmentCards(result) {
  // 구간별 카드 렌더링
  const segWrap = document.getElementById('segmentResults');
  segWrap.innerHTML = '';
  result.segments.forEach((seg) => {
    const card = document.createElement('div');
    card.className = 'segment-result-card' + (seg.index === result.selectedSegmentIndex ? ' selected' : '');
    card.innerHTML = `
      <div class="segment-result-time">${seg.startSec}-${seg.endSec}s</div>
      <div class="segment-result-tag ${seg.abnormal ? 'abnormal' : 'normal'}">${seg.abnormal ? '이상' : '정상'}</div>
    `;
    card.addEventListener('click', () => {
      document.querySelectorAll('.segment-result-card').forEach((c) => c.classList.remove('selected'));
      card.classList.add('selected');
      drawMelSpectrogram(document.getElementById('resultMelCanvas'), seg.mel);
      document.getElementById('melSegmentTag').textContent = `(${seg.startSec}-${seg.endSec}s 구간)`;
    });
    segWrap.appendChild(card);
  });

  const initialSeg = result.segments[result.selectedSegmentIndex];
  document.getElementById('melSegmentTag').textContent = `(${initialSeg.startSec}-${initialSeg.endSec}s 구간)`;
  const melCanvas = document.getElementById('resultMelCanvas');
  drawMelSpectrogram(melCanvas, initialSeg.mel);
}

document.getElementById('btnReanalyze').addEventListener('click', () => {
  resetRecordPage();
});

// ============================================================
// 초기화
// ============================================================
goToPage('home');