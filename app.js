const MP_VERSION = "0.10.32";
const MP_MODULE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const NORMAL_TRADE_WIND = -0.32;
const LEFT_ZONE_MAX = 0.42;
const RIGHT_ZONE_MIN = 0.58;
const TRACK_WINDOW_MS = 850;
const TRACK_FORGET_MS = 1200;
const INFERENCE_INTERVAL_MS = 48;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const lerp = (start, end, amount) => start + (end - start) * amount;
const expSmoothing = (current, target, dt, tau) => {
  const safeTau = Math.max(0.001, tau);
  return current + (target - current) * (1 - Math.exp(-dt / safeTau));
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  video: $("#webcam"),
  handCanvas: $("#handCanvas"),
  oceanCanvas: $("#oceanCanvas"),
  cameraStage: $("#cameraStage"),
  cameraPlaceholder: $("#cameraPlaceholder"),
  cameraBadge: $("#cameraBadge"),
  cameraStatus: $("#cameraStatus"),
  startCameraBtn: $("#startCameraBtn"),
  stopCameraBtn: $("#stopCameraBtn"),
  resetBtn: $("#resetBtn"),
  cameraModeBtn: $("#cameraModeBtn"),
  manualModeBtn: $("#manualModeBtn"),
  cameraControls: $("#cameraControls"),
  manualControls: $("#manualControls"),
  sensitivityRange: $("#sensitivityRange"),
  sensitivityOutput: $("#sensitivityOutput"),
  responseRange: $("#responseRange"),
  responseOutput: $("#responseOutput"),
  manualWindRange: $("#manualWindRange"),
  manualWindOutput: $("#manualWindOutput"),
  activeZoneValue: $("#activeZoneValue"),
  fanStrengthValue: $("#fanStrengthValue"),
  windValue: $("#windValue"),
  ensoBadge: $("#ensoBadge"),
  ensoStateLabel: $("#ensoStateLabel"),
  ensoIndexLabel: $("#ensoIndexLabel"),
  stateExplanation: $("#stateExplanation"),
  chainWind: $("#chainWind"),
  chainSea: $("#chainSea"),
  chainThermo: $("#chainThermo"),
  chainSst: $("#chainSst"),
  chainWeather: $("#chainWeather"),
  seaLevelValue: $("#seaLevelValue"),
  seaLevelHint: $("#seaLevelHint"),
  sstValue: $("#sstValue"),
  sstHint: $("#sstHint"),
  thermoclineValue: $("#thermoclineValue"),
  thermoclineHint: $("#thermoclineHint"),
  upwellingValue: $("#upwellingValue"),
  upwellingHint: $("#upwellingHint"),
  pressureValue: $("#pressureValue"),
  pressureHint: $("#pressureHint"),
  rainValue: $("#rainValue"),
  rainHint: $("#rainHint"),
  recordBtn: $("#recordBtn"),
  recordsBody: $("#recordsBody"),
  downloadCsvBtn: $("#downloadCsvBtn"),
  clearRecordsBtn: $("#clearRecordsBtn"),
};

const handContext = elements.handCanvas.getContext("2d");
const oceanContext = elements.oceanCanvas.getContext("2d");

const app = {
  mode: "camera",
  mediaPipeModule: null,
  handLandmarker: null,
  handConnections: [],
  stream: null,
  cameraRunning: false,
  initializingCamera: false,
  lastVideoTime: -1,
  lastInferenceAt: 0,
  lastHandResult: null,
  inferenceErrorShown: false,
  tracks: new Map(),
  nextTrackId: 1,
  leftEnergy: 0,
  rightEnergy: 0,
  fanStrength: 0,
  activeZone: "none",
  sensitivity: Number(elements.sensitivityRange.value),
  responseSeconds: Number(elements.responseRange.value),
  manualWind: Number(elements.manualWindRange.value),
  windTarget: NORMAL_TRADE_WIND,
  wind: NORMAL_TRADE_WIND,
  ensoTarget: 0,
  enso: 0,
  animationPhase: 0,
  lastFrameAt: performance.now(),
  lastUiUpdateAt: 0,
  records: [],
};

function setCameraBadge(kind, text) {
  elements.cameraBadge.className = `status-badge status-${kind}`;
  elements.cameraBadge.textContent = text;
}

function setCameraStatus(text, isError = false) {
  elements.cameraStatus.textContent = text;
  elements.cameraStatus.classList.toggle("is-error", isError);
}

function setMode(mode) {
  app.mode = mode;
  const cameraMode = mode === "camera";
  elements.cameraModeBtn.classList.toggle("is-active", cameraMode);
  elements.manualModeBtn.classList.toggle("is-active", !cameraMode);
  elements.cameraModeBtn.setAttribute("aria-pressed", String(cameraMode));
  elements.manualModeBtn.setAttribute("aria-pressed", String(!cameraMode));
  elements.cameraControls.hidden = !cameraMode;
  elements.manualControls.hidden = cameraMode;

  if (cameraMode) {
    if (app.cameraRunning) {
      setCameraStatus("손을 왼쪽 또는 오른쪽 구역에서 좌우로 왕복해 보세요.");
    } else {
      setCameraStatus("카메라 시작 버튼을 누르세요. HTTPS 또는 localhost에서만 카메라를 사용할 수 있습니다.");
    }
  } else {
    setCameraStatus("수동 슬라이더나 대표 상태 버튼으로 바람을 조작하고 있습니다.");
  }
}

function sensitivityLabel(value) {
  if (value < 0.82) return "둔감";
  if (value < 1.18) return "보통";
  if (value < 1.42) return "민감";
  return "매우 민감";
}

function responseLabel(value) {
  if (value < 1.05) return "즉시";
  if (value < 2.25) return "수업용";
  if (value < 3.25) return "천천히";
  return "매우 천천히";
}

function windStrengthLabel(magnitude) {
  if (magnitude < 0.12) return "매우 약함";
  if (magnitude < 0.38) return "보통";
  if (magnitude < 0.7) return "강함";
  return "매우 강함";
}

function windDescription(wind, includeDirection = true) {
  if (Math.abs(wind) < 0.05) return "거의 무풍";
  const direction = wind > 0 ? "서→동" : "동→서";
  const kind = wind > 0 ? "서풍 편차" : "무역풍";
  const strength = windStrengthLabel(Math.abs(wind));
  return includeDirection ? `${direction} · ${strength}` : `${kind} ${strength}`;
}

function updateControlOutputs() {
  elements.sensitivityOutput.textContent = sensitivityLabel(app.sensitivity);
  elements.responseOutput.textContent = responseLabel(app.responseSeconds);
  elements.manualWindOutput.textContent = windDescription(app.manualWind, false);
}

async function ensureHandLandmarker() {
  if (app.handLandmarker) return app.handLandmarker;

  setCameraBadge("loading", "모델 불러오는 중");
  setCameraStatus("MediaPipe 손 추적 라이브러리와 모델을 불러오고 있습니다.");

  app.mediaPipeModule ??= await import(MP_MODULE_URL);
  const { FilesetResolver, HandLandmarker } = app.mediaPipeModule;
  const vision = await FilesetResolver.forVisionTasks(MP_WASM_URL);

  const commonOptions = {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };

  try {
    app.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...commonOptions,
      baseOptions: {
        ...commonOptions.baseOptions,
        delegate: "GPU",
      },
    });
  } catch (gpuError) {
    console.warn("GPU delegate initialization failed; retrying with CPU.", gpuError);
    app.handLandmarker = await HandLandmarker.createFromOptions(vision, commonOptions);
  }

  app.handConnections = HandLandmarker.HAND_CONNECTIONS ?? [];
  return app.handLandmarker;
}

function isLocalhost() {
  return ["localhost", "127.0.0.1", "::1"].includes(location.hostname);
}

async function startCamera() {
  if (app.initializingCamera || app.cameraRunning) return;
  app.initializingCamera = true;
  elements.startCameraBtn.disabled = true;
  setMode("camera");

  try {
    if (!window.isSecureContext && !isLocalhost()) {
      throw new Error("INSECURE_CONTEXT");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("MEDIA_DEVICES_UNSUPPORTED");
    }

    await ensureHandLandmarker();
    setCameraStatus("카메라 사용 권한을 요청하고 있습니다.");

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 60 },
      },
    });

    app.stream = stream;
    elements.video.srcObject = stream;
    await elements.video.play();

    app.cameraRunning = true;
    app.lastVideoTime = -1;
    app.lastInferenceAt = 0;
    app.inferenceErrorShown = false;
    elements.cameraPlaceholder.classList.add("is-hidden");
    elements.stopCameraBtn.disabled = false;
    setCameraBadge("ready", "손 추적 중");
    setCameraStatus("손바닥 전체가 보이도록 한쪽 구역에서 좌우로 반복해 흔드세요.");
  } catch (error) {
    console.error(error);
    stopCamera({ preserveMessage: true });
    setCameraBadge("error", "시작 실패");
    setCameraStatus(cameraErrorMessage(error), true);
  } finally {
    app.initializingCamera = false;
    elements.startCameraBtn.disabled = app.cameraRunning;
  }
}

function cameraErrorMessage(error) {
  if (error?.message === "INSECURE_CONTEXT") {
    return "카메라는 HTTPS로 열린 GitHub Pages 또는 localhost에서만 사용할 수 있습니다. 파일을 더블클릭해 연 화면에서는 수동 모드를 사용하세요.";
  }
  if (error?.message === "MEDIA_DEVICES_UNSUPPORTED") {
    return "이 브라우저는 카메라 API를 지원하지 않습니다. 최신 Chrome, Edge 또는 Safari에서 다시 열어 주세요.";
  }
  if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
    return "카메라 권한이 거부되었습니다. 주소창의 카메라 권한을 허용한 뒤 다시 시작하세요.";
  }
  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return "사용할 수 있는 카메라를 찾지 못했습니다. 카메라 연결 상태를 확인하세요.";
  }
  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return "다른 앱이 카메라를 사용 중일 수 있습니다. 화상회의 앱을 닫고 다시 시도하세요.";
  }
  return "손 추적 모델 또는 카메라를 시작하지 못했습니다. 네트워크 연결을 확인하거나 수동 모드를 사용하세요.";
}

function stopCamera({ preserveMessage = false } = {}) {
  if (app.stream) {
    app.stream.getTracks().forEach((track) => track.stop());
  }
  app.stream = null;
  elements.video.srcObject = null;
  app.cameraRunning = false;
  app.lastHandResult = null;
  app.tracks.clear();
  app.leftEnergy = 0;
  app.rightEnergy = 0;
  app.fanStrength = 0;
  app.activeZone = "none";
  elements.cameraPlaceholder.classList.remove("is-hidden");
  elements.startCameraBtn.disabled = false;
  elements.stopCameraBtn.disabled = true;
  clearHandCanvas();

  if (!preserveMessage) {
    setCameraBadge("idle", "대기 중");
    setCameraStatus("카메라가 중지되었습니다. 수동 모드는 계속 사용할 수 있습니다.");
  }
}

function clearHandCanvas() {
  const { width, height } = resizeCanvas(elements.handCanvas, handContext);
  handContext.clearRect(0, 0, width, height);
}

function getDisplayPoint(landmarks) {
  const palmIndices = [0, 5, 9, 13, 17];
  const sum = palmIndices.reduce(
    (acc, index) => {
      acc.x += 1 - landmarks[index].x;
      acc.y += landmarks[index].y;
      return acc;
    },
    { x: 0, y: 0 },
  );
  return {
    x: sum.x / palmIndices.length,
    y: sum.y / palmIndices.length,
  };
}

function zoneFromX(x) {
  if (x < LEFT_ZONE_MAX) return "left";
  if (x > RIGHT_ZONE_MIN) return "right";
  return "center";
}

function assignDetectionsToTracks(detections, now) {
  const availableTracks = [...app.tracks.values()]
    .filter((track) => now - track.lastSeen < 650)
    .map((track) => ({ track, used: false }));

  for (const detection of detections) {
    let best = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const candidate of availableTracks) {
      if (candidate.used) continue;
      const dx = candidate.track.x - detection.x;
      const dy = candidate.track.y - detection.y;
      const distance = Math.hypot(dx, dy * 0.65);
      if (distance < bestDistance && distance < 0.33) {
        best = candidate;
        bestDistance = distance;
      }
    }

    let track;
    if (best) {
      best.used = true;
      track = best.track;
    } else {
      track = {
        id: app.nextTrackId++,
        x: detection.x,
        y: detection.y,
        side: zoneFromX(detection.x),
        samples: [],
        intensity: 0,
        lastSeen: now,
      };
      app.tracks.set(track.id, track);
    }

    updateTrack(track, detection, now);
    detection.trackId = track.id;
  }
}

function updateTrack(track, detection, now) {
  const newSide = zoneFromX(detection.x);
  const sideChanged = newSide !== track.side;

  if (sideChanged) {
    track.samples = [];
    track.intensity *= 0.35;
    track.side = newSide;
  }

  track.x = detection.x;
  track.y = detection.y;
  track.lastSeen = now;
  track.samples.push({ t: now, x: detection.x, y: detection.y });
  track.samples = track.samples.filter((sample) => now - sample.t <= TRACK_WINDOW_MS);

  const rawIntensity = calculateFanningIntensity(track.samples, track.side);
  track.intensity = lerp(track.intensity, rawIntensity, 0.48);
}

function calculateFanningIntensity(samples, side) {
  if (side === "center" || samples.length < 3) return 0;

  let path = 0;
  let reversals = 0;
  let previousSign = 0;
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let validDurationMs = 0;

  for (let index = 1; index < samples.length; index += 1) {
    const previous = samples[index - 1];
    const current = samples[index];
    const dtMs = current.t - previous.t;
    if (dtMs <= 0 || dtMs > 180) continue;

    const dx = current.x - previous.x;
    const distance = Math.abs(dx);
    path += distance;
    validDurationMs += dtMs;
    minX = Math.min(minX, current.x, previous.x);
    maxX = Math.max(maxX, current.x, previous.x);

    if (distance > 0.004) {
      const sign = Math.sign(dx);
      if (previousSign !== 0 && sign !== previousSign) reversals += 1;
      previousSign = sign;
    }
  }

  if (validDurationMs < 120 || path < 0.025 || !Number.isFinite(minX)) return 0;

  const durationSeconds = validDurationMs / 1000;
  const meanSpeed = path / durationSeconds;
  const movementRange = maxX - minX;
  const speedScore = clamp((meanSpeed * app.sensitivity - 0.08) / 0.95, 0, 1);
  const rangeScore = clamp((movementRange - 0.018) / 0.15, 0, 1);
  const reversalScore = clamp(0.58 + reversals * 0.16, 0.58, 1);
  const sustainedScore = clamp(samples.length / 8, 0.45, 1);

  return clamp(speedScore * (0.35 + 0.65 * rangeScore) * reversalScore * sustainedScore, 0, 1);
}

function processHandResult(result, now) {
  const detections = (result?.landmarks ?? []).map((landmarks, index) => {
    const point = getDisplayPoint(landmarks);
    return { ...point, landmarks, index, trackId: null };
  });

  assignDetectionsToTracks(detections, now);
  app.lastHandResult = { result, detections, capturedAt: now };
}

function runHandInference(now) {
  if (
    app.mode !== "camera" ||
    !app.cameraRunning ||
    !app.handLandmarker ||
    elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    now - app.lastInferenceAt < INFERENCE_INTERVAL_MS ||
    elements.video.currentTime === app.lastVideoTime
  ) {
    return;
  }

  app.lastInferenceAt = now;
  app.lastVideoTime = elements.video.currentTime;

  try {
    const result = app.handLandmarker.detectForVideo(elements.video, now);
    processHandResult(result, now);
  } catch (error) {
    console.error("Hand inference failed", error);
    if (!app.inferenceErrorShown) {
      setCameraStatus("손 추적 처리 중 오류가 발생했습니다. 카메라를 다시 시작하거나 수동 모드를 사용하세요.", true);
      app.inferenceErrorShown = true;
    }
  }
}

function updateTrackDecay(now, dt) {
  for (const [id, track] of app.tracks) {
    const sinceSeen = now - track.lastSeen;
    if (sinceSeen > INFERENCE_INTERVAL_MS * 2.2) {
      track.intensity *= Math.exp(-dt / 0.26);
    }
    if (sinceSeen > TRACK_FORGET_MS) {
      app.tracks.delete(id);
    }
  }
}

function aggregateGestureEnergy() {
  let leftProduct = 1;
  let rightProduct = 1;

  for (const track of app.tracks.values()) {
    const intensity = clamp(track.intensity, 0, 1);
    if (track.side === "left") leftProduct *= 1 - intensity;
    if (track.side === "right") rightProduct *= 1 - intensity;
  }

  const rawLeft = 1 - leftProduct;
  const rawRight = 1 - rightProduct;
  app.leftEnergy = lerp(app.leftEnergy, rawLeft, 0.28);
  app.rightEnergy = lerp(app.rightEnergy, rawRight, 0.28);
  app.fanStrength = Math.max(app.leftEnergy, app.rightEnergy);

  const minimumActive = 0.045;
  if (app.leftEnergy > minimumActive && app.rightEnergy > minimumActive && Math.abs(app.leftEnergy - app.rightEnergy) < 0.06) {
    app.activeZone = "both";
  } else if (app.leftEnergy > app.rightEnergy && app.leftEnergy > minimumActive) {
    app.activeZone = "left";
  } else if (app.rightEnergy > minimumActive) {
    app.activeZone = "right";
  } else {
    app.activeZone = "none";
  }
}

function gestureToWindTarget() {
  const left = app.leftEnergy;
  const right = app.rightEnergy;
  const dominance = left - right;
  const total = left + right;

  if (total < 0.055) return NORMAL_TRADE_WIND;
  if (Math.abs(dominance) < 0.06) return NORMAL_TRADE_WIND;

  if (dominance > 0) {
    const normalized = clamp(dominance / Math.max(left, 0.001), 0, 1) * left;
    return clamp(0.06 + 0.94 * normalized, 0.06, 1);
  }

  const normalized = clamp(-dominance / Math.max(right, 0.001), 0, 1) * right;
  return clamp(-(Math.abs(NORMAL_TRADE_WIND) + (1 - Math.abs(NORMAL_TRADE_WIND)) * normalized), -1, NORMAL_TRADE_WIND);
}

function windToEnsoTarget(wind) {
  if (wind >= NORMAL_TRADE_WIND) {
    return clamp((wind - NORMAL_TRADE_WIND) / (1 - NORMAL_TRADE_WIND), 0, 1);
  }
  return clamp((wind - NORMAL_TRADE_WIND) / (NORMAL_TRADE_WIND + 1), -1, 0);
}

function updatePhysics(now, dt) {
  updateTrackDecay(now, dt);
  aggregateGestureEnergy();

  app.windTarget = app.mode === "manual" ? app.manualWind : gestureToWindTarget();
  app.wind = expSmoothing(app.wind, app.windTarget, dt, 0.18);
  app.ensoTarget = windToEnsoTarget(app.wind);
  app.enso = expSmoothing(app.enso, app.ensoTarget, dt, app.responseSeconds);
  app.enso = clamp(app.enso, -1, 1);

  const direction = Math.sign(app.wind || NORMAL_TRADE_WIND);
  app.animationPhase += dt * direction * (0.2 + Math.abs(app.wind) * 0.9);
}

function classifyEnso(enso) {
  if (enso >= 0.25) return { key: "elnino", label: "엘니뇨" };
  if (enso <= -0.25) return { key: "lanina", label: "라니냐" };
  return { key: "neutral", label: "중립" };
}

function deriveModel() {
  const enso = app.enso;
  const state = classifyEnso(enso);
  const seaLevelDifference = clamp(35 - 25 * enso, 8, 62);
  const eastSstAnomaly = 2.8 * enso;
  const westSst = 29.2 - 0.45 * enso;
  const eastSst = 23.5 + 3.0 * enso;
  const westThermocline = clamp(165 - 35 * enso, 125, 205);
  const eastThermocline = clamp(80 + 45 * enso, 32, 128);
  const upwelling = clamp(65 - 50 * enso, 10, 100);
  const westPressure = 1008 + 4 * enso;
  const eastPressure = 1012 - 4 * enso;
  const westRain = clamp(70 - 45 * enso, 20, 100);
  const eastRain = clamp(30 + 55 * enso, 5, 92);

  return {
    enso,
    state,
    wind: app.wind,
    seaLevelDifference,
    eastSstAnomaly,
    westSst,
    eastSst,
    westThermocline,
    eastThermocline,
    upwelling,
    westPressure,
    eastPressure,
    westRain,
    eastRain,
  };
}

function formatSigned(value, digits = 1) {
  if (Math.abs(value) < 0.05) return (0).toFixed(digits);
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function updateUi(model) {
  const { state, enso } = model;

  if (app.mode === "manual") {
    elements.activeZoneValue.textContent = "수동 조작";
    elements.fanStrengthValue.textContent = Math.round(Math.abs(app.manualWind) * 100);
  } else {
    const zoneText = {
      left: "왼쪽 · 서풍 편차",
      right: "오른쪽 · 무역풍",
      both: "양쪽 · 상쇄",
      none: app.cameraRunning ? "감지 없음" : "카메라 꺼짐",
    };
    elements.activeZoneValue.textContent = zoneText[app.activeZone];
    elements.fanStrengthValue.textContent = Math.round(app.fanStrength * 100);
  }

  elements.windValue.textContent = windDescription(model.wind);
  elements.ensoStateLabel.textContent = state.label;
  elements.ensoIndexLabel.textContent = `모형 지수 ${formatSigned(enso, 2)}`;
  elements.ensoBadge.className = `enso-badge enso-${state.key}`;

  elements.seaLevelValue.textContent = `서쪽이 약 ${Math.round(model.seaLevelDifference)} cm 높음`;
  elements.sstValue.textContent = `${formatSigned(model.eastSstAnomaly, 1)} ℃`;
  elements.thermoclineValue.textContent = `약 ${Math.round(model.eastThermocline)} m`;
  elements.upwellingValue.textContent = `${Math.round(model.upwelling)}%`;

  const westIsLowerPressure = model.westPressure < model.eastPressure;
  elements.pressureValue.textContent = westIsLowerPressure
    ? "서쪽 저기압 · 동쪽 고기압"
    : "서쪽 고기압 · 동쪽 저기압";

  const rainDifference = model.eastRain - model.westRain;
  elements.rainValue.textContent = rainDifference > 12 ? "중·동태평양" : rainDifference < -12 ? "서태평양" : "태평양 중앙부";

  if (state.key === "elnino") {
    elements.stateExplanation.textContent = "무역풍이 약해지거나 서풍 편차가 나타나 따뜻한 표층수가 동쪽으로 퍼집니다. 동태평양의 해수면과 수온 약층이 깊어지고, 용승이 약해져 표층 수온이 높아지며 저기압·대류·강수 중심도 동쪽으로 이동합니다.";
    elements.seaLevelHint.textContent = "동태평양 해수면이 높아져 동서 경사가 크게 완화됩니다.";
    elements.sstHint.textContent = "용승 약화와 따뜻한 물의 동진으로 평년보다 따뜻합니다.";
    elements.thermoclineHint.textContent = "약층이 깊어져 용승수가 표층을 냉각하기 어려워집니다.";
    elements.upwellingHint.textContent = "차가운 심층수가 올라오는 흐름이 약합니다.";
    elements.pressureHint.textContent = "서쪽 기압은 높아지고 동쪽 기압은 낮아지는 남방진동이 나타납니다.";
    elements.rainHint.textContent = "따뜻한 바다와 상승기류가 중앙·동태평양 쪽으로 이동합니다.";
    elements.chainWind.textContent = "무역풍 약화·서풍 편차";
    elements.chainSea.textContent = "동쪽 해수면 상승";
    elements.chainThermo.textContent = "동쪽 약층 깊어짐";
    elements.chainSst.textContent = "동쪽 수온 상승";
    elements.chainWeather.textContent = "기압·비 중심 동진";
  } else if (state.key === "lanina") {
    elements.stateExplanation.textContent = "무역풍이 평년보다 강해져 따뜻한 표층수가 서태평양에 더 많이 쌓입니다. 동태평양의 해수면은 더 낮아지고 수온 약층은 더 얕아지며, 강한 용승이 차가운 물을 올려 표층 수온을 낮춥니다. 대류와 강수는 서쪽에 더욱 집중됩니다.";
    elements.seaLevelHint.textContent = "서태평양에 물이 더 쌓여 동서 해수면 경사가 커집니다.";
    elements.sstHint.textContent = "강한 용승으로 평년보다 차가운 표층수가 넓어집니다.";
    elements.thermoclineHint.textContent = "차가운 심층수가 표층에 더 가까워집니다.";
    elements.upwellingHint.textContent = "차가운 물이 표층으로 강하게 공급됩니다.";
    elements.pressureHint.textContent = "서쪽 저기압과 동쪽 고기압의 차이가 커져 무역풍이 강화됩니다.";
    elements.rainHint.textContent = "상승기류와 비가 서태평양에 더욱 집중됩니다.";
    elements.chainWind.textContent = "무역풍 강화";
    elements.chainSea.textContent = "해수면 경사 증가";
    elements.chainThermo.textContent = "동쪽 약층 얕아짐";
    elements.chainSst.textContent = "동쪽 수온 하강";
    elements.chainWeather.textContent = "비는 서쪽에 집중";
  } else {
    elements.stateExplanation.textContent = "평상시 무역풍이 따뜻한 표층수를 서쪽으로 이동시켜 서태평양의 해수면과 수온 약층이 더 높고 깊게 나타납니다. 동태평양에서는 얕은 수온 약층과 용승 때문에 표층수가 상대적으로 차갑습니다.";
    elements.seaLevelHint.textContent = "따뜻한 표층수가 서쪽에 쌓여 기본적인 동서 경사가 생깁니다.";
    elements.sstHint.textContent = "평년과 비슷한 상태입니다.";
    elements.thermoclineHint.textContent = "차가운 물이 표층 가까이에 있어 용승 냉각이 가능합니다.";
    elements.upwellingHint.textContent = "동태평양의 평상시 용승이 유지됩니다.";
    elements.pressureHint.textContent = "동서 기압 차가 평상시 무역풍과 워커 순환을 유지합니다.";
    elements.rainHint.textContent = "따뜻한 서태평양에서 상승기류와 강수가 우세합니다.";
    elements.chainWind.textContent = "무역풍 보통";
    elements.chainSea.textContent = "해수면 경사 유지";
    elements.chainThermo.textContent = "동쪽 약층 얕음";
    elements.chainSst.textContent = "동쪽 표층수 차가움";
    elements.chainWeather.textContent = "비는 서쪽에 집중";
  }
}

function resizeCanvas(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height, dpr };
}

function videoLandmarkToCanvas(landmark, canvasWidth, canvasHeight) {
  const videoWidth = elements.video.videoWidth || 16;
  const videoHeight = elements.video.videoHeight || 9;
  const scale = Math.max(canvasWidth / videoWidth, canvasHeight / videoHeight);
  const drawnWidth = videoWidth * scale;
  const drawnHeight = videoHeight * scale;
  const offsetX = (canvasWidth - drawnWidth) / 2;
  const offsetY = (canvasHeight - drawnHeight) / 2;

  return {
    x: offsetX + (1 - landmark.x) * drawnWidth,
    y: offsetY + landmark.y * drawnHeight,
  };
}

function renderHands() {
  const { width, height } = resizeCanvas(elements.handCanvas, handContext);
  handContext.clearRect(0, 0, width, height);

  if (!app.cameraRunning || !app.lastHandResult?.result?.landmarks?.length) return;

  const result = app.lastHandResult.result;
  for (let handIndex = 0; handIndex < result.landmarks.length; handIndex += 1) {
    const landmarks = result.landmarks[handIndex];
    const displayPoint = getDisplayPoint(landmarks);
    const side = zoneFromX(displayPoint.x);
    const stroke = side === "left" ? "#ffad7f" : side === "right" ? "#77c7ff" : "#f3f7f8";

    handContext.lineWidth = Math.max(2, width / 260);
    handContext.lineCap = "round";
    handContext.lineJoin = "round";
    handContext.strokeStyle = stroke;
    handContext.fillStyle = stroke;
    handContext.shadowColor = "rgba(0,0,0,0.45)";
    handContext.shadowBlur = 4;

    for (const connection of app.handConnections) {
      const startIndex = connection.start ?? connection[0];
      const endIndex = connection.end ?? connection[1];
      const start = landmarks[startIndex];
      const end = landmarks[endIndex];
      if (!start || !end) continue;
      const p1 = videoLandmarkToCanvas(start, width, height);
      const p2 = videoLandmarkToCanvas(end, width, height);
      handContext.beginPath();
      handContext.moveTo(p1.x, p1.y);
      handContext.lineTo(p2.x, p2.y);
      handContext.stroke();
    }

    for (const landmark of landmarks) {
      const point = videoLandmarkToCanvas(landmark, width, height);
      handContext.beginPath();
      handContext.arc(point.x, point.y, Math.max(2.2, width / 180), 0, Math.PI * 2);
      handContext.fill();
    }

    handContext.shadowBlur = 0;
  }
}

function temperatureColor(temperature) {
  const t = clamp((temperature - 20) / 10, 0, 1);
  const stops = [
    { t: 0, c: [38, 105, 162] },
    { t: 0.35, c: [61, 169, 190] },
    { t: 0.6, c: [242, 191, 86] },
    { t: 1, c: [230, 91, 52] },
  ];

  let left = stops[0];
  let right = stops[stops.length - 1];
  for (let index = 1; index < stops.length; index += 1) {
    if (t <= stops[index].t) {
      left = stops[index - 1];
      right = stops[index];
      break;
    }
  }
  const local = (t - left.t) / Math.max(0.0001, right.t - left.t);
  const rgb = left.c.map((channel, index) => Math.round(lerp(channel, right.c[index], local)));
  return `rgb(${rgb.join(",")})`;
}

function drawArrow(context, x1, y1, x2, y2, options = {}) {
  const { color = "#ffffff", width = 3, head = 8, alpha = 1, dashed = false } = options;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  context.save();
  context.globalAlpha = alpha;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = width;
  context.lineCap = "round";
  if (dashed) context.setLineDash([5, 5]);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.setLineDash([]);
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
  context.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
  context.closePath();
  context.fill();
  context.restore();
}

function drawRoundedLabel(context, text, x, y, options = {}) {
  const {
    align = "center",
    background = "rgba(5, 42, 53, 0.72)",
    color = "#fff",
    font = "700 12px sans-serif",
    paddingX = 7,
    paddingY = 4,
  } = options;
  context.save();
  context.font = font;
  const metrics = context.measureText(text);
  const width = metrics.width + paddingX * 2;
  const height = 18 + paddingY;
  const left = align === "left" ? x : align === "right" ? x - width : x - width / 2;
  const top = y - height / 2;
  context.fillStyle = background;
  context.beginPath();
  context.roundRect(left, top, width, height, 6);
  context.fill();
  context.fillStyle = color;
  context.textAlign = "left";
  context.textBaseline = "middle";
  context.fillText(text, left + paddingX, y);
  context.restore();
}

function drawCloud(context, x, y, scale, rainIntensity, opacity = 1) {
  context.save();
  context.globalAlpha = opacity;
  context.fillStyle = "rgba(255,255,255,0.96)";
  context.strokeStyle = "rgba(78,108,119,0.35)";
  context.lineWidth = Math.max(1, scale * 0.055);

  const circles = [
    [0, 0, 0.36],
    [-0.32, 0.08, 0.26],
    [0.34, 0.1, 0.29],
    [0.05, -0.2, 0.3],
  ];
  for (const [dx, dy, radius] of circles) {
    context.beginPath();
    context.arc(x + dx * scale, y + dy * scale, radius * scale, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }

  const rainCount = Math.round(clamp(rainIntensity / 18, 0, 6));
  context.strokeStyle = "rgba(34,123,184,0.82)";
  context.lineWidth = Math.max(1.4, scale * 0.045);
  for (let index = 0; index < rainCount; index += 1) {
    const offset = (index - (rainCount - 1) / 2) * scale * 0.18;
    context.beginPath();
    context.moveTo(x + offset, y + scale * 0.38);
    context.lineTo(x + offset - scale * 0.06, y + scale * 0.67);
    context.stroke();
  }
  context.restore();
}

function drawLand(context, width, height, surfaceAtLeft, surfaceAtRight) {
  context.save();
  context.fillStyle = "#4f7d58";
  context.strokeStyle = "rgba(31,73,49,0.65)";
  context.lineWidth = 1.5;

  context.beginPath();
  context.moveTo(0, surfaceAtLeft - height * 0.025);
  context.lineTo(width * 0.035, surfaceAtLeft - height * 0.09);
  context.lineTo(width * 0.065, surfaceAtLeft - height * 0.045);
  context.lineTo(width * 0.09, surfaceAtLeft - height * 0.12);
  context.lineTo(width * 0.125, surfaceAtLeft - height * 0.035);
  context.lineTo(width * 0.16, surfaceAtLeft - height * 0.055);
  context.lineTo(width * 0.18, surfaceAtLeft + height * 0.035);
  context.lineTo(0, surfaceAtLeft + height * 0.065);
  context.closePath();
  context.fill();
  context.stroke();

  context.beginPath();
  context.moveTo(width, surfaceAtRight - height * 0.14);
  context.lineTo(width * 0.965, surfaceAtRight - height * 0.17);
  context.lineTo(width * 0.935, surfaceAtRight - height * 0.1);
  context.lineTo(width * 0.91, surfaceAtRight - height * 0.04);
  context.lineTo(width * 0.895, surfaceAtRight + height * 0.08);
  context.lineTo(width, surfaceAtRight + height * 0.1);
  context.closePath();
  context.fill();
  context.stroke();
  context.restore();
}

function drawOcean(model) {
  const { width, height } = resizeCanvas(elements.oceanCanvas, oceanContext);
  const context = oceanContext;
  context.clearRect(0, 0, width, height);

  const sky = context.createLinearGradient(0, 0, 0, height * 0.45);
  sky.addColorStop(0, "#b8e8f5");
  sky.addColorStop(1, "#eefbff");
  context.fillStyle = sky;
  context.fillRect(0, 0, width, height);

  const surfaceCenter = height * 0.405;
  const tilt = height * clamp(0.024 - 0.018 * model.enso, 0.004, 0.045);
  const surfaceY = (normalizedX) => surfaceCenter + tilt * (normalizedX - 0.5) * 2;
  const westSurface = surfaceY(0);
  const eastSurface = surfaceY(1);
  const oceanBottom = height * 0.94;

  const depthToY = (depthMeters, normalizedX) => {
    const available = oceanBottom - surfaceY(normalizedX);
    return surfaceY(normalizedX) + clamp(depthMeters / 240, 0, 0.92) * available;
  };
  const thermoclineY = (normalizedX) => {
    const smoothX = normalizedX * normalizedX * (3 - 2 * normalizedX);
    const depth = lerp(model.westThermocline, model.eastThermocline, smoothX);
    const curvature = Math.sin(normalizedX * Math.PI) * 5 * (1 - Math.abs(model.enso));
    return depthToY(depth, normalizedX) + curvature;
  };

  context.fillStyle = "#195f92";
  context.beginPath();
  context.moveTo(0, westSurface);
  for (let index = 1; index <= 100; index += 1) {
    const x = index / 100;
    context.lineTo(x * width, surfaceY(x));
  }
  context.lineTo(width, oceanBottom);
  context.lineTo(0, oceanBottom);
  context.closePath();
  context.fill();

  const deepGradient = context.createLinearGradient(0, surfaceCenter, 0, oceanBottom);
  deepGradient.addColorStop(0, "rgba(38,122,174,0.22)");
  deepGradient.addColorStop(1, "rgba(5,37,76,0.82)");
  context.fillStyle = deepGradient;
  context.fillRect(0, Math.min(westSurface, eastSurface), width, oceanBottom - Math.min(westSurface, eastSurface));

  const columns = Math.max(90, Math.round(width / 7));
  for (let index = 0; index < columns; index += 1) {
    const x0 = index / columns;
    const x1 = (index + 1) / columns;
    const xMid = (x0 + x1) / 2;
    const localSst = lerp(model.westSst, model.eastSst, Math.pow(xMid, 1.15));
    context.fillStyle = temperatureColor(localSst);
    const top = surfaceY(xMid);
    const bottom = thermoclineY(xMid);
    context.fillRect(x0 * width - 1, top, (x1 - x0) * width + 2, Math.max(1, bottom - top));
  }

  context.save();
  context.strokeStyle = "#fff1a6";
  context.lineWidth = Math.max(2.2, width / 360);
  context.shadowColor = "rgba(0,0,0,0.42)";
  context.shadowBlur = 4;
  context.beginPath();
  context.moveTo(0, thermoclineY(0));
  for (let index = 1; index <= 120; index += 1) {
    const x = index / 120;
    context.lineTo(x * width, thermoclineY(x));
  }
  context.stroke();
  context.restore();

  context.save();
  context.strokeStyle = "rgba(255,255,255,0.92)";
  context.lineWidth = Math.max(1.8, width / 430);
  context.beginPath();
  context.moveTo(0, westSurface);
  for (let index = 1; index <= 120; index += 1) {
    const x = index / 120;
    const wave = Math.sin(index * 0.85 + app.animationPhase * 7) * height * 0.0018;
    context.lineTo(x * width, surfaceY(x) + wave);
  }
  context.stroke();
  context.restore();

  drawLand(context, width, height, westSurface, eastSurface);

  const windMagnitude = Math.abs(model.wind);
  const windDirection = model.wind >= 0 ? 1 : -1;
  const arrowCount = Math.max(3, Math.round(4 + windMagnitude * 4));
  const arrowLength = width * (0.07 + windMagnitude * 0.055);
  const windY = height * 0.16;
  for (let index = 0; index < arrowCount; index += 1) {
    const base = ((index / arrowCount + app.animationPhase * 0.17) % 1 + 1) % 1;
    const x = windDirection > 0 ? base * width : (1 - base) * width;
    const x2 = x + windDirection * arrowLength;
    drawArrow(context, x, windY + (index % 2) * height * 0.065, x2, windY + (index % 2) * height * 0.065, {
      color: model.wind > 0 ? "#d4552d" : "#166fac",
      width: Math.max(2, width / 430),
      head: Math.max(7, width / 100),
      alpha: 0.35 + windMagnitude * 0.6,
    });
  }

  const windLabel = model.wind > 0.05 ? "서풍 편차: 서 → 동" : model.wind < -0.05 ? "무역풍: 동 → 서" : "바람 매우 약함";
  drawRoundedLabel(context, windLabel, width * 0.5, height * 0.075, {
    background: model.wind > 0.05 ? "rgba(170,61,28,0.78)" : "rgba(16,93,139,0.8)",
    font: `800 ${Math.max(10, width / 68)}px sans-serif`,
  });

  const upwellingAlpha = clamp(model.upwelling / 100, 0.12, 1);
  const upwellingX = width * 0.86;
  const upwellingTop = surfaceY(0.86) + height * 0.045;
  const upwellingBottom = Math.min(oceanBottom - height * 0.04, thermoclineY(0.86) + height * 0.18);
  const upwellingCount = Math.max(1, Math.round(model.upwelling / 28));
  for (let index = 0; index < upwellingCount; index += 1) {
    const offset = (index - (upwellingCount - 1) / 2) * width * 0.022;
    drawArrow(context, upwellingX + offset, upwellingBottom, upwellingX + offset, upwellingTop, {
      color: "#ccefff",
      width: Math.max(1.8, width / 500),
      head: Math.max(6, width / 120),
      alpha: upwellingAlpha,
    });
  }
  drawRoundedLabel(context, `용승 ${Math.round(model.upwelling)}%`, upwellingX, Math.min(oceanBottom - 18, upwellingBottom + 15), {
    background: "rgba(5,66,105,0.78)",
    font: `700 ${Math.max(9, width / 85)}px sans-serif`,
  });

  const cloudScale = Math.max(24, width / 18);
  drawCloud(context, width * 0.22, height * 0.25, cloudScale * (0.75 + model.westRain / 250), model.westRain, 0.55 + model.westRain / 220);
  drawCloud(context, width * 0.72, height * 0.25, cloudScale * (0.65 + model.eastRain / 250), model.eastRain, 0.5 + model.eastRain / 220);

  const westRising = model.westRain >= model.eastRain;
  drawArrow(context, width * 0.22, height * 0.36, width * 0.22, height * 0.22, {
    color: westRising ? "#cf522a" : "#527d8a",
    width: Math.max(2, width / 480),
    head: Math.max(6, width / 120),
    alpha: 0.55,
    dashed: !westRising,
  });
  drawArrow(context, width * 0.72, height * 0.22, width * 0.72, height * 0.36, {
    color: westRising ? "#527d8a" : "#cf522a",
    width: Math.max(2, width / 480),
    head: Math.max(6, width / 120),
    alpha: 0.55,
    dashed: westRising,
  });

  const westLow = model.westPressure < model.eastPressure;
  drawRoundedLabel(context, `${westLow ? "L" : "H"} ${Math.round(model.westPressure)} hPa`, width * 0.1, height * 0.12, {
    background: westLow ? "rgba(196,70,46,0.8)" : "rgba(45,100,164,0.8)",
    font: `800 ${Math.max(9, width / 78)}px sans-serif`,
  });
  drawRoundedLabel(context, `${westLow ? "H" : "L"} ${Math.round(model.eastPressure)} hPa`, width * 0.9, height * 0.12, {
    background: westLow ? "rgba(45,100,164,0.8)" : "rgba(196,70,46,0.8)",
    font: `800 ${Math.max(9, width / 78)}px sans-serif`,
  });

  drawRoundedLabel(context, "서태평양 · 인도네시아", width * 0.12, height * 0.47, {
    align: "left",
    font: `700 ${Math.max(9, width / 85)}px sans-serif`,
  });
  drawRoundedLabel(context, "동태평양 · 남아메리카", width * 0.88, height * 0.47, {
    align: "right",
    font: `700 ${Math.max(9, width / 85)}px sans-serif`,
  });

  drawRoundedLabel(context, `표층 ${model.westSst.toFixed(1)}℃`, width * 0.2, surfaceY(0.2) + height * 0.07, {
    background: "rgba(151,71,35,0.72)",
    font: `700 ${Math.max(9, width / 86)}px sans-serif`,
  });
  drawRoundedLabel(context, `표층 ${model.eastSst.toFixed(1)}℃`, width * 0.78, surfaceY(0.78) + height * 0.07, {
    background: "rgba(27,87,132,0.75)",
    font: `700 ${Math.max(9, width / 86)}px sans-serif`,
  });

  const thermoLabelX = width * 0.52;
  drawRoundedLabel(context, "수온 약층", thermoLabelX, thermoclineY(0.52) + 15, {
    background: "rgba(79,69,28,0.77)",
    font: `700 ${Math.max(9, width / 88)}px sans-serif`,
  });

  context.save();
  context.strokeStyle = "rgba(255,255,255,0.72)";
  context.lineWidth = 1.3;
  context.setLineDash([4, 4]);
  context.beginPath();
  context.moveTo(width * 0.055, westSurface);
  context.lineTo(width * 0.055, eastSurface);
  context.moveTo(width * 0.945, eastSurface);
  context.lineTo(width * 0.945, westSurface);
  context.stroke();
  context.restore();

  const seaLabel = model.enso > 0.25 ? "동서 해수면 경사 완화" : model.enso < -0.25 ? "동서 해수면 경사 증가" : "평상시 해수면 경사";
  drawRoundedLabel(context, seaLabel, width * 0.5, surfaceCenter + height * 0.018, {
    background: "rgba(7,64,83,0.72)",
    font: `700 ${Math.max(9, width / 88)}px sans-serif`,
  });
}

function recordCurrentState() {
  const model = deriveModel();
  app.records.push({
    number: app.records.length + 1,
    state: model.state.label,
    wind: windDescription(model.wind),
    sst: formatSigned(model.eastSstAnomaly, 1),
    thermocline: Math.round(model.eastThermocline),
    upwelling: Math.round(model.upwelling),
    index: model.enso.toFixed(2),
  });
  renderRecords();
}

function renderRecords() {
  elements.recordsBody.replaceChildren();

  if (app.records.length === 0) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.textContent = "아직 기록이 없습니다.";
    row.append(cell);
    elements.recordsBody.append(row);
  } else {
    for (const record of app.records) {
      const row = document.createElement("tr");
      const values = [
        record.number,
        `${record.state} (${record.index})`,
        record.wind,
        `${record.sst} ℃`,
        `${record.thermocline} m`,
        `${record.upwelling}%`,
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.recordsBody.append(row);
    }
  }

  const hasRecords = app.records.length > 0;
  elements.downloadCsvBtn.disabled = !hasRecords;
  elements.clearRecordsBtn.disabled = !hasRecords;
}

function csvEscape(value) {
  const text = String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function downloadRecordsCsv() {
  if (app.records.length === 0) return;
  const rows = [
    ["번호", "ENSO 상태", "모형 지수", "바람", "동태평양 수온 편차(℃)", "동태평양 수온 약층 깊이(m)", "용승 세기(%)"],
    ...app.records.map((record) => [
      record.number,
      record.state,
      record.index,
      record.wind,
      record.sst,
      record.thermocline,
      record.upwelling,
    ]),
  ];
  const csv = `\uFEFF${rows.map((row) => row.map(csvEscape).join(",")).join("\r\n")}`;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `enso-observations-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function clearRecords() {
  app.records = [];
  renderRecords();
}

function resetNeutral() {
  app.tracks.clear();
  app.leftEnergy = 0;
  app.rightEnergy = 0;
  app.fanStrength = 0;
  app.activeZone = "none";
  app.manualWind = NORMAL_TRADE_WIND;
  app.windTarget = NORMAL_TRADE_WIND;
  app.wind = NORMAL_TRADE_WIND;
  app.ensoTarget = 0;
  app.enso = 0;
  elements.manualWindRange.value = String(NORMAL_TRADE_WIND);
  updateControlOutputs();
}

function applyPreset(preset) {
  const values = {
    lanina: -0.9,
    neutral: NORMAL_TRADE_WIND,
    elnino: 0.78,
  };
  setMode("manual");
  app.manualWind = values[preset] ?? NORMAL_TRADE_WIND;
  elements.manualWindRange.value = String(app.manualWind);
  updateControlOutputs();
}

function animationLoop(now) {
  const dt = clamp((now - app.lastFrameAt) / 1000, 0.001, 0.08);
  app.lastFrameAt = now;

  runHandInference(now);
  updatePhysics(now, dt);
  const model = deriveModel();
  drawOcean(model);
  renderHands();

  if (now - app.lastUiUpdateAt > 85) {
    updateUi(model);
    app.lastUiUpdateAt = now;
  }

  requestAnimationFrame(animationLoop);
}

function bindEvents() {
  elements.startCameraBtn.addEventListener("click", startCamera);
  elements.stopCameraBtn.addEventListener("click", () => stopCamera());
  elements.resetBtn.addEventListener("click", resetNeutral);
  elements.cameraModeBtn.addEventListener("click", () => setMode("camera"));
  elements.manualModeBtn.addEventListener("click", () => setMode("manual"));

  elements.sensitivityRange.addEventListener("input", (event) => {
    app.sensitivity = Number(event.currentTarget.value);
    updateControlOutputs();
  });

  elements.responseRange.addEventListener("input", (event) => {
    app.responseSeconds = Number(event.currentTarget.value);
    updateControlOutputs();
  });

  elements.manualWindRange.addEventListener("input", (event) => {
    app.manualWind = Number(event.currentTarget.value);
    updateControlOutputs();
  });

  $$(".preset-btn").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });

  elements.recordBtn.addEventListener("click", recordCurrentState);
  elements.downloadCsvBtn.addEventListener("click", downloadRecordsCsv);
  elements.clearRecordsBtn.addEventListener("click", clearRecords);

  window.addEventListener("beforeunload", () => stopCamera({ preserveMessage: true }));
  document.addEventListener("visibilitychange", () => {
    app.lastFrameAt = performance.now();
  });
}

function initialize() {
  bindEvents();
  updateControlOutputs();
  setMode("camera");
  renderRecords();
  const initialModel = deriveModel();
  updateUi(initialModel);
  drawOcean(initialModel);
  requestAnimationFrame(animationLoop);
}

initialize();
