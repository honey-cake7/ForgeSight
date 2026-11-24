'use strict';

const byId = id => document.getElementById(id);
let pluginState = null;
let pluginChanging = false;
function renderPlugin(value) {
  pluginState = value;
  byId('plugin-control').hidden = !value;
  byId('active-plugin-title').hidden = !value;
  if (!value) return;
  const select = byId('active-plugin');
  if (Array.from(select.options).map(o => o.value).join(',') !== value.plugins.map(p => p.id).join(',')) {
    select.options.length = 0;
    for (const plugin of value.plugins) select.add(new Option(plugin.label, plugin.id));
  }
  for (const option of select.options) option.disabled = !value.plugins.find(p => p.id === option.value).ready;
  select.value = value.active_plugin;
  select.disabled = busy || pluginChanging;
  byId('active-plugin-title').textContent = 'QUALITY CONTROL / ' + value.label.toUpperCase();
  byId('benchmark-part-label').textContent = 'This is an original-size MVTec ' + value.active_plugin + ' PNG.';
}
byId('active-plugin').addEventListener('change', async () => {
  if (!pluginState || pluginChanging) return;
  const selected = byId('active-plugin').value;
  pluginChanging = commandPending = true;
  commandEpoch += 1;
  byId('active-plugin').disabled = true;
  updateButton();
  try {
    const result = await requestJSON('/api/plugin/select', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({plugin_id: selected})});
    pluginState.active_plugin = result.active_plugin;
    pluginState.label = result.label;
    byId('plugin-status').textContent = 'DEMO / ' + result.switch_ms.toFixed(1) + ' ms / labels provisional';
    currentLatest = null;
    byId('diagnostic-file').value = '';
    byId('mvtec-benchmark').checked = false;
    resetResult();
  } catch {
    byId('plugin-status').textContent = 'Change unavailable; current recipe retained.';
  } finally {
    pluginChanging = commandPending = false;
    renderPlugin(pluginState);
    await refreshState();
  }
});

/* ---------- Theme (dark by default, light on request) ---------- */
let theme = 'dark';
try { theme = localStorage.getItem('forgesight-theme') === 'light' ? 'light' : 'dark'; } catch { /* storage blocked */ }
function applyTheme(next) {
  theme = next;
  document.documentElement.dataset.theme = next;
  const label = next === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
  byId('theme-toggle').setAttribute('aria-label', label);
  byId('theme-toggle').title = label;
}
applyTheme(theme);
byId('theme-toggle').addEventListener('click', () => {
  applyTheme(theme === 'dark' ? 'light' : 'dark');
  try { localStorage.setItem('forgesight-theme', theme); } catch { /* storage blocked */ }
});

const preview = byId('preview');
const overlay = byId('overlay');
const reviewSource = byId('review-source');
const reviewOverlay = byId('review-overlay');
let priorInspection = null;
let connected = false;
let commandPending = false;
let commandError = null;
let commandEpoch = 0;
let refreshing = false;
let adviceState = null;
let previewPending = false;
let previewTimer = null;
let savedView = 'overlay';
let mediaAvailable = false;
let scoreMapUrl = null;
let scoreController = null;
let scoreRequest = null;
let scoreValues = null;
let scoreError = '';
let scoreRange = [];
const drawnScores = new Map();
let reviewInspection = null;
let adviceTextKey = '';
let reviewSignature = '';
let cameraReady = false;
let cameraRunning = false;
let groundedAutoAdvice = false;
let cameraControlPending = false;
let busy = false;
let currentLatest = null;
const requestedInspection = new URLSearchParams(location.search).get('inspection');
const reportInspection = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(requestedInspection || '')
  ? requestedInspection : null;
let currentView = 'inspect';
let currentPane = 'camera';
// After the operator starts an inspection, jump to Results once the new saved
// result arrives (see refreshState).
let awaitingUntil = 0;
let awaitingBaseline = null;
const regionNote = byId('response-regions');

const savedModes = ['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC',
  'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'];

/* ---------- Routing: one view at a time; Inspect has Camera / Upload tabs ---------- */
const VIEWS = {
  inspect: {title: 'New inspection', intro: ''},
  results: {title: 'Inspection results', intro: 'Compare the evidence, check the advice, and record your finding.'},
  review: {title: 'Operator review', intro: ''},
  history: {title: 'History', intro: ''}
};

function parseHash() {
  const match = location.hash.match(/^#\/(\w+)(?:\/(\w+))?/);
  if (!match || !Object.prototype.hasOwnProperty.call(VIEWS, match[1])) return null;
  const pane = match[1] === 'inspect' && ['camera', 'upload'].includes(match[2]) ? match[2] : null;
  return {view: match[1], pane};
}

function setPane(pane) {
  currentPane = pane;
  for (const tab of document.querySelectorAll('.tab')) {
    const on = tab.dataset.pane === pane;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
  }
  for (const section of document.querySelectorAll('.pane')) {
    section.hidden = section.dataset.pane !== pane;
  }
}

function showView(target, moveFocus) {
  currentView = target.view;
  if (target.view === 'inspect') setPane(target.pane || currentPane);
  for (const section of document.querySelectorAll('.view')) {
    section.hidden = section.dataset.view !== target.view;
  }
  for (const link of document.querySelectorAll('[data-view-link]')) {
    if (link.dataset.viewLink === target.view) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  const info = VIEWS[target.view];
  byId('view-title').textContent = info.title;
  byId('view-intro').textContent = info.intro;
  byId('view-intro').hidden = !info.intro;
  document.title = info.title + ' · ForgeSight';
  window.scrollTo(0, 0);
  if (target.view === 'history' && !reportInspection) loadHistory();
  if (moveFocus) byId('view-title').focus({preventScroll: true});
}

window.addEventListener('hashchange', () => {
  const target = parseHash();
  if (target) showView(target, true);
});

const tabs = [...document.querySelectorAll('.tab')];
for (const tab of tabs) {
  tab.addEventListener('click', () => {
    setPane(tab.dataset.pane);
    history.replaceState(null, '', '#/inspect/' + tab.dataset.pane);
  });
  tab.addEventListener('keydown', event => {
    const keys = {ArrowLeft: -1, ArrowRight: 1, Home: -tabs.length, End: tabs.length};
    if (!(event.key in keys)) return;
    event.preventDefault();
    const index = tabs.indexOf(tab);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (index + keys[event.key] + tabs.length) % tabs.length;
    tabs[next].focus();
    tabs[next].click();
  });
}

function updateButton() {
  const locked = !connected || commandPending || busy;
  for (const id of ['show-overlay', 'show-source', 'show-compare']) {
    byId(id).disabled = !mediaAvailable;
  }
  for (const id of ['show-heatmap', 'show-contours']) {
    byId(id).disabled = locked || !mediaAvailable || adviceState?.status === 'PENDING';
  }
  byId('open-review').disabled = locked || currentLatest?.status !== 'ANALYZED';
  byId('camera-start').disabled = locked || cameraControlPending || cameraRunning;
  byId('camera-stop').disabled = locked || cameraControlPending || !cameraRunning;
  byId('inspect').disabled = locked || !cameraReady || preview.hidden;
  byId('diagnostic-submit').disabled = locked || !byId('diagnostic-file').files.length;
  byId('review-submit').disabled = locked || currentLatest?.status !== 'ANALYZED' ||
    !/^[A-Za-z0-9 _.-]{1,64}$/.test(byId('review-actor').value) ||
    !byId('review-outcome').value || byId('review-reason').value.trim().length < 3;
  const eligible = currentLatest?.status === 'ANALYZED' &&
    currentLatest?.candidate === 'ANOMALY_CANDIDATE' && savedModes.includes(currentLatest?.source_mode);
  // A failed or unavailable attempt may be retried; the server preserves the
  // earlier report aside and starts a fresh attempt.
  byId('advice-request').hidden = groundedAutoAdvice;
  byId('advice-request').disabled = groundedAutoAdvice || locked || !eligible ||
    !['NOT_REQUESTED', 'ADVICE_UNAVAILABLE'].includes(adviceState?.status);
  // The second advisory button has been removed; auto-routing is handled
  // server-side when the UI sends mode "auto".
  byId('inspect').querySelector('.button-label').textContent = busy ? 'Inspection in progress…' : 'Analyze';
  byId('diagnostic-submit').querySelector('.button-label').textContent = busy ? 'Inspection in progress…' : 'Analyze uploaded PNG';
  for (const form of ['capture-form', 'upload-form', 'review-form']) {
    byId(form).setAttribute('aria-busy', String(commandPending || busy));
    for (const input of byId(form).querySelectorAll('input, select, textarea')) {
      input.disabled = commandPending || busy;
    }
  }
}

async function requestJSON(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.method ? 30000 : 5000);
  try {
    const response = await fetch(path, {cache: 'no-store', ...options, signal: controller.signal});
    const text = await response.text();
    let result = {};
    try { result = JSON.parse(text); } catch { if (text) result = {error: text}; }
    if (!response.ok) throw new Error(
      (result.error || result.status || 'Station request failed')
      + (result.support_id ? ' [support ' + result.support_id + ']' : ''));
    return result;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Station did not respond in time. Check history before retrying.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function cameraStatus(label, message) {
  byId('camera-status').textContent = label;
  byId('camera-status').className = 'status ' + label.toLowerCase();
  byId('camera-detail').textContent = message;
  if (label !== 'LIVE') {
    preview.hidden = true;
    byId('preview-empty').textContent = message;
    byId('preview-empty').hidden = false;
  }
}

function stopPreview() {
  preview.removeAttribute('src');
  preview.hidden = true;
  previewPending = false;
  clearTimeout(previewTimer);
}

async function refreshState() {
  if (refreshing || commandPending || cameraControlPending) return;
  refreshing = true;
  const epoch = commandEpoch;
  try {
    const state = await requestJSON('/api/state');
    groundedAutoAdvice = state.grounded_auto_advice === true;
    // A state request begun before a user action must not undo its UI lock.
    if (epoch !== commandEpoch) return;
    connected = true;
    byId('connection-status').textContent = 'Station connected';
    byId('connection-status').className = 'connection ready';
    byId('connection-error').hidden = true;
    cameraReady = !!state.camera.ready;
    cameraRunning = state.camera.running !== false;
    busy = !!state.busy;
    renderPlugin(state.plugin || null);
    if (currentLatest?.inspection_id !== state.latest?.inspection_id) {
      byId('review-outcome').value = '';
      byId('review-reason').value = '';
    }
    currentLatest = state.latest;
    adviceState = state.advice;
    if (!cameraRunning) {
      stopPreview();
      cameraStatus('STOPPED', state.camera.message || 'Camera stopped by operator.');
    } else if (!cameraReady) {
      stopPreview();
      const starting = state.camera.message === 'Starting camera';
      cameraStatus(starting ? 'CONNECTING' : 'ERROR', state.camera.message || 'Camera unavailable. Check the USB connection.');
    } else if (preview.hidden && !previewPending) {
      cameraStatus('CONNECTING', 'Loading a live camera frame…');
    }
    if (state.latest) renderLatest(state.latest, state.busy);
    else resetResult();
    // The result lives in the Results view: bring the operator there once a
    // newly started inspection has finished.
    if (Date.now() < awaitingUntil && !state.busy && state.latest &&
        state.latest.status !== 'PROCESSING' && state.latest.inspection_id !== awaitingBaseline) {
      awaitingUntil = 0;
      if (currentView !== 'results') location.hash = '#/results';
    }
    if (['Waiting for camera.', 'Camera unavailable. You can still upload a saved PNG.',
      'Position the part in the camera view and select Analyze.'].includes(byId('request-status').textContent)) {
      byId('request-status').textContent = cameraReady
        ? 'Position the part in the camera view and select Analyze.'
        : 'Camera unavailable. You can still upload a saved PNG.';
    }
    renderReviews(state.manual_reviews || [], state.latest);
    renderAdvice(state.advice);
    if (commandError && commandError.inspectionId === currentLatest?.inspection_id) {
      byId(commandError.target).textContent = commandError.message;
    }
  } catch (error) {
    if (epoch !== commandEpoch) return;
    connected = false;
    cameraReady = false;
    cameraRunning = false;
    stopPreview();
    cameraStatus('ERROR', 'Camera status unavailable while the station is disconnected.');
    byId('connection-status').textContent = 'Station disconnected';
    byId('connection-status').className = 'connection error';
    byId('connection-error').textContent = 'Connection lost. Displayed results may be out of date. ' + error.message;
    byId('connection-error').hidden = false;
  } finally {
    refreshing = false;
    updateButton();
  }
}

function resetResult() {
  priorInspection = null;
  byId('result-content').hidden = true;
  byId('result-empty').hidden = false;
  byId('result-empty').textContent = 'The saved decision and next action will appear here after analysis.';
  byId('result-part').textContent = 'No saved result selected';
  byId('inspection-status').textContent = 'AWAITING IMAGE';
  byId('review-target').textContent = 'Inspect an image first, then compare its original and response map.';
  clearMedia('No image inspected yet. Capture a camera view or upload a PNG.');
}

function clearMedia(message) {
  mediaAvailable = false;
  if (scoreController) scoreController.abort();
  scoreMapUrl = scoreController = scoreRequest = scoreValues = null;
  scoreError = '';
  drawnScores.clear();
  byId('score-view').hidden = true;
  byId('score-legend').hidden = true;
  byId('compare-split').hidden = true;
  byId('compare-divider').hidden = true;
  for (const img of [overlay, byId('source-image')]) {
    img.hidden = true;
    img.removeAttribute('src');
    img.dataset.loaded = '';
  }
  for (const img of [reviewSource, reviewOverlay]) {
    img.hidden = true;
    img.removeAttribute('src');
  }
  byId('overlay-empty').textContent = message;
  byId('overlay-empty').hidden = false;
  byId('source-link').hidden = true;
  byId('evidence-caption').textContent = 'Latest saved inspection · EfficientAD';
}


function selectView(view) {
  savedView = view;
  const controls = {
    overlay: 'show-overlay', source: 'show-source', compare: 'show-compare',
    heatmap: 'show-heatmap', contours: 'show-contours'
  };
  for (const [key, id] of Object.entries(controls)) {
    byId(id).setAttribute('aria-pressed', String(key === view));
  }
  const score = view === 'heatmap' || view === 'contours';
  const compare = view === 'compare';
  const original = byId('source-image');
  const canvas = byId('score-view');
  const shell = byId('saved-image-shell');
  const empty = byId('overlay-empty');
  shell.classList.toggle('comparing', compare);
  overlay.hidden = !mediaAvailable || (!compare && view !== 'overlay') || overlay.dataset.loaded !== 'yes';
  original.hidden = !mediaAvailable || (!compare && view !== 'source') || original.dataset.loaded !== 'yes';
  const comparable = compare && !overlay.hidden && !original.hidden;
  byId('compare-divider').hidden = !comparable;
  byId('compare-split').hidden = !comparable;
  canvas.hidden = !mediaAvailable || !score || !drawnScores.has(view);
  byId('score-legend').hidden = canvas.hidden;
  if (!mediaAvailable) return;
  byId('evidence-view-help').textContent = {
    overlay: 'Highlighted pixels exceed the saved response threshold.',
    source: 'Original image for this inspection.',
    compare: 'Drag the divider or use the arrow keys to compare.',
    heatmap: 'Color indicates anomaly response, separate from defect severity.',
    contours: 'Lines separate bands of anomaly response.'
  }[view];
  if (score) {
    empty.hidden = !canvas.hidden;
    empty.textContent = scoreError || 'Loading saved anomaly response…';
    if (!canvas.hidden) {
      canvas.getContext('2d').putImageData(drawnScores.get(view), 0, 0);
      canvas.setAttribute('aria-label', view === 'heatmap' ? 'Anomaly heatmap' : 'Anomaly contours');
      byId('score-legend').dataset.palette = view;
      byId('score-min').textContent = scoreRange[0].toPrecision(3);
      byId('score-max').textContent = scoreRange[1].toPrecision(3);
    }
    return;
  }
  const failed = (compare && [overlay, original].some(img => img.dataset.loaded === 'error'))
    || (view === 'overlay' ? overlay : original).dataset.loaded === 'error';
  empty.hidden = compare ? comparable : view === 'overlay' ? !overlay.hidden : !original.hidden;
  empty.textContent = failed ? 'Saved image could not load. The inspection record is retained.' : 'Loading saved image…';
}

function decodeScoreMap(buffer) {
  const bytes = new Uint8Array(buffer);
  const file = new DataView(buffer);
  if (bytes.length < 10 || bytes[0] !== 147 || String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error('Invalid stored anomaly map.');
  }
  const version = bytes[6];
  if (![1, 2, 3].includes(version) || (version !== 1 && bytes.length < 12)) {
    throw new Error('Unsupported stored anomaly map.');
  }
  const offset = version === 1 ? 10 : 12;
  const size = version === 1 ? file.getUint16(8, true) : file.getUint32(8, true);
  const data = offset + size;
  if (size > 4096 || data > bytes.length) throw new Error('Invalid stored anomaly map.');
  const header = new TextDecoder().decode(bytes.subarray(offset, data));
  const shape = header.match(/['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)/);
  const format = header.match(/['"]descr['"]\s*:\s*['"]([<>])f([48])['"]/);
  if (!shape || !format || !/['"]fortran_order['"]\s*:\s*False/.test(header)) {
    throw new Error('Unsupported anomaly map format.');
  }
  const height = Number(shape[1]), width = Number(shape[2]), stride = Number(format[2]);
  if (width < 1 || height < 1 || width > 1024 || height > 1024
      || data + width * height * stride !== bytes.length) throw new Error('Invalid anomaly map dimensions.');
  const little = format[1] === '<';
  const read = stride === 4 ? file.getFloat32.bind(file) : file.getFloat64.bind(file);
  const values = new Float32Array(width * height);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const value = read(data + i * stride, little);
    if (!Number.isFinite(value) || Math.abs(value) > 1e10) throw new Error('Non-finite anomaly response.');
    values[i] = value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return {values, width, height, min, max};
}

function renderScoreMap(map, view) {
  const scale = Math.min(1, 512 / Math.max(map.width, map.height));
  const width = Math.max(1, Math.round(map.width * scale));
  const height = Math.max(1, Math.round(map.height * scale));
  const canvas = byId('score-view');
  const image = new ImageData(width, height);
  const range = map.max - map.min || 1;
  const stops = [[33, 26, 86], [38, 99, 184], [23, 170, 187], [101, 204, 96], [247, 222, 76], [239, 114, 51], [171, 30, 41]];
  const bands = [[255, 250, 203], [254, 226, 148], [254, 189, 104], [250, 142, 72], [229, 92, 53], [202, 52, 42], [157, 23, 37], [107, 17, 31]];
  const levels = view === 'contours' ? new Uint8Array(width * height) : null;
  for (let y = 0; y < height; y++) {
    const row = Math.floor(y * map.height / height) * map.width;
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      const t = Math.max(0, Math.min(1, (map.values[row + Math.floor(x * map.width / width)] - map.min) / range));
      const offset = pixel * 4;
      if (levels) {
        const band = Math.min(bands.length - 1, Math.floor(t * bands.length));
        levels[pixel] = band;
        const edge = (x && levels[pixel - 1] !== band) || (y && levels[pixel - width] !== band);
        image.data.set(edge ? [95, 56, 41] : bands[band], offset);
      } else {
        const position = t * (stops.length - 1);
        const low = Math.min(stops.length - 2, Math.floor(position));
        const fraction = position - low;
        for (let c = 0; c < 3; c++) {
          image.data[offset + c] = stops[low][c] + (stops[low + 1][c] - stops[low][c]) * fraction;
        }
      }
      image.data[offset + 3] = 255;
    }
  }
  canvas.width = width;
  canvas.height = height;
  scoreRange = [map.min, map.max];
  return image;
}

async function activateScoreView(view) {
  if (!mediaAvailable || busy || commandPending || adviceState?.status === 'PENDING') return;
  const url = scoreMapUrl;
  scoreError = '';
  selectView(view);
  if (drawnScores.has(view)) return;
  if (!scoreRequest) {
    const controller = new AbortController();
    scoreController = controller;
    const timer = setTimeout(() => controller.abort(), 5000);
    scoreRequest = fetch(url, {signal: controller.signal, cache: 'no-store'}).then(response => {
      if (!response.ok) throw new Error('The saved anomaly map is unavailable for this inspection.');
      return response.arrayBuffer();
    }).finally(() => clearTimeout(timer));
  }
  const request = scoreRequest;
  try {
    const bytes = await request;
    if (url !== scoreMapUrl || request !== scoreRequest || savedView !== view) return;
    if (busy || commandPending || adviceState?.status === 'PENDING') {
      selectView('overlay');
      return;
    }
    scoreValues = scoreValues || decodeScoreMap(bytes);
    drawnScores.set(view, renderScoreMap(scoreValues, view));
    selectView(view);
  } catch (error) {
    if (url !== scoreMapUrl || request !== scoreRequest) return;
    scoreRequest = null;
    scoreError = error.name === 'AbortError' ? 'Loading the saved anomaly map timed out.' : error.message;
    if (savedView === view) selectView(view);
  }
}

for (const img of [overlay, byId('source-image')]) {
  img.addEventListener('load', () => {
    if (!mediaAvailable) return;
    img.dataset.loaded = 'yes';
    if (img.id === 'source-image') byId('source-link').hidden = false;
    selectView(savedView);
  });
  img.addEventListener('error', () => {
    if (!mediaAvailable) return;
    img.dataset.loaded = 'error';
    selectView(savedView);
  });
}
for (const img of [reviewSource, reviewOverlay]) {
  img.addEventListener('error', () => { img.hidden = true; });
}
byId('show-overlay').addEventListener('click', () => selectView('overlay'));
byId('show-source').addEventListener('click', () => selectView('source'));
byId('show-compare').addEventListener('click', () => selectView('compare'));
byId('show-heatmap').addEventListener('click', () => activateScoreView('heatmap'));
byId('show-contours').addEventListener('click', () => activateScoreView('contours'));
byId('compare-split').addEventListener('input', event => {
  byId('saved-image-shell').style.setProperty('--split', event.target.value + '%');
});


function renderAdvice(advice) {
  const status = advice?.status || 'NOT_CONNECTED';
  const labels = {
    NOT_CONNECTED: 'On-device Llama runtime is unavailable.',
    NOT_ELIGIBLE: groundedAutoAdvice
      ? 'An eligible flagged inspection starts advice automatically.'
      : 'Advice is available after a flagged saved-image result.',
    LLAMA_ADVICE: 'Llama-selected check · verify before acting.',
    CHECKLIST_FALLBACK: 'Checklist fallback · verify before acting.',
    ADVICE_BUSY: 'Llama is processing another inspection. The vision result has been saved.',
    NOT_REQUESTED: 'Flagged image saved. You can request a separate Llama note.',
    PENDING: 'Llama is generating on this Jetson through ExecuTorch…',
    ADVICE_UNAVAILABLE: 'Llama advice was unavailable. The inspection grade is unchanged. You can try again.',
    GENERATED_REQUIRES_HUMAN_REVIEW: 'Generated operator note · human review required.'
  };
  byId('advice-status').textContent = status === 'GENERATED_REQUIRES_HUMAN_REVIEW' &&
    advice?.advisory?.note_origin === 'LLAMA_SELECTED_REVIEWED_CHECK'
    ? 'Llama-selected check · human review required.'
    : labels[status] || 'Llama advice status is unknown.';
  const runtime = advice?.llama_runtime === 'PERSISTENT_WORKER'
    ? 'This request used the warm Llama worker.'
    : advice?.llama_runtime === 'NOT_REQUESTED'
    ? 'No Llama generation was requested; a checklist may be shown.'
    : advice?.llama_runtime === 'PERSISTENT_WORKER_UNAVAILABLE'
    ? 'The warm Llama request was unavailable; a checklist may be shown.'
    : (advice?.llama_runtime || '').includes('COLD')
    ? 'Cold process: the model loads for each note.' : 'No runtime mode reported.';
  byId('advice-runtime').textContent = runtime + (advice?.advice_worker_boot_error
    ? ' The persistent worker did not start; the slower cold path is available.' : '') +
    (typeof advice?.advice_worker_available_now === 'boolean'
      ? (advice.advice_worker_available_now ? ' Worker now: ready.' : ' Worker now: unavailable.')
      : '');
  const note = advice?.advisory?.note;
  byId('advice-note').hidden = !note ||
    !['GENERATED_REQUIRES_HUMAN_REVIEW', 'LLAMA_ADVICE', 'CHECKLIST_FALLBACK'].includes(status);
  const adviceKey = note || '';
  if (adviceTextKey !== adviceKey) {
    adviceTextKey = adviceKey;
    byId('advice-note').replaceChildren();
    const pieces = adviceKey.split(/\s+(?=Vision flagged|The camera result)|\r?\n+/).filter(Boolean);
    pieces.forEach((piece, index) => {
      const entry = document.createElement('li');
      if (pieces.length === 2 && /^(Vision flagged|The camera result)/.test(pieces[1])) {
        const label = document.createElement('strong');
        label.textContent = index === 0 ? 'Recommended Action (Verified SOP)' : 'Visual Finding & Disposition';
        entry.append(label);
      }
      entry.append(document.createTextNode(piece));
      byId('advice-note').append(entry);
    });
  }
  const inspected = !busy && currentLatest?.status === 'ANALYZED';
  const ready = inspected && !byId('advice-note').hidden;
  if (ready) reviewInspection = currentLatest.inspection_id;
  const reviewOpen = inspected && reviewInspection === currentLatest.inspection_id;
  byId('review-panel').hidden = !reviewOpen;
  byId('review-empty').hidden = reviewOpen;
  byId('advice-request').querySelector('.button-label').textContent = status === 'PENDING' ? 'Generating advice…' : 'Generate Llama advice';
}

function renderLatest(item, processing) {
  if (processing || item.status === 'PROCESSING') {
    priorInspection = null;
    byId('result-content').hidden = true;
    byId('result-empty').hidden = false;
    byId('result-empty').textContent = 'Analyzing and saving this inspection…';
    byId('result-part').textContent = item.part_id || 'New inspection';
    byId('inspection-status').textContent = 'PROCESSING';
    clearMedia('Analyzing image… The new response map will appear when it is saved.');
    if (['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC', 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'].includes(item.source_mode)) byId('diagnostic-status').textContent = 'Analyzing image…';
    else byId('request-status').textContent = 'Analyzing captured frame… camera preview remains live.';
    return;
  }
  const key = JSON.stringify([item.inspection_id, item.part_id, item.status, item.error]);
  if (key === priorInspection) return;
  priorInspection = key;
  byId('result-part').textContent = item.part_id || 'Inspection';
  byId('inspection-status').textContent = item.status === 'ANALYZED' ? 'SAVED' :
    'ANALYSIS ERROR' + (item.support_id ? ' · Support ID ' + item.support_id : '');
  byId('review-target').textContent = item.status === 'ANALYZED'
    ? 'Reviewing ' + item.part_id + '. Compare the original and response map, then record your observation.'
    : 'Analysis failed. A completed result is required for review.';
  const statusText = item.status === 'ANALYZED' ? 'Result saved locally.' :
    (item.error?.message || 'Analysis failed.') + (item.support_id ? ' Support ID ' + item.support_id + '.' : '');
  byId('review-status').textContent = item.status === 'ANALYZED'
    ? 'Compare the saved original and overlay, then record your observation.'
    : 'Manual review is available after a completed analysis.';
  if (['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC', 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD', 'USER_DECLARED_MVTEC_SCREW_UPLOAD'].includes(item.source_mode)) byId('diagnostic-status').textContent = statusText;
  else byId('request-status').textContent = statusText;
  byId('result-empty').hidden = true;
  byId('result-content').hidden = false;
  const grade = byId('grade');
  const benchmark = ['USER_DECLARED_MVTEC_METAL_NUT_UPLOAD', 'USER_DECLARED_MVTEC_SCREW_UPLOAD'].includes(item.source_mode);
  grade.textContent = item.grade || 'ERROR';
  byId('decision-context').textContent = benchmark ? 'Provisional benchmark decision' : 'Prototype decision';
  byId('decision-context').hidden = !item.grade;
  grade.className = 'grade ' + (item.grade || 'ERROR').toLowerCase();
  const uploaded = item.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC';
  const partName = (item.plugin_id === 'screw') ? 'screw' : 'metal_nut';
  byId('source-mode').textContent = benchmark ? 'User-declared MVTec ' + partName + ' PNG · origin unverified · no physical part inspected' : uploaded ? 'User uploaded PNG · diagnostic only · no physical part inspected' : item.source_mode === 'SAVED_IMAGE_REPLAY' ? 'Original dataset replay · ' + item.dataset_path + ' · no physical part inspected' : item.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'Screen photo diagnostic · no physical part inspected' : item.source_mode === 'LIVE_CAMERA' ? 'Marked physical by operator · not independently verified' : 'Camera capture';
  byId('candidate').textContent = item.candidate === 'ANOMALY_CANDIDATE' ? 'Above MVTec threshold' : item.candidate === 'NORMAL_CANDIDATE' ? 'At/below MVTec threshold' : 'Unavailable';
  byId('score').textContent = item.score == null ? '—' : Number(item.score).toFixed(4);
  byId('latency').textContent = item.model_forward_ms == null ? '—' : Math.round(item.model_forward_ms) + ' ms';
  const responseFraction = item.response_area_px == null ? null : Number(item.response_area_px) / Number(item.source_area_px || 640 * 480);
  byId('area').textContent = responseFraction == null ? '—' : Number(item.response_area_px).toLocaleString() + ' px · ' + (responseFraction * 100).toFixed(1) + '%';
  byId('candidate').textContent += item.threshold == null ? '' : ' · MVTec reference cutoff ' + Number(item.threshold).toFixed(4);
  const regions = item.response_regions || [];
  byId('region-summary').textContent = regions[0]?.approximate_nut_area ||
    (regions.length ? 'Area name unavailable' : item.response_area_px === 0 ? 'None above threshold' : 'Unavailable');
  if (!regions.length) regionNote.textContent = item.response_area_px === 0 ? 'No pixels above the provisional response threshold.' : 'Region measurements unavailable.';
  else {
    const first = regions[0];
    const area = first.approximate_nut_area || 'nut area unavailable for this source';
    const otherArea = regions.slice(1).reduce((sum, region) => sum + region.area_px, 0);
    const otherLabels = regions.slice(1, 4).map(region => region.approximate_nut_area || 'area unavailable');
    regionNote.textContent = 'Approximate area of largest response: ' + area + ' / ' +
      first.area_px.toLocaleString() + ' px. Other regions: ' + (regions.length - 1) +
      ' / ' + otherArea.toLocaleString() + ' px combined' +
      (otherLabels.length ? ' / ' + otherLabels.join(', ') : '') +
      '. Percent is of the full image; response pixels are not measured damage.';
  }
  byId('response-note').textContent = benchmark
    ? 'Provisional MVTec benchmark score verdict. This does not confirm a defect or grade a physical nut.'
    : uploaded
    ? 'Uploaded-image diagnostic. The MVTec threshold has not been validated for this file; highlighted pixels are model response, not confirmed damage.'
    : responseFraction != null && responseFraction > 0.5
    ? (item.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC'
      ? 'Broad model response. A phone-screen capture can change focus, color and texture; this does not identify a localized defect. Use the original saved image for benchmark evaluation.'
      : 'Broad model response. Check the capture and camera setup; this does not identify a localized defect.')
    : 'Highlighted pixels are model response, not confirmed physical damage.';
  byId('reason').textContent = item.status === 'ERROR' && item.error?.message ? item.error.message :
    item.reason === 'CAMERA_DOMAIN_UNVALIDATED'
    ? 'Camera setup not calibrated (CAMERA_DOMAIN_UNVALIDATED)'
    : item.reason || (item.error && (item.error.message || item.error)) || 'Inspection could not complete';
  byId('action').textContent = item.action || (item.status === 'ERROR' ? 'Check the analysis error and input image before trying again.' : 'No operator action was provided. Review the inspection evidence.');
  const assessment = byId('defect-assessment');
  assessment.hidden = !item.assessment;
  assessment.textContent = assessmentLabel(item.assessment);
  byId('inspection-id').textContent = item.inspection_id ? 'Inspection ID: ' + item.inspection_id : '';
  clearMedia(item.status === 'ANALYZED' ? 'Loading saved image…' : 'No response map is available for this failed inspection.');
  if (item.status === 'ANALYZED') {
    mediaAvailable = true;
    byId('evidence-caption').textContent = 'Saved image · ' + item.part_id;
    // Bound to this inspection's stored record, so a newer inspection can
    // never replace the evidence under a result already being viewed.
    const media = '/api/inspection/' + encodeURIComponent(item.inspection_id);
    scoreMapUrl = media + '/score-map.npy';
    const stamp = '?t=' + Date.now();
    overlay.src = media + '/overlay.png' + stamp;
    byId('source-image').src = media + '/source.png' + stamp;
    byId('source-link').href = media + '/source.png' + stamp;
    // Same images, shown beside the review form (lazy: loads when that view opens).
    reviewSource.src = media + '/source.png' + stamp;
    reviewOverlay.src = media + '/overlay.png' + stamp;
    reviewSource.hidden = false;
    reviewOverlay.hidden = false;
    selectView('overlay');
  }
}

function assessmentLabel(assessment) {
  if (!assessment) return '';
  const names = new Map([
    ['discoloration', 'Possible discoloration (provisional)'],
    ['bent', 'Possible bent nut (provisional)'],
    ['flip', 'Possible flipped nut (provisional)'],
    ['scratch', 'Possible scratch (provisional)'],
    ['manipulated_front', 'Possible altered screw tip (provisional)'],
    ['scratch_head', 'Possible head scratch (provisional)'],
    ['scratch_neck', 'Possible shank scratch (provisional)'],
    ['thread_side', 'Possible thread-flank defect (provisional)'],
    ['thread_top', 'Possible thread-crest defect (provisional)'],
  ]);
  const label = names.get(assessment.defect_type) || 'Defect type unknown';
  const severity = assessment.severity;
  if (!severity) return label;
  if (severity.reason === 'ORIENTATION_ISSUE') return label + ' · Orientation issue · Severity unassessed';
  const levels = {LOW: 'Low', MEDIUM: 'Medium', HIGH: 'High'};
  return label + (levels[severity.level] ? ' · Provisional severity: ' + levels[severity.level] : ' · Severity unassessed');
}


let historyPage = 0;
let historySequence = 0;
let detailSequence = 0;
let openInspection = null;
let pendingDelete = null;
let editingReview = null;

function historyButton(label, css, action) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn ' + css;
  button.textContent = label;
  const name = {'Open result': 'external', 'Add review': 'review', 'Edit review': 'edit',
    'Delete test': 'delete', 'Delete review': 'delete'}[label];
  if (name) {
    const namespace = 'http://www.w3.org/2000/svg';
    const icon = document.createElementNS(namespace, 'svg');
    icon.setAttribute('class', 'ui-icon');
    icon.setAttribute('aria-hidden', 'true');
    icon.setAttribute('focusable', 'false');
    const use = document.createElementNS(namespace, 'use');
    use.setAttribute('href', '#icon-' + name);
    icon.append(use);
    button.prepend(icon);
  }
  button.addEventListener('click', action);
  return button;
}

function historyTime(value) {
  return new Date(value).toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'medium'});
}

async function loadHistory(next = historyPage) {
  const sequence = ++historySequence;
  try {
    const data = await requestJSON('/api/history?page=' + next);
    if (sequence !== historySequence || currentView !== 'history') return;
    if (next > 0 && !data.rows.length) {
      loadHistory(next - 1);
      return;
    }
    historyPage = next;
    byId('history-page-note').textContent = 'Page ' + (next + 1);
    byId('history-prev').disabled = next === 0;
    byId('history-next').disabled = !data.more;
    renderHistory(data.rows);
  } catch (error) {
    if (sequence === historySequence) byId('history-feedback').textContent = error.message;
  }
}

function renderHistory(rows) {
  const host = byId('history');
  host.replaceChildren();
  if (!rows.length) {
    host.textContent = 'No inspections saved.';
    return;
  }
  const table = document.createElement('table');
  table.className = 'history-table';
  const header = table.createTHead().insertRow();
  for (const title of ['Inspection ID', 'Test time', 'Result', 'Actions']) {
    const column = document.createElement('th');
    column.scope = 'col';
    column.textContent = title;
    header.append(column);
  }
  const list = table.createTBody();
  for (const item of rows) {
    const line = list.insertRow();
    line.className = 'history-record';
    line.addEventListener('click', event => {
      if (!event.target.closest('button, a')) openReportWindow(item.inspection_id);
    });
    line.insertCell().textContent = item.part_id;
    const time = document.createElement('time');
    time.dateTime = item.timestamp_utc;
    time.textContent = historyTime(item.timestamp_utc);
    line.insertCell().append(time);
    line.insertCell().textContent = item.grade + ' / ' + item.status +
      (item.support_id ? ' / Support ID ' + item.support_id : '');
    const actions = line.insertCell();
    actions.className = 'history-actions';
    actions.append(historyButton('Open result', 'secondary', () => openReportWindow(item.inspection_id)),
      historyButton('Delete test', 'danger', () => confirmHistory(
        {action: 'delete', inspection_id: item.inspection_id},
        'Remove ' + item.part_id + ' from History? The original audit record and evidence files will be kept.')));
  }
  host.append(table);
}

for (const [img, empty] of [['saved-original', 'original-unavailable'], ['saved-overlay', 'overlay-unavailable']]) {
  byId(img).addEventListener('error', () => { byId(img).hidden = true; byId(empty).hidden = false; });
  byId(img).addEventListener('load', () => { byId(img).hidden = false; byId(empty).hidden = true; });
}

function openReportWindow(id) {
  const target = '/?inspection=' + encodeURIComponent(id) + '#/history';
  const report = window.open(target, 'forgesight-' + id, 'popup,width=1160,height=820');
  if (report) report.focus();
  else byId('history-feedback').textContent = 'Allow the report window to open, then try again.';
}

async function openSavedInspection(id, moveFocus = true) {
  const sequence = ++detailSequence;
  try {
    const item = await requestJSON('/api/history/' + encodeURIComponent(id));
    if (currentView !== 'history' || sequence !== detailSequence) return;
    openInspection = id;
    byId('saved-title').textContent = item.part_id;
    if (reportInspection) document.title = item.part_id + ' / ForgeSight report';
    const assessment = item.assessment;
    byId('saved-summary').textContent = [historyTime(item.timestamp_utc), item.grade, item.status,
      assessment ? assessmentLabel(assessment) : null,
      item.support_id ? 'Support ID ' + item.support_id : null].filter(Boolean).join(' / ');
    byId('saved-advice-origin').textContent = item.advice.status === 'CHECKLIST_FALLBACK'
      ? 'Checklist fallback' : item.advice.note ? 'Saved Llama advice' : '';
    const advice = byId('saved-advice');
    advice.replaceChildren();
    if (item.advice.note) {
      const list = document.createElement('ul');
      for (const part of item.advice.note.split(/\s+(?=Vision flagged|The camera result)|\r?\n+/).filter(Boolean)) {
        const li = document.createElement('li');
        li.textContent = part;
        list.append(li);
      }
      advice.append(list);
    } else advice.textContent = 'No advice was saved.';
    const download = byId('download-report');
    download.href = '/api/history/' + encodeURIComponent(id) + '/report.html';
    if (/\bForgeSight\//.test(navigator.userAgent)) {
      download.removeAttribute('download');
      download.querySelector('.button-label').textContent = 'Save PDF';
      download.title = 'Save to Desktop / ForgeSight Results';
    } else {
      download.download = 'ForgeSight-' + item.part_id + '.html';
    }
    const media = '/api/inspection/' + encodeURIComponent(id);
    for (const [img, filename, empty] of [['saved-original','source.png','original-unavailable'],
      ['saved-overlay','overlay.png','overlay-unavailable']]) {
      byId(img).hidden = true;
      byId(empty).hidden = true;
      byId(img).src = media + '/' + filename;
    }
    const reviews = byId('saved-reviews');
    reviews.replaceChildren();
    if (!item.reviews.length) reviews.textContent = 'No operator review was saved.';
    if (item.status === 'ANALYZED') {
      reviews.append(historyButton('Add review', 'secondary', () => startReviewEdit(id)));
    }
    for (const review of item.reviews) {
      const row = document.createElement('div');
      row.className = 'saved-review';
      const heading = document.createElement('strong');
      heading.textContent = review.actor + ' / ' + historyTime(review.timestamp_utc);
      const description = document.createElement('p');
      const findings = {no_visible_issue: 'No visible issue', visible_issue: 'Visible issue',
        uncertain_or_recapture: 'Uncertain / capture again'};
      description.textContent = (findings[review.review_outcome] || 'Operator finding') + ': ' + review.reason;
      const actions = document.createElement('div');
      actions.className = 'history-actions';
      actions.append(historyButton('Edit review', 'secondary', () => startReviewEdit(id, review)),
        historyButton('Delete review', 'danger', () => confirmHistory(
          {action: 'delete_review', inspection_id: id, correction_id: review.correction_id},
          'Remove this operator review from History? The original claim remains in the audit records.')));
      row.append(heading, description, actions);
      reviews.append(row);
    }
    byId('saved-inspection').hidden = false;
    if (moveFocus) {
      byId('saved-title').focus();
      byId('saved-inspection').scrollIntoView({block: 'start'});
    }
  } catch (error) {
    if (sequence === detailSequence) showReportFeedback(error.message);
  }
}

function showReportFeedback(message) {
  const feedback = byId(reportInspection ? 'report-feedback' : 'history-feedback');
  feedback.hidden = false;
  feedback.textContent = message;
}

function changeHistory(body) {
  return requestJSON('/api/history', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
}

window.addEventListener('forgesight-report-status', event => showReportFeedback(event.detail));

function confirmHistory(payload, text) {
  pendingDelete = payload;
  byId('history-confirm-text').textContent = text;
  byId('history-confirm').showModal();
  byId('history-cancel').focus();
}

function startReviewEdit(id, review) {
  editingReview = {inspection_id: id, correction_id: review?.correction_id,
    action: review ? 'edit_review' : 'add_review'};
  byId('edit-review-title').textContent = review ? 'Edit operator review' : 'Add operator review';
  byId('edit-actor').value = '';
  byId('edit-outcome').value = review?.review_outcome || 'uncertain_or_recapture';
  byId('edit-reason').value = review?.reason || '';
  byId('edit-feedback').textContent = '';
  byId('edit-review-dialog').showModal();
  byId('edit-actor').focus();
}

byId('history-prev').addEventListener('click', () => loadHistory(historyPage - 1));
byId('history-next').addEventListener('click', () => loadHistory(historyPage + 1));
byId('history-cancel').addEventListener('click', () => byId('history-confirm').close());
byId('history-delete').addEventListener('click', async () => {
  const payload = pendingDelete;
  if (!payload) return;
  byId('history-delete').disabled = true;
  try {
    await changeHistory(payload);
    byId('history-confirm').close();
    if (payload.action === 'delete') {
      if (openInspection === payload.inspection_id) {
        openInspection = null;
        byId('saved-inspection').hidden = true;
      }
      loadHistory();
    } else if (openInspection === payload.inspection_id) {
      openSavedInspection(openInspection, false);
    }
    showReportFeedback('Removed from History.');
  } catch (error) {
    byId('history-confirm-text').textContent = error.message;
  } finally {
    byId('history-delete').disabled = false;
  }
});
byId('edit-cancel').addEventListener('click', () => byId('edit-review-dialog').close());
byId('edit-review-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!editingReview) return;
  const payload = {...editingReview, actor: byId('edit-actor').value,
    outcome: byId('edit-outcome').value, reason: byId('edit-reason').value};
  byId('edit-save').disabled = true;
  try {
    await changeHistory(payload);
    byId('edit-review-dialog').close();
    openSavedInspection(payload.inspection_id, false);
    showReportFeedback('Review saved.');
  } catch (error) {
    byId('edit-feedback').textContent = error.message;
  } finally {
    byId('edit-save').disabled = false;
  }
});

byId('sidebar-toggle').addEventListener('click', () => {
  const hidden = document.querySelector('.app').classList.toggle('sidebar-closed');
  byId('station-sidebar').hidden = hidden;
  byId('sidebar-toggle').setAttribute('aria-expanded', String(!hidden));
  byId('sidebar-toggle').setAttribute('aria-label', hidden ? 'Show navigation' : 'Hide navigation');
  byId('sidebar-toggle').title = hidden ? 'Show navigation' : 'Hide navigation';
});

function renderReviews(reviews, latest) {
  const signature = JSON.stringify([reviews, latest?.inspection_id, latest?.status]);
  if (signature === reviewSignature) return;
  reviewSignature = signature;
  const host = byId('review-history');
  host.replaceChildren();
  if (!reviews.length) {
    host.textContent = latest?.status === 'ANALYZED' ? 'No operator finding recorded for this image.' : '';
    return;
  }
  for (const review of reviews) {
    const row = document.createElement('p');
    const label = {no_visible_issue: 'No visible issue', visible_issue: 'Visible issue',
      uncertain_or_recapture: 'Uncertain / capture again'}[review.review_outcome] || 'Operator finding';
    row.textContent = new Date(review.timestamp_utc).toLocaleString() + ' / ' + review.actor +
      ' / ' + label + ' / proposed ' + review.proposed_grade + ' / unverified: ' + review.reason;
    host.append(row);
  }
}

// Opening review: the sidebar link navigates through its own href; the
// "Open operator review" button on Results navigates here.
function openReview(event) {
  if (!busy && !commandPending && currentLatest?.status === 'ANALYZED') {
    reviewInspection = currentLatest.inspection_id;
    byId('review-panel').hidden = false;
    byId('review-empty').hidden = true;
  }
  if (event.currentTarget.id === 'open-review') location.hash = '#/review';
}
byId('open-review').addEventListener('click', openReview);
byId('review-navigation').addEventListener('click', openReview);
byId('diagnostic-file').addEventListener('change', updateButton);
byId('review-actor').addEventListener('input', updateButton);
byId('review-outcome').addEventListener('change', updateButton);
byId('review-reason').addEventListener('input', updateButton);
// One mutation at a time. Polling cannot re-enable a button during its POST.
async function submitAction(button, status, path, options, onAccepted) {
  if (byId(button).disabled || commandPending) return;
  commandPending = true;
  commandError = null;
  commandEpoch += 1;
  updateButton();
  try {
    await requestJSON(path, options);
    onAccepted();
  } catch (error) {
    commandError = {target: status, message: error.message, inspectionId: currentLatest?.inspection_id};
    byId(status).textContent = error.message;
  } finally {
    commandPending = false;
    await refreshState();
    updateButton();
  }
}

function expectResult() {
  awaitingBaseline = currentLatest?.inspection_id ?? null;
  awaitingUntil = Date.now() + 180000;
}

byId('capture-form').addEventListener('submit', event => {
  event.preventDefault();
  // Selecting Analyze confirms that the operator placed the part in view.
  const body = {confirmed: true, source_kind: 'physical_nut'};
  if (byId('inspect').disabled) return;
  byId('request-status').textContent = 'Capturing one camera frame…';
  submitAction('inspect', 'request-status', '/api/inspect',
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)},
    () => {
      busy = true;
      expectResult();
      byId('request-status').textContent = 'Analyzing captured frame…';
    });
});

byId('upload-form').addEventListener('submit', event => {
  event.preventDefault();
  if (byId('diagnostic-submit').disabled) return;
  const file = byId('diagnostic-file').files[0];
  if (!file || file.size < 1 || file.size > 8 * 1024 * 1024) {
    byId('diagnostic-status').textContent = 'Choose one PNG under 8 MiB.';
    return;
  }
  byId('diagnostic-status').textContent = 'Uploading your PNG…';
  submitAction('diagnostic-submit', 'diagnostic-status', '/api/upload-diagnostic',
    {method: 'POST', headers: {'Content-Type': 'image/png',
      'X-Mvtec-Benchmark': byId('mvtec-benchmark').checked ? 'yes' : 'no'}, body: file},
    () => {
      busy = true;
      expectResult();
      byId('diagnostic-status').textContent = 'Analyzing uploaded PNG…';
    });
});

byId('advice-request').addEventListener('click', () => {
  if (byId('advice-request').disabled) return;
  byId('advice-status').textContent = 'Requesting on-device Llama advice…';
  submitAction('advice-request', 'advice-status', '/api/advice',
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode: 'auto'})},
    () => { byId('advice-status').textContent = 'Llama is generating through ExecuTorch…'; });
});

byId('review-form').addEventListener('submit', event => {
  event.preventDefault();
  if (byId('review-submit').disabled) return;
  const body = {inspection_id: currentLatest?.inspection_id, actor: byId('review-actor').value,
    outcome: byId('review-outcome').value, reason: byId('review-reason').value};
  byId('review-status').textContent = 'Saving your observation…';
  submitAction('review-submit', 'review-status', '/api/manual-review',
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)},
    () => {
      byId('review-status').textContent = 'Operator finding saved. Original model grade is unchanged.';
      byId('review-reason').value = '';
      byId('review-outcome').value = '';
    });
});

async function cameraControl(action) {
  if (cameraControlPending || commandPending || busy || !connected) return;
  cameraControlPending = true;
  commandEpoch += 1;
  updateButton();
  try {
    const state = await requestJSON('/api/camera/' + action, {method: 'POST'});
    cameraRunning = state.running !== false;
    cameraReady = !!state.ready;
    if (!cameraRunning) {
      stopPreview();
      cameraStatus('STOPPED', state.message || 'Camera stopped by operator.');
    } else {
      cameraStatus('CONNECTING', state.message || 'Starting camera…');
    }
  } catch (error) {
    cameraStatus('ERROR', error.message);
  } finally {
    cameraControlPending = false;
    updateButton();
    await refreshState();
  }
}
byId('camera-start').addEventListener('click', () => cameraControl('start'));
byId('camera-stop').addEventListener('click', () => cameraControl('stop'));

preview.addEventListener('load', () => {
  previewPending = false;
  clearTimeout(previewTimer);
  if (!connected || !cameraReady) return;
  preview.hidden = false;
  byId('preview-empty').hidden = true;
  cameraStatus('LIVE', preview.naturalWidth + ' × ' + preview.naturalHeight + ' · live camera view');
  updateButton();
});
function previewFailed() {
  previewPending = false;
  clearTimeout(previewTimer);
  if (!connected || !cameraReady) return;
  cameraStatus('ERROR', 'Camera preview unavailable. Checking for a new frame…');
  updateButton();
}
preview.addEventListener('error', previewFailed);

async function pollState() {
  await refreshState();
  setTimeout(pollState, busy ? 250 : 1500);
}

if (reportInspection) {
  document.querySelector('.app').classList.add('report-window');
  showView({view: 'history', pane: null}, false);
  byId('view-title').textContent = 'Saved inspection';
  byId('view-intro').hidden = true;
  byId('close-report').hidden = false;
  byId('close-report').addEventListener('click', () => window.close());
  openSavedInspection(reportInspection);
} else {
  showView(parseHash() || {view: 'inspect', pane: null}, false);
  pollState();
}
// The live frame is only fetched while the Camera tab is on screen.
setInterval(() => {
  if (currentView === 'inspect' && currentPane === 'camera' &&
      connected && cameraReady && !previewPending && !document.hidden) {
    previewPending = true;
    previewTimer = setTimeout(() => {
      preview.removeAttribute('src');
      previewFailed();
    }, 4000);
    preview.src = '/api/frame.jpg?t=' + Date.now();
  }
}, 250);
