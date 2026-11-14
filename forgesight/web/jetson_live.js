'use strict';

const byId = id => document.getElementById(id);
const preview = byId('preview');
const overlay = byId('overlay');
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
let historySignature = '';
let reviewSignature = '';
let cameraReady = false;
let busy = false;
let currentLatest = null;
const regionNote = byId('response-regions');

const savedModes = ['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC',
  'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'];
const validId = value => /^[A-Za-z0-9_-]{1,64}$/.test(value);

function updateButton() {
  const locked = !connected || commandPending || busy;
  byId('inspect').disabled = locked || !cameraReady || preview.hidden ||
    !byId('confirmed').checked || !validId(byId('part-id').value);
  byId('diagnostic-submit').disabled = locked || !byId('diagnostic-file').files.length ||
    !validId(byId('diagnostic-id').value);
  byId('review-submit').disabled = locked || currentLatest?.status !== 'ANALYZED' ||
    !/^[A-Za-z0-9 _.-]{1,64}$/.test(byId('review-actor').value) ||
    !byId('review-outcome').value || byId('review-reason').value.trim().length < 3;
  const eligible = currentLatest?.status === 'ANALYZED' &&
    currentLatest?.candidate === 'ANOMALY_CANDIDATE' && savedModes.includes(currentLatest?.source_mode);
  byId('advice-request').disabled = locked || !eligible || adviceState?.status !== 'NOT_REQUESTED';
  byId('inspect').textContent = busy ? 'Inspection in progress…' : 'Capture & analyze';
  byId('diagnostic-submit').textContent = busy ? 'Inspection in progress…' : 'Analyze uploaded PNG';
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
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || result.status || 'Station request failed');
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
  if (refreshing || commandPending) return;
  refreshing = true;
  const epoch = commandEpoch;
  try {
    const state = await requestJSON('/api/state');
    // A state request begun before a user action must not undo its UI lock.
    if (epoch !== commandEpoch) return;
    connected = true;
    byId('connection-status').textContent = 'Station connected';
    byId('connection-status').className = 'connection ready';
    byId('connection-error').hidden = true;
    cameraReady = !!state.camera.ready;
    busy = !!state.busy;
    if (currentLatest?.inspection_id !== state.latest?.inspection_id) {
      byId('review-outcome').value = '';
      byId('review-reason').value = '';
    }
    currentLatest = state.latest;
    adviceState = state.advice;
    if (!cameraReady) {
      stopPreview();
      // Current API exposes ready/message, not a stopped state. Never infer STOPPED.
      const starting = state.camera.message === 'Starting camera';
      cameraStatus(starting ? 'CONNECTING' : 'ERROR', state.camera.message || 'Camera unavailable. Check the USB connection.');
    } else if (preview.hidden && !previewPending) {
      cameraStatus('CONNECTING', 'Loading a live camera frame…');
    }
    if (state.latest) renderLatest(state.latest, state.busy);
    else resetResult();
    if (['Waiting for camera.', 'Camera unavailable. You can still upload a saved PNG.',
      'Position the part, enter a new ID and confirm the camera view.'].includes(byId('request-status').textContent)) {
      byId('request-status').textContent = cameraReady
        ? 'Position the part, enter a new ID and confirm the camera view.'
        : 'Camera unavailable. You can still upload a saved PNG.';
    }
    renderReviews(state.manual_reviews || [], state.latest);
    renderAdvice(state.advice);
    renderHistory(state.history || []);
    if (commandError && commandError.inspectionId === currentLatest?.inspection_id) {
      byId(commandError.target).textContent = commandError.message;
    }
  } catch (error) {
    if (epoch !== commandEpoch) return;
    connected = false;
    cameraReady = false;
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
  byId('review-target').textContent = 'Inspect an image first, then compare its original and response map above.';
  clearMedia('No image inspected yet. Capture a camera view or upload a PNG below.');
}

function clearMedia(message) {
  mediaAvailable = false;
  for (const img of [overlay, byId('source-image')]) {
    img.hidden = true;
    img.removeAttribute('src');
    img.dataset.loaded = '';
  }
  byId('overlay-empty').textContent = message;
  byId('overlay-empty').hidden = false;
  byId('source-link').hidden = true;
  byId('evidence-caption').textContent = 'Latest saved inspection · EfficientAD';
}

function selectView(view) {
  savedView = view;
  byId('show-overlay').setAttribute('aria-pressed', String(view === 'overlay'));
  byId('show-source').setAttribute('aria-pressed', String(view === 'source'));
  if (!mediaAvailable) return;
  overlay.hidden = view !== 'overlay' || overlay.dataset.loaded !== 'yes';
  byId('source-image').hidden = view !== 'source' || byId('source-image').dataset.loaded !== 'yes';
  const active = view === 'overlay' ? overlay : byId('source-image');
  byId('overlay-empty').hidden = active.dataset.loaded === 'yes';
  byId('overlay-empty').textContent = active.dataset.loaded === 'error'
    ? 'Saved image could not load. The inspection record is still retained.'
    : 'Loading saved image…';
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
byId('show-overlay').addEventListener('click', () => selectView('overlay'));
byId('show-source').addEventListener('click', () => selectView('source'));

function renderAdvice(advice) {
  const status = advice?.status || 'NOT_CONNECTED';
  const labels = {
    NOT_CONNECTED: 'On-device Llama runtime is unavailable.',
    NOT_ELIGIBLE: 'Advice is available after a flagged saved-image result.',
    NOT_REQUESTED: 'Flagged image saved. You can request a separate Llama note.',
    PENDING: 'Llama is generating on this Jetson through ExecuTorch…',
    ADVICE_UNAVAILABLE: 'Llama advice was unavailable. The inspection grade is unchanged.',
    GENERATED_REQUIRES_HUMAN_REVIEW: 'Generated operator note · human review required.'
  };
  byId('advice-status').textContent = labels[status] || 'Llama advice status is unknown.';
  const runtime = advice?.llama_runtime === 'PERSISTENT_WORKER'
    ? 'Warm worker: the model stays loaded between notes.'
    : advice?.llama_runtime === 'COLD_SUBPROCESS'
    ? 'Cold process: the model loads for each note.' : 'No runtime mode reported.';
  byId('advice-runtime').textContent = runtime + (advice?.advice_worker_boot_error
    ? ' The persistent worker did not start; the slower cold path is available.' : '');
  const note = advice?.advisory?.note;
  byId('advice-note').hidden = status !== 'GENERATED_REQUIRES_HUMAN_REVIEW' || !note;
  byId('advice-note').textContent = note || '';
  byId('advice-request').textContent = status === 'PENDING' ? 'Generating advice…' : 'Generate Llama advice';
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
  byId('inspection-status').textContent = item.status === 'ANALYZED' ? 'SAVED' : 'ANALYSIS ERROR';
  byId('review-target').textContent = item.status === 'ANALYZED'
    ? 'Reviewing ' + item.part_id + '. Compare Original and Response map above.'
    : 'Analysis failed. A completed result is required for review.';
  const statusText = item.status === 'ANALYZED' ? 'Result saved locally.' : 'Analysis failed; see error below.';
  byId('review-status').textContent = item.status === 'ANALYZED'
    ? 'Compare the saved original and overlay, then record your observation.'
    : 'Manual review is available after a completed analysis.';
  if (['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC', 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'].includes(item.source_mode)) byId('diagnostic-status').textContent = statusText;
  else byId('request-status').textContent = statusText;
  byId('result-empty').hidden = true;
  byId('result-content').hidden = false;
  const grade = byId('grade');
  const benchmark = item.source_mode === 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD';
  grade.textContent = item.grade || 'ERROR';
  byId('decision-context').textContent = benchmark ? 'Provisional benchmark decision' : 'Prototype decision';
  byId('decision-context').hidden = !item.grade;
  grade.className = 'grade ' + (item.grade || 'ERROR').toLowerCase();
  const uploaded = item.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC';
  byId('source-mode').textContent = benchmark ? 'User-declared MVTec metal_nut PNG · origin unverified · no physical nut inspected' : uploaded ? 'User uploaded PNG · diagnostic only · no physical nut inspected' : item.source_mode === 'SAVED_IMAGE_REPLAY' ? 'Original dataset replay · ' + item.dataset_path + ' · no physical nut inspected' : item.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'Screen photo diagnostic · no physical nut inspected' : item.source_mode === 'LIVE_CAMERA' ? 'Marked physical by operator · not independently verified' : 'Camera capture';
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
  byId('reason').textContent = item.reason === 'CAMERA_DOMAIN_UNVALIDATED'
    ? 'Camera setup not calibrated (CAMERA_DOMAIN_UNVALIDATED)'
    : item.reason || (item.error && (item.error.message || item.error)) || 'Inspection could not complete';
  byId('action').textContent = item.action || (item.status === 'ERROR' ? 'Check the analysis error and input image before trying again.' : 'No operator action was provided. Review the inspection evidence.');
  byId('inspection-id').textContent = item.inspection_id ? 'Inspection ID: ' + item.inspection_id : '';
  clearMedia(item.status === 'ANALYZED' ? 'Loading saved image…' : 'No response map is available for this failed inspection.');
  if (item.status === 'ANALYZED') {
    mediaAvailable = true;
    byId('evidence-caption').textContent = 'Saved image · ' + item.part_id;
    const stamp = '?t=' + Date.now();
    overlay.src = '/api/overlay.png' + stamp;
    byId('source-image').src = '/api/source.png' + stamp;
    byId('source-link').href = '/api/source.png' + stamp;
    selectView(savedView);
  }
}

function renderHistory(rows) {
  const signature = JSON.stringify(rows);
  if (signature === historySignature) return;
  historySignature = signature;
  const host = byId('history');
  host.replaceChildren();
  if (!rows.length) { host.textContent = 'No inspections yet.'; return; }
  const list = document.createElement('div');
  list.className = 'history-list';
  for (const row of rows) {
    const line = document.createElement('div');
    line.className = 'history-row';
    const name = document.createElement('strong');
    name.textContent = row.part_id;
    const details = document.createElement('time');
    details.dateTime = row.timestamp_utc;
    details.textContent = new Date(row.timestamp_utc).toLocaleString();
    const decision = document.createElement('div');
    const badge = document.createElement('span');
    badge.className = 'grade history-grade ' + (row.grade || '').toLowerCase();
    badge.textContent = row.grade;
    const grade = document.createElement('span');
    grade.className = 'history-meta';
    grade.textContent = (row.source_mode === 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD' ? 'BENCHMARK · ' : row.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC' ? 'USER PNG · ' : row.source_mode === 'SAVED_IMAGE_REPLAY' ? 'DATASET REPLAY · ' : row.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'SCREEN PHOTO · ' : '') + row.status;
    decision.append(badge, grade);
    line.append(name, details, decision);
    list.append(line);
  }
  host.append(list);
}

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

byId('confirmed').addEventListener('change', updateButton);
byId('part-id').addEventListener('input', updateButton);
byId('diagnostic-file').addEventListener('change', updateButton);
byId('diagnostic-id').addEventListener('input', updateButton);
byId('review-actor').addEventListener('input', updateButton);
byId('review-outcome').addEventListener('change', updateButton);
byId('review-reason').addEventListener('input', updateButton);
for (const choice of document.querySelectorAll('input[name="source-kind"]')) {
  choice.addEventListener('change', () => {
    const physical = document.querySelector('input[name="source-kind"]:checked').value === 'physical_nut';
    byId('confirm-text').textContent = physical
      ? 'I confirm an actual metal nut, not a photo on any screen, is in view.'
      : 'I confirm a displayed nut photo, not a physical part, is in view.';
    byId('confirmed').checked = false;
    updateButton();
  });
}
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

byId('capture-form').addEventListener('submit', event => {
  event.preventDefault();
  const body = {part_id: byId('part-id').value, confirmed: byId('confirmed').checked,
    source_kind: document.querySelector('input[name="source-kind"]:checked').value};
  if (byId('inspect').disabled) return;
  byId('request-status').textContent = 'Capturing one camera frame…';
  submitAction('inspect', 'request-status', '/api/inspect',
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)},
    () => {
      busy = true;
      byId('confirmed').checked = false;
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
      'X-Upload-Id': byId('diagnostic-id').value,
      'X-Mvtec-Benchmark': byId('mvtec-benchmark').checked ? 'yes' : 'no'}, body: file},
    () => {
      busy = true;
      byId('diagnostic-status').textContent = 'Analyzing uploaded PNG…';
    });
});

byId('advice-request').addEventListener('click', () => {
  if (byId('advice-request').disabled) return;
  byId('advice-status').textContent = 'Requesting on-device Llama advice…';
  submitAction('advice-request', 'advice-status', '/api/advice', {method: 'POST'},
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
pollState();
setInterval(() => {
  if (connected && cameraReady && !previewPending && !document.hidden) {
    previewPending = true;
    previewTimer = setTimeout(() => {
      preview.removeAttribute('src');
      previewFailed();
    }, 4000);
    preview.src = '/api/frame.jpg?t=' + Date.now();
  }
}, 250);
