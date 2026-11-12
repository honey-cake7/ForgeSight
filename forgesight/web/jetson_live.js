const byId = id => document.getElementById(id);
const preview = byId('preview');
const overlay = byId('overlay');
let priorInspection = null;
let cameraReady = false;
let busy = false;
let replaySamples = [];
let localReplaySamples = [];

function updateButton() {
  byId('inspect').disabled = !cameraReady || busy || !byId('confirmed').checked || !/^[A-Za-z0-9_-]{1,64}$/.test(byId('part-id').value);
  byId('replay-local').disabled = busy || !localReplaySamples.includes(byId('replay-sample').value) ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(byId('replay-id').value);
  byId('diagnostic-submit').disabled = busy || !byId('diagnostic-file').files.length ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(byId('diagnostic-id').value);
}

async function refreshState() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error('Backend unavailable');
    const state = await response.json();
    cameraReady = !!state.camera.ready;
    busy = !!state.busy;
    if (Array.isArray(state.replay_samples) && state.replay_samples.join('|') !== replaySamples.join('|')) {
      replaySamples = state.replay_samples;
      const selected = byId('replay-sample').value;
      byId('replay-sample').replaceChildren(new Option('Choose image', ''),
        ...replaySamples.map(path => new Option(path, path)));
      byId('replay-sample').value = selected;
    }
    localReplaySamples = Array.isArray(state.local_replay_samples) ? state.local_replay_samples : [];
    const badge = byId('camera-status');
    badge.textContent = state.camera.message;
    badge.classList.toggle('ready', cameraReady);
    if (cameraReady) {
      byId('preview-empty').hidden = true;
    } else {
      preview.removeAttribute('src');
      byId('preview-empty').hidden = false;
    }
    updateButton();
    if (state.latest) renderLatest(state.latest, state.busy);
    renderHistory(state.history);
  } catch (error) {
    cameraReady = false;
    busy = false;
    byId('camera-status').textContent = error.message;
    updateButton();
  }
}

function renderLatest(item, processing) {
  if (processing || item.status === 'PROCESSING') {
    if (item.source_mode === 'SAVED_IMAGE_REPLAY') byId('replay-status').textContent = 'Analyzing original dataset image…';
    else if (item.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC') byId('diagnostic-status').textContent = 'Analyzing uploaded PNG…';
    else byId('request-status').textContent = 'Analyzing captured frame… camera preview remains live.';
    return;
  }
  if (item.inspection_id === priorInspection && item.status !== 'ERROR') return;
  priorInspection = item.inspection_id || null;
  const statusText = item.status === 'ANALYZED' ? 'Result saved locally.' : 'Analysis failed; see error below.';
  if (item.source_mode === 'SAVED_IMAGE_REPLAY') byId('replay-status').textContent = statusText;
  else if (item.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC') byId('diagnostic-status').textContent = statusText;
  else byId('request-status').textContent = statusText;
  byId('result-empty').hidden = true;
  byId('result-content').hidden = false;
  const grade = byId('grade');
  grade.textContent = item.grade || 'ERROR';
  grade.className = 'grade ' + (item.grade || '').toLowerCase();
  const uploaded = item.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC';
  byId('source-mode').textContent = uploaded ? 'User uploaded PNG · diagnostic only · no physical nut inspected' : item.source_mode === 'SAVED_IMAGE_REPLAY' ? 'Original dataset replay · ' + item.dataset_path + ' · no physical nut inspected' : item.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'Screen photo diagnostic · no physical nut inspected' : item.source_mode === 'LIVE_CAMERA' ? 'Marked physical by operator · not independently verified' : 'Camera capture';
  byId('candidate').textContent = item.candidate === 'ANOMALY_CANDIDATE' ? 'Above MVTec threshold' : item.candidate === 'NORMAL_CANDIDATE' ? 'At/below MVTec threshold' : 'Unavailable';
  byId('score').textContent = item.score == null ? '—' : Number(item.score).toFixed(4) + ' / MVTec reference ' + Number(item.threshold).toFixed(4);
  byId('latency').textContent = item.model_forward_ms == null ? '—' : Math.round(item.model_forward_ms) + ' ms';
  const responseFraction = item.response_area_px == null ? null : Number(item.response_area_px) / Number(item.source_area_px || 640 * 480);
  byId('area').textContent = responseFraction == null ? '—' : Number(item.response_area_px).toLocaleString() + ' px · ' + (responseFraction * 100).toFixed(1) + '%';
  byId('response-note').textContent = uploaded
    ? 'Uploaded-image diagnostic. The MVTec threshold has not been validated for this file; highlighted pixels are model response, not confirmed damage.'
    : responseFraction != null && responseFraction > 0.5
    ? (item.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC'
      ? 'Broad model response. A phone-screen capture can change focus, color and texture; this does not identify a localized defect. Use the original saved image for benchmark evaluation.'
      : 'Broad model response. Check the capture and camera setup; this does not identify a localized defect.')
    : 'Highlighted pixels are model response, not confirmed physical damage.';
  byId('reason').textContent = item.reason === 'CAMERA_DOMAIN_UNVALIDATED'
    ? 'Camera setup not calibrated (CAMERA_DOMAIN_UNVALIDATED)'
    : item.reason || (item.error && (item.error.message || item.error)) || 'Inspection could not complete';
  byId('action').textContent = item.action || '';
  byId('inspection-id').textContent = item.inspection_id ? 'Inspection ID: ' + item.inspection_id : '';
  if (item.status === 'ANALYZED') {
    overlay.src = '/api/overlay.png?t=' + Date.now();
    overlay.hidden = false;
    byId('overlay-empty').hidden = true;
  } else {
    overlay.hidden = true;
    byId('overlay-empty').hidden = false;
  }
}

function renderHistory(rows) {
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
    const details = document.createElement('span');
    details.textContent = new Date(row.timestamp_utc).toLocaleString();
    const grade = document.createElement('span');
    grade.textContent = (row.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC' ? 'USER PNG · ' : row.source_mode === 'SAVED_IMAGE_REPLAY' ? 'DATASET REPLAY · ' : row.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'SCREEN PHOTO · ' : '') + row.grade + ' · ' + row.status;
    line.append(name, details, grade);
    list.append(line);
  }
  host.append(list);
}

byId('confirmed').addEventListener('change', updateButton);
byId('part-id').addEventListener('input', updateButton);
for (const id of ['replay-sample', 'replay-id']) {
  byId(id).addEventListener(id === 'replay-id' ? 'input' : 'change', updateButton);
}
byId('diagnostic-file').addEventListener('change', updateButton);
byId('diagnostic-id').addEventListener('input', updateButton);
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
byId('inspect').addEventListener('click', async () => {
  busy = true;
  updateButton();
  byId('request-status').textContent = 'Capturing one camera frame…';
  try {
    const response = await fetch('/api/inspect', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({part_id: byId('part-id').value, confirmed: byId('confirmed').checked,
        source_kind: document.querySelector('input[name="source-kind"]:checked').value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Inspection request rejected');
    byId('request-status').textContent = 'Analyzing captured frame…';
    byId('confirmed').checked = false;
  } catch (error) {
    byId('request-status').textContent = error.message;
  } finally {
    await refreshState();
  }
});

byId('replay-local').addEventListener('click', async () => {
  busy = true;
  updateButton();
  byId('replay-status').textContent = 'Opening the original PNG stored on the Jetson…';
  try {
    const response = await fetch('/api/replay-local', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({replay_id: byId('replay-id').value,
        dataset_path: byId('replay-sample').value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Jetson replay request rejected');
    byId('replay-status').textContent = 'Analyzing original dataset image…';
  } catch (error) {
    byId('replay-status').textContent = error.message;
  } finally {
    await refreshState();
  }
});

byId('diagnostic-submit').addEventListener('click', async () => {
  const file = byId('diagnostic-file').files[0];
  if (!file || file.size < 1 || file.size > 8 * 1024 * 1024) {
    byId('diagnostic-status').textContent = 'Choose one PNG under 8 MiB.';
    return;
  }
  busy = true;
  updateButton();
  byId('diagnostic-status').textContent = 'Uploading your PNG…';
  try {
    const response = await fetch('/api/upload-diagnostic', {method: 'POST',
      headers: {'Content-Type': 'image/png', 'X-Upload-Id': byId('diagnostic-id').value}, body: file});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Upload rejected');
    byId('diagnostic-status').textContent = 'Analyzing uploaded PNG…';
  } catch (error) {
    byId('diagnostic-status').textContent = error.message;
  } finally {
    await refreshState();
  }
});

refreshState();
setInterval(refreshState, 700);
setInterval(() => {
  if (cameraReady) preview.src = '/api/frame.jpg?t=' + Date.now();
}, 250);
