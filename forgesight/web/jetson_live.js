const byId = id => document.getElementById(id);
const preview = byId('preview');
const overlay = byId('overlay');
let priorInspection = null;
let cameraReady = false;
let busy = false;

function updateButton() {
  byId('inspect').disabled = !cameraReady || busy || !byId('confirmed').checked || !/^[A-Za-z0-9_-]{1,64}$/.test(byId('part-id').value);
}

async function refreshState() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error('Backend unavailable');
    const state = await response.json();
    cameraReady = !!state.camera.ready;
    busy = !!state.busy;
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
    byId('request-status').textContent = 'Analyzing captured frame… camera preview remains live.';
    return;
  }
  if (item.inspection_id === priorInspection && item.status !== 'ERROR') return;
  priorInspection = item.inspection_id || null;
  byId('request-status').textContent = item.status === 'ANALYZED' ? 'Inspection saved locally.' : 'Inspection failed; see error below.';
  byId('result-empty').hidden = true;
  byId('result-content').hidden = false;
  const grade = byId('grade');
  grade.textContent = item.grade || 'ERROR';
  grade.className = 'grade ' + (item.grade || '').toLowerCase();
  byId('candidate').textContent = item.candidate === 'ANOMALY_CANDIDATE' ? 'Anomaly response' : item.candidate === 'NORMAL_CANDIDATE' ? 'Normal candidate' : 'Unavailable';
  byId('score').textContent = item.score == null ? '—' : Number(item.score).toFixed(4);
  byId('latency').textContent = item.model_forward_ms == null ? '—' : Math.round(item.model_forward_ms) + ' ms';
  byId('area').textContent = item.response_area_px == null ? '—' : Number(item.response_area_px).toLocaleString() + ' px';
  byId('reason').textContent = item.reason || (item.error && (item.error.message || item.error)) || 'Inspection could not complete';
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
    grade.textContent = row.grade + ' · ' + row.status;
    line.append(name, details, grade);
    list.append(line);
  }
  host.append(list);
}

byId('confirmed').addEventListener('change', updateButton);
byId('part-id').addEventListener('input', updateButton);
byId('inspect').addEventListener('click', async () => {
  busy = true;
  updateButton();
  byId('request-status').textContent = 'Capturing one camera frame…';
  try {
    const response = await fetch('/api/inspect', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({part_id: byId('part-id').value, confirmed: byId('confirmed').checked})});
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

refreshState();
setInterval(refreshState, 700);
setInterval(() => {
  if (cameraReady) preview.src = '/api/frame.jpg?t=' + Date.now();
}, 250);
