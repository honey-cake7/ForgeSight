const byId = id => document.getElementById(id);
const preview = byId('preview');
const overlay = byId('overlay');
let priorInspection = null;
let cameraReady = false;
let busy = false;
let currentLatest = null;
const regionNote = document.createElement('p');
regionNote.id = 'response-regions';
regionNote.className = 'response-note';
byId('response-note').before(regionNote);

function updateButton() {
  byId('inspect').disabled = !cameraReady || busy || !byId('confirmed').checked || !/^[A-Za-z0-9_-]{1,64}$/.test(byId('part-id').value);
  byId('diagnostic-submit').disabled = busy || !byId('diagnostic-file').files.length ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(byId('diagnostic-id').value);
  byId('review-submit').disabled = busy || currentLatest?.status !== 'ANALYZED' ||
    !/^[A-Za-z0-9 _.-]{1,64}$/.test(byId('review-actor').value) ||
    !byId('review-outcome').value || byId('review-reason').value.trim().length < 3;
}

async function refreshState() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error('Backend unavailable');
    const state = await response.json();
    cameraReady = !!state.camera.ready;
    busy = !!state.busy;
    currentLatest = state.latest;
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
    renderReviews(state.manual_reviews || [], state.latest);
    renderAdvice(state.advice, state.latest, state.busy);
    renderHistory(state.history);
  } catch (error) {
    cameraReady = false;
    busy = false;
    byId('camera-status').textContent = error.message;
    updateButton();
  }
}

function renderAdvice(advice, latest, inspectionBusy) {
  const status = advice?.status || 'NOT_CONNECTED';
  const eligible = latest?.status === 'ANALYZED' && latest?.candidate === 'ANOMALY_CANDIDATE' &&
    ['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC',
      'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'].includes(latest?.source_mode);
  byId('advice-request').disabled = inspectionBusy || !eligible || status !== 'NOT_REQUESTED';
  const labels = {
    NOT_CONNECTED: 'On-device Llama runtime is unavailable.',
    NOT_ELIGIBLE: 'Advice is available after a flagged saved-image result.',
    NOT_REQUESTED: 'Flagged image saved. You can request a separate Llama note.',
    PENDING: 'Llama is generating on this Jetson through ExecuTorch…',
    ADVICE_UNAVAILABLE: 'Llama advice was unavailable. The inspection grade is unchanged.',
    GENERATED_REQUIRES_HUMAN_REVIEW: 'ExecuTorch generation verified · human review required.'
  };
  byId('advice-status').textContent = labels[status] || 'Llama advice status is unknown.';
  const note = advice?.advisory?.note;
  byId('advice-note').hidden = status !== 'GENERATED_REQUIRES_HUMAN_REVIEW' || !note;
  byId('advice-note').textContent = note || '';
}

function renderLatest(item, processing) {
  if (processing || item.status === 'PROCESSING') {
    if (['SAVED_IMAGE_REPLAY', 'USER_UPLOADED_IMAGE_DIAGNOSTIC', 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'].includes(item.source_mode)) byId('diagnostic-status').textContent = 'Analyzing image…';
    else byId('request-status').textContent = 'Analyzing captured frame… camera preview remains live.';
    return;
  }
  if (item.inspection_id === priorInspection && item.status !== 'ERROR') return;
  priorInspection = item.inspection_id || null;
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
  grade.textContent = benchmark && item.grade ? 'BENCHMARK ' + item.grade : item.grade || 'ERROR';
  grade.className = 'grade ' + (item.grade || '').toLowerCase();
  const uploaded = item.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC';
  byId('source-mode').textContent = benchmark ? 'User-declared MVTec metal_nut PNG · origin unverified · no physical nut inspected' : uploaded ? 'User uploaded PNG · diagnostic only · no physical nut inspected' : item.source_mode === 'SAVED_IMAGE_REPLAY' ? 'Original dataset replay · ' + item.dataset_path + ' · no physical nut inspected' : item.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'Screen photo diagnostic · no physical nut inspected' : item.source_mode === 'LIVE_CAMERA' ? 'Marked physical by operator · not independently verified' : 'Camera capture';
  byId('candidate').textContent = item.candidate === 'ANOMALY_CANDIDATE' ? 'Above MVTec threshold' : item.candidate === 'NORMAL_CANDIDATE' ? 'At/below MVTec threshold' : 'Unavailable';
  byId('score').textContent = item.score == null ? '—' : Number(item.score).toFixed(4) + ' / MVTec reference ' + Number(item.threshold).toFixed(4);
  byId('latency').textContent = item.model_forward_ms == null ? '—' : Math.round(item.model_forward_ms) + ' ms';
  const responseFraction = item.response_area_px == null ? null : Number(item.response_area_px) / Number(item.source_area_px || 640 * 480);
  byId('area').textContent = responseFraction == null ? '—' : Number(item.response_area_px).toLocaleString() + ' px · ' + (responseFraction * 100).toFixed(1) + '%';
  const regions = item.response_regions || [];
  if (!regions.length) regionNote.textContent = 'No pixels above the provisional response threshold.';
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
    grade.textContent = (row.source_mode === 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD' ? 'BENCHMARK · ' : row.source_mode === 'USER_UPLOADED_IMAGE_DIAGNOSTIC' ? 'USER PNG · ' : row.source_mode === 'SAVED_IMAGE_REPLAY' ? 'DATASET REPLAY · ' : row.source_mode === 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC' ? 'SCREEN PHOTO · ' : '') + row.grade + ' · ' + row.status;
    line.append(name, details, grade);
    list.append(line);
  }
  host.append(list);
}

function renderReviews(reviews, latest) {
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
      headers: {'Content-Type': 'image/png', 'X-Upload-Id': byId('diagnostic-id').value,
        'X-Mvtec-Benchmark': byId('mvtec-benchmark').checked ? 'yes' : 'no'}, body: file});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Upload rejected');
    byId('diagnostic-status').textContent = 'Analyzing uploaded PNG…';
  } catch (error) {
    byId('diagnostic-status').textContent = error.message;
  } finally {
    await refreshState();
  }
});

byId('advice-request').addEventListener('click', async () => {
  byId('advice-request').disabled = true;
  byId('advice-status').textContent = 'Requesting on-device Llama advice…';
  try {
    const response = await fetch('/api/advice', {method: 'POST'});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || result.status || 'Advice request rejected');
    byId('advice-status').textContent = 'Llama is generating on this Jetson through ExecuTorch…';
  } catch (error) {
    byId('advice-status').textContent = error.message;
  } finally {
    await refreshState();
  }
});

byId('review-submit').addEventListener('click', async () => {
  byId('review-submit').disabled = true;
  try {
    const response = await fetch('/api/manual-review', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({inspection_id: currentLatest?.inspection_id,
        actor: byId('review-actor').value, outcome: byId('review-outcome').value,
        reason: byId('review-reason').value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Review could not be saved');
    byId('review-status').textContent = 'Operator finding saved. Original model grade is unchanged.';
    byId('review-reason').value = '';
    byId('review-outcome').value = '';
  } catch (error) {
    byId('review-status').textContent = error.message;
  } finally {
    await refreshState();
  }
});

async function pollState() {
  await refreshState();
  setTimeout(pollState, busy ? 250 : 1500);
}

pollState();
setInterval(() => {
  if (cameraReady) preview.src = '/api/frame.jpg?t=' + Date.now();
}, 250);
