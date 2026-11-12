'use strict';
const $ = id => document.getElementById(id);
const state = {ready:false, busy:false, selected:null, offset:0, limit:10, hasNext:false};
const number = value => Number(value).toLocaleString();
function element(tag, text, className) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e; }
function finding(record) { return record.status === 'ERROR' ? 'Inspection error' : record.candidate === 'ANOMALY_CANDIDATE' ? 'Anomaly flagged' : 'No anomaly flagged'; }
function timestamp(value) { return new Date(value).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'}); }
function message(text) { $('message').textContent = text || ''; $('message').hidden = !text; }
async function api(path, options) {
  const response = await fetch(path, {cache:'no-store', ...options});
  let data; try { data = await response.json(); } catch { throw new Error('The local service returned an unreadable response. Check the server.'); }
  if (!response.ok) { let detail = data.detail || 'The local request failed.'; if (Array.isArray(detail)) detail = detail.map(x => x.msg).join('; '); throw new Error(detail); }
  return data;
}
function showImage(kind, url) {
  const img = $(kind + '-image'), empty = $(kind + '-empty');
  img.hidden = true; empty.hidden = false;
  if (!url) { img.removeAttribute('src'); return; }
  img.onload = () => { img.hidden = false; empty.hidden = true; };
  img.onerror = () => { img.hidden = true; empty.hidden = false; empty.replaceChildren(element('span','Saved image unavailable'), element('small','The inspection record is still retained.')); };
  img.src = url;
}
async function showAdvice(id) {
  $('advice-heading').textContent = 'Checking advice';
  $('advice-note').textContent = 'Reading the separate local advisory report…';
  $('advice-source').textContent = 'ForgeGrade remains the decision source.';
  let data;
  try { data = await api('/api/inspections/' + encodeURIComponent(id) + '/advice'); }
  catch { data = {status:'ADVICE_UNAVAILABLE', advisory:null}; }
  if (state.selected !== id) return;
  const labels = {NOT_CONNECTED:'Advice not connected', NOT_REQUESTED:'No advice requested', PENDING:'Advice is being prepared', ADVICE_UNAVAILABLE:'Advice unavailable', GENERATED_REQUIRES_HUMAN_REVIEW:'AI note · human review required'};
  $('advice-heading').textContent = labels[data.status] || 'Advice unavailable';
  $('advice-note').textContent = data.advisory?.note || (data.status === 'PENDING' ? 'The inspection decision is already saved. Advice is still pending.' : 'Use the deterministic action above.');
  $('advice-source').textContent = data.advisory ? 'Reported Jetson ExecuTorch · ' + data.advisory.source_mode.replaceAll('_',' ').toLowerCase() + ' · prototype guidance, not factory approved.' : 'ForgeGrade remains the decision source.';
  if (data.status === 'PENDING') setTimeout(() => { if (state.selected === id) showAdvice(id); }, 2000);
}
function showInspection(item) {
  if (!item) return;
  const record = item.record, decision = record.decision;
  state.selected = record.inspection_id;
  $('result-heading').textContent = record.part_id;
  $('grade').textContent = decision.grade === 'PASS' ? 'PASS · PROVISIONAL' : decision.grade;
  $('grade').className = 'grade ' + decision.grade.toLowerCase();
  $('finding').textContent = finding(record);
  $('defect-type').textContent = record.status === 'ERROR' ? 'Not assessed' : record.candidate === 'NORMAL_CANDIDATE' ? 'None flagged' : record.predicted_defect_type === 'unknown' ? 'Unknown' : record.predicted_defect_type;
  $('area').textContent = record.geometry ? number(record.geometry.total_area_px) + ' pixels' : 'Not measured';
  $('result-time').textContent = timestamp(record.timestamp_utc);
  $('model-time').textContent = record.timings.model_forward_ms === undefined ? 'Input failed; no model timing.' : 'Model forward: ' + record.timings.model_forward_ms.toFixed(1) + ' ms · excludes saving and display';
  $('action-heading').textContent = decision.grade === 'PASS' ? 'Provisional pass' : decision.grade === 'REJECT' ? 'Keep the part aside' : 'Hold for human review';
  $('operator-action').textContent = record.status === 'ERROR' ? 'Image could not be inspected. Check Details.' : decision.grade === 'PASS' ? 'No anomaly flagged. This pass is provisional.' : decision.grade === 'REJECT' ? 'Confirm the result against the approved part specification.' : record.geometry?.total_area_px > 0 ? 'Check the highlighted area before making a decision.' : 'Check the part and photo before making a decision.';
  $('recorded-action').textContent = decision.operator_action;
  $('error-detail').hidden = record.status !== 'ERROR';
  $('error-detail').textContent = record.status === 'ERROR' ? record.error.message : '';
  showImage('source', item.media.source); showImage('overlay', item.media.overlay);
  showAdvice(record.inspection_id);
  document.querySelectorAll('#history-body tr').forEach(row => row.classList.toggle('selected', row.dataset.inspection === state.selected));
}
async function refreshHistory() {
  const grade = $('grade-filter').value;
  const params = new URLSearchParams({limit:state.limit, offset:state.offset}); if (grade) params.set('grade', grade);
  const [history, summary] = await Promise.all([api('/api/history?' + params), api('/api/summary')]);
  $('total-count').textContent = number(summary.inspection_count);
  const counts = summary.original_grade_counts;
  $('counts').textContent = counts.PASS + ' provisional PASS  /  ' + counts.REVIEW + ' REVIEW  /  ' + counts.REJECT + ' REJECT  ·  ' + summary.processing_status_counts.ERROR + ' input/processing errors included';
  const body = $('history-body'); body.replaceChildren();
  for (const item of history.items) {
    const record = item.record, row = element('tr'); row.dataset.inspection = record.inspection_id;
    row.classList.toggle('selected', state.selected === record.inspection_id);
    row.append(element('td',record.part_id), element('td',timestamp(record.timestamp_utc)));
    const decision = element('td'); decision.append(element('span',record.decision.grade,'grade ' + record.decision.grade.toLowerCase())); row.append(decision);
    row.append(element('td',finding(record)), element('td',record.geometry ? number(record.geometry.total_area_px) + ' px' : '—'));
    const action = element('td'), button = element('button','View','view-button'); button.type = 'button'; button.setAttribute('aria-label','View inspection ' + record.part_id); button.onclick = () => showInspection(item); action.append(button); row.append(action); body.append(row);
  }
  $('history-empty').hidden = history.items.length !== 0;
  state.hasNext = history.pagination.has_next;
  $('previous-page').disabled = state.offset === 0; $('next-page').disabled = !state.hasNext;
  $('page-label').textContent = history.items.length ? 'Showing ' + (state.offset + 1) + '–' + (state.offset + history.items.length) : 'No matching records';
}
function controls() { $('inspect-button').disabled = state.busy || !state.ready; $('part-id').disabled = state.busy; $('sample-id').disabled = state.busy || !state.ready; $('button-label').textContent = state.busy ? 'Inspecting and saving…' : 'Inspect part'; }
$('inspect-form').addEventListener('submit', async event => {
  event.preventDefault(); if (state.busy || !state.ready) return;
  const part = $('part-id').value;
  if (!part.trim() || part !== part.trim()) { message('Enter a part ID without leading or trailing spaces.'); return; }
  state.busy = true; controls(); message(''); $('inspect-progress').textContent = 'Running the real model and saving the result…';
  try {
    const item = await api('/api/inspect', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({part_id:part, sample_id:$('sample-id').value})});
    showInspection(item); state.offset = 0; $('grade-filter').value = ''; await refreshHistory();
    $('inspect-progress').textContent = 'Saved one inspection for ' + item.record.part_id + '.';
    if (item.record.status === 'ERROR') message('Inspection failed. Saved for review; see Details.');
  } catch (error) { message(error.message); $('inspect-progress').textContent = 'No new successful inspection is claimed. Check history before retrying.'; }
  finally { state.busy = false; controls(); }
});
function handleError(error) { message(error.message); }
$('refresh-history').onclick = () => refreshHistory().catch(handleError);
$('grade-filter').onchange = () => { state.offset = 0; refreshHistory().catch(handleError); };
$('previous-page').onclick = () => { state.offset = Math.max(0,state.offset-state.limit); refreshHistory().catch(handleError); };
$('next-page').onclick = () => { if (state.hasNext) { state.offset += state.limit; refreshHistory().catch(handleError); } };
async function start() {
  try {
    const config = await api('/api/config');
    $('sample-id').replaceChildren(...config.samples.map(sample => {const option = element('option',sample.label); option.value = sample.id; return option;}));
    await refreshHistory();
    const latest = await api('/api/latest'); showInspection(latest.inspection);
    const healthResponse = await fetch('/api/health', {cache:'no-store'});
    const health = await healthResponse.json(); state.ready = health.model_ready;
    $('health-label').textContent = state.ready ? 'Ready' : 'Vision unavailable';
    $('health-dot').className = state.ready ? 'status-dot ready' : 'status-dot unavailable';
    if (!state.ready) message('Vision model unavailable. Saved inspections and separate advice remain readable.');
  } catch (error) { $('health-label').textContent = 'Unavailable'; $('health-dot').className = 'status-dot unavailable'; message(error.message); }
  controls();
}
start();
