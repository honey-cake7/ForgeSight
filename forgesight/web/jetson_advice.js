'use strict';
const byId = id => document.getElementById(id);
let requesting = false;
async function refresh() {
  try {
    const response = await fetch('/api/state', {cache:'no-store'});
    if (!response.ok) throw new Error('State unavailable');
    const state = await response.json();
    byId('inspection').textContent = 'Inspection ' + state.inspection.inspection_id + ' · saved-image replay';
    byId('grade').textContent = state.inspection.grade;
    byId('action').textContent = state.inspection.operator_action;
    const advice = state.advice;
    const labels = {NOT_REQUESTED:'No advice requested', PENDING:'Advice pending', GENERATED_REQUIRES_HUMAN_REVIEW:'Generated · human review required', ADVICE_UNAVAILABLE:'Advice unavailable'};
    byId('advice-state').textContent = labels[advice.status] || 'Advice unavailable';
    byId('note').textContent = advice.advisory?.note || '';
    byId('request').disabled = requesting || advice.status !== 'NOT_REQUESTED';
  } catch {
    byId('advice-state').textContent = 'Local status unavailable';
    byId('request').disabled = true;
  }
}
byId('request').onclick = async () => {
  requesting = true; byId('request').disabled = true;
  try {
    await fetch('/api/advice', {method:'POST'});
  } finally { requesting = false; await refresh(); }
};
refresh(); setInterval(refresh, 500);
