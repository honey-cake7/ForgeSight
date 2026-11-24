"""Screw check choices through the same warm Llama worker as the baseline part."""
import json
import os
import time
from uuid import uuid4

from forgesight.assist_contract import canonical_hash
from forgesight.grounded_live import GroundedLive
from forgesight.grade import grade_record
from forgesight.screw_guidance import advice_request, resolve_choice


class ScrewLive(GroundedLive):
    def __init__(self, queue, report_root, worker_socket, timeout, recipe, worker_unhealthy=None, llama_root=None):
        super().__init__(queue, report_root, worker_socket, timeout, worker_unhealthy, llama_root)
        self.recipe = recipe
        self.guidance = recipe.documents['guidance']

    def _run(self, record, worker_ready):
        try:
            if (record.get('plugin_id') != 'screw'
                    or record['model']['export_sha256'] != self.recipe.model_sha256
                    or record['model']['image_threshold'] != self.recipe.threshold
                    or record['decision'] != grade_record(record, record['decision']['policy_snapshot'])):
                return {'status': 'INSPECTION_WITHHELD'}
            assessment = record.get('assessment') or {}
            label = assessment.get('defect_type', 'unknown')
            if assessment and assessment.get('profile_id') != self.recipe.documents['classifier']['profile_id']:
                return {'status': 'INSPECTION_WITHHELD'}
            payload = advice_request(label, self.guidance)
        except (KeyError, ValueError, TypeError):
            return {'status': 'INSPECTION_WITHHELD'}
        report = {'inspection_id': record['inspection_id'], 'inspection_sha256': canonical_hash(record),
                  'plugin_id': 'screw', 'status': 'CHECKLIST_FALLBACK',
                  'advice': self.guidance['fallback'], 'llama_runtime_mode': 'NOT_REQUESTED',
                  'guidance_sha256': canonical_hash(self.guidance), 'part_release_authorized': False}
        started = time.perf_counter()
        try:
            if not worker_ready():
                raise ConnectionError('Warm Llama worker unavailable.')
            from forgesight.assist_edge_worker import request
            from forgesight.assist_jetson_replay_once import _pinned_asset_hashes
            reply = request(self.worker_socket, payload, timeout_seconds=self.timeout, connect_retries=1)
            evidence = reply['runtime_evidence']
            pinned = _pinned_asset_hashes(self.llama_root)
            artifact, tokenizer = evidence['artifact']['sha256'], evidence['tokenizer']['sha256']
            if (reply['status'] != 'GENERATED_BY_PERSISTENT_WORKER' or pinned is None
                    or artifact not in pinned[0] or tokenizer not in pinned[1]):
                raise ValueError('Llama runtime evidence does not match the pinned assets.')
            choice = resolve_choice(reply['generated_text'], label, self.guidance)
            report.update(status='LLAMA_ADVICE' if choice['status'] == 'REVIEWED_CHOICE' else 'CHECKLIST_FALLBACK',
                          advice=choice['text'], llama_runtime_mode='PERSISTENT_WORKER',
                          worker={'worker_started_utc': reply['worker_started_utc'],
                                  'model_sha256': artifact, 'tokenizer_sha256': tokenizer})
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            report['fallback_reason'] = type(error).__name__
        report['advice_elapsed_ms'] = (time.perf_counter() - started) * 1000
        self.report_root.mkdir(parents=True, exist_ok=True)
        target = self.report_root / (record['inspection_id'] + '.json')
        temporary = target.with_suffix('.' + uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, target)
        return report
