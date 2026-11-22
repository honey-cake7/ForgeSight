"""Host integration tests for forgesight_token_measure.py.

Plumbing is proven here with a stub tokenizer (whitespace counter) in a
self-contained fake HOME; real token counts are device evidence only. The
production pinned-tokenizer SHA gate is verified to refuse the stub, and a
test-local copy with the pin swapped proves the end-to-end path."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from forgesight.assist_contract import DEFAULT_GUIDANCE, prepare_advisory
from forgesight.assist_prompt_token_report import VARIANT_TRANSFORMS

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / 'forgesight_token_measure.py'
PINNED = '82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55'

STUB_TOKENIZER = ('class CppTiktoken:\n'
                  '    def load(self, path): self.path = path\n'
                  '    def encode(self, text, a, b): return list(text.split())\n')


def make_home(root):
    home = root / 'home'
    stubs = root / 'stubs'
    stubs.mkdir(parents=True)
    (stubs / 'pytorch_tokenizers.py').write_text(STUB_TOKENIZER)
    dist = stubs / 'pytorch_tokenizers-1.6.0.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: pytorch-tokenizers\nVersion: 1.6.0\n')
    model = home / '.local/share/forgesight/llama_cpu_v1/model'
    model.mkdir(parents=True)
    (model / 'tokenizer.model').write_text('stub tokenizer bytes')
    (model / 'edge_llama.json').write_text(json.dumps(
        {'max_seq_length': 128, 'max_context_length': 512, 'model_id': 'stub'}))
    (model / 'llama3_2_1b_assets.json').write_text('{}')
    venv_bin = home / '.local/share/forgesight/llama_cpu_v1/venv/bin'
    venv_bin.mkdir(parents=True)
    wrapper = venv_bin / 'python'
    wrapper.write_text('#!/bin/sh\nPYTHONPATH=%s exec %s "$@"\n' % (stubs, sys.executable))
    wrapper.chmod(0o755)
    record = deepcopy(json.loads((REPO / 'manifests/operator_app_verification_v1.json').read_text())
                      ['browser_proof']['inspections'][1]['record'])
    result = home / 'forgesight-feature-audits/main-x/composed-camera/records/fix1/result'
    result.mkdir(parents=True)
    (result / 'record.json').write_text(json.dumps(record))
    return home


def run_script(home, cwd, script=SCRIPT):
    env = dict(os.environ, HOME=str(home))
    return subprocess.run([sys.executable, str(script)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


class BootstrapIntegrationTests(unittest.TestCase):
    def test_sha_gate_refuses_unpinned_tokenizer(self):
        with TemporaryDirectory() as tmp:
            home = make_home(Path(tmp))
            result = run_script(home, REPO)
            self.assertEqual(result.returncode, 2)
            self.assertIn('does not match the pinned deployed asset', result.stderr)

    def test_end_to_end_with_stub_counter(self):
        with TemporaryDirectory() as tmp:
            home = make_home(Path(tmp))
            stub_sha = hashlib.sha256(
                (home / '.local/share/forgesight/llama_cpu_v1/model/tokenizer.model').read_bytes()
            ).hexdigest()
            patched = Path(tmp) / 'ftm_test.py'
            patched.write_text(SCRIPT.read_text().replace(PINNED, stub_sha))
            # A throwaway git repo carrying the candidate package as FETCH_HEAD.
            work = Path(tmp) / 'src'
            work.mkdir()
            subprocess.run(['git', 'init', '-q'], cwd=work, check=True)
            for relative in ('forgesight', 'guidance'):
                subprocess.run(['cp', '-r', str(REPO / relative), str(work / relative)], check=True)
            subprocess.run(['git', 'add', '-A'], cwd=work, check=True)
            subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t',
                            'commit', '-qm', 'fixture'], cwd=work, check=True)
            head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=work, check=True,
                                  capture_output=True, text=True).stdout.strip()
            (work / '.git/FETCH_HEAD').write_text(head + '\t\tfixture\n')
            result = run_script(home, work, script=patched)
            self.assertEqual(result.returncode, 0, result.stderr)
            reports = list(work.glob('forgesight_token_measure_*/token_report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            rows = report['cases']
            self.assertEqual(len(rows), 24)  # 4 cases x (base + 5 variant rows)
            names = {row['name'].split('__')[0] for row in rows}
            self.assertEqual(names, {'real_review', 'synthetic_camera_variant',
                                     'synthetic_multiregion_12', 'synthetic_wide_digits'})
            for row in rows:
                self.assertEqual(row['eligibility'], 'ELIGIBLE', row['name'])
                tokens = row['tokens']
                self.assertEqual(tokens['full_formatted'],
                                 tokens['prompt_text'] + tokens['system_text']
                                 + tokens['template_overhead'], row['name'])
            base = {row['name']: row['tokens']['full_formatted'] for row in rows}
            for case in ('real_review', 'synthetic_camera_variant',
                         'synthetic_multiregion_12', 'synthetic_wide_digits'):
                self.assertLess(base[case + '__noPct-noLoc'], base[case], case)
            trimmed = {row['name']: row for row in rows if '__' in row['name']}
            for row in trimmed.values():
                if 'noDisclaimer' in row['name']:
                    self.assertEqual(row['compliance'],
                                     'NONCOMPLIANT_RESEARCH_ONLY_NEVER_A_FIX_VARIANT')
                else:
                    self.assertEqual(row['compliance'], 'GUARD_PRESERVING_CANDIDATE')
            self.assertEqual(report['recommendation_scope'],
                             'CANDIDATE_MEASURED_ON_LISTED_INPUTS_ONLY')
            # Provenance: the exact fetched commit and per-file hashes are recorded.
            provenance = report['provenance']
            self.assertEqual(provenance['candidate_commit'], head)
            self.assertEqual(provenance['source_repo'], str(work))
            self.assertEqual(len(provenance['package_files']), 10)
            for entry in provenance['package_files']:
                self.assertEqual(entry['sha256'], hashlib.sha256(
                    (work / entry['path']).read_bytes()).hexdigest(), entry['path'])
            # The executed script bytes hash ties the run to the reviewed file.
            self.assertEqual(report['script']['sha256'],
                             hashlib.sha256(patched.read_bytes()).hexdigest())
            self.assertEqual(report['script']['path'], str(patched.resolve()))
            # Input kinds: real review recorded, absent benchmark said explicitly.
            kinds = report['input_kinds']
            self.assertTrue(kinds['real_review'].endswith('record.json'))
            self.assertEqual(kinds['real_reject'], 'MISSING_NOT_FOUND_ON_THIS_DEVICE')
            self.assertTrue(kinds['synthetic_camera_variant'].startswith('DERIVED_FROM:'))
            self.assertIn('real_reject      MISSING', result.stdout)
            self.assertIn('candidate: ' + head, result.stdout)


class TrimParserCompatTests(unittest.TestCase):
    """Each viable trim must still admit a fully compliant note (not just a label)."""

    def setUp(self):
        record = deepcopy(json.loads((REPO / 'manifests/operator_app_verification_v1.json')
                                     .read_text())['browser_proof']['inspections'][1]['record'])
        guidance = json.loads(DEFAULT_GUIDANCE.read_text())
        self.context = prepare_advisory(record, guidance)

    def trimmed_context(self, tags):
        context = deepcopy(self.context)
        for tag in tags:
            context['prompt_text'] = VARIANT_TRANSFORMS[tag](context['prompt_text'])
        return context

    def note(self, with_location):
        location = ('; largest region width 139 px height 116 px at x 512 y 361 '
                    'in the right area of the frame') if with_location else ''
        return ("An anomaly is flagged; type and cause are unknown [E1]. "
                "Severity HIGH, prototype pixel-response severity, not physical-damage severity. "
                "Anomaly score 0.682809 vs threshold 0.227766. "
                "Response area 9015 px across 2 regions" + location + ". "
                "Hold for human review; check clarity and lighting; compare the highlighted "
                "region with a known-good part [G1].")

    def test_noPct_admits_compliant_note(self):
        # Semantic coupling: the v4 prompt carries the word 'frame' only inside the
        # percent clause, so a noPct note restates the location without that word.
        from forgesight.assist_contract import parse_advisory
        note = self.note(True).replace(' in the right area of the frame', ', right')
        result = parse_advisory(note, self.trimmed_context(('noPct',)))
        self.assertEqual(result['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')

    def test_noLoc_admits_compliant_note_without_location(self):
        from forgesight.assist_contract import parse_advisory
        result = parse_advisory(self.note(False), self.trimmed_context(('noLoc',)))
        self.assertEqual(result['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')

    def test_noPct_refuses_frame_wording(self):
        from forgesight.assist_contract import parse_advisory
        with self.assertRaises(ValueError):
            parse_advisory(self.note(True), self.trimmed_context(('noPct',)))

    def test_noLoc_still_refuses_invented_location(self):
        from forgesight.assist_contract import parse_advisory
        with self.assertRaises(ValueError):
            parse_advisory(self.note(True), self.trimmed_context(('noLoc',)))

    def test_combined_trims_admit_compliant_note(self):
        from forgesight.assist_contract import parse_advisory
        result = parse_advisory(self.note(False), self.trimmed_context(('noPct', 'noLoc')))
        self.assertEqual(result['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')


if __name__ == '__main__':
    unittest.main()
