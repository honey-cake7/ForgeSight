"""Versioned local SQLite history with immutable originals and separate correction claims."""

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import UUID, uuid4

from forgesight.grade import policy_digest, validate_policy


DEFAULT_DATABASE = Path('artifacts/runtime/inspections.sqlite3')
SCHEMA_VERSION = 1
APPLICATION_ID = 0x46534947  # FSIG; refuse unrelated or newer databases.
GRADES = ('PASS', 'REVIEW', 'REJECT')
STATUSES = ('ANALYZED', 'ERROR')

SCHEMA_V1 = (
    '''CREATE TABLE inspections (
        inspection_id TEXT PRIMARY KEY NOT NULL,
        part_id TEXT NOT NULL UNIQUE,
        timestamp_utc TEXT NOT NULL,
        source_path TEXT NOT NULL,
        source_sha256 TEXT,
        model_name TEXT NOT NULL,
        model_sha256 TEXT NOT NULL,
        score_policy_sha256 TEXT NOT NULL,
        grade_policy_id TEXT NOT NULL,
        grade_policy_sha256 TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('ANALYZED', 'ERROR')),
        grade TEXT NOT NULL CHECK (grade IN ('PASS', 'REVIEW', 'REJECT')),
        advice_status TEXT NOT NULL,
        record_json TEXT NOT NULL,
        record_sha256 TEXT NOT NULL,
        persisted_at_utc TEXT NOT NULL
    )''',
    'CREATE INDEX inspections_time ON inspections(timestamp_utc DESC, inspection_id DESC)',
    'CREATE INDEX inspections_grade_time ON inspections(grade, timestamp_utc DESC, inspection_id DESC)',
    '''CREATE TABLE corrections (
        correction_id TEXT PRIMARY KEY NOT NULL,
        inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE RESTRICT,
        timestamp_utc TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT NOT NULL,
        proposed_grade TEXT CHECK (proposed_grade IS NULL OR proposed_grade IN ('PASS', 'REVIEW', 'REJECT')),
        verification_status TEXT NOT NULL CHECK (verification_status = 'UNVERIFIED_OPERATOR_CLAIM'),
        correction_json TEXT NOT NULL,
        correction_sha256 TEXT NOT NULL
    )''',
    'CREATE INDEX corrections_inspection_time ON corrections(inspection_id, timestamp_utc, correction_id)',
    '''CREATE TRIGGER inspections_no_update BEFORE UPDATE ON inspections
       BEGIN SELECT RAISE(ABORT, 'Original inspections are append-only'); END''',
    '''CREATE TRIGGER inspections_no_delete BEFORE DELETE ON inspections
       BEGIN SELECT RAISE(ABORT, 'Original inspections are append-only'); END''',
    '''CREATE TRIGGER corrections_no_update BEFORE UPDATE ON corrections
       BEGIN SELECT RAISE(ABORT, 'Correction claims are append-only'); END''',
    '''CREATE TRIGGER corrections_no_delete BEFORE DELETE ON corrections
       BEGIN SELECT RAISE(ABORT, 'Correction claims are append-only'); END''',
)


class RecordConflict(ValueError):
    """An ID already identifies different immutable content."""


class PartAlreadyInspected(ValueError):
    """A distinct inspection tried to count an already recorded part again."""


def _text(value, name, maximum=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'Invalid {name}.')
    return value


def _uuid(value, name):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError(f'Expected canonical UUID for {name}.')
    return value


def _part_id(value):
    _text(value, 'part_id', 128)
    if value != value.strip():
        raise ValueError('Part IDs cannot have leading or trailing whitespace.')
    return value


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError(f'Invalid SHA-256 for {name}.')
    return value


def _utc(value):
    stamp = datetime.fromisoformat(_text(value, 'timestamp', 64))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('An explicit UTC timestamp is required.')
    return stamp.astimezone(timezone.utc).isoformat(timespec='microseconds')


def _canonical(document):
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > 1024 * 1024:
        raise ValueError('Metadata exceeds 1 MiB; images/maps belong in referenced files.')
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def _validate_record(record):
    if not isinstance(record, dict) or type(record.get('schema_version')) is not int or record['schema_version'] != 1:
        raise ValueError('Expected InspectionRecord schema version 1.')
    try:
        _uuid(record['inspection_id'], 'inspection_id')
        _part_id(record['part_id'])
        stamp = _utc(record['timestamp_utc'])
        _text(record['source']['path'], 'source path')
        if record['source']['sha256'] is not None:
            _sha(record['source']['sha256'], 'source')
        model, decision = record['model'], record['decision']
        _text(model['name'], 'model name', 128)
        _sha(model['checkpoint_sha256'], 'model')
        _sha(model['score_policy_sha256'], 'score policy')
        validate_policy(decision['policy_snapshot'])
        if (policy_digest(decision['policy_snapshot']) != decision['policy_sha256']
                or decision['policy_id'] != decision['policy_snapshot']['policy_id']
                or decision['grade'] not in GRADES or record['status'] not in STATUSES):
            raise ValueError('Invalid grade, processing status or policy fingerprint.')
        _text(decision['reason'], 'decision reason')
        _text(record['advice_status'], 'advice status', 128)
        _text(record['predicted_defect_type'], 'predicted type', 128)
        _text(record['type_status'], 'type status', 128)
        if record['score'] is not None and (type(record['score']) not in (int, float) or not math.isfinite(record['score'])):
            raise ValueError('Nonfinite or invalid score.')
        if record['status'] == 'ERROR':
            if decision['grade'] != 'REVIEW' or not isinstance(record['error'], dict):
                raise ValueError('Failed inspection requires explicit error evidence and REVIEW.')
            _text(record['error']['type'], 'error type', 128)
            if not isinstance(record['error']['message'], str):
                raise ValueError('Missing error message.')
        elif (record['score'] is None or record['candidate'] not in ('NORMAL_CANDIDATE', 'ANOMALY_CANDIDATE')
              or not isinstance(record['geometry'], dict) or record['error'] is not None
              or record['source']['sha256'] is None):
            raise ValueError('Analyzed inspection requires real score/map evidence and no error.')
        if not isinstance(record['artifacts'], dict) or not isinstance(record['timings'], dict):
            raise ValueError('Artifact references and timings must be metadata objects.')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Incomplete inspection metadata.') from exc
    encoded, digest = _canonical(record)
    return stamp, encoded, digest


def new_correction(inspection_id, actor, reason, proposed_grade=None):
    """A proposal is not a confirmed label/cause and never overwrites the decision."""
    return {'schema_version': 1, 'correction_id': str(uuid4()), 'inspection_id': inspection_id,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'actor': actor,
            'reason': reason, 'proposed_grade': proposed_grade,
            'verification_status': 'UNVERIFIED_OPERATOR_CLAIM'}


class InspectionStore:
    """Short-lived connections, explicit transactions, and no external dependencies."""
    def __init__(self, path=DEFAULT_DATABASE, *, create=True):
        self.path = Path(path).resolve()
        if not create and not self.path.is_file():
            raise FileNotFoundError(f'Inspection database does not exist: {self.path}')
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect(create=create)) as connection, self._transaction(connection):
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            application = connection.execute('PRAGMA application_id').fetchone()[0]
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if create and version == 0 and application == 0 and not tables:
                for statement in SCHEMA_V1:
                    connection.execute(statement)
                connection.execute(f'PRAGMA application_id = {APPLICATION_ID}')
                connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            elif version != SCHEMA_VERSION or application != APPLICATION_ID or tables != {'inspections', 'corrections'}:
                raise ValueError('Unrelated, unversioned or unsupported database; original file preserved.')

    def _connect(self, *, create=False, readonly=False):
        mode = 'ro' if readonly else ('rwc' if create else 'rw')
        connection = sqlite3.connect(self.path.as_uri() + '?mode=' + mode, uri=True,
                                     timeout=5.0, isolation_level=None)
        try:
            connection.execute('PRAGMA foreign_keys = ON')
            connection.execute('PRAGMA synchronous = FULL')
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    @contextmanager
    def _transaction(connection):
        connection.execute('BEGIN IMMEDIATE')
        try:
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def has_part(self, part_id):
        _part_id(part_id)
        with closing(self._connect(readonly=True)) as connection:
            return connection.execute('SELECT 1 FROM inspections WHERE part_id = ?', (part_id,)).fetchone() is not None

    def save_inspection(self, record):
        stamp, encoded, digest = _validate_record(record)
        with closing(self._connect()) as connection, self._transaction(connection):
            existing = connection.execute('SELECT record_sha256 FROM inspections WHERE inspection_id = ?',
                                          (record['inspection_id'],)).fetchone()
            if existing:
                if existing[0] == digest:
                    return False  # Exact retry; never count twice or replace content.
                raise RecordConflict('Inspection ID already has different content; original preserved.')
            if connection.execute('SELECT 1 FROM inspections WHERE part_id = ?', (record['part_id'],)).fetchone():
                raise PartAlreadyInspected('Part ID already recorded in this database; original preserved.')
            model, decision = record['model'], record['decision']
            connection.execute('''INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (record['inspection_id'], record['part_id'], stamp, record['source']['path'], record['source']['sha256'],
                 model['name'], model['checkpoint_sha256'], model['score_policy_sha256'],
                 decision['policy_id'], decision['policy_sha256'], record['status'], decision['grade'],
                 record['advice_status'], encoded, digest, datetime.now(timezone.utc).isoformat()))
        return True

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        encoded, expected = row
        if hashlib.sha256(encoded.encode()).hexdigest() != expected:
            raise ValueError('Stored metadata checksum mismatch; refusing changed history.')
        return json.loads(encoded)

    def get_inspection(self, inspection_id):
        _uuid(inspection_id, 'inspection_id')
        with closing(self._connect(readonly=True)) as connection:
            return self._decode(connection.execute('SELECT record_json, record_sha256 FROM inspections WHERE inspection_id = ?',
                                                   (inspection_id,)).fetchone())

    def history(self, *, limit=50, offset=0, grade=None, status=None, part_id=None):
        if type(limit) is not int or not 1 <= limit <= 1000 or type(offset) is not int or offset < 0:
            raise ValueError('History requires limit 1..1000 and nonnegative offset.')
        clauses, values = [], []
        for field, value, allowed in [('grade', grade, GRADES), ('status', status, STATUSES)]:
            if value is not None:
                if value not in allowed:
                    raise ValueError(f'Invalid {field} filter.')
                clauses.append(field + ' = ?')
                values.append(value)
        if part_id is not None:
            clauses.append('part_id = ?')
            values.append(_part_id(part_id))
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with closing(self._connect(readonly=True)) as connection:
            rows = connection.execute('SELECT record_json, record_sha256 FROM inspections' + where +
                ' ORDER BY timestamp_utc DESC, inspection_id DESC LIMIT ? OFFSET ?', (*values, limit, offset)).fetchall()
        return [self._decode(row) for row in rows]

    def summary(self):
        with closing(self._connect(readonly=True)) as connection:
            connection.execute('BEGIN')
            total, parts = connection.execute('SELECT COUNT(*), COUNT(DISTINCT part_id) FROM inspections').fetchone()
            grades = dict(connection.execute('SELECT grade, COUNT(*) FROM inspections GROUP BY grade'))
            statuses = dict(connection.execute('SELECT status, COUNT(*) FROM inspections GROUP BY status'))
            corrections = connection.execute('SELECT COUNT(*) FROM corrections').fetchone()[0]
        return {'schema_version': SCHEMA_VERSION, 'inspection_count': total, 'distinct_part_count': parts,
                'original_grade_counts': {g: grades.get(g, 0) for g in GRADES},
                'processing_status_counts': {s: statuses.get(s, 0) for s in STATUSES},
                'unverified_correction_count': corrections}

    def save_correction(self, correction):
        try:
            if correction['schema_version'] != 1 or correction['verification_status'] != 'UNVERIFIED_OPERATOR_CLAIM':
                raise ValueError('Correction must remain an unverified operator claim.')
            _uuid(correction['correction_id'], 'correction_id')
            _uuid(correction['inspection_id'], 'inspection_id')
            stamp = _utc(correction['timestamp_utc'])
            _text(correction['actor'], 'actor', 128)
            _text(correction['reason'], 'correction reason')
            if correction['proposed_grade'] is not None and correction['proposed_grade'] not in GRADES:
                raise ValueError('Invalid proposed grade.')
        except (KeyError, TypeError) as exc:
            raise ValueError('Incomplete correction claim.') from exc
        encoded, digest = _canonical(correction)
        with closing(self._connect()) as connection, self._transaction(connection):
            existing = connection.execute('SELECT correction_sha256 FROM corrections WHERE correction_id = ?',
                                          (correction['correction_id'],)).fetchone()
            if existing:
                if existing[0] == digest:
                    return False
                raise RecordConflict('Correction ID already has different content; original preserved.')
            if not connection.execute('SELECT 1 FROM inspections WHERE inspection_id = ?', (correction['inspection_id'],)).fetchone():
                raise ValueError('Correction references an unknown inspection.')
            connection.execute('INSERT INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                               (correction['correction_id'], correction['inspection_id'], stamp,
                                correction['actor'], correction['reason'], correction['proposed_grade'],
                                correction['verification_status'], encoded, digest))
        return True

    def corrections(self, inspection_id):
        _uuid(inspection_id, 'inspection_id')
        with closing(self._connect(readonly=True)) as connection:
            rows = connection.execute('SELECT correction_json, correction_sha256 FROM corrections WHERE inspection_id = ? '
                                      'ORDER BY timestamp_utc, correction_id', (inspection_id,)).fetchall()
        return [self._decode(row) for row in rows]

    def integrity(self):
        with closing(self._connect(readonly=True)) as connection:
            return {'integrity_check': [r[0] for r in connection.execute('PRAGMA integrity_check')],
                    'foreign_key_check': connection.execute('PRAGMA foreign_key_check').fetchall(),
                    'foreign_keys_enabled': bool(connection.execute('PRAGMA foreign_keys').fetchone()[0]),
                    'journal_mode': connection.execute('PRAGMA journal_mode').fetchone()[0],
                    'synchronous': connection.execute('PRAGMA synchronous').fetchone()[0],
                    'schema_version': connection.execute('PRAGMA user_version').fetchone()[0],
                    'application_id': connection.execute('PRAGMA application_id').fetchone()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DATABASE)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('init')
    sub.add_parser('summary')
    sub.add_parser('integrity')
    history = sub.add_parser('history')
    history.add_argument('--limit', type=int, default=50)
    history.add_argument('--offset', type=int, default=0)
    history.add_argument('--grade', choices=GRADES)
    history.add_argument('--status', choices=STATUSES)
    history.add_argument('--part-id')
    for name in ('get', 'corrections'):
        sub.add_parser(name).add_argument('inspection_id')
    for name in ('import-record', 'import-correction'):
        sub.add_parser(name).add_argument('file', type=Path)
    args = parser.parse_args()
    try:
        store = InspectionStore(args.db, create=args.action in ('init', 'import-record'))
        if args.action in ('init', 'summary'):
            result = store.summary()
        elif args.action == 'integrity':
            result = store.integrity()
        elif args.action == 'history':
            result = store.history(limit=args.limit, offset=args.offset, grade=args.grade,
                                   status=args.status, part_id=args.part_id)
        elif args.action == 'get':
            result = store.get_inspection(args.inspection_id)
        elif args.action == 'corrections':
            result = store.corrections(args.inspection_id)
        elif args.action == 'import-record':
            result = {'inserted': store.save_inspection(json.loads(args.file.read_text()))}
        else:
            result = {'inserted': store.save_correction(json.loads(args.file.read_text()))}
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
