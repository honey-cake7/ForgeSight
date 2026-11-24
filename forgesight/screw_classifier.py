"""Fit a small, JSON-only screw subtype tree on the frozen classifier roles."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from forgesight.defect_type import SCREW_FEATURES, SCREW_TYPES, load_profile, predict_type, screw_features
from forgesight.edge_assets import fingerprint
from forgesight.screw_recipe import validate_manifest


def _vector(row, names=SCREW_FEATURES):
    return [row['features'][name] for name in names]


def candidate_rank(evidence):
    """Cover native classes first so aggregate accuracy cannot hide a missing type."""
    return (len(evidence['supported_labels']), evidence['validation_correct'],
            -evidence['leaf_count'], -evidence['feature_count'])


def evaluate_rows(rows, profile):
    confusion = defaultdict(Counter)
    predictions = []
    for row in rows:
        predicted = predict_type(row['features'], profile) if row['flagged'] else 'not_flagged'
        confusion[row['label']][predicted] += 1
        predictions.append({'source_path': row['source_path'], 'truth': row['label'], 'prediction': predicted})
    correct = sum(r['truth'] == r['prediction'] for r in predictions)
    called = sum(r['prediction'] in SCREW_TYPES for r in predictions)
    return {'images': len(rows), 'correct': correct, 'type_calls': called,
            'incorrect_type_calls': called - correct,
            'unknown': sum(r['prediction'] == 'unknown' for r in predictions),
            'not_flagged': sum(r['prediction'] == 'not_flagged' for r in predictions),
            'end_to_end_type_accuracy': correct / len(rows) if rows else None,
            'accuracy_on_type_calls': correct / called if called else None,
            'type_call_coverage': called / len(rows) if rows else None,
            'counts_by_truth': dict(confusion), 'predictions': predictions}


def fit_profile(partitions, model_sha256, threshold, manifest_sha256):
    """Train on fit only; retain provisional leaves with separate correct support."""
    from sklearn.tree import DecisionTreeClassifier
    if set(partitions) != {'classifier_fit', 'classifier_validation'}:
        raise ValueError('Only frozen classifier fit and validation roles are allowed.')
    seen_paths, seen_hashes = set(), set()
    for role, rows in partitions.items():
        for row in rows:
            if (row['role'] != role or row['label'] not in SCREW_TYPES
                    or row['source_path'] in seen_paths or row['source_sha256'] in seen_hashes):
                raise ValueError('Invalid classifier role, label, or cross-split duplicate.')
            seen_paths.add(row['source_path'])
            seen_hashes.add(row['source_sha256'])
    train = [r for r in partitions['classifier_fit'] if r['flagged'] and r['features'] is not None]
    if {r['label'] for r in train} != set(SCREW_TYPES):
        raise ValueError('Usable flagged training examples of every native defect type are required.')
    validation = [r for r in partitions['classifier_validation'] if r['flagged'] and r['features'] is not None]
    if not validation:
        raise ValueError('Separate usable classifier validation evidence is required.')
    groups = (
        SCREW_FEATURES,
        tuple(n for n in SCREW_FEATURES if n not in ('head_texture', 'neck_texture', 'thread_texture', 'response_contrast')),
        ('peak_axial', 'peak_transverse', 'response_axial_mean', 'response_transverse_mean',
         'head_response_fraction', 'neck_response_fraction', 'tip_response_fraction', 'response_fraction'),
    )
    candidates, best = [], None
    for group_index, names in enumerate(groups):
        for depth in (2, 3, 4):
            for leaf in (3, 5, 8):
                candidate = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                                   class_weight='balanced', random_state=42)
                candidate.fit([_vector(r, names) for r in train], [r['label'] for r in train])
                predicted = candidate.predict([_vector(r, names) for r in validation])
                correct = sum(p == r['label'] for p, r in zip(predicted, validation))
                evidence = {'feature_group': group_index, 'max_depth': depth, 'min_samples_leaf': leaf,
                            'validation_correct': int(correct), 'usable_validation_images': len(validation),
                            'supported_labels': sorted({str(p) for p, r in zip(predicted, validation) if p == r['label']}),
                            'feature_count': len(names), 'leaf_count': int(candidate.get_n_leaves())}
                candidates.append(evidence)
                # Fixed ordering resolves ties; no exposed final-set labels enter selection.
                ranking = candidate_rank(evidence)
                if best is None or ranking > best[0]:
                    best = ranking, candidate, names, evidence
    _, tree, selected_names, selected = best
    support = defaultdict(list)
    for row in validation:
        support[int(tree.apply([_vector(row, selected_names)])[0])].append(row['label'])

    def export(node=0, abstain=False):
        structure = tree.tree_
        left, right = structure.children_left[node], structure.children_right[node]
        if left == right:
            label = str(tree.classes_[structure.value[node][0].argmax()])
            if abstain and label not in support[node]:
                label = 'unknown'
            return {'label': label}
        return {'feature': selected_names[structure.feature[node]], 'threshold': float(structure.threshold[node]),
                'left': export(left, abstain), 'right': export(right, abstain)}

    profile = {'schema_version': 1, 'profile_id': 'screw-30k-tree-v3', 'category': 'screw',
        'kind': 'screw_decision_tree', 'feature_names': list(SCREW_FEATURES), 'labels': list(SCREW_TYPES),
        'model_sha256': model_sha256, 'image_threshold': threshold, 'manifest_sha256': manifest_sha256,
        'fresh_acceptance_verified': False, 'physical_camera_accuracy': 'NOT_VERIFIED',
        'feature_geometry': 'PCA dark-object axis, wider end=head, approximate axial bands; 256px grid',
        'training': {'max_depth': selected['max_depth'], 'min_samples_leaf': selected['min_samples_leaf'],
                     'selected_features': list(selected_names), 'class_weight': 'balanced', 'seed': 42},
        'leaf_policy': 'Provisional label only when at least one separate validation example correctly supports the leaf; otherwise unknown. Wrong validation calls are reported.',
        'tree': export()}
    unpruned = evaluate_rows(partitions['classifier_validation'], profile)
    profile['tree'] = export(abstain=True)
    metrics = {'fit': evaluate_rows(partitions['classifier_fit'], profile),
               'validation_before_leaf_abstention': unpruned,
               'validation_after_leaf_abstention': evaluate_rows(partitions['classifier_validation'], profile),
               'validation_scope': 'Used for 27-candidate selection and leaf support; not a fresh or unbiased final test',
               'selection': {'data_role': 'classifier_validation', 'final_holdout_used': False,
                             'metric': 'Native classes with correct validation support, then correct labels; tie: fewer leaves then fewer features',
                             'selected': selected, 'candidates': candidates},
               'final_holdout_images_read': 0, 'ground_truth_masks_read': 0}
    return profile, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', type=Path, default=Path('plugins/screw'))
    parser.add_argument('--root', type=Path, default=Path('data/mvtec_ad/screw'))
    parser.add_argument('--manifest', type=Path, default=Path('manifests/screw_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new classifier evidence directory.')
    vision = json.loads((args.plugin / 'vision.json').read_text())
    model_path = args.plugin / vision['model']['file']
    if fingerprint(model_path)[1] != vision['model']['sha256']:
        parser.error('Plugin vision model hash mismatch.')
    if fingerprint(args.manifest)[1] != vision['dataset_manifest_sha256']:
        parser.error('Frozen dataset manifest mismatch.')
    policy_path = args.plugin / vision['score_policy']['file']
    if fingerprint(policy_path)[1] != vision['score_policy']['sha256']:
        parser.error('Plugin score policy hash mismatch.')
    policy = json.loads(policy_path.read_text())
    threshold = policy['threshold']
    if threshold != vision['score_policy']['threshold']:
        parser.error('Plugin threshold mismatch.')
    document = json.loads(args.manifest.read_text())
    validate_manifest(document)
    import torch
    from forgesight.data_loader import load_verified_image
    from forgesight.regions import response_regions
    from forgesight.vision_calibrate import tensor_for
    torch.set_num_threads(2)
    model = torch.jit.load(str(model_path), map_location='cpu').eval()
    args.output.mkdir(parents=True)
    partitions = {'classifier_fit': [], 'classifier_validation': []}
    selected = [r for r in document['samples'] if r['role'] in partitions]
    with (args.output / 'features.jsonl').open('x') as log, torch.inference_mode():
        for index, row in enumerate(selected):
            values, score = model(tensor_for(row, args.root, 'cpu'))
            if not torch.isfinite(values).all().item() or not torch.isfinite(score).all().item():
                raise ValueError('Nonfinite classifier inference input.')
            score = float(score.item())
            score_map = values[0, 0].numpy()
            features = None
            if score > threshold:
                with load_verified_image(args.root, row) as image:
                    features = screw_features(image.convert('RGB').resize((256, 256), resample=2), score_map,
                                              response_regions(score_map, threshold))
            entry = {'source_path': row['path'], 'source_sha256': row['sha256'], 'role': row['role'],
                     'label': row['label'], 'flagged': score > threshold, 'score': score, 'features': features}
            partitions[row['role']].append(entry)
            log.write(json.dumps(entry, allow_nan=False) + '\n')
            log.flush()
            if (index + 1) % 10 == 0:
                print(f'Classifier features: {index + 1}/{len(selected)}', flush=True)
    profile, metrics = fit_profile(partitions, vision['model']['sha256'], threshold, fingerprint(args.manifest)[1])
    profile_path = args.output / 'defect_type.json'
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False) + '\n')
    loaded = load_profile(profile_path)
    assert loaded == profile
    report = {'status': 'PROVISIONAL_CLASSIFIER_FITTED_VALIDATION_USED_FOR_ABSTENTION',
        'metrics': metrics, 'features_sha256': fingerprint(args.output / 'features.jsonl')[1],
        'profile_sha256': fingerprint(profile_path)[1], 'model_sha256': vision['model']['sha256'],
        'image_threshold': threshold, 'dataset_manifest_sha256': fingerprint(args.manifest)[1],
        'source_sha256': {str(p): fingerprint(p)[1] for p in (Path(__file__), Path('forgesight/defect_type.py'))},
        'final_holdout_images_read': 0, 'ground_truth_masks_read': 0, 'promotion_allowed': False}
    (args.output / 'training_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in metrics['validation_after_leaf_abstention'].items()
                      if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    main()
