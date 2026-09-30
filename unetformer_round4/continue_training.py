"""Continue the round-3 UNetFormer checkpoint on the fixed training split.

Selection uses the original 700-image holdout. The existing round-3 ZIP remains
the submission unless the full-resolution dihedral-8 score increases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import MODES, eval_mode, export, run_trial, sampling_weights
from unetformer_pipeline import Pairs, evaluate, loader, make_model, save_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('images', 'masks', 'cache', 'split', 'init', 'audit',
                 'test-images', 'out', 'prior-selection'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    prior = json.loads(Path(args.prior_selection).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(args.split).read_bytes()).hexdigest() != audit['split_sha256']:
        raise ValueError('Split and class weights do not match')
    if len(split['train']) != 6296 or len(split['val']) != 700:
        raise ValueError('Unexpected train/validation split')
    if set(split['train']) & set(split['val']):
        raise ValueError('Train/validation overlap')
    save_json(out / 'config.json', vars(args))

    model = make_model().cuda()
    model.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=True)['model'])
    val = loader(Pairs(split['val'], args.images, args.masks, 512), 2, args.workers)
    from round3 import evaluate_hv
    baseline_hv = evaluate_hv(model, val)
    del val
    baseline_d8 = prior['selected']['mIoU']
    save_json(out / 'baseline.json', {'hvflip': baseline_hv, 'dihedral8_mIoU': baseline_d8})
    print(f'baseline hv={baseline_hv["mIoU"]:.6f} dihedral8={baseline_d8:.6f}', flush=True)
    del model
    torch.cuda.empty_cache()

    recipe = {'name': 'low_lr_rare_images', 'epochs': 8, 'lr': 1.5e-6,
              'balanced': True, 'strength': .08,
              'sizes': [448, 512, 448, 512, 448, 512, 448, 512],
              'sampling': True, 'hard_pixels': False}
    save_json(out / 'recipe.json', recipe)
    sample_weights = sampling_weights(split['train'], Path(args.cache) / 'masks')
    best_path, selected_hv = run_trial(args, recipe, split, audit['weights'],
                                       baseline_hv['mIoU'], sample_weights)
    if best_path is None:
        save_json(out / 'selection.json', {
            'improved': False, 'reason': 'No continuation checkpoint exceeded the four-flip baseline',
            'baseline_dihedral8_mIoU': baseline_d8, 'baseline_hvflip_mIoU': baseline_hv['mIoU']})
        print('No four-flip improvement; existing submission remains best.', flush=True)
        return

    model = make_model().cuda()
    model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)['model'])
    model.eval()
    mode = MODES['512_dihedral8']
    result = eval_mode(model, args, split['val'], mode)
    improved = result['mIoU'] > baseline_d8
    save_json(out / 'selection.json', {
        'improved': improved, 'baseline_dihedral8_mIoU': baseline_d8,
        'candidate_hvflip': selected_hv, 'candidate_dihedral8': result,
        'delta_percentage_points': 100 * (result['mIoU'] - baseline_d8),
        'official_score': None})
    print(f'candidate dihedral8={result["mIoU"]:.6f}; improved={improved}', flush=True)
    if improved:
        shutil.copyfile(best_path, out / 'best.pth')
        export(model, args.test_images, out, mode)
        (out / 'submission_unetformer_round3_test2.zip').rename(
            out / 'submission_unetformer_round4_test2.zip')
        manifest = json.loads((out / 'submission_manifest.json').read_text(encoding='utf-8'))
        manifest['file'] = 'submission_unetformer_round4_test2.zip'
        save_json(out / 'submission_manifest.json', manifest)


if __name__ == '__main__':
    main()

