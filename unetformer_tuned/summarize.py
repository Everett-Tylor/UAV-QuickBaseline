"""Write the final comparison from recorded validation evidence."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    run=Path(a.run);s=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    base=s['baseline'];selected=s['selected']
    delta={k:100*(v-base['per_class_IoU'][k]) for k,v in selected['per_class_IoU'].items()}
    comparison={'baseline_mIoU':base['mIoU'],'selected_mIoU':selected['mIoU'],
                'delta_percentage_points':s['delta_percentage_points'],
                'per_class_delta_percentage_points':delta,
                'all_classes_improved':all(v>0 for v in delta.values()),'official_score':None}
    (run/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    manifest=json.loads((run/'submission_manifest.json').read_text(encoding='utf-8'))
    lines=['# UNetFormer tuning results','',
        f"Validation: **{100*selected['mIoU']:.4f}% mIoU**, up **{s['delta_percentage_points']:.4f} percentage points** from the remeasured {100*base['mIoU']:.4f}% baseline.",'',
        'Same 6,296/700 split; test2 excluded from training and selection. These are local validation results, not official competition scores.','',
        '## Trial comparison (512 + horizontal flip)','',
        '| Trial | Best epoch | Weights | mIoU |','|---|---:|---|---:|']
    for trial in s['trials']:
        t=trial['best']
        if t:lines.append(f"| {trial['name']} | {t['epoch']} | {t['variant']} | {100*t['mIoU']:.4f}% |")
        else:lines.append(f"| {trial['name']} | — | — | No improvement |")
    chosen=s['checkpoint_selection']
    lines+=['',f"Selected checkpoint: {chosen['trial']}, epoch {chosen.get('epoch','original')}, {chosen.get('variant','original')}.",'',
        '## Inference comparison','', '| Setting | mIoU |','|---|---:|']
    for c in s['inference_candidates']:lines.append(f"| {c['name']} | {100*c['mIoU']:.4f}% |")
    lines+=['',f"Selected inference: `{selected['name']}`. One model is used; no checkpoint ensemble.",'',
        '## Per-class changes','', '| Class | Previous IoU | New IoU | Change (pp) |','|---|---:|---:|---:|']
    for k,v in selected['per_class_IoU'].items():
        lines.append(f"| {k} | {100*base['per_class_IoU'][k]:.4f}% | {100*v:.4f}% | {delta[k]:+.4f} |")
    lines+=['','## Deliverable','',f"ZIP: `{manifest['file']}`; 1,300 PNG files, 1024×1024, single-channel class IDs.",
        f"SHA-256: `{manifest['sha256']}`.",'',
        'The raw/EMA checkpoints were selected on the same holdout repeatedly. This can make the reported gain optimistic; no independent unseen validation set or competition result is available.',
        'All trials start from the same previous checkpoint. Each recipe changes multiple factors, so these results do not isolate individual effects.',
        'Source includes the upstream GPL-3.0 license and a documented reflection-padding compatibility fix. Weights and predictions are retained locally.']
    (run.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(comparison),flush=True)

if __name__=='__main__':main()
