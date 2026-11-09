"""Build v83 concept-conditioned identity control from the immutable v82 source."""
from pathlib import Path
import json,shutil,hashlib,ast
root=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
parent=root/'reve_v82_grouped_identity_weight1';out=root/'reve_v83_concept_identity_weight1'
assert not out.exists();out.mkdir();src=out/'source_gradient_monitor'
shutil.copytree(parent/'source_gradient_monitor',src)
for name in ['local.npy','local_complete.json','predictability']:(out/name).symlink_to((parent/name).resolve())
f=src/'grouped_identity.py';s=f.read_text();f.chmod(0o644)
s=s.replace('image-conditioned','concept-conditioned')
s=s.replace('        self.rng = np.random.default_rng(810033)', '''        paths = json.loads((Path(cfg['source_root'])/'data/manifest.json').read_text())['paths']
        # Strip split-specific numeric prefix; the remaining folder name is the concept.
        concept = {i: Path(path).parent.name.split('_', 1)[1] for i, path in enumerate(paths)}
        assert all('training_images' in Path(paths[int(i)]).parts for i in self.images)
        assert not ({concept[int(i)] for i in self.images} & {concept[int(i)] for i in test_ids})
        by_concept = {}
        for image in self.images: by_concept.setdefault(concept[int(image)], []).append(int(image))
        assert len(by_concept)==1654 and all(len(v)==10 for v in by_concept.values())
        self.concepts = sorted(by_concept)
        self.concept_images = [by_concept[c] for c in self.concepts]
        # Require the same four people to have >=2 trials for both selected images.
        self.concept_subjects = [sorted(set.intersection(*(set(self.groups[i]) for i in ims))) for ims in self.concept_images]
        assert all(len(v)>=4 for v in self.concept_subjects)
        self.rng = np.random.default_rng(810033)''')
a=s.index('        for im in self.rng.choice(self.images, 32, replace=False):');b=s.index('        assert len(eeg)',a)
s=s[:a]+'''        for concept_id in self.rng.choice(len(self.concepts), 16, replace=False):
            selected_images = self.rng.choice(self.concept_images[concept_id], 2, replace=False)
            selected_subjects = self.rng.choice(self.concept_subjects[concept_id], 4, replace=False)
            for subject in selected_subjects:
                for image in selected_images:
                    pool = self.groups[int(image)][subject]
                    for j in self.rng.choice(len(pool), 2, replace=False):
                        d, i = pool[j]
                        assert int(self.datasets[d].ids[i])==image
                        assert str(self.datasets[d].subjects[i])==subject
                        eeg.append(torch.from_numpy(self.datasets[d].x[i].copy()))
                        images.append(int(concept_id)); subjects.append(subject); records.append((d, i))
''' +s[b:]
s=s.replace('positives.sum(1) == 1','positives.sum(1) == 3').replace('allowed.sum(1) == 7','allowed.sum(1) == 15')
s=s.replace('return -logp.masked_fill(~positives, 0).sum(1).mean()', 'return -(logp.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()')
s=s.replace('images_per_batch=32,','images_per_batch=32, concepts_per_batch=16, concepts=len(sampler.concepts),')
s=s.replace('positives_per_anchor=1,','positives_per_anchor=3,').replace('negatives_per_anchor=6','negatives_per_anchor=12')
s=s.replace('x[:8].cuda(), im[:8].cuda(), sub[:8].cuda()', 'x[:16].cuda(), im[:16].cuda(), sub[:16].cuda()')
s=s.replace('    for step in range(50):', '    floor = float(np.log(3))  # Three equally weighted positives: SupCon lower bound.\n    for step in range(50):')
s=s.replace('float(value.detach()) < initial*.5','float(value.detach()) - floor < (initial-floor)*.5')
s=s.replace('two trials each from four people, one image.', 'two trials per image, two images, four people, one concept.')
ast.parse(s);f.write_text(s)
c=json.loads((parent/'config.json').read_text());c.update(run_root=str(out),experiment='v83: concept-conditioned identity supervision, matched v82 weight1',only_change='identity grouping: same image -> same concept, 16 concepts x4 subjects x2 shared images x2 trials',auxiliary_images=32,auxiliary_concepts=16,auxiliary_images_per_concept=2,identity_positives='same concept same subject excluding self (3 positives); same concept other subjects (12 negatives); other concepts excluded')
(out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
old=json.loads((parent/'config.json').read_text());(out/'config_diff.json').write_text(json.dumps({k:[old.get(k),v] for k,v in c.items() if old.get(k)!=v},indent=2))
manifest={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
(out/'source_gradient_monitor_sha256.json').write_text(json.dumps(manifest,indent=2))
for name in manifest:(src/name).chmod(0o444)
launch=(parent/'launch.sh').read_text().replace(str(parent),str(out)).replace('CUDA_VISIBLE_DEVICES=2','CUDA_VISIBLE_DEVICES=0').replace('gpu2','gpu0').replace('nvidia-smi -i 2','nvidia-smi -i 0')
(out/'launch.sh').write_text(launch)
(out/'EXPERIMENT_PROTOCOL.md').write_text('v83 vs v82: identity auxiliary batch groups by concept rather than image. 16 concepts x4 shared subjects x2 shared images x2 distinct trials=256. Each anchor has3 same-subject same-concept positives and12 other-subject same-concept negatives; positives averaged in SupCon. Concept IDs derive from original image directories with numeric split prefix removed, train/test concept disjointness asserted. All other training settings retained including weight1, temperature0.1, fresh pretrained REVE, seed33, original retrieval batch256 and augmentation, EMA, public test development selection. Gradient monitoring retained. This changes positive/negative composition as well as group semantics; loss raw values are not directly comparable (positive floor log3).\n')
print(out)
