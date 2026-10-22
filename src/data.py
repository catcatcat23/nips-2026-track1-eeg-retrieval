import json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class DatasetEEG(Dataset):
    def __init__(self,root,split,aux=True,local_aux=True):
        root=Path(root)
        manifest=json.loads((root/'data/manifest.json').read_text())
        if manifest.get('preprocessing_contract')!='competition_eeg_image_v1':
            raise ValueError('Legacy EEG preprocessing: regenerate competition 120 Hz cache')
        self.x=np.load(root/'data'/f'{split}_eeg.npy',mmap_mode='r')
        assert list(self.x.shape[1:])==manifest['input_shape'] and self.x.shape[-1]==120
        self.ids=np.load(root/'data'/f'{split}_ids.npy')
        self.subjects=np.load(root/'data'/f'{split}_subjects.npy')
        self.clean=np.load(root/'data/clean.npy',mmap_mode='r')
        self.aux=aux;self.local_aux=local_aux
        if aux:
            self.blur=np.load(root/'features/blur.npy',mmap_mode='r')
            if local_aux:self.local=np.load(root/'features/local.npy',mmap_mode='r')
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        k=int(self.ids[i]);out={'eeg':torch.from_numpy(self.x[i].copy()),'id':k,'subject':str(self.subjects[i]),'clean':torch.from_numpy(self.clean[k].copy())}
        if self.aux:out['blur']=torch.from_numpy(self.blur[k].copy())
        if self.aux and self.local_aux:out['local']=torch.from_numpy(self.local[k].copy())
        return out

def config():return json.loads(Path(os.environ.get('EEG_CONFIG',Path(__file__).with_name('config.json'))).read_text())
