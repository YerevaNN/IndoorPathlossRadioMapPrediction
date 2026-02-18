import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler


class WAIRDBaseDatamodule(pl.LightningDataModule):
    
    collate_fn = None
    
    def __init__(self, batch_size: int, num_workers: int, drop_last: bool, multi_gpu: bool = False, *args, **kwargs):
        super().__init__()
        
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._drop_last = drop_last
        
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self.args = args
        self.kwargs = kwargs
        
        self._multi_gpu = multi_gpu
        if multi_gpu:
            self.prepare_data()
    
    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._train_set) if self._multi_gpu else None
        return DataLoader(
            self._train_set, batch_size=self._batch_size, num_workers=self._num_workers,
            sampler=sampler, shuffle=None if self._multi_gpu else True, collate_fn=self.collate_fn,
            drop_last=self._drop_last
        )
    
    def val_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._val_set, shuffle=False) if self._multi_gpu else None
        return DataLoader(
            self._val_set, batch_size=self._batch_size, num_workers=self._num_workers, sampler=sampler,
            collate_fn=self.collate_fn, drop_last=self._drop_last
        )
    
    def test_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._test_set, shuffle=False) if self._multi_gpu else None
        return DataLoader(
            self._test_set, batch_size=self._batch_size, num_workers=self._num_workers, sampler=sampler,
            collate_fn=self.collate_fn, drop_last=self._drop_last
        )
    
    @property
    def train_set(self):
        return self._train_set
    
    @property
    def test_set(self):
        return self._test_set
    
    @property
    def val_set(self):
        return self._val_set
