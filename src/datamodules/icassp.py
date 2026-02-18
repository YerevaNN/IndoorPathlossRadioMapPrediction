from typing import Union

from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.datasets import ICASSPDataset
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class ICASSPDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
        multi_gpu: bool = False,
        *args, **kwargs
    ):
        self.dataloader_val = None
        self.dataloader_val_new_room = None
        self.dataloader_val_new_freq = None
        self.dataloader_val_new_ant = None
        self.dataloader_val_new_room_freq = None
        self.dataloader_val_new_room_ant = None
        self.dataloader_val_new_freq_ant = None
        
        self.dataloader_test = None
        self.dataloader_test_new_room = None
        self.dataloader_test_new_freq = None
        self.dataloader_test_new_ant = None
        self.dataloader_test_new_room_freq = None
        self.dataloader_test_new_room_ant = None
        self.dataloader_test_new_freq_ant = None
        
        self.val_set_new_freq = None
        self.val_set_new_room = None
        self.val_set_new_ant = None
        self.val_set_new_room_freq = None
        self.val_set_new_room_ant = None
        self.val_set_new_freq_ant = None
        
        self.test_set_new_freq = None
        self.test_set_new_room = None
        self.test_set_new_ant = None
        self.test_set_new_room_freq = None
        self.test_set_new_room_ant = None
        self.test_set_new_freq_ant = None
        
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, multi_gpu=multi_gpu,
            *args, **kwargs
        )
    
    def prepare_data(self) -> None:
        if self.kwargs["task_idx"] == 1:
            self._train_set = ICASSPDataset(split="train", *self.args, **self.kwargs)
            self._val_set = ICASSPDataset(split="val", *self.args, **self.kwargs)
            self._test_set = ICASSPDataset(split="test", *self.args, **self.kwargs)
        elif self.kwargs["task_idx"] == 2:
            self._train_set = ICASSPDataset(split="train", *self.args, **self.kwargs)
            self.val_set_new_room = ICASSPDataset(split="val_new_room", *self.args, **self.kwargs)
            self.val_set_new_freq = ICASSPDataset(split="val_new_freq", *self.args, **self.kwargs)
            self._val_set = ICASSPDataset(split="val", *self.args, **self.kwargs)
            self.test_set_new_room = ICASSPDataset(split="test_new_room", *self.args, **self.kwargs)
            self.test_set_new_freq = ICASSPDataset(split="test_new_freq", *self.args, **self.kwargs)
            self._test_set = ICASSPDataset(split="test", *self.args, **self.kwargs)
        elif self.kwargs["task_idx"] == 3:
            self._train_set = ICASSPDataset(split="train", *self.args, **self.kwargs)
            
            self._val_set = ICASSPDataset(split="val", *self.args, **self.kwargs)
            self.val_set_new_room = ICASSPDataset(split="val_new_room", *self.args, **self.kwargs)
            self.val_set_new_freq = ICASSPDataset(split="val_new_freq", *self.args, **self.kwargs)
            self.val_set_new_ant = ICASSPDataset(split="val_new_ant", *self.args, **self.kwargs)
            self.val_set_new_room_freq = ICASSPDataset(split="val_new_room_freq", *self.args, **self.kwargs)
            self.val_set_new_room_ant = ICASSPDataset(split="val_new_room_ant", *self.args, **self.kwargs)
            self.val_set_new_freq_ant = ICASSPDataset(split="val_new_freq_ant", *self.args, **self.kwargs)
            
            self._test_set = ICASSPDataset(split="test", *self.args, **self.kwargs)
            self.test_set_new_room = ICASSPDataset(split="test_new_room", *self.args, **self.kwargs)
            self.test_set_new_freq = ICASSPDataset(split="test_new_freq", *self.args, **self.kwargs)
            self.test_set_new_ant = ICASSPDataset(split="test_new_ant", *self.args, **self.kwargs)
            self.test_set_new_room_freq = ICASSPDataset(split="test_new_room_freq", *self.args, **self.kwargs)
            self.test_set_new_room_ant = ICASSPDataset(split="test_new_room_ant", *self.args, **self.kwargs)
            self.test_set_new_freq_ant = ICASSPDataset(split="test_new_freq_ant", *self.args, **self.kwargs)
    
    def val_dataloader(self) -> Union[list[DataLoader], DataLoader]:
        self.dataloader_val = DataLoader(
            self._val_set, batch_size=self._batch_size, num_workers=self._num_workers,
            sampler=DistributedSampler(
                self._val_set, shuffle=False, drop_last=self._drop_last
            ) if self._multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self._drop_last
        )
        if self.kwargs["task_idx"] == 2:
            self.dataloader_val_new_room = DataLoader(
                self.val_set_new_room, batch_size=self._batch_size, num_workers=self._num_workers,
                sampler=DistributedSampler(
                    self.val_set_new_room, shuffle=False, drop_last=self._drop_last
                ) if self._multi_gpu else None,
                collate_fn=self.collate_fn, drop_last=self._drop_last
            )
            self.dataloader_val_new_freq = DataLoader(
                self.val_set_new_freq, batch_size=self._batch_size, num_workers=self._num_workers,
                sampler=DistributedSampler(
                    self.val_set_new_freq, shuffle=False, drop_last=self._drop_last
                ) if self._multi_gpu else None,
                collate_fn=self.collate_fn, drop_last=self._drop_last
            )
            return [self.dataloader_val, self.dataloader_val_new_room, self.dataloader_val_new_freq]
        return self.dataloader_val
    
    def test_dataloader(self) -> Union[list[DataLoader], DataLoader]:
        self.dataloader_test = DataLoader(
            self._test_set, batch_size=self._batch_size, num_workers=self._num_workers,
            sampler=DistributedSampler(
                self._test_set, shuffle=False, drop_last=self._drop_last
            ) if self._multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self._drop_last
        )
        if self.kwargs["task_idx"] == 2:
            self.dataloader_test_new_room = DataLoader(
                self.test_set_new_room, batch_size=self._batch_size, num_workers=self._num_workers,
                sampler=DistributedSampler(
                    self.test_set_new_room, shuffle=False, drop_last=self._drop_last
                ) if self._multi_gpu else None,
                collate_fn=self.collate_fn, drop_last=self._drop_last
            )
            self.dataloader_test_new_freq = DataLoader(
                self.test_set_new_freq, batch_size=self._batch_size, num_workers=self._num_workers,
                sampler=DistributedSampler(
                    self.test_set_new_freq, shuffle=False, drop_last=self._drop_last
                ) if self._multi_gpu else None,
                collate_fn=self.collate_fn, drop_last=self._drop_last
            )
            return [self.dataloader_test, self.dataloader_test_new_room, self.dataloader_test_new_freq]
        return self.dataloader_test
    
    @property
    def test_set(self):
        if self.kwargs["task_idx"] == 1:
            return self._test_set
        if self.kwargs["task_idx"] == 2:
            return [self._test_set, self.test_set_new_room, self.test_set_new_freq]
        if self.kwargs["task_idx"] == 3:
            return [
                self._test_set,
                self.test_set_new_room, self.test_set_new_freq, self.test_set_new_ant,
                self.test_set_new_room_freq, self.test_set_new_room_ant, self.test_set_new_freq_ant
            ]
    
    @property
    def val_set(self):
        if self.kwargs["task_idx"] == 1:
            return self._val_set
        if self.kwargs["task_idx"] == 2:
            return [self._val_set, self.val_set_new_room, self.val_set_new_freq]
        if self.kwargs["task_idx"] == 3:
            return [
                self._val_set,
                self.val_set_new_room, self.val_set_new_freq, self.val_set_new_ant,
                self.val_set_new_room_freq, self.val_set_new_room_ant, self.val_set_new_freq_ant
            ]
