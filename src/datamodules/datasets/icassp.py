import logging
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ICASSPDataset(Dataset):
    
    def __init__(
        self,
        split: str,
        data_path: str,
        orig_data_path: str,
        task_idx: int,
        mixup_ratio: float,
        rotate_ratio: float,
        crop_ratio: float,
        crop_size: float,
        idx_to_freq: dict[str, float],
        max_freq: float,
        channels_to_use: list[int],
        *args, **kwargs
    ):
        super().__init__()
        
        self.input_dir: str = os.path.join(data_path, "input")
        self.output_dir: str = os.path.join(data_path, "output")
        self.orig_out_dir: str = os.path.join(orig_data_path, "Outputs", f"Task_{task_idx}_ICASSP")
        self.split: str = split
        self.idx_to_freq: dict[str, float] = idx_to_freq
        self.mixup_ratio: float = mixup_ratio
        self.rotate_ratio: float = rotate_ratio
        self.crop_ratio: float = crop_ratio
        self.crop_size: float = crop_size
        self.fixed_scale: bool = self.crop_size is not None
        self.channels_to_use: list[int] = channels_to_use
        
        self.max_freq: float = max_freq
        self.task_idx: int = task_idx
        
        self.paths: list[str] = self.prepare_paths()
    
    def mixup(
        self, input_img: torch.Tensor, output_img: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.mixup_ratio:
            idx = random.randint(0, len(self) - 1)
            mixup_ratio = self.mixup_ratio
            self.mixup_ratio = 0  # to avoid recursion
            input_img2, output_img2, _, mask2 = self[idx]
            self.mixup_ratio = mixup_ratio
            alpha = random.random()
            input_img = alpha * input_img + (1 - alpha) * input_img2
            output_img = alpha * output_img + (1 - alpha) * output_img2
            mask = mask & mask2
        return input_img, output_img, mask
    
    @staticmethod
    def rotate90(
        input_img: torch.Tensor, output_img: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        angle = random.randint(0, 3)
        input_img = torch.rot90(input_img, angle, [1, 2])
        output_img = torch.rot90(output_img, angle, [1, 2])
        mask = torch.rot90(mask, angle, [0, 1])
        return input_img, output_img, mask
    
    def rotate(
        self, input_img: torch.Tensor, output_img: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.rotate_ratio:
            angle = random.random() * 360
            # noinspection PyUnresolvedReferences
            input_img, output_img, mask = (
                torchvision.transforms.functional.rotate(input_img, angle, fill=-1),
                torchvision.transforms.functional.rotate(output_img, angle, fill=-1),
                torchvision.transforms.functional.rotate(mask, angle, fill=0),
            )
        return input_img, output_img, mask
    
    def crop(
        self, input_img: torch.Tensor, output_img: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = input_img.shape[1:]
        if self.fixed_scale or random.random() < self.crop_ratio:
            if self.fixed_scale:
                h_new, w_new = self.crop_size, self.crop_size
                y = random.randint(0, h - h_new)
                x = random.randint(0, w - w_new)
            else:
                h_new = random.randint(h // 2, h)
                w_new = random.randint(w // 2, w)
                y = random.randint(0, h - h_new)
                x = random.randint(0, w - w_new)
            
            input_img = input_img[:, y:y + h_new, x:x + w_new]
            output_img = output_img[:, y:y + h_new, x:x + w_new]
            mask = mask[y:y + h_new, x:x + w_new]
            
            if not self.fixed_scale:
                input_img = torch.nn.functional.interpolate(
                    input_img.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
                ).squeeze(0)
                output_img = torch.nn.functional.interpolate(
                    output_img.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
                ).squeeze(0)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).to(torch.float), (h, w), mode="nearest"
                ).squeeze(0).squeeze(0).to(torch.bool)
        
        return input_img, output_img, mask
    
    def __getitem__(self, idx: int):
        if self.split == "train":
            path_idx = idx // 4
            flip = idx % 4
            flip_h = flip // 2
            flip_v = flip % 2
        else:
            path_idx = idx
            flip_h = False
            flip_v = False
        
        path: str = self.paths[path_idx]
        
        input_path = os.path.join(self.input_dir, path)
        output_path = os.path.join(self.output_dir, path)
        inp = np.load(input_path)
        input_img = torch.from_numpy(inp["img"].astype(np.float32)).permute((2, 0, 1))
        if self.channels_to_use is not None:
            input_img = input_img[self.channels_to_use]
        output = np.load(output_path)
        output_img = torch.from_numpy(output["out"].astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(output["mask"])
        orig_out_path = os.path.join(self.orig_out_dir, path.rsplit(".", 1)[0])
        
        if self.idx_to_freq is not None and self.max_freq is not None:
            if "f1" in path or self.task_idx == 1:
                frequency = self.idx_to_freq["f1"] / self.max_freq
            elif "f2" in path:
                frequency = self.idx_to_freq["f2"] / self.max_freq
            else:
                frequency = self.idx_to_freq["f3"] / self.max_freq
            freq_channel = torch.full((1, input_img.shape[1], input_img.shape[2]), frequency, dtype=torch.float32)
            input_img = torch.cat((input_img, freq_channel), dim=0)
        
        if flip_h:
            input_img = torch.flip(input_img, [2])
            output_img = torch.flip(output_img, [2])
        if flip_v:
            input_img = torch.flip(input_img, [1])
            output_img = torch.flip(output_img, [1])
        
        if self.split == "train":
            input_img, output_img, mask = self.crop(input_img, output_img, mask)
            if self.mixup_ratio > 0:
                input_img, output_img, mask = self.mixup(input_img, output_img, mask)
            input_img, output_img, mask = ICASSPDataset.rotate90(input_img, output_img, mask)
            input_img, output_img, mask = self.rotate(input_img, output_img, mask)
        
        return input_img, output_img, orig_out_path, mask
    
    def __len__(self):
        if self.split == "train":
            return len(self.paths) * 4
        return len(self.paths)
    
    @staticmethod
    def get_room_number(path):
        return int(path.split("_", 1)[0][1:])
    
    @staticmethod
    def get_freq_number(path):
        return int(path.split("f", 1)[1][0])
    
    @staticmethod
    def get_ant_number(path):
        return int(path.split("Ant", 1)[1][0])
    
    def prepare_paths(self) -> list[str]:
        paths = sorted(os.listdir(self.input_dir))
        
        if self.task_idx == 1:
            if self.split == "train":
                return list(
                    filter(
                        lambda path: (
                            ICASSPDataset.get_room_number(path) <= 19  # or
                            # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                            # 39 <= ICASSPDataset.get_room_number(path)
                        ),
                        paths
                    )
                )
            elif self.split == "val":
                return list(
                    filter(
                        lambda path: (
                            20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                            # 35 <= ICASSPDataset.get_room_number(path) <= 38
                        ), paths
                    )
                )
            else:
                return list(filter(lambda path: 23 <= ICASSPDataset.get_room_number(path) <= 25, paths))
        
        if self.task_idx == 2:
            if self.split == "train":
                return list(
                    filter(
                        lambda path: (
                            (
                                ICASSPDataset.get_room_number(path) <= 19  # or
                                # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                                # 39 <= ICASSPDataset.get_room_number(path)
                            ) and
                            ICASSPDataset.get_freq_number(path) != 2
                        ),
                        paths
                    )
                )
            elif self.split == "val":
                return list(
                    filter(
                        lambda path: (
                            (
                                20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                                # 35 <= ICASSPDataset.get_room_number(path) <= 38
                            ) and
                            ICASSPDataset.get_freq_number(path) == 2
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_room":
                return list(
                    filter(
                        lambda path: (
                            (
                                20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                                # 35 <= ICASSPDataset.get_room_number(path) <= 38
                            ) and
                            ICASSPDataset.get_freq_number(path) != 2
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_freq":
                return list(
                    filter(
                        lambda path: (
                            (
                                ICASSPDataset.get_room_number(path) <= 10  # or
                                # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                                # 39 <= ICASSPDataset.get_room_number(path)
                            ) and
                            ICASSPDataset.get_freq_number(path) == 2
                        ),
                        paths
                    )
                )
            elif self.split == "test":
                return list(
                    filter(
                        lambda path: (
                            23 <= ICASSPDataset.get_room_number(path) <= 25 and
                            ICASSPDataset.get_freq_number(path) == 2
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_room":
                return list(
                    filter(
                        lambda path: (
                            23 <= ICASSPDataset.get_room_number(path) <= 25 and
                            ICASSPDataset.get_freq_number(path) != 2
                        ),
                        paths
                    )
                )
            else:
                return list(
                    filter(
                        lambda path: (
                            11 <= ICASSPDataset.get_room_number(path) <= 19 and
                            ICASSPDataset.get_freq_number(path) == 2
                        ),
                        paths
                    )
                )
        elif self.task_idx == 3:
            if self.split == "train":
                return list(
                    filter(
                        lambda path: (
                            (
                                ICASSPDataset.get_room_number(path) <= 19  # or
                                # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                                # 39 <= ICASSPDataset.get_room_number(path)
                            ) and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "val":
                return list(
                    filter(
                        lambda path: (
                            (
                                20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                                # 35 <= ICASSPDataset.get_room_number(path) <= 38
                            ) and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) == 4
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_room":
                return list(
                    filter(
                        lambda path: (
                            (
                                20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                                # 35 <= ICASSPDataset.get_room_number(path) <= 38
                            ) and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_freq":
                return list(
                    filter(
                        lambda path: (
                            (
                                ICASSPDataset.get_room_number(path) <= 10  # or
                                # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                                # 39 <= ICASSPDataset.get_room_number(path)
                            ) and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_ant":
                return list(
                    filter(
                        lambda path: (
                            (
                                ICASSPDataset.get_room_number(path) <= 10  # or
                                # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                                # 39 <= ICASSPDataset.get_room_number(path)
                            ) and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) == 4
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_room_freq":
                return list(
                    filter(
                        lambda path: (
                            (
                                20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                                # 35 <= ICASSPDataset.get_room_number(path) <= 38
                            ) and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_room_ant":
                return list(
                    filter(
                        lambda path: (
                            (
                                20 <= ICASSPDataset.get_room_number(path) <= 22  # or
                                # 35 <= ICASSPDataset.get_room_number(path) <= 38
                            ) and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) == 4
                        ),
                        paths
                    )
                )
            elif self.split == "val_new_freq_ant":
                return list(
                    filter(
                        lambda path: (
                            (
                                ICASSPDataset.get_room_number(path) <= 10  # or
                                # 26 <= ICASSPDataset.get_room_number(path) <= 34 or
                                # 39 <= ICASSPDataset.get_room_number(path)
                            ) and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) == 4
                        ),
                        paths
                    )
                )
            elif self.split == "test":
                return list(
                    filter(
                        lambda path: (
                            23 <= ICASSPDataset.get_room_number(path) <= 25 and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) == 5
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_room":
                return list(
                    filter(
                        lambda path: (
                            23 <= ICASSPDataset.get_room_number(path) <= 25 and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_freq":
                return list(
                    filter(
                        lambda path: (
                            11 <= ICASSPDataset.get_room_number(path) <= 19 and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_ant":
                return list(
                    filter(
                        lambda path: (
                            11 <= ICASSPDataset.get_room_number(path) <= 19 and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) == 5
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_room_freq":
                return list(
                    filter(
                        lambda path: (
                            23 <= ICASSPDataset.get_room_number(path) <= 25 and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) not in {4, 5}
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_room_ant":
                return list(
                    filter(
                        lambda path: (
                            23 <= ICASSPDataset.get_room_number(path) <= 25 and
                            ICASSPDataset.get_freq_number(path) != 2 and
                            ICASSPDataset.get_ant_number(path) == 5
                        ),
                        paths
                    )
                )
            elif self.split == "test_new_freq_ant":
                return list(
                    filter(
                        lambda path: (
                            11 <= ICASSPDataset.get_room_number(path) <= 19 and
                            ICASSPDataset.get_freq_number(path) == 2 and
                            ICASSPDataset.get_ant_number(path) == 5
                        ),
                        paths
                    )
                )
