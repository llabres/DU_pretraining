import os
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

class SP_IDL_denoising(Dataset):
    def __init__(self, config):
        super().__init__()
        self.imdb_path = config['data_path']

        self.image_path = config['image_path']

        self.gt_answers = config['gt_answers'] if 'gt_answers' in config else False
        self.use_images = config['use_images'] if 'use_images' in config else False
        self.use_ocr = config['use_ocr'] if 'use_ocr' in config else False
        self.mp_mode = config['mp_mode'] if 'mp_mode' in config else False

        self.current_imdb = None
        self.imdb_todo = list(range(0, 11)) #list(range(0, 11)) #list(range(0, 54))
        random.shuffle(self.imdb_todo)
        self.i = 0

    def __len__(self):
        return 5325608 #5325608 #26600964

    def __getitem__(self, idx):
        if self.current_imdb is None:
            if self.i == len(self.imdb_todo):
                self.i = 0
            imdb_file = self.imdb_todo[self.i]
            self.current_imdb = np.load(f"{self.imdb_path}/imdb_pretrain_p{imdb_file}.npy", allow_pickle=True)
            self.current_imdb = self.current_imdb[1:]
            self.idxs = list(range(len(self.current_imdb)))
            random.shuffle(self.idxs)

        sample = self.current_imdb[self.idxs.pop()]
        
        if len(self.idxs) == 0:
            self.current_imdb = None

        try:
            new_sample = {}
            if self.use_images:
                image_id = sample['image_id']
                page = sample['ucsf_document_page']
                image_path = os.path.join(self.image_path, image_id[0], image_id[1], f"{image_id}.tif")

                image = Image.open(image_path)
                image.seek(page)
                new_image = image.copy()
                image.close()
                image = new_image.convert("RGB")

                if self.mp_mode:
                    new_sample['image'] = [image]
                else:
                    new_sample['image'] = image

            
            if self.use_ocr:
                if self.mp_mode:
                    new_sample['ocr_tokens'] = [sample['ocr_tokens']]
                    new_sample['ocr_boxes'] = [sample['ocr_normalized_boxes']]
                
                else:
                    new_sample['ocr_tokens'] = sample['ocr_tokens']
                    new_sample['ocr_boxes'] = sample['ocr_normalized_boxes']
        except Exception as e:
            return self.__getitem__(0)


        return new_sample


class SP_IDL_denoising(Dataset):
    def __init__(self, imdb_indx, config):
        super().__init__()
        self.imdb_path = config['data_path']

        self.image_path = config['image_path']

        self.gt_answers = config['gt_answers'] if 'gt_answers' in config else False
        self.use_images = config['use_images'] if 'use_images' in config else False
        self.use_ocr = config['use_ocr'] if 'use_ocr' in config else False
        self.mp_mode = config['mp_mode'] if 'mp_mode' in config else False


        self.imdbs = list(range(0, 11)) #list(range(0, 11)) #list(range(0, 54))
        random.shuffle(self.imdbs)

        self.imdb = np.load(f"{self.imdb_path}/imdb_pretrain_p{imdb_indx}.npy", allow_pickle=True)[1:]

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        sample = self.imdb[idx]

        try:
            new_sample = {}
            if self.use_images:
                image_id = sample['image_id']
                page = sample['ucsf_document_page']
                image_path = os.path.join(self.image_path, image_id[0], image_id[1], f"{image_id}.tif")

                image = Image.open(image_path)
                image.seek(page)
                new_image = image.copy()
                image.close()
                image = new_image.convert("RGB")

                if self.mp_mode:
                    new_sample['image'] = [image]
                else:
                    new_sample['image'] = image

            
            if self.use_ocr:
                if self.mp_mode:
                    new_sample['ocr_tokens'] = [sample['ocr_tokens']]
                    new_sample['ocr_boxes'] = [sample['ocr_normalized_boxes']]
                
                else:
                    new_sample['ocr_tokens'] = sample['ocr_tokens']
                    new_sample['ocr_boxes'] = sample['ocr_normalized_boxes']
        except Exception as e:
            return self.__getitem__(idx-1)


        return new_sample