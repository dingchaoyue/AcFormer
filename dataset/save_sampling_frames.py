import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing as mp
import time
import datetime
from transformers import *
from torch.utils.data import Dataset
from torch.utils.data import Dataset 
from contractions import contractions_dict # 导入缩写字典
from torchvision.transforms._transforms_video import (
    ResizedVideo,)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,)
from torchvision.transforms import (
    Compose,
    Lambda,)


num_frames = 8
video_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            ShortSideScale(
                size=224
            ),
            ResizedVideo(size=(224, 224))
        ]
    ),
)

def video_to_npy(video_file, save_file):
    video = EncodedVideo.from_path(video_file, decode_audio=False, decoder="pyav")
    clip_duration = video.duration.__ceil__()
    end_sec = clip_duration
    video_data = video.get_clip(start_sec=0, end_sec=end_sec)
    video_data = video_transform(video_data)
    video_inputs = video_data["video"]
    # video_inputs = video_inputs.cpu().detach().numpy() # direct save tensor
    with open(save_file, 'wb') as f:
        np.save(f, video_inputs)


class MSAPretrainDataset(Dataset):
    def __init__(self, split='train', pretrain_transform=None, **kwargs):
        """
        This dataset handles the storage, loading, decoding and clip sampling for a
        video dataset. It assumes each video is stored as either an encoded video
        (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)        
        
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.
            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.
            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().
            decode_audio (bool): If True, decode audio from video.
            decode_video (bool): If True, decode video frames from a video container.
            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        video_dir, audio_dir, transcript_dir = kwargs['video_dir'], kwargs['audio_dir'], kwargs['transcript_dir']
        if split == 'train':
            self.split_file = kwargs['train_file']
        else:
            raise Exception("check your split mode")        
        self.data_list = self._get_data(self.split_file, video_dir, audio_dir, transcript_dir)             
        side_size = 224
        crop_size = 224
        self.num_frames = num_frames = kwargs['num_frames']
        self.video_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    ShortSideScale(
                        size=side_size
                    ),
                    ResizedVideo(size=(crop_size, crop_size))
                ]
            ),
        )
    # too slow
    # def save_video(self, save_dir):
    #     for item in self.data_list:
    #         video_file = item[0]
    #         name = video_file.split('/')[-1]
    #         match = re.search(r'(.*)\.mp4', name)
    #         if match:
    #             result_name = match.group(1)
    #         else:
    #             print('No match')                    
    #             raise Exception(f"check your video file {video_file}")
    #         video = EncodedVideo.from_path(video_file, decode_audio=False, decoder="pyav")
    #         clip_duration = video.duration.__ceil__() 
    #         end_sec = self.start_sec + clip_duration
    #         video_data = video.get_clip(start_sec=self.start_sec, end_sec=end_sec) # video_data['video'].shape torch.size(3, frames, 360, 640)            
    #         video_data = self.video_transform(video_data) # video_data.keys() dict_keys(['video', 'audio'])
    #         video_inputs = video_data["video"] # video_inputs.shape  torch.Size([3, 8, 256, 256])
    #         save_file = os.path.join(save_dir, result_name+'.npy')
    #         with open(save_file, 'wb') as f:
    #             np.save(save_file, video_inputs)
    #     print("Save video complete")

    def video_to_npy(self, video_file, save_file):
        video = EncodedVideo.from_path(video_file, decode_audio=False, decoder="pyav")
        clip_duration = video.duration.__ceil__()
        end_sec = clip_duration
        video_data = video.get_clip(start_sec=0, end_sec=end_sec)
        video_data = self.video_transform(video_data)
        video_inputs = video_data["video"] # torch.Size([3, 8, 224, 224]) <class 'torch.Tensor'>
        # video_inputs = video_inputs.cpu().detach().numpy()
        with open(save_file, 'wb') as f:
            np.save(f, video_inputs)

    def save_video(self, save_dir):
        n = mp.cpu_count()
        print("Max cores {}".format(n))
        pool=mp.Pool(processes=24)        
        start_time  = time.time()
        for video_file in self.data_list:                
            name = video_file.split('/')[-1]
            match = re.search(r'(.*).mp4', name)
            if match:
                result_name = match.group(1)
            else:
                print('No match')
                raise Exception("check your video file {}".format({video_file}))
            save_file = os.path.join(save_dir, result_name+'.npy')
            # self.video_to_npy(video_file, save_file)        
            pool.apply_async(video_to_npy, args=(video_file, save_file))
        pool.close()
        pool.join()
        end_time = time.time()
        duration = str(datetime.timedelta(seconds=int(end_time-start_time)))
        print("Duration time is {0}".format(duration))
        print("Save video complete")

    def _get_data(self, split_file, video_dir, audio_dir, transcript_dir):
        data_list = []
        df = pd.read_csv(split_file, header=0)
        for index, row in df.iterrows():
            id = 'dia' + str(row['Dialogue_ID']) +'_'+'utt' + str(row['Utterance_ID'])
            video_id = id + '.mp4'
            if os.path.exists(os.path.join(video_dir, video_id)):
                video_file = os.path.join(video_dir, video_id)
            else:
                print("Missing Data {}".format(os.path.join(video_dir, video_id)))
                continue
            data_list.append(video_file)
        print("Length of dataframe is {}".format(len(df)))
        print("Length of datalist is {}".format(len(data_list)))
        return data_list

if __name__ == "__main__":
    msa_ptrainer=MSAPretrainDataset(
        video_dir='/home/mnt/cv/acformer/data/meld/segmented_video/',
        audio_dir='/home/mnt/cv/acformer/data/meld/segmented_audio/',
        transcript_dir= '/home/mnt/cv/acformer/data/meld/segmented_text/',
        train_file= '/home/mnt/cv/acformer/data/meld/MELD_subset.csv',
        num_frames= 8)
    msa_ptrainer.save_video('/home/mnt/cv/acformer/data/meld/segmented_video/')



                          