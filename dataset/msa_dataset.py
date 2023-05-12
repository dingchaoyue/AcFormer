from collections import defaultdict
import copy
from torch.utils.data import Dataset
from transformers import *
from dataset.create_dataset import MOSI, MOSEI, UR_FUNNY, PAD, UNK
import os
import re
import torch
import torchaudio
import subprocess
from torch.utils.data import Dataset 
import pandas as pd
import re
import nltk
import numpy as np
# nltk.download('punkt')
from contractions import contractions_dict # 导入缩写字典
from PIL import Image
import torchvision

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    ResizedVideo,)

from pytorchvideo.data.encoded_video import EncodedVideo

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,)

from torchvision.transforms import (
    Compose,
    Lambda,)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class MSADataset(Dataset):
    def __init__(self, config, split='train'):
        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()
        # train/dev/test data split
        assert split in ['train', 'dev', 'test'], "check your dataset split" 
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(split)
        
        if "ur_funny" in str(config.data_dir).lower():
            print(f"before data clean || length of self.data is: {len(self.data)}")
            # bad_cases = [item for item in self.data if item[0][3]==[""]]
            # for bad_case in bad_cases:
            #     self.data.remove(bad_case)
            self.data = list(filter(lambda item: item[0][3]!=[''], self.data))
            print(f"after data clean || length of self.data is: {len(self.data)}")

        self.len = len(self.data)  # 16315

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id  # len(): 2729, len() = 16819
        config.pretrained_emb = self.pretrained_emb  # torch.Size([16819, 300])

    def __getitem__(self, index):
        return self.data[index]  # word_token, vidio75, audio81, word

    def __len__(self):
        return self.len


class MSAPretrainDataset(Dataset):
    def __init__(self, config, split='train', pretrain_transform=None, **kwargs):
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
        video_dir, audio_dir, transcript_dir = config['video_dir'], config['audio_dir'], config['transcript_dir']
        if split == 'train':
            self.split_file = config['train_file']
        else:
            raise Exception("check your split mode")
        self.data_list = self._get_data(self.split_file, video_dir, audio_dir, transcript_dir)
        self.pretrain_transform = pretrain_transform     
        # initialize video processor
        side_size = 224
        crop_size = 224
        self.num_frames = num_frames = config['num_frames']
        self.start_sec = 0
        self.video_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    # NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    ResizedVideo(size=(crop_size, crop_size))
                ]
            ),
        )
        # initialize audio processor
        self.mel_bins = config['freq_bins'] # default 128

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx] 
        video_file = data[0]
        audio_file = data[1]
        text_data = data[2] # list, ['REALLY', 'BAD', 'MOVIE']
        target = data[3] # int, 0
        # load video and audio
        video_data = self._load_video(video_file) # video_data.shape torch.Size([3, 8, 256, 256])
        # video_data_aug = self._load_video(video_file) # !!! repeat calls
        video_data_aug = copy.deepcopy(video_data)
        # if self.pretrain_transform is not None:
        #     for i,j in zip(range(self.num_frames),range(self.num_frames)):
        #         image = torchvision.transforms.ToPILImage(mode='RGB')(video_data[:,i,:,:])
        #         image_aug = torchvision.transforms.ToPILImage(mode='RGB')(video_data_aug[:,j,:,:])
        #         # check sampling image frames 
        #         # image.save('./{}.test.png'.format(i)) 
        #         # image_aug.save('./{}.test_aug.png'.format(j)) 
        #         video_data[:,i,:,:] = self.pretrain_transform(image)
        #         video_data_aug[:,j,:,:] = self.pretrain_transform(image_aug)
        audio_data, wav_lens = self._load_audio_fbank(audio_file, self.mel_bins) #  audio_data.shape torch.Size([1024, 128])
        output_dict = {
            'video': video_data,
            'video_aug': video_data_aug,
            'audio': audio_data,
            'wav_lens': wav_lens,
            'text':text_data,
            'label':target
            }
        return output_dict
    
    def _load_video(self, video_file):
        assert os.path.exists(video_file), f"check your video_file path:{video_file}"
        video = np.load(video_file)
        return video
    
    def _load_video_raw(self, video_file):
        # Initialize an EncodedVideo helper class
        video = EncodedVideo.from_path(video_file, decode_audio=False, decoder="pyav")
        # The duration of the input clip is also specific to the model
        # self.clip_duration = (num_frames * sampling_rate) / fps
        clip_duration = video.duration.__ceil__() 
        end_sec = self.start_sec + clip_duration
        # print("start sec {0} and end sec {1}".format(self.start_sec,end_sec))
        # Load the desired clip
        video_data = video.get_clip(start_sec=self.start_sec, end_sec=end_sec) # video_data['video'].shape torch.size(3, frames, 360, 640)
        
        # Apply a transform to normalize the video input
        video_data = self.video_transform(video_data) # video_data.keys() dict_keys(['video', 'audio'])
        # move the inputs to the desired device
        video_inputs = video_data["video"] # video_inputs.shape  torch.Size([3, 8, 256, 256])
        return video_inputs

    def _load_audio_fbank(self, audio_file, mel_bins, target_length=1024):
        '''        
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        '''
        # waveform, sample_rate = torchaudio.load(audio_file)
        waveform, sr = torchaudio.load(audio_file)
        assert sr == 16000, 'input audio sampling rate must be 16kHz'
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_length=25, frame_shift=10)   
        # padding note: n_frames of 10s wav is around 1035
        n_frames = fbank.shape[0]
        # wav_lens = float(n_frames / target_length)
        wav_lens = n_frames
        # p = target_length - n_frames
        # if p > 0:
        #     m = torch.nn.ZeroPad2d((0, 0, 0, p))
        #     fbank = m(fbank)
        # elif p < 0:
        #     fbank = fbank[0:target_length, :]
        # fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank, wav_lens

    def _get_data(self, split_file, video_dir, audio_dir, transcript_dir):
        data_list = []
        # iemocap ['Happy', 'Sad', 'Angry', 'Neutral']  
        label_names=['anger','disgust','fear','joy','neutral','sadness','surprise']
        df = pd.read_csv(split_file, header=0)
        for index, row in df.iterrows():
            id = 'dia' + str(row['Dialogue_ID']) +'_'+'utt' + str(row['Utterance_ID'])
            video_id = id + '.npy'
            audio_id = id + '.wav'
            if os.path.exists(os.path.join(video_dir, video_id)) and os.path.exists(os.path.join(audio_dir, audio_id)):
                video_file = os.path.join(video_dir, video_id)
                audio_file = os.path.join(audio_dir, audio_id)
            else:
                print("Missing Data {0} {1}".format(os.path.join(video_dir, video_id),os.path.join(audio_dir, audio_id)))
                continue
            raw_text = row['Utterance'].replace('',"'")
            tokens = [contractions_dict.get(token.lower(), token) for token in raw_text.split()]
            new_text = " ".join(tokens)
            text_file = self.process_text(new_text)
            target = label_names.index(row['Emotion'].strip().lower())
            assert target in [0,1,2,3,4,5,6]
            data_list.append((video_file, audio_file, text_file, target))
        print("Length of Dataframe is {0}".format(len(df)))
        print("Length of Datalist is {0}".format(len(data_list)))
        return data_list

    def _get_data_raw(self, split_file, video_dir, audio_dir, transcript_dir):
        data_list = []
        # iemocap ['Happy', 'Sad', 'Angry', 'Neutral']  
        label_names=['anger','disgust','fear','joy','neutral','sadness','surprise']
        df = pd.read_csv(split_file, header=0)
        for index, row in df.iterrows():
            id = 'dia' + str(row['Dialogue_ID']) +'_'+'utt' + str(row['Utterance_ID'])
            video_id = id + '.mp4'
            audio_id = id + '.wav'
            if os.path.exists(os.path.join(video_dir, video_id)) and os.path.exists(os.path.join(audio_dir, audio_id)):
                video_file = os.path.join(video_dir, video_id)
                audio_file = os.path.join(audio_dir, audio_id)
            else:
                print("Missing Data {0} {1}".format(os.path.join(video_dir, video_id),os.path.join(audio_dir, audio_id)))
                continue
            raw_text = row['Utterance'].replace('',"'")
            tokens = [contractions_dict.get(token.lower(), token) for token in raw_text.split()]
            new_text = " ".join(tokens)
            text_file = self.process_text(new_text)
            target = label_names.index(row['Emotion'].strip().lower())
            assert target in [0,1,2,3,4,5,6]
            data_list.append((video_file, audio_file, text_file, target))
        print("Length of dataframe is {0}".format(len(df)))
        print("Length of datalist is {0}".format(len(data_list)))
        return data_list
    
    def _get_video_fps(self, video_path: str):
        ext = os.path.splitext(video_path)[-1]
        if ext != '.mp4' and ext != '.avi' and ext != '.flv':
            raise Exception('format not support')
        ffprobe_cmd = 'ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {}'
        p = subprocess.Popen(
            ffprobe_cmd.format(video_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
        out, err = p.communicate()
        print("subprocess excuse results :{} err:{}".format(out, err))
        fps_info = str(out, 'utf-8').strip()
        if fps_info:
            if fps_info.find("/") > 0:
                video_fps_str = fps_info.split('/', 1)
                fps_result = int(int(video_fps_str[0]) / int(video_fps_str[1]))
            else:
                fps_result = int(fps_info)
        else:
            raise Exception('get fps error')
        print('fps of video is :{}'.format(fps_result))
        return fps_result
    def process_text(self, text):
        # 移除特殊字符和标点符号
        text = re.sub(r'[^\w\s]', '', text)    
        # 分词
        tokens = nltk.word_tokenize(text)
        # 返回处理后的结果
        return tokens


class MSAPretrainDataset_(Dataset):
    def __init__(self, config, image_transform, audio_transform, split='train'):
        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()
        # train/dev/test data split
        assert split in ['train', 'dev', 'test'], "check your dataset split" 
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(split)
        self.len = len(self.data)  # 16315

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id  # len(): 2729, len() = 16819
        config.pretrained_emb = self.pretrained_emb  # torch.Size([16819, 300])

        self.image_transform = image_transform
        self.audio_transform = audio_transform
        
    def __getitem__(self, index):
        data = self.data[index]
        # train.append(((words, visual, acoustic, actual_words), label, segment))
        words = data[0][0]
        visual = data[0][1]
        visual_aug = visual
        acoustic = data[0][2]
        acoustic_aug = acoustic
        actual_words = data[0][3]
        label = data[1]
        segment = data[2]
        return ((words, visual, acoustic, actual_words, visual_aug, acoustic_aug), label, segment)
    def __len__(self):
        return self.len
    


if __name__ == "__main__":
    import torchvision
    pretrain_transform = torchvision.transforms.Compose([                        
        # torchvision.transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.RandomApply([
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        # torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
        #                                 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        torchvision.transforms.ToTensor(),
        # torchvision.normalize,
    ])
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                        std=[1/0.229, 1/0.224, 1/0.225]),
                    torchvision.transforms.ToPILImage(),
                ])


    # import torchvision.transforms as transforms
    # from PIL import Image
    # import torch
    # input_tensor = torch.randn(3, 224, 224)

    # transform = transforms.Compose([
    #                 transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #                                     std=[1/0.229, 1/0.224, 1/0.225]),
    #                 transforms.ToPILImage(),
    #             ])
    # image = transform(input_tensor)
    # image.save('image.jpg')

    video_dir = '/home/mnt/cv/acformer/data/meld/segmented_video/'
    audio_dir = '/home/mnt/cv/acformer/data/meld/segmented_audio/'
    transcript_dir = '/home/mnt/cv/acformer/data/meld/segmented_text/'
    train_file = '/home/mnt/cv/acformer/data/meld/MELD_subset.csv'
    frames = 4
    freq_bins = 128
    ds = MSAPretrainDataset(num_frames = frames, freq_bins = freq_bins, 
                            train_file=train_file,  audio_dir=audio_dir,
                            video_dir=video_dir, transcript_dir=transcript_dir, 
                            split = 'train', pretrain_transform=pretrain_transform)
    ds[1]