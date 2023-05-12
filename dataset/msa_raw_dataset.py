import os
import re
import torch
import torchaudio
import subprocess
from torch.utils.data import Dataset 

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
    RandomResizedCropVideo,
    ResizedVideo,
)

from pytorchvideo.data.encoded_video import EncodedVideo

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
)

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

class RawMosiDataset(Dataset):
    def __init__(self, config, split='train'):
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
        elif split == 'dev':
            self.split_file = config['val_file']
        elif split == 'test':
            self.split_file = config['test_file']
        else:
            raise Exception("check your split mode")
        self.data_list = self._get_data(self.split_file, video_dir, audio_dir, transcript_dir)       
        # initialize video processor
        side_size = 224
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 224
        num_frames = config['num_frames']
        self.start_sec = 0

        self.video_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    # RandomHorizontalFlipVideo(p=0.5),
                    ResizedVideo(size=(crop_size, crop_size))
                    # CenterCropVideo(crop_size=(crop_size, crop_size))
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
        target = data[3] # float, -2.6
        # load video and audio
        video_data = self._load_video(video_file) # video_data.shape torch.Size([3, 8, 256, 256])
        audio_data, wav_lens = self._load_audio_fbank(audio_file, self.mel_bins) #  audio_data.shape torch.Size([1024, 128])
        output_dict = {
            'video': video_data,
            'audio': audio_data,
            'wav_lens': wav_lens,
            'text':text_data,
            'label':target
            }
        return output_dict
    
    def _load_video(self, video_file):
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
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank, wav_lens

    def _get_data(self, split_file, video_dir, audio_dir, transcript_dir):
        data_list = []
        with open(split_file, 'r') as f:
            for file in f.readlines():
                name = file.split()[0].strip()
                target = float(file.split()[1].strip())
                video_file = os.path.join(video_dir, name+'.mp4')
                audio_file = os.path.join(audio_dir, name+'.wav')
                trans_match = re.search('.*(?=_)', name, re.M|re.I)
                trans_txt_id = re.search('_(\d+)$', name, re.M|re.I)
                if trans_match and trans_txt_id:
                    text_seed = trans_match.group(0)
                    text_id = int(trans_txt_id.group(1))
                    raw_text = os.path.join(transcript_dir, text_seed+'.annotprocessed')
                    with open(raw_text, 'r') as f:
                        raw_texts = f.readlines()
                        text_file = raw_texts[text_id-1].split()
                        trans_match = re.search('_([^_]*)$', text_file[0], re.M|re.I)
                        if trans_match.group(1)!='':                            
                            first_word = trans_match.group(1)
                            del text_file[0]
                            text_file.insert(0,first_word)
                        else:
                            del text_file[0]
                data_list.append((video_file, audio_file, text_file, target))    
        print("len {0} is {1}".format(split_file.split('/')[-1],len(data_list)))
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

if __name__ == "__main__":
    audio_dir = '/home/mnt/cv/acformer/data/mosi/segmented_audio/WAV_16000/segmented'
    video_dir = '/home/mnt/cv/acformer/data/mosi/segmented_video'
    transcript_dir = '/home/mnt/cv/acformer/data/mosi/segmented_text'

    train_file = '/home/mnt/cv/acformer/data/mosi/train.txt'
    val_file =  '/home/mnt/cv/acformer/data/mosi/dev.txt'
    test_file =  '/home/mnt/cv/acformer/data/mosi/test.txt'
    ds = RawMosiDataset(train_file, 'test', audio_dir, video_dir, transcript_dir)
    ds[1]