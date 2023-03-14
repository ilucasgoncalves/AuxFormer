import os
import torch
import torchaudio

# set the path to the folder containing the video files
folder_path = "VideoFlash/"

# set the path to the folder for saving the processed audio files
output_folder_path = "processed_auds/"

# create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# set the sampling rate and number of channels
sampling_rate = 16000
num_channels = 1

# loop through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith(".flv") or file.endswith(".mp4"):
        # read the video file
        video, sample_rate = torchaudio.backend.sox_backend.load(folder_path + file)
        
        # convert the video to mono-channel audio
        audio = torchaudio.functional.convert_audio_dtype(video.mean(dim=0, keepdim=True), dtype=torch.float32).squeeze()
        
        # resample the audio
        audio = torchaudio.transforms.Resample(sample_rate, sampling_rate)(audio)
        
        # set the output file path and name
        output_file = output_folder_path + os.path.splitext(file)[0] + ".wav"
        
        # save the audio file
        torchaudio.save(output_file, audio, sampling_rate=sampling_rate, channels=num_channels)
        
        print(f"Converted {file} to {output_file}")
