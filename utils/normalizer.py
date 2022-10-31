import numpy as np
from tqdm import tqdm
def get_norm_stat_for_frame_repr_list(repr_list, feat_dim):
    """
    mel_spec: (D, T)
    """
    sum_vec = np.zeros(feat_dim)
    sqsum_vec = np.zeros(feat_dim)
    count = 0
    for feat_mat in repr_list:
        assert(feat_mat.shape[0]) == feat_dim
        
        feat_sum = np.sum(feat_mat, axis=1)
        feat_sqsum = np.sum(feat_mat**2, axis=1)

        sum_vec += feat_sum
        sqsum_vec += feat_sqsum

        count += feat_mat.shape[1]

    feat_mean = sum_vec / count
    feat_var = (sqsum_vec / count) - (feat_mean**2)

    return feat_mean, feat_var

def get_norm_stat_for_melspec(spec_list, feat_dim=128):
    feat_mean, feat_var = get_norm_stat_for_frame_repr_list(spec_list, feat_dim)
    return feat_mean, feat_var

def get_norm_stat_for_wav(wav_list, verbose=False):
    count = 0
    wav_sum = 0
    wav_sqsum = 0
    
    for cur_wav in tqdm(wav_list):
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav**2)
        count += len(cur_wav)
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)

    return wav_mean, wav_std

def get_norm_stat_for_vid(vid_list, verbose=False):
    count = 0
    vid_sum = 0
    vid_sqsum = 0
    for cur_vid in tqdm(vid_list):
        vid_sum += np.sum(cur_vid, axis=0)
        vid_sqsum += np.sum(cur_vid**2, axis=0)
        count += np.shape(cur_vid)[0]
    
    vid_mean = vid_sum / count
    vid_var = (vid_sqsum / count) - (vid_mean**2)
    vid_std = np.sqrt(vid_var)

    return vid_mean, vid_std