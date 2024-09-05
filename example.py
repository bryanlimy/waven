import zebrAnalysis3
import numpy as np
import gc

path = "path_to_zebra_movie_folder"
zebrAnalysis.downsample_video_binary(path+'/zebramovie.mp4')

## wavelet transform of the input stimuli movie
videodata=np.load(path[:-3]+'npy')
zebrAnalysis.waveletDecomposition(videodata, 0, path)
zebrAnalysis.waveletDecomposition(videodata, 1, path)


## parameter for the recorded neural datas
pathdir='/media/sophie/Seagate Basic/video/2screens/10/'
dirs = ['/media/sophie/Seagate Basic/datasets']
exp_info = ('SS002','2024-08-21', 1)
n_planes=1
block_end=9009
nx=135
ny=54
nb_frames=18000
n_trial2keep=3
path='/media/sophie/Expansion1/UCL/datatest/videos/2screens/10/'#'/media/sophie/Seagate Basic/video/2screens/10/'
pathdata=dirs[0]+'/'+exp_info[0]+'/'+exp_info[1]+'/'
pathsuite2p=pathdata+'/suite2p'
downsampling=False


##

spks, neuron_pos=zebrAnalysis.LoadPinkNoise.loadSPKMesoscope(exp_info, dirs, pathsuite2p, block_end, n_planes, nb_frames, first=True,  method='photosensor')
neuron_pos=zebrAnalysis.LoadPinkNoise.correctNeuronPos(neuron_pos)


respcorr_zebra = zebrAnalysis.Analysis_Utils.repetability_trial2(spks, neuron_pos)

wavelets0, wavelets1, wavelet_c = zebrAnalysis.LoadPinkNoise.coarseWavelet(path, downsampling)


rfs_zebra =  zebrAnalysis.Analysis_Utils.PearsonCorrelationPinkNoise(wavelet_c.reshape(18000, -1), np.mean(spks[:, :18000], axis=0),
                                  neuron_pos, 27, 11, plotting=True)
