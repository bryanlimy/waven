
import numpy as n
import os

import numpy as np
from scipy.io import loadmat
from suite2p.utils import mpeppy as mp
from suite2p.utils import cortex_lab_utils as clu
from suite2p.utils import utils
import suite2p.utils.twophopy as twop
import suite2p.utils.tiff_utils as tif
from scipy.ndimage import filters
from scipy import stats
from skimage import transform
import matplotlib.pyplot as plt
# import pycolormap_2d as cmap2d
# from suite2p.suite2p.detection import svd_utils as svu
from scipy import ndimage
from suite2p.modules import RetinotopicMapping as rm
plt.rcParams["figure.figsize"] = [10.0,6.0]
plt.style.use('ggplot')
plt.style.use('seaborn-muted')

def do_retinotopy(expt_info, clu_dirs, n_resized = 16, f_interp_Hz=10,response_index=5, meso_params=None):
    clu.dirs = clu_dirs
    exp_path = clu.find_expt_file(expt_info,'root' )
    exp = twop.load_experiment(expt_info)
    n_planes = exp['ops'][0]['nplanes']
    ts_stim, stims = load_sparse_noise(expt_info)

    if meso_params is None:
        stacks = tif.load_downsampled_stacks(exp_path, n_resized, n_planes)
        ts_stacks, stacks_z = tif.interp_stack(stacks, exp['plane_times'], zscore=True)
        one_stack_z = stacks_z.mean(axis=1)[:,n.newaxis]
    else:
        stacks = tif.load_downsampled_meso_tifs(exp_path, **meso_params)
        ts_stacks, stacks_z = tif.interp_stack(stacks[:,n.newaxis], exp['plane_times'], zscore=True)
        one_stack_z = stacks_z.mean(axis=1)[:,n.newaxis]

    responses = stim_triggered_average_2D(one_stack_z, ts_stacks, stims, ts_stim, event_type='both')[response_index,0]
    maps = get_neural_maps(responses)

    plot_neural_maps(maps)

    plot_tiled_stim_maps(responses)

def load_sparse_noise(expt_info, dirs=None, all_repeats=True):
    exp_path = clu.find_expt_file(expt_info,'root', dirs=dirs)
    t_on_off, ts_frame = mp.get_stim_times(expt_info, True, verbose=True, dirs=dirs)
    try:
        frame_onsets = ts_frame[0][:-1]
    except:
        print('only one ts_frame')
        t_on_off, ts_frame = mp.get_stim_times(expt_info, True, dirs=dirs, type='block')
        frame_onsets = ts_frame[0]
    try:
        img_seq = loadmat(os.path.join(exp_path,'ImageSequence.mat'))['ImgSeq'][0]
        img_tex = loadmat(os.path.join(exp_path,'ImageTextures.mat'))['ImgTexs']
        stims = n.array([i[0] for i in img_tex])
    except:
        try:
            print('load npy')
            img_seq = n.load(os.path.join(exp_path, 'sparseNoise.sequence.npy'))[0]
            img_tex = n.load(os.path.join(exp_path, 'sparseNoise.textures.npy'))
            stims = img_tex
            print(stims.shape)
        except:
            print('try loading from event data ... signals block definition file')
            block=loadmat(os.path.join(exp_path,'2023-10-04_2_SP045_Block.mat'))
            Bevent=block['block']['events'][0][0][0][0][14]
            Wevent = block['block']['events'][0][0][0][0][16]
            img_tex = (Wevent > 0).astype(int) - (Bevent > 0).astype(int)
            img_tex = img_tex.reshape((8, -1, 20))
            img_tex=np.swapaxes(img_tex, 0,1)
            # img_tex = np.swapaxes(img_tex, 2, 1)
            img_seq=n.arange(img_tex.shape[-1])
            stims = img_tex
    stim_idxs = n.concatenate([[0], n.where(n.diff(img_seq))[0]])

    if all_repeats:
        n_reps = len(t_on_off[0])
        # print(n_reps, t_on_off)
        stims = n.concatenate([stims]*n_reps, axis=0)
        frame_onsets = n.concatenate([tx[:-1] for tx in ts_frame])
        stim_ts = n.concatenate([ts_frame_rep[:-1][stim_idxs] for ts_frame_rep in ts_frame])
    else:
        stim_ts = frame_onsets[stim_idxs]

    
    return stim_ts, stims

def stim_triggered_average_2D(neural_frames, ts_neural, stimulus_frames, ts_stimulus, 
                              event_type='both', response_window_s = 1.0, pre_stim_n_ts = 0,
                              return_resp_window_ts=False):
    """get a stimulus onset/offset triggered average for each patch of the downsampled 2D
    tiff stack, corresponding to each pixel of the stimulus

    Args:
        neural_frames (ndarray): (n_scans, n_planes, n_y_neural, n_x_neural) patches from downsampled tiff stacks, many planes
        ts_neural (ndarray): (n_scans) timestamps of neural frames
        stimulus_frames (ndarray): (n_stim, n_y_stim, n_x_stim) frames of the stimulus presented
        ts_stimulus (ndarray): (n_stim) timestamps of the stimulus frames
        event_type (str, optional): 'on' takes pixel-on events,'off' takes pixel-off events, or 'both'. Defaults to 'both'.
        response_window_s (float, optional): time window after event to capture in the response. Defaults to 1.0.
        f_interp_Hz (int, optional): [description]. Defaults to 10.
    """
    n_scan, n_planes, n_y_neural, n_x_neural = neural_frames.shape
    n_stim, n_y_stim, n_x_stim = stimulus_frames.shape
    
    f_neural_Hz = 1/(n.mean(n.diff(ts_neural)))
    n_response_window = int(response_window_s*f_neural_Hz)
    responses = n.zeros((n_response_window + pre_stim_n_ts, n_planes, n_y_stim, n_x_stim, n_y_neural, n_x_neural))

    response_window_ts = (n.arange(n_response_window+pre_stim_n_ts) - pre_stim_n_ts) / f_neural_Hz

    for y_idx in range(n_y_stim):
#         print(y_idx)
        for x_idx in range(n_x_stim):
#             print(x_idx)
            on_events =  n.where(stimulus_frames[:,y_idx,x_idx] == 1)[0]
            off_events = n.where(stimulus_frames[:,y_idx,x_idx] == -1)[0]
            if event_type=='on':  events = on_events
            elif event_type=='off': events = off_events
            else:
                assert event_type=='both'
                events = n.union1d(on_events, off_events)
            
            event_ts = ts_stimulus[events]
            event_ts = event_ts[event_ts>ts_neural.min()+2.0]
            neural_t_idxs = [n.where(t < ts_neural)[0][0] for t in event_ts]
            
            # print("For %d, %d, event_ts:" % (y_idx, x_idx))
            # print(event_ts)
            # print("Corresponding neural ts")
            # print(ts_neural[neural_t_idxs])
            mean_response = n.mean([neural_frames[neural_t_idx - pre_stim_n_ts:neural_t_idx+n_response_window]\
                                    for neural_t_idx in neural_t_idxs], axis=0)
            responses[:, :,y_idx, x_idx] = mean_response
    if return_resp_window_ts:
        return response_window_ts, responses
    return responses

def get_rf_max_coords(cc_sq, scale=(90,270), mins=(-45,-135), norm=False, abs=False, flip_y=True, flip_x=True):
    squeeze = False
    if len(cc_sq.shape) < 4:
        squeeze = True
        cc_sq = cc_sq[n.newaxis]
    ny, nx, n_stim_y, n_stim_x = cc_sq.shape
    if abs: cc_sq = n.abs(cc_sq)
    cc_sq_f = cc_sq.reshape(ny,nx, n_stim_y * n_stim_x)
    yxmax = n.argmax(cc_sq_f, axis=-1).reshape(ny,nx)
    xmax = (yxmax % n_stim_x).astype(float)
    ymax = (yxmax // n_stim_x).astype(float)
    if flip_y:
        ymax = n_stim_y - ymax
    if flip_x:
        xmax = n_stim_x - xmax

    if scale is not None:
        ymax /= (n_stim_y / scale[0])
        xmax /= (n_stim_x / scale[1])
    if mins is not None:
        ymax += mins[0]
        xmax += mins[1]
    if squeeze:
        ymax = ymax[0]; xmax = xmax[0]
    return ymax, xmax

def filter_rfs(cc, scale=1, sigma=0):
    sigma = sigma * scale
    n_ypix, n_xpix = cc.shape[:2]
    n_data_dim = len(cc.shape) - 2
    data_shape = cc.shape[2:]
    n_ypix *= scale; n_xpix *= scale
    if scale != 1:
        cc = transform.resize(cc, (n_ypix, n_xpix) + data_shape)
    filter_size = (sigma, sigma) + (0,) * n_data_dim
    if sigma > 0:
        cc = filters.gaussian_filter(cc, filter_size, mode='constant')
    cc_f = cc.reshape(n_ypix * n_xpix, -1)
    return cc_f, (n_ypix, n_xpix)
        

def split_trials(stims_all, resps_all, mean_over_window=True, nan_trials = None):
    if nan_trials is None: nan_trials = n.isnan(resps_all.mean(axis=(1,2)))
    resps = resps_all[~nan_trials]
    stims = stims_all[~nan_trials]
    # stim_ts = stim_ts_all[~nan_trials]

    if type(mean_over_window) == bool:
        if mean_over_window:
            resps = resps.mean(axis=-1)
    else:
        resps = resps[:,:,mean_over_window].mean(axis=-1)

    blanks = n.abs(stims).sum(axis=(1,2)) == 0
    bresp = resps[blanks]
    sresp = resps[~blanks]
    sstims = stims[~blanks]

    return sstims, sresp, bresp, nan_trials, blanks

def dot_stim_resp(stims, resps, z_resp=False, abs_stim = True,scale_stim=False):
    reshape=False
    if len(stims.shape) > 2:
        reshape=True
        ns = stims.shape[0]; ndata = stims.shape[1:]
        stims = stims.reshape(stims.shape[0], -1)
    nc = resps.shape[1]
    if z_resp:
        resps = utils.zscore(resps, ax=0)
    if abs_stim:
        stims = n.abs(stims)
    if scale_stim:
        stims = stims / stims.sum(axis=0, keepdims=True)
    # return stims, resps
    cc = stims.T @ resps
    if reshape:
        cc = cc.reshape(*ndata, nc)
    return cc





def get_neural_maps(resps, sigma=None):
    """get two 2D maps, each the size of the neural FOV, colored by the pixel
       of the stimulus to which the neural FOV has the highest response.

    Args:
        resps (ndarray): (n_stim_y, n_stim_x, n_resp_y, n_resp_x)

    Returns:
        maps: (2,n_resp_y, n_resp_x), 0 is the map for the y axis of the stimulus, 1 for the x axis 
    """
    n_stim_y, n_stim_x, n_resp_y, n_resp_x = resps.shape

    maps = n.zeros((2, n_resp_y, n_resp_x))

    for i in range(n_resp_y):
        for j in range(n_resp_x):
            rx = resps[:,:,i,j]
            if sigma is not None:
                rx = filters.gaussian_filter(rx, sigma)
            top_y, top_x = n.unravel_index(n.argmax(rx), (n_stim_y,n_stim_x))
            maps[0,i,j] = top_y
            maps[1,i,j] = top_x
    return maps

def plot_neural_maps(maps, figsize=(15,7), ymin=0, ymax=8, xmin=0, xmax=30, cmap='jet', interp='none', dpi=150, alpha=None):
    f, axs = plt.subplots(1,2, figsize=figsize, dpi=dpi)

    im0 = axs[0].imshow(maps[0], vmin=ymin, vmax=ymax, interpolation=interp, cmap=cmap, alpha=alpha)
    plt.colorbar(im0, ax=axs[0],shrink=0.5)
    axs[0].grid(False)
    axs[0].set_title("Altitude")
    # plt.show()

    im1 = axs[1].imshow(maps[1], vmin=xmin, vmax=xmax, interpolation=interp, cmap=cmap, alpha=alpha)
    plt.colorbar(im1, ax=axs[1],shrink=0.5)
    axs[1].grid(False)
    axs[1].set_title("Azimuth")
    plt.tight_layout()

    
def plot_tiled_stim_maps(resp, num_plots = 5, figsize=(25,8), interp='bicubic', avg=False):
    __, __, n_y, n_x = resp.shape
    prevy = prevx = 0
    f, axs = plt.subplots(num_plots,num_plots, figsize=figsize)
    for i in range(num_plots):
        prevx=0
        yval = int((i+1)*(n_y-1)/num_plots)
        for j in range(num_plots):
            xval = int((j+1)*(n_x-1)/num_plots)
#             print(prevy, yval)
#             print(prevx,xval)
            if avg:
                axs[i][j].imshow(resp[:,:,prevy:yval,prevx:xval].mean(axis=(2,3)),interpolation=interp)
            else:
                axs[i][j].imshow(resp[:,:,yval,xval],interpolation=interp)
            
            axs[i][j].grid(False)
            axs[i][j].set_yticks([])
            axs[i][j].set_xticks([])
            prevx = xval
        prevy = yval
    plt.tight_layout()
    plt.suptitle("")


import math
#https://github.com/zhuangjun1981/retinotopic_mapping/blob/master/retinotopic_mapping/examples/signmap_analysis/retinotopic_mapping_example.ipynb
def visualSignMap(phasemap1, phasemap2):
    """
    calculate visual sign map from two orthogonally oriented phase maps
    """
    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    # gradmap1 = ni.filters.median_filter(gradmap1,100.)
    # gradmap2 = ni.filters.median_filter(gradmap2,100.)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    # gradmag1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))
    # gradmag2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])

            # gradmag1[i,j] = np.sqrt((gradmap1[1][i,j]**2)+(gradmap1[0][i,j]**2))
            # gradmag2[i,j] = np.sqrt((gradmap2[1][i,j]**2)+(gradmap2[0][i,j]**2))

    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))

    return areamap

def plot_retinotopy_summary(img, alt_map, azi_map, sign_map, figsize=(12,12),dpi=150, transpose=False,
                            alt_map_range = (0,4), azi_map_range=(0,8), sign_map_scale = 0.5, aspect=1, img_range=None):

    f,axs = plt.subplots(2,2, figsize=figsize,dpi=dpi)

    if transpose:
        img = img.T; alt_map = alt_map.T
        azi_map = azi_map.T; sign_map = sign_map.T

    other_args={'aspect': aspect}
    if img_range is not None:
        other_args['vmin'] = img_range[0]
        other_args['vmax'] = img_range[1]
    _,ax, im = tif.show_tif(img, ax=axs[0][0], other_args=other_args)
    plt.colorbar(im,ax=ax,shrink=0.75)
    ax.set_title("Mean Img")
    # ax.imshow(sign_map_f,cmap='RdYlBu_r',alpha=0.3)


    _,ax,im = tif.show_tif(alt_map, ax=axs[0][1],cmap='hsv_r',
                           other_args={'vmin':alt_map_range[0],'vmax':alt_map_range[1], 'aspect':aspect},flip=1)
    plt.colorbar(im,ax=ax,shrink=0.75)
    ax.set_title("Altitude")
    _,ax,im = tif.show_tif(azi_map, ax=axs[1][1],cmap='hsv_r', 
                           other_args={'vmin':azi_map_range[0],'vmax':azi_map_range[1], 'aspect':aspect},flip=1)
    ax.set_title("Azimuth")
    plt.colorbar(im,ax=ax,shrink=0.75)



    vscale = n.max(n.abs(sign_map))*sign_map_scale
    _,ax,im = tif.show_tif(sign_map, cmap='RdYlBu_r', other_args={'vmin':-vscale,'vmax':vscale, 'aspect':aspect}, flip=-1, ax=axs[1][0])
    plt.colorbar(im,ax=ax,shrink=0.75)
    ax.set_title("Sign Map")
    # axs[1][0].imshow(sign_map_f, cmap='RdYlBu_r', vmin=-vscale, vmax=vscale)
    plt.tight_layout()
    
    return f

def plot_rf(map, ax=None, figsize=(8,6), dpi=150, auto_scale = 2, scale=None, center_cmap=True, cmap='bwr_r'):
    f = None
    if ax is None:
        f,ax = plt.subplots(figsize=figsize, dpi=dpi)

    if scale is None:
        scale = auto_scale*map.std()
    
    center = map.mean()

    vmin, vmax = center - scale, center + scale

    if center_cmap:
        scale = max(n.abs(center-scale), n.abs(center + scale))
        vmin, vmax = -scale, scale
    # print(vmin,vmax)
    ax.imshow(map, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])

    return f,ax

def compute_centroid(map2d, xx=None, yy=None,sign='abs', method='w_mean'):
    n_ypix,n_xpix = map2d.shape
    map2d = map2d.copy()
    if sign == 'pos':
        map2d[map2d < 0] = 0
    elif sign == 'neg':
        map2d[map2d > 0] = 0
        map2d = map2d * -1
    elif sign == 'abs':
        map2d = n.abs(map2d)

    if method == 'w_mean':
        if xx is None or yy is None:
            yy,xx = n.meshgrid(n.arange(n_ypix), n.arange(n_xpix), indexing='ij')
            
        xcp = (xx * map2d).sum() / (map2d.sum())
        ycp = (yy * map2d).sum() / (map2d.sum())

    elif method == 'max':
        ycp, xcp = n.unravel_index(n.argmax(map2d), map2d.shape)

    return ycp, xcp

def round_coords(yc, xc, shape=None, ravel=True):
    best_x = int(n.round(xc))
    best_y = int(n.round(yc))
    if ravel:
        n_ypix, n_xpix = shape
        best_xy = best_y * n_xpix + best_x
        return best_xy, best_y, best_x
    return best_y, best_x


def get_on_off_responses(spks,spk_ts,stim, stim_ts, win=(-0.5,1.5), compute_sem=True):
    if len(spks.shape) == 1: spks = spks[n.newaxis]
    n_cells = spks.shape[0]
    stim_on = stim_ts[n.where(stim == 1)]
    print(stim_on)
    stim_off = stim_ts[n.where(stim == -1)]
    resp_ts, resps_on = utils.extract_event_responses(spk_ts, 
                                 spks,stim_on, window=win)
    resp_ts, resps_off = utils.extract_event_responses(spk_ts, 
                                 spks,stim_off, window=win)
    resps_all = n.concatenate([resps_on, resps_off])
    
    if n_cells == 1:
        resps_on = resps_on[:,0]
        resps_off = resps_off[:,0]
        resps_all = resps_all[:,0]

    resp_on  = resps_on.mean(axis=0)
    resp_off = resps_off.mean(axis=0)
    resp_all = resps_all.mean(axis=0)
    
    if compute_sem:
        sem_on =  stats.sem(resp_on, axis=0)
        sem_off = stats.sem(resp_off, axis=0)
        sem_all = stats.sem(resp_all, axis=0)
        return resp_ts, resp_all, resp_on, resp_off, sem_all, sem_on, sem_off
    
    return resp_ts, resp_all, resp_on, resp_off

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl 

def add_colorbar(fig, axis, mappable, ticks=None, orientation='horizontal', loc = None):
    if orientation == 'horizontal':
        height = '5%'; width = '60%'
        if loc is None: loc = 'upper center'
    elif orientation == 'vertical':
        height = '60%'; width = '5%'
        if loc is None: loc = 'center left'
    cax = inset_axes(axis, height=height, width=width,loc=loc)
    fig.colorbar(mappable, cax=cax, orientation=orientation)
    if ticks is not None:
        if orientation == 'horizontal':
            cax.set_xticks(ticks)
        else: cax.set_yticks(ticks)
    return cax


def plot_cell_rf_summary(cell_idx, rfs_sq, yc, xc,resp, resp_pred, r2s, on_offs,
                         rtx=None, rty=None, alphas=None, hor_clims=None, vert_clims=None, my=None, mx=None,
                         figsize=(15,10), dpi=150, rf_auto_scale = 3):
    c1 = 'lightseagreen'
    c2 = 'maroon'
    c3 = 'black'

    rf = rfs_sq[cell_idx]
    n_ypix, n_xpix = rf.shape

    xyc, yc_int, xc_int = round_coords(yc, xc, rf.shape)
    resp_ts, resp_all, resp_on, resp_off, sem_all, sem_on, sem_off = on_offs

    plot_rtmap = True
    if rtx is None:
        plot_rtmap = False


    f = plt.figure(figsize=(15, 10),dpi=dpi)
    if plot_rtmap:
        gs = f.add_gridspec(2,3)
    else:
        gs = f.add_gridspec(2,2)

    ax1 = f.add_subplot(gs[0,:2])
    ax2 = f.add_subplot(gs[1,0])
    ax3 = f.add_subplot(gs[1,1])
    
    __,ax1 = plot_rf(rf, center_cmap=True, auto_scale=rf_auto_scale, ax=ax1)
    ax1.grid(False)
    ax1.set_title("Receptive Field for cell %d, centroid: %.2f, %.2f" % (cell_idx,yc,xc))
    ax1.set_xticks(n.arange(0,n_xpix, 2))
    ax1.set_yticks(n.arange(0,n_ypix, 2))
    ax1.scatter([xc], [yc], s=100, color='blue', edgecolor='k', linewidth=2)
    
    ax2.set_facecolor('white')
    ax2.set_title("Prediction, R2 : %.04f" % r2s[cell_idx])
    ax2.hist2d(resp[:,cell_idx], resp_pred[:,cell_idx], bins = 20,density=True, 
                alpha=1.0, cmap='Greens', norm=mpl.colors.LogNorm(vmin=1e-4, vmax=5e-2));
    ax2.scatter(resp[:,cell_idx], resp_pred[:,cell_idx], s = 2, 
                color='k', alpha=1.0)
    ax2.set_xlabel("True response")
    ax2.set_ylabel("Predicted Response")

    ax3.set_facecolor('white')

    ax3.set_title("Average response to change in pixel (%d, %d)" % (yc_int,xc_int)) 
    ax3.grid(False)
    ax3.plot(resp_ts, resp_on, color=c1, label='on', linewidth=3)
    ax3.fill_between(resp_ts, resp_on-sem_on,
                    resp_on+sem_on, color=c1, alpha=0.3)

    ax3.plot(resp_ts, resp_off, color=c2, label='off', linewidth=3)
    ax3.fill_between(resp_ts, resp_off-sem_off,
                    resp_off+sem_off, color=c2, alpha=0.3)

    ax3.plot(resp_ts, resp_all, color=c3, label='all', linewidth=3)
    ax3.fill_between(resp_ts, resp_all-sem_all,
                    resp_all+sem_all, color=c3, alpha=0.3)
    ax3.set_xlabel("Time from stimulus")
    plt.legend(frameon=False)

    if plot_rtmap:
        ax4 = f.add_subplot(gs[1,2])
        __, __, im4= tif.show_tif(rtx.T, cmap='jet', vminmax=hor_clims, ax=ax4, alpha=alphas.T)
        add_colorbar(f, ax4, im4, ticks=[hor_clims[0], xc, hor_clims[1]], loc='upper center')
        ax4.scatter([my], [mx], s=200,linewidth=3,edgecolor='k',facecolors='none', alpha=1)

        ax5 = f.add_subplot(gs[0,2])
        __, __, im5 = tif.show_tif(rty.T, cmap='jet', vminmax=vert_clims, ax=ax5, alpha=alphas.T)
        add_colorbar(f, ax5, im5, ticks=[vert_clims[0], yc, vert_clims[1]], loc='upper center')
        ax5.scatter([my], [mx], s=200,linewidth=3,edgecolor='k',facecolors='none', alpha=1)

    plt.tight_layout()

    return f
#
# def plot_rf_map_2d(altmap, azimap, clim_azi = (35,90), clim_alt = (-15,30), alpha=None, cbar=True, ax = None,
#                     cbar_loc = 'left', figsize=(8,8), cmap_func=None, plot=True):
#
#     if cmap_func is None:
#         cmap_func = cmap2d.ColorMap2DSchumann
#     cmap = cmap_func(range_x = clim_azi, range_y = clim_alt)
#     ny,nx = altmap.shape
#     imrgb = n.ones(altmap.shape + (4,), dtype=int)
#     for yi in range(ny):
#         for xi in range(nx):
#             imrgb[yi,xi,:3] = cmap(azimap[yi,xi],altmap[yi,xi])
#             if alpha is not None: imrgb[yi,xi,3] = int(alpha[yi,xi] * 255)
#             else: imrgb[yi,xi,3] = 255
#     if not plot: return imrgb / 255
#     if ax is None:
#         plt.figure(figsize=figsize)
#         f,ax = plt.subplots()
#     ax.imshow(imrgb)
#     ax.grid(False)
#
#     if cbar:
#         if cbar_loc == 'left':
#             cbar_loc = [0.1, 0.75, 0.15, 0.15]; cbar_ori='vertical'
#         elif cbar_loc == 'right':
#             cbar_loc = [0.88, 0.4, 0.15, 0.15]; cbar_ori='vertical'
#         cax = ax.inset_axes(cbar_loc)
#         key_shape = (100)
#         key_mat = n.zeros((2,key_shape,key_shape))
#         ycoords = n.linspace(*clim_alt[::-1], key_shape)
#         xcoords = n.linspace(*clim_azi, key_shape)
#         key_mat[0,:] = ycoords[:,n.newaxis]
#         key_mat[1,:] = xcoords[n.newaxis]
#         key_color = n.zeros(key_mat[0].shape + (3,), int)
#         for i in range(key_shape):
#             for j in range(key_shape):
#                 key_color[i,j] = cmap(key_mat[1][i,j], key_mat[0,i,j])
#         cax.imshow(key_color)
#         cax.set_xticks([0,99], [clim_azi[0], clim_azi[1]])
#         cax.set_yticks([0,99], [clim_alt[1], clim_alt[0]])
#         cax.grid(False)
#
#
# #
# def do_svd_retinotopy(u,s,resp,shape,vstim,
#         n_comps, scale, sigma_rf, sigma_us, pmap_sigma,
#         smap_sigma, smap_thresh, scale_stims=True, z_resp=False, return_cc_sq=False, full_trial=False, plot=False):
#     ny,nx = shape
#     n_comps = int(n_comps); scale = int(scale)
#     u_sq = u.reshape(ny,nx,-1)[:,:,:n_comps]
#     resp_f = resp[:,:n_comps]
#     s_f = s[:n_comps]
#     u_f = ndimage.gaussian_filter(u_sq,sigma=(sigma_us,sigma_us,0)).reshape(ny*nx,n_comps)
#     pow_map = svu.reconstruct(u_f,s,resp_f.mean(axis=0),n_comp=n_comps).compute().reshape(ny,nx)
#     cc = dot_stim_resp(vstim, resp_f, scale_stim=scale_stims,z_resp=z_resp)
#     ccbig_gf, (n_ypix_big, n_xpix_big) = filter_rfs(cc, scale=scale, sigma = sigma_rf)
#     cc_sq = svu.reconstruct(u_f, s_f, ccbig_gf.T, n_comp = n_comps)\
#             .compute().reshape(ny, nx, n_ypix_big, n_xpix_big)
#     altmap, azimap = get_rf_max_coords(cc_sq, abs=True)
#
#     clim_x = (0,130)
#     clim_y = (-30,30)
#     rfmap_2d = plot_rf_map_2d(altmap, azimap, clim_azi = clim_x, clim_alt=clim_y, plot=plot)
#
#     rtx = rm.RetinotopicMappingTrial(altmap, azimap,pow_map, pow_map,pow_map, 'x',1)
#     rtx.params['phaseMapFilterSigma'] = pmap_sigma
#     rtx.params['signMapFilterSigma'] = smap_sigma
#     rtx.params['signMapThr'] = smap_thresh
#     altmap, azimap, __, __, sigmap, signmap_f = rtx._getSignMap(isPlot=plot);
#     patchmap = rtx._getRawPatchMap(isPlot=plot)
#     if not full_trial:
#         return pow_map, altmap, azimap, rfmap_2d, sigmap, signmap_f, patchmap
#     else:
#         rtx.processTrial(isPlot=plot)
#         return rtx, pow_map, altmap, azimap, rfmap_2d, sigmap, signmap_f, patchmap


def load_retinotopy(rt_dir, nz = None):
    patches_path = os.path.join(rt_dir, 'patches.npy')
    maps_path = os.path.join(rt_dir, 'maps.npy')

    patches = n.load(patches_path, allow_pickle=True).item()
    maps = n.load(maps_path, allow_pickle=True).item()
    areas = list(patches.keys())
    shape = patches[areas[0]].getBorder().shape
    borders = n.zeros(shape, dtype=int)
    for i,patch_name in enumerate(areas):
        borders[~n.isnan(patches[patch_name].getBorder())] = i + 1
    if nz is not None:
        borders = n.array([borders] * nz)
    return areas, borders, maps, patches
        