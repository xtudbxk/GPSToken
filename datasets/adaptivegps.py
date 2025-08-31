import cv2
import math
import torch
import numpy as np

from .base import ImagePaths

class AdaptiveGPSDataset(ImagePaths):
    def __init__(self, paths_or_file, size, random_crop=False, hflip=True, gps_num=64, adaptive_weight=2.5, adaptive_minhw=8):
        super().__init__(paths_or_file, size, random_crop, hflip)
        self.ps = 256
        self.gps_num = gps_num
        self.adaptive_weight = adaptive_weight
        self.adaptive_minhw = adaptive_minhw

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img = example["image"]
        path = example["path"]

        # grey_image for calculating entropy
        grey_img = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

        assert img.shape[0] % self.ps == 0 and img.shape[1] % self.ps == 0
        gpscodes, regions = [], []
        for hi in range(img.shape[0]//self.ps):
            for wi in range(img.shape[1]//self.ps):
                _gpscodes, _regions = adaptive_initialize(self.gps_num, img=grey_img[hi*self.ps:(hi+1)*self.ps,wi*self.ps:(wi+1)*self.ps], minhw=self.adaptive_minhw, weight=self.adaptive_weight)
                gpscodes.append( _gpscodes )
                regions.append( _regions )
        gpscodes = np.concatenate(gpscodes, axis=0)
        regions = np.concatenate(regions, axis=0)
        example["gpscodes"] = gpscodes
        example["regions"] = regions

        return example

def calculate_complexity_degree(img, weight=1.0):
    try:
        img = (img + 1.0) / 2.0 # (0, 1)
        h, w = img.shape
        img = cv2.resize(img, dsize=(64,64))
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE) # (-4, 4)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2) # [-4*2**0.5, 4*2**0.5]
        hist, _ = np.histogram(edge_magnitude, bins=512, range=(-5.5, 5.5))
        hist = hist / (hist.sum()+1e-5)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # 避免 log(0) 的情况

        complexity_degree = entropy**weight*h*w
    except Exception as e:
        print(f"img: {img.shape}")
    return complexity_degree

def adaptive_initialize(gps_num, img, minhw=8, weight=1.0, cdbias=None):

    # initialize
    h, w = img.shape
    if cdbias is None:
        cdbias = 0.0*img
    region_candidate = [(img, cdbias, calculate_complexity_degree(img, weight=weight)+np.sum(cdbias[:h,:w]), (0,h-1), (0,w-1))]

    while(len(region_candidate) < gps_num):

        # find the one with larget-entropy
        region_candidate.sort(key=lambda x:x[2], reverse=True)
        for _i in range(len(region_candidate)):
            pick_one = region_candidate[_i]
            _, _, _, _hrange, _wrange = pick_one
            if _hrange[1]-_hrange[0] <= minhw and _wrange[1]-_wrange[0] <= minhw:
                continue
            else:
                del region_candidate[_i]
                break

        # print([(_[1],_[2],_[3]) for _ in region_candidate])

        # split it to two region
        _img, _cdbias, _, _hrange, _wrange = pick_one
        _h, _w = _img.shape
        # for x-cross
        if _w > minhw and _w >= _h:
            _img1 = _img[:,:_w//2]
            _cdbias1 = _cdbias[:,:_w//2]
            _ep1 = calculate_complexity_degree(_img1, weight=weight) + np.sum(_cdbias1)
            _hrange1 = _hrange
            _wrange1 = (_wrange[0], _wrange[0]+_w//2-1 )

            _img2 = _img[:,_w//2:]
            _cdbias2 = _cdbias[:,_w//2:]
            _ep2 = calculate_complexity_degree(_img2, weight=weight) + np.sum(_cdbias2)
            _hrange2 = _hrange
            _wrange2 = (_wrange[0] + _w//2, _wrange[1])
        else:
            _ep1, _ep2 = 1000000, 1000000

        # for y-cross
        if _h > minhw and _h >= _w:
            _img3 = _img[:_h//2,:]
            _cdbias3 = _cdbias[:_h//2,:]
            _ep3 = calculate_complexity_degree(_img3, weight=weight) + np.sum(_cdbias3)
            _hrange3 = (_hrange[0], _hrange[0]+_h//2-1)
            _wrange3 = _wrange

            _img4 = _img[_h//2:,:]
            _cdbias4 = _cdbias[_h//2:,:]
            _ep4 = calculate_complexity_degree(_img4, weight=weight) + np.sum(_cdbias4)
            _hrange4 = (_hrange[0]+_h//2, _hrange[1])
            _wrange4 = _wrange
        else:
            _ep3, _ep4 = 1000000, 1000000

        if _w > _h:
            region_candidate.append((_img1, _cdbias1, _ep1, _hrange1, _wrange1))
            region_candidate.append((_img2, _cdbias2, _ep2, _hrange2, _wrange2))
        elif _w < _h:
            region_candidate.append((_img3, _cdbias3, _ep3, _hrange3, _wrange3))
            region_candidate.append((_img4, _cdbias4, _ep4, _hrange4, _wrange4))
        elif min(_ep1,_ep2) < min(_ep3, _ep4):
            region_candidate.append((_img1, _cdbias1, _ep1, _hrange1, _wrange1))
            region_candidate.append((_img2, _cdbias2, _ep2, _hrange2, _wrange2))
        else:
            region_candidate.append((_img3, _cdbias3, _ep3, _hrange3, _wrange3))
            region_candidate.append((_img4, _cdbias4, _ep4, _hrange4, _wrange4))

    # --- convert to gaussian parameters ---
    gpscodes, regions = [], []
    for pick_one in region_candidate:
        _, _, _, hrange, wrange = pick_one
        sigma_x = (wrange[1]-wrange[0])/w*2.0/3.0
        sigma_y = (hrange[1]-hrange[0])/h*2.0/3.0
        rho = 0.0
        x = ((wrange[1]+wrange[0])/2.0 /(w-1) -0.5) * 2.0
        y = ((hrange[1]+hrange[0])/2.0 /(h-1) -0.5) * 2.0

        gpscodes.append((sigma_x, sigma_y, rho, x, y))
        regions.append((
            (wrange[0]/(w-1)-0.5)*2,
            (hrange[0]/(h-1)-0.5)*2,
            (wrange[1]/(w-1)-0.5)*2,
            (hrange[1]/(h-1)-0.5)*2,))

    return np.array(gpscodes).astype(np.float32), np.array(regions).astype(np.float32)
