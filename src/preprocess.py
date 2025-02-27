import signal
from collections import defaultdict
from tqdm import tqdm
from skimage.measure import regionprops
from src.utils import get_config
import os
import skimage.feature as skfeat
import cv2
import numpy as np
import igraph as ig
import json
import logging
import joblib

import multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

try:
    openslide_home = get_config()['openslide-home']
    # if windows
    if os.name == 'nt':
        os.add_dll_directory(openslide_home)
    from openslide import OpenSlide
except Exception as e:
    logging.warning(f'Error in loading OpenSlide: {e}')
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def worker_initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def getRegionPropFromContour(contour, bbox, extention=2):
    (left, top), (right, bottom) = bbox
    height, width = bottom - top, right - left
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    image = np.zeros((height + extention * 2,
                      width + extention * 2),
                     dtype=np.uint8)
    contour = np.array(contour)
    contour[:, 0] = contour[:, 0] - left + extention
    contour[:, 1] = contour[:, 1] - top + extention
    cv2.drawContours(image, [contour], 0, 1, -1)
    regionProp = regionprops(image)[0]
    return regionProp


def getCurvature(contour, n_size=5):
    contour = np.array(contour)
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    samplekeep = np.zeros((len(contour)), dtype=np.bool_)
    samplekeep[0] = True
    flag = 0
    for i in range(1, len(contour)):
        if np.abs(contour[i] - contour[flag]).sum() > 2:
            flag = i
            samplekeep[flag] = True

    contour = contour[samplekeep]
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    ds = np.sqrt(np.sum(dxy ** 2, axis=1, keepdims=True))
    ddxy = dxy / ds
    ds = (ds + np.roll(ds, shift=1)) / 2
    Cxy = np.diff(np.concatenate([ddxy, ddxy[0:1]], axis=0), axis=0) / ds
    Cxy = (Cxy + np.roll(Cxy, shift=1, axis=0)) / 2
    k = (ddxy[:, 1] * Cxy[:, 0] - ddxy[:, 0] * Cxy[:, 1]) / ((ddxy ** 2).sum(axis=1) ** (3 / 2))

    curvMean = k.mean()
    curvMin = k.min()
    curvMax = k.max()
    curvStd = k.std()

    n_protrusion = 0
    n_indentation = 0
    if n_size > len(k):
        n_size = len(k) // 2
    k_circle = np.concatenate([k[-n_size:], k, k[:n_size]], axis=0)
    for i in range(n_size, len(k_circle) - n_size):
        neighbor = k_circle[i - 5:i + 5]
        if k_circle[i] > 0:
            if k_circle[i] == neighbor.max():
                n_protrusion += 1
        elif k_circle[i] < 0:
            if k_circle[i] == neighbor.min():
                n_indentation += 1
    n_protrusion /= len(contour)
    n_indentation /= len(contour)

    return curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation


def SingleMorphFeatures(args):
    ids, name, contours, bboxes = args
    featuresDict = defaultdict(list)
    featuresDict['name'] = name
    for contour, bbox in zip(contours, bboxes):
        regionProps = getRegionPropFromContour(contour, bbox)
        featuresDict['Area'] += [regionProps.area]
        featuresDict['AreaBbox'] += [regionProps.bbox_area]
        featuresDict['CellEccentricities'] += [regionProps.eccentricity]
        featuresDict['Circularity'] += [(4 * np.pi * regionProps.area) / (regionProps.perimeter ** 2)]
        featuresDict['Elongation'] += [regionProps.major_axis_length / regionProps.minor_axis_length]
        featuresDict['Extent'] += [regionProps.extent]
        featuresDict['MajorAxisLength'] += [regionProps.major_axis_length]
        featuresDict['MinorAxisLength'] += [regionProps.minor_axis_length]
        featuresDict['Perimeter'] += [regionProps.perimeter]
        featuresDict['Solidity'] += [regionProps.solidity]

        curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation = getCurvature(contour)
        featuresDict['CurvMean'] += [curvMean]
        featuresDict['CurvStd'] += [curvStd]
        featuresDict['CurvMax'] += [curvMax]
        featuresDict['CurvMin'] += [curvMin]

    return featuresDict


def getMorphFeatures(name, contours, bboxes, desc, process_n=1):
    name = [int(i) for i in name]
    if process_n == 1:
        return SingleMorphFeatures([0, name, contours, bboxes])
    else:
        featuresDict = defaultdict(list)
        vertex_len = len(name)
        batch_size = vertex_len // 8
        for batch in tqdm(range(0, vertex_len, batch_size)):
            p_slice = [slice(batch + i, min(batch + batch_size, vertex_len), process_n) for i in range(process_n)]
            args = [[ids, name[i], contours[i], bboxes[i]] for ids, i in enumerate(p_slice)]
            try:
                with mp.Pool(process_n, initializer=worker_initializer) as p:
                    ans = p.map(SingleMorphFeatures, args)
            except KeyboardInterrupt:
                p.terminate()
                p.join()
                raise KeyboardInterrupt
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    featuresDict[k] += v
    return featuresDict


def getCellImg(slidePtr, bbox, pad=2, level=0):
    bbox = np.array(bbox)
    bbox[0] = bbox[0] - pad
    bbox[1] = bbox[1] + pad
    cellImg = slidePtr.read_region(location=bbox[0] * 2 ** level, level=level, size=bbox[1] - bbox[0])
    cellImg = np.array(cv2.cvtColor(np.asarray(cellImg), cv2.COLOR_RGB2GRAY))
    return cellImg


def getCellMask(contour, bbox, pad=2, level=0):
    if level != 0:
        raise KeyError('Not support level now')
    (left, top), (right, bottom) = bbox
    height, width = bottom - top, right - left
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    cellMask = np.zeros((height + pad * 2,
                         width + pad * 2),
                        dtype=np.uint8)
    contour = np.array(contour)
    contour[:, 0] = contour[:, 0] - left + pad
    contour[:, 1] = contour[:, 1] - top + pad
    cv2.drawContours(cellMask, [contour], 0, 1, -1)
    return cellMask


def mygreycoprops(P):
    # reference https://murphylab.web.cmu.edu/publications/boland/boland_node26.html
    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    Pxplusy = np.zeros((num_level + num_level2 - 1, num_dist, num_angle))
    Ixplusy = np.expand_dims(np.arange(num_level + num_level2 - 1), axis=(1, 2))
    P_flip = np.flip(P, axis=0)
    for i, offset in enumerate(range(num_level - 1, -num_level2, -1)):
        Pxplusy[i] = np.trace(P_flip, offset)
    SumAverage = np.sum(Ixplusy * Pxplusy, axis=0)
    Entropy = - np.sum(Pxplusy * np.log(Pxplusy + 1e-15), axis=0)
    SumVariance = np.sum((Ixplusy - Entropy) ** 2 * Pxplusy, axis=0)

    Ix = np.tile(np.arange(num_level).reshape(-1, 1, 1, 1), [1, num_level2, 1, 1])
    Average = np.sum(Ix * P, axis=(0, 1))
    Variance = np.sum((Ix - Average) ** 2 * P, axis=(0, 1))
    return SumAverage, Entropy, SumVariance, Average, Variance


def SingleGLCMFeatures(args):
    ids, wsiPath, name, contours, bboxes, pad, level = args
    slidePtr = OpenSlide(wsiPath)
    # Use wsipath as parameter because multiprocess can't use pointer like the object OpenSlide() as parameter
    featuresDict = defaultdict(list)
    featuresDict['name'] = name
    for contour, bbox in zip(contours, bboxes):
        cellImg = getCellImg(slidePtr, bbox, pad, level)
        cellMask = getCellMask(contour, bbox, pad).astype(np.bool_)
        cellImg[~cellMask] = 0

        outMatrix = skfeat.graycomatrix(cellImg, [1], [0])
        outMatrix[0, :, ...] = 0
        outMatrix[:, 0, ...] = 0

        homogeneity = skfeat.graycoprops(outMatrix, 'homogeneity')[0][0]
        ASM = skfeat.graycoprops(outMatrix, 'ASM')[0][0]
        contrast = skfeat.graycoprops(outMatrix, 'contrast')[0][0]
        correlation = skfeat.graycoprops(outMatrix, 'correlation')[0][0]
        SumAverage, Entropy, SumVariance, Average, Variance = mygreycoprops(outMatrix)

        featuresDict['ASM'] += [ASM]
        featuresDict['Contrast'] += [contrast]
        featuresDict['Correlation'] += [correlation]
        featuresDict['Entropy'] += [Entropy[0][0]]
        featuresDict['Homogeneity'] += [homogeneity]

        featuresDict['IntensityMean'] += [cellImg[cellMask].mean()]
        featuresDict['IntensityStd'] += [cellImg[cellMask].std()]
        featuresDict['IntensityMax'] += [cellImg[cellMask].max().astype('int16')]
        featuresDict['IntensityMin'] += [cellImg[cellMask].min().astype('int16')]
    return featuresDict


def getGLCMFeatures(wsiPath, name, contours, bboxes, pad=2, level=0, process_n=1):
    name = [int(i) for i in name]
    if process_n == 1:
        return SingleGLCMFeatures([0, wsiPath, name, contours, bboxes, pad, level])
    else:
        featuresDict = defaultdict(list)
        vertex_len = len(name)
        batch_size = vertex_len // 8
        for batch in tqdm(range(0, vertex_len, batch_size)):
            p_slice = [slice(batch + i, min(batch + batch_size, vertex_len), process_n) for i in range(process_n)]
            args = [[ids, wsiPath, name[i], contours[i], bboxes[i], pad, level] for ids, i in enumerate(p_slice)]
            try:
                with mp.Pool(process_n, initializer=worker_initializer) as p:
                    ans = p.map(SingleGLCMFeatures, args)
            except KeyboardInterrupt:
                p.terminate()
                p.join()
                raise KeyboardInterrupt
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    featuresDict[k] += v
    return featuresDict


def basicFeatureExtraction(
        wsiPath, nucleusInfo, level, featureSet
):
    offset = np.array([0, 0])

    bboxes, centroids, contours, types = [], [], [], []

    for nucInfo in tqdm(nucleusInfo['nuc'].values()):
        tmpCnt = np.array(nucInfo['contour'])
        left, top = tmpCnt.min(0)
        right, bottom = tmpCnt.max(0)
        bbox = [[left + offset[0], top + offset[1]], [right + offset[0], bottom + offset[1]]]
        bboxes.append(bbox)
        centroids.append(nucInfo['centroid'])
        contours.append(nucInfo['contour'])
        types.append(nucInfo['type'])
    assert len(bboxes) == len(centroids) == len(
        types), 'The attribute of nodes (bboxes, centroids, types) must have same length'
    vertex_len = len(bboxes)
    globalGraph = ig.Graph()
    names = [str(i) for i in range(vertex_len)]

    globalGraph.add_vertices(vertex_len, attributes={
        'name': names, 'Bbox': bboxes, 'Centroid': centroids,
        'Contour': contours, 'CellType': types})

    if 'Morph' in featureSet or 'morph' in featureSet:
        logging.info('Getting Morph features')
        morphFeats = getMorphFeatures(names, contours, bboxes, 'MorphFeatures', process_n=8)
        for k, v in zip(morphFeats.keys(), morphFeats.values()):
            if k != 'name':
                globalGraph.vs[morphFeats['name']]['Morph_' + k] = v

    if 'Texture' in featureSet or 'texture' in featureSet:
        logging.info('Getting Texture features')
        GLCMFeats = getGLCMFeatures(wsiPath, names, contours, bboxes, pad=2, level=level, process_n=8)
        for k, v in zip(GLCMFeats.keys(), GLCMFeats.values()):
            if k != 'name':
                globalGraph.vs[GLCMFeats['name']]['Texture_' + k] = v

    return globalGraph


def read_data_as_json(dat_path):
    data = joblib.load(dat_path)
    if isinstance(data, dict):
        return {key: read_data_as_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [read_data_as_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return np.asscalar(data)
    else:
        return data


def process(seg_path, wsi_path, output_path, level, feature_set, cell_types):
    seg_path, wsi_path, output_path = str(seg_path), str(wsi_path), str(output_path)
    assert os.path.exists(seg_path) and os.path.isfile(seg_path), \
        f"json_path: {seg_path} is not allowed, please make sure it's a file and exists"
    assert os.path.exists(wsi_path) and os.path.isfile(wsi_path), \
        f"wsi_path: {wsi_path} is not allowed, please make sure it's a file and exists"
    try:
        os.makedirs(output_path, exist_ok=True)
    except PermissionError:
        logging.warning(f"Permission denied to create directory: {output_path}")

    sample_name = os.path.basename(wsi_path).split('.')[0]
    if seg_path.endswith('.json'):
        with open(seg_path) as fp:
            logging.info('Loading json')
            nucleusInfo = json.load(fp)
    elif seg_path.endswith('.dat'):
        logging.info('Loading dat')
        nucleusInfo = read_data_as_json(seg_path)
    else:
        raise ValueError(f"Unsupported file format: {seg_path.split('.')[-1]}, expected .json or .dat")

    global_graph = basicFeatureExtraction(wsi_path, nucleusInfo, level, feature_set)
    vertex_dataframe = global_graph.get_vertex_dataframe()

    col_dist = defaultdict(list)
    for feat_name in vertex_dataframe.columns.values:
        for cell in cell_types:
            col_dist[cell] += [feat_name] if feat_name != 'Contour' else []
    cellType_save = {'T': [1],  # Neopla
                     'I': [2],  # Inflam
                     'S': [3],  # Connec
                     'N': [5]}  # Normal

    for i in col_dist.keys():
        vertex_csvfile = os.path.join(output_path, sample_name + '_Feats_' + i + '.csv')
        save_index = vertex_dataframe['CellType'].isin(cellType_save[i]).values
        vertex_dataframe.iloc[save_index].to_csv(vertex_csvfile, index=False, columns=col_dist[i])


# Core function
def preprocess_files(args, configs):
    process_queue = list(args.seg.glob(f'*.json')) + list(args.seg.glob(f'*.dat'))
    output_dir = args.buffer
    ext = args.ext.split('.')[-1]
    if len(process_queue) > 1 and ext is None:
        logging.warning('No file extension provided, use --ext to specify the extension.')
    elif len(process_queue) == 1 and ext is None:
        ext = process_queue[0].suffix[1:]
    logging.info(f'Total {len(process_queue)} files to process.')

    for i, seg_path in enumerate(process_queue):
        logging.info(f'Phase 1 Preprocessing \t {i + 1} / {len(process_queue)} \t  {seg_path} ')
        slide_name = seg_path.stem
        wsi_path = args.wsi / f"{slide_name}.{ext}"
        output_path = output_dir / f"{slide_name}_Feats_T.csv"

        if args.auto_skip and output_path.exists():
            logging.info(f'Skip {slide_name} as it is already processed.')
            continue

        process(seg_path, wsi_path, output_dir, args.level, configs['feature-set'], configs['cell-types'])


def run_wsi(args, configs):
    logging.info(f'Phase 1 Preprocessing \t 1 / 1 \t {args.seg} ')
    process(args.seg, args.wsi, args.buffer, args.level, configs['feature-set'], configs['cell-types'])

