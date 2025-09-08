#!/usr/bin/env python3

from . import geom, segmentation, strokify, planning, imaging
from easydict import EasyDict as edict
import numpy as np
import pdb
from PIL import ImageOps


def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum() 


def init_paths_random(img, n, num_ctrl, sigma=2.5, k=7, tau=0.2, use_attention=True, intersect_xdog=True, width=None, get_data=False, sample_around_radius=0):
    if use_attention:
        attn_map = segmentation.clip_saliency(img) #painter.input_img)
        edges = 1-segmentation.xdog(img, sigma=sigma, k=k)
    else:
        edges = 1
        attn_map = np.array(img)
        if np.max(attn_map) > 1:
            attn_map = attn_map/255
        attn_map = 1-attn_map
        edges = np.array(attn_map)

    if intersect_xdog:
        attn_map *= edges
    attn_map_soft = np.copy(attn_map)
    attn_map_soft[attn_map > 0] = softmax(attn_map[attn_map > 0], tau=tau)
    paths = []
    
    def attn_val(p):
        x, y = p
        return attn_map[int(y), int(x)]
    def attn_dist(a, b):
        v = attn_val((a + b)/2)
        return (1-v) + geom.distance(a, b)/img.width

    if sample_around_radius > 0:
        inds = np.random.choice(range(attn_map.flatten().shape[0]), size=n, replace=False, p=attn_map_soft.flatten())
        inds = np.array(np.unravel_index(inds, attn_map.shape))[::-1].T
        P = inds.astype(np.float64)
        
        for p in P:
            blob = []
            for t in np.linspace(0, np.pi*2, num_ctrl, endpoint=False):
                blob.append(p + np.array([np.cos(t), np.sin(t)])*np.random.uniform(0.2, 1.0)*sample_around_radius)
            blob = np.array(blob)
            if width is not None:
                blob = np.hstack([blob, np.ones((len(blob), 1))*width])
            paths.append(blob)
    else:
        for i in range(n):
            inds = np.random.choice(range(attn_map.flatten().shape[0]), size=num_ctrl, replace=False, p=attn_map_soft.flatten())
            inds = np.array(np.unravel_index(inds, attn_map.shape))[::-1].T
            P = inds.astype(np.float64)
            I = planning.path_tsp(P, distfn=attn_dist) #lambda a, b: attn_val((a + b)/2))
            P = P[I]
            if width is not None:
                P = np.hstack([P, np.ones((len(P), 1))*width])
            paths.append(P)

    if not get_data:
        return paths
    return edict(dict(paths=paths,
                      saliency=attn_map,
                      edges=edges))
    

    #
def init_paths(img, n=5, join_thresh=100, simplify_eps=1.5, sigma=2.5, k=7,
               width_range=None,
               relevance_thresh=2.5**2,
               saliency=None, get_data=False,
               thresh_img=None, contours=False):
    if saliency is None:
        saliency = segmentation.clip_saliency(img) #painter.input_img)
    if thresh_img is None:
        segments = segmentation.find_segments(img, 'instance')
        if not segments:
            print("found no objects attempting panoptic seg")
            segments = segmentation.find_segments(img, 'panoptic')
    else:
        segments = []
    if segments:
        masks = [seg['visual_mask'] for seg in segments]
        edges = 1-segmentation.xdog(img, sigma=sigma, k=7) #2.5, k=7)
    else:
        if thresh_img is not None:
            edges = (1-np.array(thresh_img.convert('L'))/255)
        else:
            thresh_img = (1-np.array(img.convert('L'))/255)
            edges = 1-segmentation.xdog(thresh_img, sigma=sigma, k=k)
        masks = [np.ones(edges.shape[:2])]

    w, h = img.size
    rect = geom.make_rect(0, 0, w, h)

    im = np.array(img)
    paths = []
    for mask in masks:
        mask = imaging.morpho_dilate(np.array(mask), iterations=1)
        masked = edges*mask
        #masked = filters.gaussian(masked, 2.0)
        #masked = imaging.morpho_open(masked, iterations=1)
        # masked = imaging.morpho_dilate(masked, iterations=1)
        if not contours:
            if width_range is not None:
                strokes = strokify.incremental_strokify_binary_map((masked).astype(bool), width_range,
                                                                   relevance_thresh=relevance_thresh,
                                                                   max_strokes=n)
            else:
                strokes = strokify.strokify_binary_map((masked).astype(bool))
        else:
            strokes = imaging.find_contours((masked*255).astype(np.uint8)) #.astype(bool))
            strokes = [geom.close(P) for P in strokes]
            strokes = [np.hstack([P, np.ones((len(P), 1))]) for P in strokes]
        strokes = [s for s in strokes if len(s) > 1]
        if not strokes:
            continue

        if join_thresh > 0 and len(strokes) > 1:
            # print("Joining %d strokes"%len(strokes))
            strokes = planning.sort_paths(strokes, join_thresh=join_thresh) #100)
        # print('Simplifying strokes')
        strokes = strokify.simplify_strokes(strokes, simplify_eps, perimeter=w)
        strokes = [s for s in strokes if len(s) > 1]
        #strokes = spray.simplify_strokes(strokes)
        if not strokes:
            continue

        # print('Computing saliencies and sorting')
        saliencies = []
        for stroke in strokes:
            stroke = np.array(stroke)
            box = geom.bounding_box(stroke[:,:2])
            if np.max(geom.rect_size(box)) > 512:
                print("Stroke too long?")
                print(len(stroke))
                print(geom.chord_length(stroke[:,:2]))
                continue

            stroke = strokify.smooth_stroke(stroke, ds=1, degree=1)
            sim = strokify.raster_stroke_for_image(stroke, im)/255
            #sal = saliency*sim # softmax(sal*sim)
            #s = np.mean(sim) #

            s = np.mean(saliency[sim > 0])

            saliencies.append(s*geom.chord_length(stroke[:,:2]))
            #plt.imshow(sal*sim)
            #print(im.dtype, im.shape)
            #break
        saliencies = np.array(saliencies)
        I = np.argsort(saliencies)[::-1][:n]
        saliencies = [saliencies[i] for i in I]
        paths += [strokes[i] for i in I]
    if get_data:
        return edict(dict(paths=paths,
                         saliency=saliency,
                         edges=edges,
                         segments=segments))
    return paths


def init_path_tsp(img, n, nb_iter=30, mask=None, startup_w=None, minutes_limit=1/30, **kwargs):
    # Saliency
    #sal = segmentation.clip_saliency(input_img)
    from . import ood_saliency, tsp_art
    sal = ood_saliency.compute_saliency(img.convert('RGB'))[0]

    density_map = ((sal-sal.min())/(sal.max() - sal.min()))**2 #(sal*(1-img))
    if mask is not None:
        density_map *= mask
    #density_map = density_map*(1-img)
    points = tsp_art.weighted_voronoi_sampling(density_map, n, nb_iter=nb_iter)
    # Find top left point
    n = len(points)
    P = [tuple(p) for p in points]
    # sorted according to x,y
    I = sorted(list(range(n)), key=lambda i: P[i])
    points = np.array([points[i] for i in I])
    # TSP func assumes first and last points are fixed if cylce is True and end_to_end is True
    I = tsp_art.heuristic_solve(points, time_limit_minutes=minutes_limit, cycle=False, end_to_end=True) #, logging=True, verbose=True)
    P = points[I]
    if startup_w is not None:
        P = np.hstack([P, np.ones((len(P), 1))*startup_w])
    return [P], density_map
