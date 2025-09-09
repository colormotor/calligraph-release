#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch, time

from transformers import get_cosine_schedule_with_warmup

from calligraph import (
    plut,
    geom,
    bezier,
    dce,
    config,
    util,
    fs,
    diffvg_utils,
    imaging,
    stroke_init,
    segmentation,
    spline_losses,
    image_losses,
)


device = config.device
dtype = torch.float32


def params():
    verbose = 0

    text = "A"  # If an empty string load an image instead
    size = 512
    padding = 25
    font = "./data/fonts/Bradley Hand Bold.ttf"
    image_path = "./data/spock.jpg"

    minw, maxw = 0.5, 7  # stroke width range
    degree = 5  # Spline degree
    deriv = 3  # Smoothing derivative order
    multiplicity = 1  # Keypoint multiplicity
    b_spline = True  # Use B-splines

    fill = 0
    closed = False
    if not closed:
        fill = 0
    seed = 133
    alpha = 1  # 0.5 #0.5 #0.5 # 1.0 #0.5 #0.5
    image_alpha = 0.5

    output_path = "./outputs/"
    style_img = "./data/chinese.jpg"
    # style_img = "./data/style-picasso-1.jpg"
    # style_img = "/home/danielberio/Dropbox/transfer_box/data/calligraph/pat5.jpg"
    # style_img = "/home/danielberio/Dropbox/transfer_box/data/calligraph/flo8.jpg"
    # style_img = "/home/danielberio/Dropbox/transfer_box/data/calligraph/flo4.jpg"

    vary_width = 1  # If 1 vary stroke width
    single_path = 1

    num_voronoi_points = 100  # For single path only
    # Otherwise, params for multiple paths
    num_paths = 10  #
    num_vertices_per_path = 35
    spread_radius = 30

    # Optimization params
    lr_shape = 3
    lr_width = 0.01
    num_opt_steps = 300  # 500

    # Loss weights
    smoothing_w = 0.1
    use_clipag = 1
    style_w = 1.0
    overlap_w = 0  # 1000.0
    blur = 0
    if overlap_w > 0:
        alpha = 0.5

    repulsion_subd = 15
    repulsion_w = 0
    # Don't use repulsion for open curves
    if not closed:
        repulsion_w = 0

    mse_w = 5.0
    mse_mul = 1  # Factor multiplying each mse blur level (> 1 emph low freq)

    return locals()


# Parse command line and update parameters
cfg = util.ConfigArgs(params())

output_path = cfg.output_path
if not cfg.save:
    output_path = "___"

# Configure saving
saver = util.SaveHelper(
    __file__,
    output_path,
    use_wandb=cfg.headless,
    dropbox_folder=cfg.dropbox_folder,
    cfg=cfg,
)

if cfg.text:
    input_img = plut.font_to_image(
        cfg.text,
        image_size=(cfg.size, cfg.size),
        padding=cfg.padding,
        font_path=cfg.font,
    )
    img = np.array(input_img) / 255
else:
    input_img = Image.open(cfg.image_path).convert("L").resize((cfg.size, cfg.size))

    img = np.array(input_img) / 255
    img = segmentation.xdog(img, sigma=2.5, k=7)

h, w = img.shape

style_img = Image.open(cfg.style_img).convert('L') #.resize((512, 512)) #224, 244)) #resize((cfg.size, cfg.size))
target_img = 1-(1.0-img)*(cfg.alpha*cfg.image_alpha)

def add_multiplicity(Q, noise=0.0):
    Q = np.kron(Q, np.ones((cfg.multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)


##############################################
# Settings
background_image = np.ones((h, w))

##############################################
# Initialization paths


def init_path_tsp(img, n, **kwargs):
    from calligraph import tsp_art

    density_map = 1 - img
    points = tsp_art.weighted_voronoi_sampling(density_map, n, nb_iter=50)
    # Find top left point
    n = len(points)
    if cfg.closed:
        P = [tuple(p) for p in points]
        # sorted according to x,y
        I = sorted(list(range(n)), key=lambda i: P[i])
        points = np.array([points[i] for i in I])
    else:
        points = np.array(points)
    # TSP func assumes first and last points are fixed if cycle is True and end_to_end is True
    I = tsp_art.heuristic_solve(
        points,
        time_limit_minutes=1 / 10,
        penalty_func=tsp_art.make_coverage_penalty(img),
        penalty_weight=10000000.0,
        cycle=cfg.closed,
        end_to_end=not cfg.closed,
    )  # , logging=True, verbose=True)
    P = points[I]
    return np.hstack([P, np.ones((len(P), 1))])


def init_path_weight(img, num_paths, subd, radius):
    from calligraph import tsp_art

    density_map = 1 - img
    points = tsp_art.weighted_voronoi_sampling(density_map, num_paths, nb_iter=30)
    paths = []
    for p in points:
        blob = []
        for t in np.linspace(0, np.pi * 2, subd, endpoint=False):
            blob.append(
                p
                + np.array([np.cos(t), np.sin(t)])
                * np.random.uniform(0.2, 1.0)
                * radius
            )
        paths.append(np.array(blob))
    return [np.hstack([P, np.ones((len(P), 5))]) for P in paths]


if cfg.single_path:
    startup_paths = [init_path_tsp(img, cfg.num_voronoi_points)]
else:
    startup_paths = init_path_weight(
        img, cfg.num_paths, cfg.num_vertices_per_path, cfg.spread_radius
    )


##############################################
# Create the scene

scene = diffvg_utils.Scene()

fill_color = None

if cfg.fill:
    fill_color = ([0.0, 0.0, 0.0, cfg.alpha * cfg.image_alpha], False)

for Pw in startup_paths:
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(
            Pw[:, :2],
            stroke_width=(Pw[:, 2], True),
            degree=cfg.degree,
            split_pieces=cfg.alpha < 1 and not cfg.fill,
            closed=cfg.closed,
        )
    else:
        Q = bezier.cubic_bspline_to_bezier_chain(Pw, periodic=cfg.closed)
        if cfg.closed:
            Q = Q[:-1]
        path = diffvg_utils.Path(
            Q[:, :2], stroke_width=(Q[:, 2], True), degree=3, closed=cfg.closed
        )
    scene.add_shapes(
        [path],
        stroke_color=([cfg.alpha], False),
        fill_color=fill_color,
        split_primitives=True,
    )


##############################################
# Optimization

Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.999), eps=1e-6)

optimizers = [Opt(scene.get_points(), lr=cfg.lr_shape)]
if cfg.vary_width and not cfg.fill:
    optimizers += [Opt(scene.get_stroke_widths(), lr=0.5)]

schedulers = [util.step_cosine_lr_scheduler(opt, 0.0, 0.2, cfg.num_opt_steps)
              for opt in optimizers]


##############################################
# Losses

losses = util.MultiLoss(verbose=cfg.verbose)
# mse = image_losses.MSELoss(rgb=False, blur=3)
mse = image_losses.MultiscaleMSELoss(rgb=False)
losses.add('mse',
           mse, cfg.mse_w)

if cfg.degree > 3 and cfg.b_spline:
    losses.add('deriv',
               spline_losses.make_deriv_loss(cfg.deriv, 1), cfg.smoothing_w)

losses.add('repulsion',
               spline_losses.make_repulsion_loss(15, False, dist=5, signed=True), cfg.repulsion_w)
overlap_loss = spline_losses.make_overlap_loss(cfg.alpha, blur=cfg.blur, subtract_widths=False)
losses.add('overlap', overlap_loss, cfg.overlap_w) # 10000) #1000.0)

losses.add('bbox',
           spline_losses.make_bbox_loss(geom.make_rect(0, 0, w, h)), 1.0)


style_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
                                        model='CLIPAG', #'CLIPAG',
                                        min_size = 128,
                                        # model='ViT-L-14', #'CLIPAG',
                                        use_negative=False, n_cuts=24)
losses.add('style',
               style_loss, cfg.style_w)


##############################################
# Begin visualization and optimize

plut.set_theme()
fig = plt.figure(figsize=(8,8))
gs = GridSpec(3, 2, height_ratios=[1.0, 0.3, 0.3])
gs_sub = GridSpecFromSubplotSpec(1, 2, gs[1,:])
first_frame = None

render_shapes = []

time_count = 0

def frame(step):
    global first_frame, render_shapes, time_count

    perf_t = time.perf_counter()

    for opt in optimizers:
       opt.zero_grad()

    # Rasterize
    with util.perf_timer('render', verbose=cfg.verbose):
        im = scene.render(background_image, num_samples=2)[:,:,0].to(device)

    im_no_gamma = im
    if cfg.alpha < 1:
        im = im**300 # # ** gamma

    # Losses
    # Note that if loss is not added or weight is 0 it is not computed
    loss = losses(
        mse=(im_no_gamma, target_img, 1.0),
        offset=(scene.shapes,),
        deriv=(scene.shapes,),
        overlap=(im_no_gamma, scene.shapes,),
        style=(im,),
        repulsion=(scene.shapes,),
        bbox=(scene.shapes,),
    )

    with util.perf_timer('Opt step', verbose=cfg.verbose):
        loss.backward()
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

    # Constrain
    with torch.no_grad():
        for path in scene.shapes:
            path.param('stroke_width').data.clamp_(cfg.minw, cfg.maxw)
            path.param('stroke_width').data[0] = cfg.minw
            path.param('stroke_width').data[-1] = cfg.minw

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed

    # Viz
    im = im.detach().cpu().numpy()
    if first_frame is None:
        first_frame = im

    plt.subplot(gs[0,0])
    plt.title('Startup - time: %.3f'%(time_count))
    amt = 0.5
    plt.imshow(first_frame*amt + target_img*(1-amt) , cmap='gray')
    #plt.imshow(mseim, cmap='gray')

    # for i, path in enumerate(paths):
    #     Q = path.param('points').detach().cpu().numpy()
    #     P = Q[::multiplicity]
    #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
    #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)


    plt.subplot(gs[0,1])
    plt.title('Step %d'%step)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    if losses.has_loss('style'):
        plt.subplot(gs_sub[0])
        plt.imshow(style_img, cmap='gray')

    plut.setup(box=geom.make_rect(0, 0, w, h))
    plt.subplot(gs[2,:])
    plt.title('Loss')
    for i, (key, kloss) in enumerate(losses.losses.items()):
        if key=='total' or not losses.has_loss(key):
            # There is a bug in 'total'
            continue
        plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
    plt.legend()

    if saver.valid:
        save_every = 10
        if step%save_every == save_every-1 or cfg.num_opt_steps==1:
            scene.save_json(saver.with_ext('.json'))
            plut.figure_image().save(saver.with_ext('.png'))
            saver.copy_file()



plut.show_animation(fig, frame, cfg.num_opt_steps, filename=saver.with_ext('.mp4'))
