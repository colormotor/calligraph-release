#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch, time, os

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
    has_dropbox = os.path.isdir(os.path.expanduser('~/Dropbox/transfer_box/real'))
    if has_dropbox:
        output_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/outputs/'
    else:
        output_path = './generated/tests'

    text = 'G' #'T' #'R' #'R' #'G'
    size = 256
    padding = 15

    font = "./data/fonts/UnifrakturMaguntia-Regular.ttf"
    image_path = "./data/spock.jpg"

    minw, maxw = 0.0, 3.5  # stroke width range
    degree = 5 #5
    deriv = 3 #3
    multiplicity = 1
    b_spline = 1
    pspline = 0
    if deriv >= degree:
        pspline = 1
    catmull = 1
    ref_size_factor = 0.5
    fill = 1
    closed = 0 #True #False #True
    if not closed:
        fill = 0
    seed = 133
    alpha =  1 #0.5 #0.5 #0.5 # 1.0 #0.5 #0.5
    image_alpha = 0.5 #6 #5 #0.6 #0.45 #5 #5 #4 #5 #5 #0.4

    ablation = 'x'
    suffix = text


    style_img = './data/chinese.jpg'

    #style_img = './data/style-picasso-1.jpg'
    single_path = 1

    # For single paths
    point_density = 0.015 #0.003 # 0.015
    # For multiple paths
    num_paths = 30 #30
    num_vertices_per_path = 25
    spread_radius = 15

    lr_shape = 2.0 #1 #2.0 #1.5 #1.5 #1.0 #2.0
    lr_width = 0.3 # 0.5 #0.1 #25

    num_opt_steps = 300 # 500
    schedule_decay_factor = 1.5

    smoothing_w = 30 #1 #10 #50 #500 #100 # 100 #500 #100 #10 #10.0
    use_clipag = 1
    style_w = 10.0 #10.0 #0.3 #5 #1.0
    distortion_scale = 0.3 #0.5 #0 #0.3 #5 #5 #0.25 #0.0 #5 #0
    patch_size = 128 #64 #128 # 128 #64
    overlap_w = 0 #1000.0
    blur = 0
    if overlap_w > 0:
        alpha = 0.5

    repulsion_subd = 10 #5
    repulsion_w = 0 #100 #1000 #3000 # 500
    if not closed:
        repulsion_w = 0
    repulsion_d = 10

    mse_w = 20.0
    mse_mul = 1 #0.5 # Factor multiplying each mse blur level (> 1 emph low freq)
    sw = 2.0
    vary_width = 1

    return locals()



# Parse command line and update parameters
cfg = util.ConfigArgs(params())

output_path = cfg.output_path
if not cfg.save:
    output_path = '___'

cfg.ablation = cfg.ablation.lower()
if 'r' in cfg.ablation:
    cfg.repulsion_w = 0.0
    cfg.resolve_self_ins_every = 0
if 's' in cfg.ablation:
    cfg.smoothing_w = 0.0
if 'o' in cfg.ablation:
    cfg.style_w = 0.0
if 'b' in cfg.ablation:
    cfg.b_spline = 0
if cfg.ablation:
    cfg.suffix += '_abl-' + cfg.ablation

if not cfg.has_dropbox:
    print("Running remote. Init dropbox dir")
    cfg.headless = 1
    cfg.dropbox_folder = '/transfer_box/igor/calligraph/outputs'

saver = util.SaveHelper(__file__, output_path, use_wandb=False,
                        dropbox_folder=cfg.dropbox_folder,
                        suffix=cfg.suffix,
                        cfg=cfg)

if cfg.text:
    input_img, outline = plut.font_to_image(cfg.text, image_size=(cfg.size, cfg.size), padding=cfg.padding,
                                   font_path=cfg.font, return_outline=True)
    outline = geom.fix_shape_winding(outline)
    img = np.array(input_img)/255
    holes = geom.get_holes(outline)
else:
    input_img = Image.open(cfg.image_path).convert('L').resize((cfg.size, cfg.size))

    img = np.array(input_img)/255
    img = segmentation.xdog(img, sigma=2.5, k=7)
    holes = []

h, w = img.shape

style_img = Image.open(cfg.style_img).convert('L').resize((512, 512))
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
    density_map = 1-img
    points = tsp_art.weighted_voronoi_sampling(density_map, n, nb_iter=50)
    print("Density done, tsp now")
    # Find top left point
    n = len(points)
    if cfg.closed:
        P = [tuple(p) for p in points]
        # sorted according to x,y
        I = sorted(list(range(n)), key=lambda i: P[i])
        points = np.array([points[i] for i in I])
    else:
        points = np.array(points)
    # TSP func assumes first and last points are fixed if cylce is True and end_to_end is True
    I = tsp_art.heuristic_solve(points, time_limit_minutes=1/10, cycle=cfg.closed, end_to_end=not cfg.closed) #, logging=True, verbose=True)
    #P = points
    P = points[I]
    return np.hstack([P, np.ones((len(P), 1))*cfg.sw])


def init_path_weight(img, num_paths, subd, radius):
    from calligraph import tsp_art
    density_map = 1-img
    points = tsp_art.weighted_voronoi_sampling(density_map, num_paths, nb_iter=30)
    paths = []
    for p in points:
        blob = []
        for t in np.linspace(0, np.pi*2, subd, endpoint=False):
            blob.append(p + np.array([np.cos(t), np.sin(t)])*np.random.uniform(0.2, 1.0)*radius)
        paths.append(np.array(blob))
    return [np.hstack([P, np.ones((len(P), 2))]) for P in paths]



if not cfg.single_path:
    startup_paths = init_path_weight(img, cfg.num_paths,
                                     cfg.num_vertices_per_path,
                                     cfg.spread_radius)

else:
    num_points = int(np.sum(1-img)*cfg.point_density)
    startup_paths = [init_path_tsp(img, num_points)]
    if sum(holes):
        box = geom.make_rect(0, 0, w, h)

        for P, hole in zip(outline, holes):
            if hole:
                rast = imaging.ShapeRasterizer(box, 256)
                rast.fill_shape(P)
                himg = 1-rast.get_image()/255
                # plut.figure_pixels(w, h)
                # plut.fill(P, 'k')
                # plut.setup()
                # himg = np.array(plut.figure_image(close=True).convert('L'))/255
                num_points = max(5, int(np.sum(1-himg)*cfg.point_density))
                print('nump', num_points)
                #pdb.set_trace()

                startup_paths += [init_path_tsp(himg, num_points)]
print("Done")
startup_paths = [add_multiplicity(P) for P in startup_paths]


##############################################
# Create the scene

scene = diffvg_utils.Scene()

fill_color = None
if cfg.fill:  #cfg.closed:
    fill_color=([0.0, 0.0, 0.0, cfg.alpha*cfg.image_alpha], False)

paths = []
for Pw in startup_paths:
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(Pw[:,:2],
                stroke_width=(Pw[:,2], True),
                degree=cfg.degree,
                pspline=cfg.pspline,
                split_pieces=cfg.alpha < 1 and not cfg.fill, #cfg.overlap_w > 0, #overlap,
                closed=cfg.closed)
    else:
        if cfg.catmull:
            path = diffvg_utils.CardinalSpline(Pw[:,:2],
                                               stroke_width=(Pw[:,2], True),
                                           closed=cfg.closed)
        else:
            Q = diffvg_utils.cardinal_spline(Pw, 0.5, closed=cfg.closed) #bezier.cubic_bspline_to_bezier_chain(Pw, periodic=cfg.closed)
            if cfg.closed:
                Q = Q[:-1]
            path = diffvg_utils.Path(Q[:,:2],
                                    stroke_width=(Q[:,2], True),
                                    degree=3,
                                    closed=cfg.closed)
    paths.append(path)
scene.add_shapes(paths, stroke_color=([cfg.alpha], False), fill_color=fill_color,
                     split_primitives=False )


##############################################
# Optimization

Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.8, 0.999))

optimizers = [Opt(scene.get_points(), lr=cfg.lr_shape)]
if cfg.vary_width and not cfg.fill:
    optimizers += [Opt(scene.get_stroke_widths(), lr=0.5)]

schedulers = [util.step_cosine_lr_scheduler(opt, 0.0, 0.5, cfg.num_opt_steps)
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
               spline_losses.make_deriv_loss(cfg.deriv, w), cfg.smoothing_w)

losses.add('repulsion',
               spline_losses.make_repulsion_loss(15, False, dist=5, signed=True), cfg.repulsion_w)
overlap_loss = spline_losses.make_overlap_loss(cfg.alpha, blur=cfg.blur, subtract_widths=False)
losses.add('overlap', overlap_loss, cfg.overlap_w) # 10000) #1000.0)

losses.add('bbox',
           spline_losses.make_bbox_loss(geom.make_rect(0, 0, w, h)), 1.0)


style_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
                                        model='CLIPAG', #'ViT-B-16', #'ViT-B-32-256', #'ViT-SO400M-14-SigLIP', #'ViT-L-14', #'CLIPAG',
                                        min_size=cfg.patch_size, #128, #64, #64, #64, #128, #32, #128, #64, #64, #100, #64, #128,
                                        cut_scale=0.0 if cfg.distortion_scale > 0.0 else 0.35, #5, #0.5, #0.1,
                                        distortion_scale=cfg.distortion_scale, #0.5, #5,
                                        blur_sigma=0.0, #1.0,
                                        thresh=0.0,
                                        n_cuts=64, #32, #64,
                                        use_negative=False) #, n_cuts=24) #16)


losses.add('style',
               style_loss, cfg.style_w)


##############################################
# Begin visualization and optimize


plut.set_theme()

fig = plt.figure(figsize=(8,6))
gs = GridSpec(2, 2, height_ratios=[1.0, 0.5])
gs_sub = GridSpecFromSubplotSpec(1, 3, width_ratios=[0.15, 0.15, 0.9], subplot_spec=gs[1, :])

first_frame = None
time_count = 0

render_shapes = []

startup_splines = [path.samples(100).detach().cpu().numpy() for path in scene.shapes]

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

    # Losses (see weights above)
    loss = losses(
        mse=(im_no_gamma, target_img, cfg.mse_mul),
        offset=(scene.shapes,),
        curv=(scene.shapes,),
        orient=(scene.shapes,),
        flow=(scene.shapes,),

        deriv=(scene.shapes,),
        overlap=(im_no_gamma, scene.shapes,),
        style=(im,),
        repulsion=(scene.shapes,),
        bbox=(scene.shapes,),
    )

    if cfg.num_opt_steps > 1:
        with util.perf_timer('Opt step', verbose=cfg.verbose):
            loss.backward()
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

    # Constrain
    with torch.no_grad():
        for path in paths:
            path.param('stroke_width').data.clamp_(cfg.minw, cfg.maxw)
            #path.param('stroke_width').data[0] = cfg.minw
            #path.param('stroke_width').data[-1] = cfg.minw

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed


    save_every = 10
    must_save = step%save_every == save_every-1 or cfg.num_opt_steps==1
    must_show = True
    if cfg.headless and not must_save:
        must_show = must_save

    if must_show:
        # Viz
        im = im.detach().cpu().numpy()
        if first_frame is None:
            first_frame = im

        plt.subplot(gs[0,0])
        plt.title('Startup')
        amt = 0.5
        #mseim =(mse.target.detach().cpu().numpy() + mse.im.detach().cpu().numpy())/2
        plt.imshow(target_img*amt + first_frame*(1-amt) , cmap='gray', vmin=0.0, vmax=1.0)
        #for P in startup_splines:
        #    plut.stroke(P, 'k', lw=2)

        #plt.imshow(mseim, cmap='gray')

        # for i, path in enumerate(paths):
        #     Q = path.param('points').detach().cpu().numpy()
        #     P = Q[::multiplicity]
        #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
        #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
        plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)


        plt.subplot(gs[0,1])
        plt.title('Step %d'%step)
        plt.imshow(background_image, cmap='gray', vmin=0, vmax=1)
        if cfg.num_opt_steps < 2:
            plt.imshow(img, cmap='gray', vmin=0, vmax=1, alpha=0.5)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        if losses.has_loss('curv'):
            plut.fill_circle([cfg.target_radius, cfg.target_radius], cfg.target_radius, 'c')
        # step_skip = 1
        # if cfg.fill or not cfg.vary_width:
        #     step_skip = 1
        # if step % step_skip == 0: #with util.perf_timer('Thick curves', verbose=verbose) :
        #     render_shapes = []
        #     for i, path in enumerate(paths):
        #         Q = path.param('points').detach().cpu().numpy()
        #         X = path.samples(len(Q)*10).detach().cpu().numpy()
        #         #X[:,2] *= 0.75
        #         if cfg.fill: # and not overlap:
        #             plut.fill(X[:,:2], 'k', alpha=cfg.alpha*cfg.image_alpha)
        #             plut.stroke(X[:,:2], 'k')
        #         else:
        #             #if not cfg.vary_width:
        #             #    X[:,2] *= 0.5
        #             if cfg.vary_width:
        #                 S = geom.thick_curve(X, add_cap=False)
        #                 render_shapes.append(S)
        #             else:
        #                 plut.stroke(X[:,:2], 'k', lw=cfg.sw)
        #         #plut.fill(S, 'k')

        # if render_shapes:
        #     for S in render_shapes:
        #         plut.fill(S, 'k')

        #plt.legend()
        plut.setup(box=geom.make_rect(0, 0, w, h))


        plt.subplot(gs_sub[0])
        plt.imshow(style_img, cmap='gray')
        plt.axis('off')

        if losses.has_loss('semantic'):
            plt.subplot(gs_sub[1])
            plt.imshow(patch_loss.test_cutout, cmap='gray')
            plt.axis('off')

        plt.subplot(gs_sub[2])
        plt.title(f'Loss - step {step} - t: ' + '%.1f'%time_count)
        for i, (key, kloss) in enumerate(losses.losses.items()):
            if key=='total' or not losses.has_loss(key):
                # There is a bug in 'total'
                continue
            plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
        plt.legend()

    if saver.valid:
        if must_save:
            print("Saving ", step)
            saver.clear_collected_paths()
            scene.save_json(saver.with_ext('.json'))
            plut.figure_image().save(saver.with_ext('.png'), adjust=False)
            saver.copy_file()
            saver.collected_to_dropbox()


if cfg.headless:
    filename = ''
else:
    filename = saver.with_ext('.mp4')
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=filename, headless=cfg.headless)
saver.finish()
