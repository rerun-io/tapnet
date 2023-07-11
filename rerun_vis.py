"""Visualize TAPIR result on a video with rerun."""
from typing import Optional

import cv2
import haiku as hk
import jax
import matplotlib
import mediapy as media
import numpy as np
import rerun as rr
import tree

from tapnet import tapir_model
from tapnet.utils import transforms, viz_utils

NUM_PIPS_ITER = 4


def build_model(frames, query_points, highlight_track_id):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(num_pips_iter=NUM_PIPS_ITER)
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=64,
        highlight_track_id=highlight_track_id,
    )
    return outputs


def preprocess_frames(frames: np.ndarray):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32
      expected_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    # visibles = occlusions < 0
    visibles = (1 - jax.nn.sigmoid(occlusions)) * (
        1 - jax.nn.sigmoid(expected_dist)
    ) > 0.5
    return visibles


def inference(frames, query_points, highlight_track_id=None):
    """Inference on one video.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
      query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
      tracks: [num_points, 3], [-1, 1], [t, y, x]
      visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(
        params, state, rng, frames, query_points, highlight_track_id
    )
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    log_outputs(outputs)

    tracks, occlusions, expected_dist = (
        outputs["tracks"],
        outputs["occlusion"],
        outputs["expected_dist"],
    )
    visibles = postprocess_occlusions(occlusions, expected_dist)

    return tracks, visibles


def log_outputs(outputs: dict) -> None:
    """Log outputs of TAPIR model to rerun."""
    tracks, occlusions, expected_dist = (
        outputs["tracks"],
        outputs["occlusion"],
        outputs["expected_dist"],
    )
    unrefined_tracks, unrefined_occlusions, unrefined_expected_dist = (
        outputs["unrefined_tracks"],
        outputs["unrefined_occlusion"],
        outputs["unrefined_expected_dist"],
    )

    # Initial predictions
    unrefined_visibles_0 = postprocess_occlusions(
        unrefined_occlusions[0], unrefined_expected_dist[0]
    )
    log_tracks(unrefined_tracks[0], unrefined_visibles_0, colors, "_initial")
    if highlight_track_id is not None:
        log_track_scalars(
            unrefined_occlusions[0][highlight_track_id],
            unrefined_expected_dist[0][highlight_track_id],
            unrefined_visibles_0[highlight_track_id],
            suffix="_initial",
        )

    # Refined (intermediate) predictions
    # following are iterative refinements, which might have to be averaged over
    # different resolutions, similar to final output (see tapir_model.py)
    for i in range(1, NUM_PIPS_ITER - 1):
        occ = np.mean(unrefined_occlusions[i::NUM_PIPS_ITER], axis=0)
        tra = np.mean(unrefined_tracks[i::NUM_PIPS_ITER], axis=0)
        exp_d = np.mean(unrefined_expected_dist[i::NUM_PIPS_ITER], axis=0)
        vis = postprocess_occlusions(occ, exp_d)
        log_tracks(tra, vis, colors, f"_{i}")
        if highlight_track_id is not None:
            log_track_scalars(
                occ[highlight_track_id],
                exp_d[highlight_track_id],
                vis[highlight_track_id],
                suffix=f"_{i}",
            )

    # Final predictions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    log_tracks(tracks, visibles, colors, "_final")
    if highlight_track_id is not None:
        log_track_scalars(
            occlusions[highlight_track_id],
            expected_dist[highlight_track_id],
            visibles[highlight_track_id],
            suffix="_final",
        )


def log_track_scalars(occlusions, expected_dists, visibles, suffix=""):
    """Log scalars associated with track to rerun."""
    for frame_id in range(len(occlusions)):
        rr.set_time_sequence("frameid", frame_id)
        rr.log_scalar(
            "occluded_prob" + suffix, 1 - jax.nn.sigmoid(occlusions[frame_id])
        )
        rr.log_scalar(
            "accurate_prob" + suffix, 1 - jax.nn.sigmoid(expected_dists[frame_id])
        )
        rr.log_scalar("visible" + suffix, visibles[frame_id])


def log_query(
    query_frame: np.ndarray, query_xys: np.ndarray, colors: Optional[np.ndarray] = None
) -> None:
    """Log query image and points to rerun."""
    rr.set_time_sequence("frameid", 0)
    rr.log_image("query_frame", query_frame)
    rr.log_points("query_frame/query_points", query_xys, radii=4, colors=colors)


def log_video(frames) -> None:
    """Log video frames to rerun."""
    for i, frame in enumerate(frames):
        rr.set_time_sequence("frameid", i)
        rr.log_image("frame", frame)


def log_tracks(
    tracks: np.ndarray,
    visibles: np.ndarray,
    colors: Optional[np.ndarray] = None,
    suffix="",
) -> None:
    """Log predicted point tracks to rerun."""
    tracks = transforms.convert_grid_coordinates(
        tracks, (resize_width, resize_height), (original_width, original_height)
    )

    # tracks has shape (num_tracks, num_frames, 2)
    num_tracks = tracks.shape[0]
    num_frames = tracks.shape[1]

    for frame_id in range(num_frames):
        rr.set_time_sequence("frameid", frame_id)
        rr.log_points(
            "frame/points" + suffix,
            tracks[visibles[:, frame_id], frame_id],
            radii=4,
            colors=colors[visibles[:, frame_id]],
        )

        if frame_id == 0:
            continue

        for track_id in range(num_tracks):
            if visibles[track_id, frame_id - 1] and visibles[track_id, frame_id]:
                rr.log_line_segments(
                    f"frame/tracks{suffix}/#{track_id}",
                    tracks[track_id, frame_id - 1 : frame_id + 1],
                    color=colors[track_id].tolist(),
                )
            else:
                rr.log_cleared(f"frame/tracks{suffix}/#{track_id}")


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


# TODO argparse this stuff
# TODO option to save as rrd instead of spawn
resize_factor = 0.5
num_points = 20

# settings for grid points on mask
mask_file = "./tennis-vest.png"
mask_id = 2
grid_spacing = 20
highlight_track_id = 39  # none to not highlight any

video_file = "./tennis-vest.mp4"
video_out_file = "./tennis-vest-out.mp4"

rr.init("track test", spawn=True)

model = hk.transform_with_state(build_model)
model_apply = model.apply  # can't use jit if we want to visualize inside the model
checkpoint_path = "checkpoint/tapir_checkpoint.npy"
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]

video = media.read_video(video_file)

log_video(video)

original_height, original_width = video.shape[1:3]
resize_height = round(original_height * resize_factor)
resize_width = round(original_width * resize_factor)
ij_resize_factor = np.array([resize_height, resize_width]) / np.array(
    [original_height, original_width]
)
uv_resize_factor = ij_resize_factor[::-1]
resized_frames = media.resize_video(video, (resize_height, resize_width))

if mask_file is None:
    resized_query_tijs = sample_random_points(
        0, resized_frames.shape[1], resized_frames.shape[2], num_points
    )  # t, row, col
    colors = None
else:
    mask = (media.read_image(mask_file) == mask_id).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel)
    ijs = np.indices(mask.shape)
    grid_mask = np.all(ijs % grid_spacing == 0, axis=0)
    original_query_ijs = np.stack(np.nonzero(mask * grid_mask), axis=-1)

    resized_query_ijs = (original_query_ijs * ij_resize_factor).astype(np.int64)
    num_points = len(resized_query_ijs)
    resized_query_tijs = np.pad(resized_query_ijs, ((0, 0), (1, 0)))

    cmap = matplotlib.colormaps["rainbow"]
    norm = matplotlib.colors.Normalize(
        vmin=original_query_ijs[:, 0].min(), vmax=original_query_ijs[:, 0].max()
    )
    colors = cmap(norm(original_query_ijs[:, 0]))

original_query_uvs = (
    resized_query_tijs[:, 2:0:-1] / uv_resize_factor + 0.5
)  # convert to continuous coordinates with pixel center being at 0.5
log_query(video[0], original_query_uvs, colors)

print("Running inference... ", end="")
tracks, visibles = inference(resized_frames, resized_query_tijs, highlight_track_id)
print("Done.")

tracks = transforms.convert_grid_coordinates(
    tracks, (resize_width, resize_height), (original_width, original_height)
)
out_video = viz_utils.paint_point_track(video, tracks, visibles)

with media.VideoWriter(
    video_out_file, out_video.shape[1:3], fps=video.metadata.fps
) as writer:
    for image in out_video:
        writer.add_image(image)

t = np.linspace(0, 5, 1000)
