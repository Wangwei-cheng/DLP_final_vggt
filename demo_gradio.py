# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.nn.functional as F

# Import SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model, detected_objects=None) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    if detected_objects:
        print(f"Backend: Detecting {len(detected_objects)} objects: {detected_objects}")
    else:
        print("Backend: No objects specified for detection.")

    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Object Detection and Segmentation Pipeline
# -------------------------------------------------------------------------

def run_groundingdino(target_dir, detected_objects):
    """
    Run GroundingDINO on all images in target_dir/images.
    Return detections_per_frame.
    """
    if not detected_objects:
        return {}

    print(f"Backend: Running GroundingDINO on objects: {detected_objects}")
    
    # Load model and processor
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    text_prompt = " . ".join(detected_objects) + " ."
    
    image_dir = os.path.join(target_dir, "images")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    
    detections_per_frame = {}
    
    detection_dir = os.path.join(target_dir, "detections")
    os.makedirs(detection_dir, exist_ok=True)
    
    for img_path in image_paths:
        frame_id = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        
        # Post-process
        width, height = image.size
        target_sizes = torch.tensor([[height, width]]).to(device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=target_sizes
        )[0]
        
        frame_detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            frame_detections.append({
                "label": label,
                "score": score.item(),
                "box": box.cpu().numpy().tolist()
            })
        
        detections_per_frame[frame_id] = frame_detections
        
        # Save to JSON for debugging and reuse
        json_path = os.path.join(detection_dir, f"{os.path.splitext(frame_id)[0]}.json")
        with open(json_path, "w") as f:
            json.dump(frame_detections, f, indent=4)
            
    print(f"GroundingDINO detection complete for {len(image_paths)} frames.")
    return detections_per_frame


def run_sam2(target_dir, detections_per_frame):
    """
    Run SAM2 using GroundingDINO boxes as prompts.
    Return masks_per_frame and mask metadata.
    """
    if not detections_per_frame:
        return {}, {}
        
    print("Backend: Running SAM2 segmentation...")
    
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    
    masks_per_frame = {}
    mask_metadata = {}
    
    mask_dir = os.path.join(target_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    for frame_id, detections in detections_per_frame.items():
        img_path = os.path.join(target_dir, "images", frame_id)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        predictor.set_image(image)
        
        frame_masks = []
        frame_metadata = []
        
        frame_mask_dir = os.path.join(mask_dir, os.path.splitext(frame_id)[0])
        os.makedirs(frame_mask_dir, exist_ok=True)
        
        for i, det in enumerate(detections):
            label = det["label"]
            box = np.array(det["box"])
            
            masks, scores, logits = predictor.predict(
                box=box,
                multimask_output=False
            )
            
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            mask_filename = f"{label}_{i}.png"
            mask_path = os.path.join(frame_mask_dir, mask_filename)
            
            # Save mask
            cv2.imwrite(mask_path, (best_mask * 255).astype(np.uint8))
            
            frame_masks.append(best_mask)
            frame_metadata.append({
                "object_id": f"{label}_{i}",
                "label": label,
                "box": det["box"],
                "mask_path": os.path.relpath(mask_path, target_dir),
                "detection_score": det["score"],
                "sam_score": float(best_score)
            })
            
        masks_per_frame[frame_id] = frame_masks
        mask_metadata[frame_id] = frame_metadata
        
    # Save metadata JSON
    with open(os.path.join(mask_dir, "mask_metadata.json"), "w") as f:
        json.dump(mask_metadata, f, indent=4)
        
    print("SAM2 segmentation complete.")
    return masks_per_frame, mask_metadata


def align_masks_to_vggt(masks_per_frame, mask_metadata, predictions):
    """
    Resize SAM2 masks to match VGGT output resolution and keep metadata.
    """
    if not masks_per_frame:
        return {}
        
    # predictions["depth"] shape is (S, H, W, 1)
    target_h, target_w = predictions["depth"].shape[1:3]
    
    aligned_results = {}
    for frame_id, masks in masks_per_frame.items():
        frame_results = []
        for mask, meta in zip(masks, mask_metadata[frame_id]):
            mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            frame_results.append({
                "mask": mask_resized > 0,
                "label": meta["label"]
            })
        aligned_results[frame_id] = frame_results
    return aligned_results


def extract_object_point_clouds(predictions, aligned_results, conf_thres):
    """
    Apply object masks to VGGT point maps and confidence maps.
    Return object-specific point clouds with colors.
    """
    if not aligned_results:
        return {}
        
    object_data = {} # {label: {"points": [], "colors": []}}
    frame_ids = sorted(aligned_results.keys())
    
    # Handle image format for colors
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW
        images_nhwc = np.transpose(images, (0, 2, 3, 1))
    else:
        images_nhwc = images

    for i, frame_id in enumerate(frame_ids):
        point_map = predictions["world_points_from_depth"][i]
        conf_map = predictions["depth_conf"][i]
        color_map = (images_nhwc[i] * 255).astype(np.uint8)
        
        flat_conf = conf_map.reshape(-1)
        threshold_val = np.percentile(flat_conf, conf_thres) if conf_thres > 0 else 0.0
        
        for res in aligned_results[frame_id]:
            mask = res["mask"]
            label = res["label"]
            
            valid_pixels = mask & (conf_map > threshold_val)
            object_points = point_map[valid_pixels]
            object_colors = color_map[valid_pixels]
            
            if label not in object_data:
                object_data[label] = {"points": [], "colors": []}
            object_data[label]["points"].append(object_points)
            object_data[label]["colors"].append(object_colors)
            
    # Concatenate
    final_object_clouds = {}
    for label in list(object_data.keys()):
        if object_data[label]["points"]:
            pts = np.concatenate(object_data[label]["points"], axis=0)
            clrs = np.concatenate(object_data[label]["colors"], axis=0)
            final_object_clouds[label] = {"points": pts, "colors": clrs}
        else:
            continue
            
    return final_object_clouds


def export_object_outputs(object_point_clouds, target_dir, predictions):
    """
    Save each object point cloud as PLY and GLB, applying scene alignment.
    """
    import trimesh
    from visual_util import get_opengl_conversion_matrix, apply_scene_alignment
    
    if not object_point_clouds:
        return
        
    output_base_dir = os.path.join(target_dir, "object_outputs")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get alignment matrix from first camera
    extrinsics = predictions["extrinsic"]
    num_cameras = len(extrinsics)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = extrinsics
    extrinsics_matrices[:, 3, 3] = 1
    
    for label, data in object_point_clouds.items():
        points = data["points"]
        colors = data["colors"]
        
        if points.shape[0] == 0:
            continue
            
        obj_dir = os.path.join(output_base_dir, label)
        os.makedirs(obj_dir, exist_ok=True)
        
        # Create PointCloud
        pc = trimesh.PointCloud(vertices=points, colors=colors)
        
        # Save PLY (original coordinates)
        pc.export(os.path.join(obj_dir, "point_cloud.ply"))
        
        # Create Scene for GLB and apply alignment
        scene = trimesh.Scene(pc)
        scene = apply_scene_alignment(scene, extrinsics_matrices)
        scene.export(os.path.join(obj_dir, "object.glb"))
        
    print(f"Backend: Object outputs exported to {output_base_dir}")


# -------------------------------------------------------------------------
# 5) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    detected_objects=None,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model, detected_objects=detected_objects)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    object_choices = ["Full Scene"]
    
    # --- Object-level Reconstruction Pipeline ---
    if detected_objects:
        try:
            detections_per_frame = run_groundingdino(
                target_dir=target_dir,
                detected_objects=detected_objects
            )

            masks_per_frame, mask_metadata = run_sam2(
                target_dir=target_dir,
                detections_per_frame=detections_per_frame
            )

            aligned_results = align_masks_to_vggt(
                masks_per_frame=masks_per_frame,
                mask_metadata=mask_metadata,
                predictions=predictions
            )

            object_point_clouds = extract_object_point_clouds(
                predictions=predictions,
                aligned_results=aligned_results,
                conf_thres=conf_thres
            )

            export_object_outputs(
                object_point_clouds=object_point_clouds,
                target_dir=target_dir,
                predictions=predictions
            )
            
            object_choices += sorted(list(object_point_clouds.keys()))
        except Exception as e:
            print(f"Error in object-level pipeline: {e}")
            # Continue to show full scene even if object extraction fails

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True), gr.Dropdown(choices=object_choices, value="Full Scene", interactive=True), "Full Scene"


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def switch_object_view(target_dir, object_name):
    """
    Switches the 3D viewer to show a specific object or the full scene.
    """
    if not target_dir or target_dir == "None":
        return None
        
    if object_name == "Full Scene" or not object_name:
        # Full scene path is tricky because it depends on all params. 
        # For simplicity, we just look for the first GLB in target_dir that isn't in object_outputs.
        glbs = glob.glob(os.path.join(target_dir, "*.glb"))
        return glbs[0] if glbs else None
    else:
        obj_glb = os.path.join(target_dir, "object_outputs", object_name, "object.glb")
        return obj_glb if os.path.exists(obj_glb) else None


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            detected_objects = gr.Dropdown(
                label="Objects to detect (Type and press Enter)",
                choices=[],
                multiselect=True,
                allow_custom_value=True,
                info="Enter the names of objects you want to detect in the scene.",
            )

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)
            
            with gr.Row():
                object_selector = gr.Dropdown(
                    choices=["Full Scene"],
                    value="Full Scene",
                    label="Select Object to Visualize",
                    interactive=True,
                )

    # ---------------------- Examples section ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True", []],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True", []],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True", []],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True", []],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True", []],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True", []],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True", []],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
        detected_objects,
    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery
        We do NOT return is_example. It's just an input.
        """
        target_dir, image_paths = handle_uploads(input_video, input_images)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown, obj_dropdown, selected_obj = gradio_demo(
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            detected_objects=detected_objects,
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths, obj_dropdown

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
            detected_objects,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery, object_selector],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            detected_objects,
        ],
        outputs=[reconstruction_output, log_output, frame_filter, object_selector, object_selector],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    object_selector.change(
        fn=switch_object_view,
        inputs=[target_dir_output, object_selector],
        outputs=[reconstruction_output]
    )

    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
