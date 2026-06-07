# Planned Implementation Workflow and Data Format

The current implementation is based on the original `demo_gradio.py` from VGGT. The system already supports image or video upload through a Gradio interface, automatic frame extraction from video, image preview, VGGT inference, camera parameter conversion, depth-to-point unprojection, prediction saving, and GLB visualization. The current pipeline also includes a user interface field for specifying target objects through `detected_objects`, but the actual object detection, segmentation, and object-level 3D extraction have not yet been implemented.

The next stage of implementation will extend the current VGGT reconstruction pipeline into a text-guided object-level 3D reconstruction system. The planned pipeline will use the user-provided object names as text prompts, apply GroundingDINO to detect corresponding bounding boxes, use SAM2 to generate binary object masks from those boxes, and finally apply the masks to VGGT outputs to extract separate 3D point clouds for each object.

The planned workflow is:

```text
Input video or image set
        ↓
Frame extraction / image copy into target_dir/images
        ↓
User enters target object names in the Gradio dropdown
        ↓
VGGT inference generates depth, confidence, camera parameters, and point maps
        ↓
GroundingDINO detects object boxes for each frame using the object text prompts
        ↓
SAM2 generates binary masks from GroundingDINO boxes
        ↓
SAM2 masks are aligned with VGGT output resolution
        ↓
Masks are applied to VGGT point maps and confidence maps
        ↓
Object-specific 3D points are extracted
        ↓
Each object is exported as an individual point cloud or GLB model
```

## Current Implemented Components

The current code already implements the input and VGGT reconstruction stages.

First, the upload handler creates a unique working directory for each run. The directory structure is currently:

```text
input_images_<timestamp>/
    images/
        000000.png
        000001.png
        ...
```

If the user uploads images, the files are copied directly into the `images/` folder. If the user uploads a video, frames are extracted at one frame per second and saved as PNG files.

Second, the uploaded images are passed into VGGT using:

```python
images = load_and_preprocess_images(image_names).to(device)
predictions = model(images)
```

After inference, the pose encoding is converted into camera extrinsic and intrinsic matrices:

```python
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    images.shape[-2:]
)
```

The current prediction dictionary contains VGGT outputs such as:

```text
pose_enc
depth
depth_conf
world_points
world_points_conf
images
extrinsic
intrinsic
```

The implementation also computes additional 3D world points from the predicted depth map:

```python
world_points_from_depth = unproject_depth_map_to_point_map(
    depth_map,
    predictions["extrinsic"],
    predictions["intrinsic"]
)
```

The predictions are saved as:

```text
input_images_<timestamp>/
    predictions.npz
```

The current system can then convert the VGGT predictions into a GLB scene for visualization:

```text
glbscene_<settings>.glb
```

This provides the full-scene 3D reconstruction result.

## Planned Object Prompt Format

The existing Gradio UI already includes a multi-select dropdown for object names:

```python
detected_objects = gr.Dropdown(
    label="Objects to detect (Type and press Enter)",
    choices=[],
    multiselect=True,
    allow_custom_value=True
)
```

This field will be used as the object prompt input for GroundingDINO.

The expected input format is a list of object names:

```python
detected_objects = ["cup", "bottle", "toy"]
```

Before passing the object list into GroundingDINO, the list will be converted into a text prompt format:

```text
"cup . bottle . toy ."
```

Each object name will be treated as a target category. The system will run GroundingDINO on every frame in `target_dir/images` and detect bounding boxes corresponding to these object names.

## Planned GroundingDINO Output Format

For each input frame, GroundingDINO will output detected object boxes, labels, and confidence scores.

The planned detection result format is:

```python
detections_per_frame = {
    frame_id: [
        {
            "label": "cup",
            "score": 0.83,
            "box": [x1, y1, x2, y2]
        },
        {
            "label": "bottle",
            "score": 0.79,
            "box": [x1, y1, x2, y2]
        }
    ]
}
```

The bounding box format will be:

```text
[x1, y1, x2, y2]
```

where `x1, y1` are the top-left coordinates and `x2, y2` are the bottom-right coordinates in image pixel space.

A detection confidence threshold will be applied to remove weak detections:

```python
keep = detection_score > detection_threshold
```

The detection results will be saved for debugging and reuse:

```text
input_images_<timestamp>/
    detections/
        frame_000000.json
        frame_000001.json
        ...
```

Each JSON file will contain all detected boxes for one frame.

## Planned SAM2 Mask Generation Format

After GroundingDINO produces bounding boxes, each box will be passed to SAM2 as a box prompt.

For each detected object, SAM2 will receive:

```text
image: RGB image, shape H × W × 3
box: [x1, y1, x2, y2]
```

SAM2 will output one or more candidate masks. If multiple masks are returned, the mask with the highest SAM2 confidence score will be selected:

```python
best_mask = masks[argmax(scores)]
```

The selected mask will be stored as a binary mask:

```text
mask[y, x] = 1 if the pixel belongs to the object
mask[y, x] = 0 otherwise
```

The planned mask format is:

```text
shape: H × W
dtype: bool or uint8
```

The masks will be saved under:

```text
input_images_<timestamp>/
    masks/
        frame_000000/
            cup_0.png
            bottle_0.png
        frame_000001/
            cup_0.png
            bottle_0.png
```

A corresponding metadata file will also be saved:

```text
input_images_<timestamp>/
    masks/
        mask_metadata.json
```

The metadata will record the relationship between frame IDs, object labels, instance IDs, boxes, and mask paths:

```python
mask_metadata = {
    "frame_000000": [
        {
            "object_id": "cup_0",
            "label": "cup",
            "box": [x1, y1, x2, y2],
            "mask_path": "masks/frame_000000/cup_0.png",
            "detection_score": 0.83,
            "sam_score": 0.91
        }
    ]
}
```

For the initial implementation, the system will assume that each object category appears at most once in the scene. Under this assumption, masks can be associated across views using their object labels. For example, all masks labeled `"cup"` will be treated as the same 3D object.

## Planned Alignment Between SAM2 Masks and VGGT Outputs

A key implementation detail is that the SAM2 masks must align with the VGGT point maps. VGGT uses preprocessed images for inference, while GroundingDINO and SAM2 may operate on the original uploaded images. Therefore, the implementation must ensure that the mask resolution matches the VGGT output resolution before applying the mask to VGGT predictions.

The current VGGT prediction outputs have per-frame spatial maps such as:

```text
depth: S × H × W × 1
depth_conf: S × H × W
world_points: S × H × W × 3
world_points_conf: S × H × W
world_points_from_depth: S × H × W × 3
```

where:

```text
S = number of input frames
H = VGGT output height
W = VGGT output width
```

The planned implementation will resize each SAM2 mask to the VGGT output size if necessary:

```python
mask_resized = resize(mask, target_size=(H, W))
```

The resized mask must remain binary:

```python
mask_resized = mask_resized > 0
```

This produces a mask that can be directly applied to VGGT point maps.

## Planned Object-Level Point Extraction

After obtaining resized object masks, the system will extract object-specific 3D points from VGGT outputs.

For each frame and each detected object:

```python
object_mask = masks[frame_id][object_id]
point_map = predictions["world_points_from_depth"][frame_id]
confidence_map = predictions["depth_conf"][frame_id]
```

The valid pixels will be selected using both the SAM2 mask and VGGT confidence:

```python
valid_pixels = (object_mask == 1) & (confidence_map > confidence_threshold)
```

Then the object-specific 3D points will be extracted:

```python
object_points = point_map[valid_pixels]
```

For multiple frames, points with the same object label will be concatenated:

```python
object_point_clouds["cup"].append(object_points_from_frame_i)
object_point_clouds["bottle"].append(object_points_from_frame_i)
```

After all frames are processed:

```python
object_point_clouds["cup"] = concatenate(all cup points)
object_point_clouds["bottle"] = concatenate(all bottle points)
```

The expected output is a dictionary of separated object-level point clouds:

```python
object_point_clouds = {
    "cup": np.ndarray of shape N × 3,
    "bottle": np.ndarray of shape M × 3,
    "toy": np.ndarray of shape K × 3
}
```

Each point cloud contains only the 3D points selected by the corresponding SAM2 object mask.

## Planned Confidence Filtering

The current Gradio interface already includes a confidence threshold slider:

```python
conf_thres = gr.Slider(
    minimum=0,
    maximum=100,
    value=50,
    step=0.1,
    label="Confidence Threshold (%)"
)
```

This threshold will also be used for object-level extraction. The slider value will be converted into a confidence threshold and applied to VGGT confidence maps.

The planned filtering rule is:

```python
valid_pixels = (object_mask == 1) & (confidence_map > conf_thres)
```

This step removes low-confidence VGGT points and reduces noisy geometry in the final object-level reconstruction.

## Planned Object-Level Output Files

The current implementation already exports a full-scene GLB file. The next implementation stage will add object-level outputs.

The planned output directory structure is:

```text
input_images_<timestamp>/
    predictions.npz
    detections/
    masks/
    object_outputs/
        cup/
            point_cloud.ply
            object.glb
        bottle/
            point_cloud.ply
            object.glb
        toy/
            point_cloud.ply
            object.glb
```

The point cloud file will store the extracted 3D object points:

```text
point_cloud.ply
```

The GLB file will be used for visualization in the Gradio 3D viewer:

```text
object.glb
```

At the first stage, the implementation will focus on exporting separated point clouds. Object-level GLB or mesh generation can then be implemented by adapting the existing `predictions_to_glb` visualization logic or by constructing a new object-only point cloud scene.

## Planned Gradio UI Extension

The current UI already supports object name input through the `detected_objects` dropdown. The next update will connect this input to the backend detection and segmentation pipeline.

The planned Gradio behavior is:

```text
1. User uploads video or images.
2. Uploaded frames are displayed in the preview gallery.
3. User enters object names in the "Objects to detect" field.
4. User clicks "Reconstruct".
5. VGGT runs full-scene reconstruction.
6. GroundingDINO detects object boxes for the selected object names.
7. SAM2 generates masks for the detected boxes.
8. The masks are applied to VGGT point maps.
9. The system exports both the full-scene reconstruction and separated object-level outputs.
10. The user can download the full scene and each object-level 3D result.
```

The Gradio interface may later include an additional dropdown for selecting which object to visualize:

```python
object_selector = gr.Dropdown(
    choices=["Full Scene", "cup", "bottle", "toy"],
    label="Select Object to Visualize"
)
```

When the user selects an object, the viewer will display the corresponding object-level GLB file.

## Planned Backend Functions

The next implementation will add several backend functions to the current codebase.

The planned function for GroundingDINO detection is:

```python
def run_groundingdino(target_dir, detected_objects):
    """
    Run GroundingDINO on all images in target_dir/images.
    Return detections_per_frame.
    """
```

The planned function for SAM2 segmentation is:

```python
def run_sam2(target_dir, detections_per_frame):
    """
    Run SAM2 using GroundingDINO boxes as prompts.
    Return masks_per_frame and mask metadata.
    """
```

The planned function for mask resizing and alignment is:

```python
def align_masks_to_vggt(masks_per_frame, predictions):
    """
    Resize SAM2 masks to match VGGT output resolution.
    Return aligned binary masks.
    """
```

The planned function for object-level point extraction is:

```python
def extract_object_point_clouds(predictions, aligned_masks, conf_thres):
    """
    Apply object masks to VGGT point maps and confidence maps.
    Return object-specific point clouds.
    """
```

The planned function for exporting object outputs is:

```python
def export_object_outputs(object_point_clouds, target_dir):
    """
    Save each object point cloud as PLY and optionally GLB.
    """
```

These functions will be integrated into the existing `gradio_demo()` function after VGGT inference and before or after full-scene GLB generation.

## Planned Integration into the Existing `gradio_demo()` Function

The current `gradio_demo()` function runs VGGT, saves predictions, and exports a full-scene GLB. The planned integration point is after:

```python
predictions = run_model(target_dir, model, detected_objects=detected_objects)
np.savez(prediction_save_path, **predictions)
```

The new object-level pipeline will be added as:

```python
if detected_objects:
    detections_per_frame = run_groundingdino(
        target_dir=target_dir,
        detected_objects=detected_objects
    )

    masks_per_frame = run_sam2(
        target_dir=target_dir,
        detections_per_frame=detections_per_frame
    )

    aligned_masks = align_masks_to_vggt(
        masks_per_frame=masks_per_frame,
        predictions=predictions
    )

    object_point_clouds = extract_object_point_clouds(
        predictions=predictions,
        aligned_masks=aligned_masks,
        conf_thres=conf_thres
    )

    export_object_outputs(
        object_point_clouds=object_point_clouds,
        target_dir=target_dir
    )
```

This keeps the original full-scene VGGT reconstruction intact while adding a new object-level extraction branch.

## Planned Final Output

The final implementation will produce both full-scene and object-level results.

The full-scene output will remain:

```text
input_images_<timestamp>/
    glbscene_<settings>.glb
```

The new object-level outputs will be:

```text
input_images_<timestamp>/
    object_outputs/
        <object_name>/
            point_cloud.ply
            object.glb
```

For example:

```text
input_images_20260608_120000/
    object_outputs/
        cup/
            point_cloud.ply
            object.glb
        bottle/
            point_cloud.ply
            object.glb
```

The expected demonstration is that the original VGGT output reconstructs the whole scene, while the new object-level branch reconstructs only the objects specified by the user through text prompts.
