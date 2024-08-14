import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from PIL import Image
import io

# Load the SAM2 model (you may want to do this outside the main app for efficiency)
@st.cache_resource
def load_model():
    checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

predictor = load_model()

def segment_image(input_image, mask):
    # Convert PIL Image to numpy array
    input_image = np.array(input_image)
    
    # Convert RGBA to RGB if necessary
    if input_image.shape[2] == 4:
        input_image = input_image[:,:,:3]
    
    # Prepare input prompts from the mask
    y, x = np.where(mask > 0)
    point_coords = np.array(list(zip(x, y)))
    point_labels = np.ones(len(point_coords))
    
    # Reduce the number of points if there are too many
    if len(point_coords) > 500:
        indices = np.random.choice(len(point_coords), 500, replace=False)
        point_coords = point_coords[indices]
        point_labels = point_labels[indices]
    
    # Set the image and predict the mask
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(input_image)
        masks, _, _ = predictor.predict(
            point_coords=point_coords, 
            point_labels=point_labels,
            multimask_output=False
        )
    
    # Use the predicted mask
    result_mask = masks[0].astype(np.uint8) * 255
    
    # Apply the mask to the image
    segmented_image = cv2.bitwise_and(input_image, input_image, mask=result_mask)
    
    # Create an output image where the background is transparent
    alpha = result_mask
    output_image = np.dstack([segmented_image, alpha])
    
    return Image.fromarray(output_image)

st.title("SAM2 Image Segmentation")
st.write("Upload an image, use the brush to select the area you want to keep, and see the segmented result.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Slider for brush size
    brush_size = st.slider("Brush size", 1, 50, 20)
    
    # Create a canvas for drawing with a brush
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",  # semi-transparent white
        stroke_width=brush_size,
        stroke_color="#ffffff",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Segment Image"):
        if canvas_result.image_data is not None:
            # Get the mask from the canvas
            mask = canvas_result.image_data[:,:,3]  # Alpha channel
            
            # Perform segmentation
            result = segment_image(image, mask)
            
            # Display the result
            st.image(result, caption="Segmented Image", use_column_width=True)
            
            # Provide download button for the result
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            btn = st.download_button(
                label="Download Segmented Image",
                data=buf.getvalue(),
                file_name="segmented_image.png",
                mime="image/png"
            )
else:
    st.write("Please upload an image to get started.")