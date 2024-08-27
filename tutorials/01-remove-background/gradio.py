import gradio as gr
from common import remove_background_points, remove_background_box

def process_points(image, points, labels):
    points = [[float(p.split(',')[0]), float(p.split(',')[1])] for p in points.split()]
    labels = [int(l) for l in labels.split()]
    return remove_background_points(image, points, labels)

def process_box(image, x1, y1, x2, y2):
    return remove_background_box(image, x1, y1, x2, y2)

def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Product Background Removal with SAM2")
        gr.Markdown("Upload a product image and provide either points or a bounding box to remove the background.")
        
        with gr.Tab("Points"):
            with gr.Row():
                image_input_points = gr.Image(type="numpy", label="Upload Product Image")
                output_points = [
                    gr.Image(type="pil", label="Processed Image"),
                    gr.Image(type="pil", label="Before/After Comparison")
                ]
            points_input = gr.Textbox(label="Points (format: 'x1,y1 x2,y2 ...')")
            labels_input = gr.Textbox(label="Labels (format: '1 0 ...' where 1=foreground, 0=background)")
            points_button = gr.Button("Remove Background (Points)")
        
        with gr.Tab("Bounding Box"):
            with gr.Row():
                image_input_box = gr.Image(type="numpy", label="Upload Product Image")
                output_box = [
                    gr.Image(type="pil", label="Processed Image"),
                    gr.Image(type="pil", label="Before/After Comparison")
                ]
            with gr.Row():
                x1_input = gr.Number(label="X1")
                y1_input = gr.Number(label="Y1")
                x2_input = gr.Number(label="X2")
                y2_input = gr.Number(label="Y2")
            box_button = gr.Button("Remove Background (Box)")
        
        points_button.click(
            process_points,
            inputs=[image_input_points, points_input, labels_input],
            outputs=output_points
        )
        
        box_button.click(
            process_box,
            inputs=[image_input_box, x1_input, y1_input, x2_input, y2_input],
            outputs=output_box
        )
    
    return demo

if __name__ == "__main__":
    create_gradio_app().launch()