import gradio as gr
import numpy as np


def flip(im):
    return np.flipud(im)


with gr.Blocks() as demo:
    gr.Markdown("# EfficientNet Inference Demo App")
    with gr.Tab("Upload"):
        upload = gr.Interface(
            fn=flip, inputs="image", outputs="image", title="Upload an image"
        )

    with gr.Tab("Webcam"):
        with gr.Row():
            with gr.Column():
                inp = gr.Webcam()
            with gr.Column():
                out = gr.Image()
            inp.stream(fn=flip, inputs=[inp], outputs=[out])

demo.launch()
