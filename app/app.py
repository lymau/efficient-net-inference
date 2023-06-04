import gradio as gr
import numpy as np
from PIL import Image
from helper import initialize_model
import keras.utils as image

model = initialize_model("model/model.h5")
CLASS_NAMES = ["Mask", "Mask_Chin", "Mask_Mouth_Chin", "Mask_Nose_Mouth"]


def prepare(img_path):
    IMG_SIZE = 224
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def flip(im: np.ndarray) -> np.ndarray:
    return np.flipud(im)


def predict_from_file(img):
    img = Image.fromarray(np.uint8(img))
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict([img])
    label = CLASS_NAMES[int(np.argmax(prediction, axis=1))]

    return label


with gr.Blocks() as demo:
    gr.Markdown("# EfficientNet Inference Demo App")
    with gr.Tab("Upload"):
        upload = gr.Interface(
            fn=predict_from_file,
            inputs="image",
            outputs="text",
            title="Upload an image",
        )

    with gr.Tab("Webcam"):
        with gr.Row():
            with gr.Column():
                inp = gr.Webcam()
            with gr.Column():
                out = gr.Image()
            inp.stream(fn=flip, inputs=[inp], outputs=[out])


demo.launch()
