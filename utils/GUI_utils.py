from tkinter import *
import os
from PIL import ImageTk, Image
def tensor_to_image(tensor):
    # Convert tensor to numpy array
    tensor = tensor.cpu().numpy()

    # Convert to PIL Image
    image = Image.fromarray((tensor * 255).astype('uint8'))

    return image
import numpy as np
from tkinter import Tk, Canvas, Scale, NW, BOTTOM, X, HORIZONTAL
from PIL import Image, ImageTk

def array_to_image(arr):
    # Normalize to 0-255 and convert to PIL Image
    arr = arr.astype(np.float32)
    arr = (255 * (arr - arr.min()) / (np.ptp(arr) + 1e-8)).astype(np.uint8)
    return Image.fromarray(arr)

def load_and_show_image_array(img_array, scale_mult = 3):
    def nex_img(i):
        canvas.delete('image')
        canvas.delete('index')
        canvas.create_image(0, 0, anchor=NW, image=listimg[int(i) - 1][0], tags='image')
        canvas.create_text(10, 10, anchor=NW, fill="darkblue", font="Times 16 italic bold",
                           text=f"Slice {listimg[int(i) - 1][1]}", tags='index')

    # Assume img_array shape = [N, H, W]
    first_shape = img_array[0].shape
    h, w = first_shape

    root = Tk()
    root.title("3D Image Viewer")

    # Convert each slice to ImageTk.PhotoImage without resizing
    listimg = [(ImageTk.PhotoImage(array_to_image(slice_2d).resize((w * scale_mult, h * scale_mult)), master=root), idx)
               for idx, slice_2d in enumerate(img_array)]

    scale = Scale(master=root, orient=HORIZONTAL, from_=1, to=len(listimg), resolution=1,
                  showvalue=False, command=nex_img)
    scale.pack(side=BOTTOM, fill=X)

    canvas = Canvas(root, width=900, height=900)
    canvas.pack()

    nex_img(1)

    root.mainloop()
