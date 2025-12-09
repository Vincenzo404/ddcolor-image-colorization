1) Install dependencies:
   !pip install modelscope==1.9.5 opencv-python pillow matplotlib addict scipy

2) Upload gambar grayscale:
   from google.colab import files
   uploaded = files.upload()
   filename = list(uploaded.keys())[0]

3) Jalankan pipeline DDColor untuk colorization:
   from modelscope.pipelines import pipeline
   from modelscope.hub.snapshot_download import snapshot_download

   model_dir = snapshot_download("damo/cv_ddcolor_image-colorization", cache_dir="models")
   ddcolor = pipeline(task="image-colorization", model=model_dir, device="cuda")

   result = ddcolor(filename)

4) Simpan dan tampilkan hasil:
   from PIL import Image
   import numpy as np

   img_np = result["output_img"]
   img_rgb = img_np[:, :, ::-1]
   img_pil = Image.fromarray(img_rgb)
   img_pil.save("colorized_output.png")

5) Bandingkan hasilnya:
   import matplotlib.pyplot as plt
   import cv2

   gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
   rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

   plt.figure(figsize=(12,4))
   plt.subplot(1,3,1)
   plt.imshow(gray_img, cmap='gray')
   plt.title("Grayscale")
   plt.axis("off")
  
   plt.subplot(1,3,2)
   plt.imshow(rgb_img)
   plt.title("RGB (3-channel)")
   plt.axis("off")
  
   plt.subplot(1,3,3)
   plt.imshow(Image.open("colorized_output.png"))
   plt.title("Colorized (DDColor)")
   plt.axis("off")
   plt.show()




