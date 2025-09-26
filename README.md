# 🎨 Vintage Vibrance

Vintage Vibrance is a web application that transforms **black-and-white images into vibrant color** using a deep learning–based colorization model.  
Built with **Streamlit**, **OpenCV**, and pretrained **Caffe models**, it offers a simple way to upload, process, customize, and download enhanced images.

---

## Features
- 🖼️ **Image Upload** – Upload black-and-white or color images.  
- 🎨 **Deep Learning Colorization** – Restore colors with pretrained Caffe models.  
- ✨ **Image Enhancements** – Adjust tones, brightness, blur, and add stickers/text.  
- ⚡ **Quick Processing** – Fast and accurate results.  
- 📥 **Download & Share** – Save results locally for easy sharing.  

---

## Tech Stack

- **Backend / Runtime:** Python 3 (pip)  
- **Web UI:** Streamlit (app layout, sidebar, image display)  
- **Computer Vision:** OpenCV (`cv2`) with DNN Caffe model (colorization)  
- **Numerical Computing:** NumPy  
- **Imaging:** Pillow (PIL) – image loading, enhancements, stickers, text overlays  
- **Data / Utility (via Streamlit):** pandas, pyarrow, altair, requests (indirect dependencies)  
- **Model Assets:**  
  - `colorization_deploy_v2.prototxt`  
  - `colorization_release_v2.caffemodel`  
  - `pts_in_hull.npy` (cluster centers)  
- **Platform:** Runs locally (Windows/PowerShell) with  

  python -m streamlit run app.py

* **Key Files:**
  * `app.py` → Main Streamlit application
  * `models/` → Caffe + NumPy model assets
  * `Input_images/` → User-uploaded images
  * `stickers/` → Sticker assets for customization
  * `Result_images/` → Generated outputs
---

## Workflow

1. **Upload** a black-and-white image.
2. **Colorization** via OpenCV DNN with pretrained Caffe models.
3. **Enhancements** (brightness, blur, stickers, text overlays).
4. **Preview & Save** results with `st.image` (container width for responsiveness).
5. **Download** the final image.

---

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/vintage-vibrance.git
   cd vintage-vibrance
   ```
2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:

   ```bash
   python -m streamlit run app.py
   ```
5. Open in your browser:

   ```
   http://localhost:8501/
   ```

## 👩‍💻 Developed By

* Shravani Dakve

---

## 📝 License
This project is licensed under the MIT License – feel free to use and modify.  
