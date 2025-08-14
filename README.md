Corrosion Detection App

A **Streamlit-based and CNN-powered corrosion detection** application that identifies the presence of corrosion on metal surfaces. Users can either **upload images** or use a **live webcam feed** for real-time analysis, making it useful for **industrial-grade inspection** and **maintenance automation**.

---

##  Features

- 🏭 **Binary Classification** – Differentiates between “Corrosion” and “No Corrosion”.  
- 📷 **Image Upload or Webcam Input** – Easy interface for both static images and live stream.  
- ⏱ **Real-Time Predictions** – Immediate results through a sleek UI.  
- 🖥️ **Built with Streamlit** – Highly interactive and user-friendly frontend.  
- 🧠 **Custom CNN Model** – Built in `model.py`, trained via `train_model.py`, and saved/loaded using `save_model.py`.

---

Installation & Usage :
**Clone the repository**
   git clone https://github.com/sakshik1712/Corrosion-Detection-App.git
   cd Corrosion-Detection-App

Install dependencies :

'pip install -r requirements.txt'


Run the Streamlit app :

'streamlit run app.py'


Training a new model (optional) :

'python train_model.py'
'python save_model.py'


Use the app interface
Choose between uploading an image or activating your webcam to detect corrosion in real time.

Future Enhancements :
- Batch Image Processing – Enable processing of multiple images in one go.
- Localization – Highlight and outline corroded regions rather than just classifying them.
- Model Expansion – Train on more corrosion types, surfaces, and lighting conditions.
- Cross-Platform Deployment – Deploy as a web service (Flask/Django) or desktop app (Electron/PyInstaller).
- User Feedback Loop – Logging incorrect predictions for continual training improvements.

License
This project is licensed under the MIT License.
