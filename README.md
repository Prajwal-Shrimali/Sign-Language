# Sign Language Recognition with MediaPipe & Deep Learning

This project uses MediaPipe to extract hand landmarks from images and videos, then trains a deep neural network to recognize American Sign Language (ASL) letters. It supports both batch prediction and real-time webcam inference.

---

## 📁 Project Structure

```
Sign Language/
├── mediapipemodel.ipynb      # Main notebook for data processing & training
├── realtime_test.py          # Real-time ASL prediction from webcam
├── working/
│   ├── hand_landmarks_with_features.csv  # Extracted features for training
│   ├── label_encoder.pkl                 # Saved label encoder
│   └── models/
│       └── asl_model.h5                  # Trained model
├── Synthetic ASL Alphabet/   # Dataset folders
│   ├── Train_Alphabet/
│   └── Test_Alphabet/
└── README.md                 # This file
```

---

## 🛠️ Setup

1. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe tensorflow scikit-learn pandas tqdm joblib matplotlib
   ```

2. **Download Datasets:**
   - Place the `Synthetic ASL Alphabet` dataset in the project folder.
   - Make sure paths in the notebook/scripts match your folder structure.

---

## 🧑‍💻 Training the Model

1. **Run `mediapipemodel.ipynb`:**
   - Extracts hand landmarks and features from images using MediaPipe.
   - Saves features to `working/hand_landmarks_with_features.csv`.
   - Trains a multi-layer perceptron (MLP) on these features.
   - Saves the trained model to `working/models/asl_model.h5` and the label encoder to `working/label_encoder.pkl`.

2. **Feature Extraction:**
   - Uses normalized landmarks, pairwise distances, and angles between fingers for robust recognition.

3. **Model Architecture:**
   - 4–5+ dense layers with dropout for regularization.
   - Output layer matches the number of ASL classes.

---

## 📷 Real-Time ASL Prediction

1. **Run `realtime_test.py`:**
   - Uses your webcam to capture hand images.
   - Extracts features using the same pipeline as training.
   - Loads the trained model and label encoder.
   - Predicts and displays the ASL letter in real time.

   ```bash
   python realtime_test.py
   ```

   - Press `ESC` to exit.

2. **Troubleshooting:**
   - If predictions are poor, ensure the feature extraction in `realtime_test.py` matches the training pipeline (`extract_features`).
   - Good lighting and clear hand pose improve detection.

---

## 🖼️ Testing on Static Images

To test a single image instead of webcam, modify `realtime_test.py`:

```python
img_path = r"Synthetic ASL Alphabet\Test_Alphabet\A\your_image.png"
image = cv2.imread(img_path)
# ... (use the same feature extraction and prediction code as in real-time)
```

---

## 📝 Notes

- **Label Encoder:** Always use the same encoder for training and inference (`label_encoder.pkl`).
- **Feature Consistency:** The model expects features in the same format as training (`extract_features`).
- **Dataset:** You can expand with more ASL letters or custom gestures by adding more images and retraining.

---

## 🧩 Customization

- **Model Depth:** You can increase the number of layers in the MLP for better accuracy.
- **Other Gestures:** Add new classes by updating your dataset and retraining.
- **Performance:** Use GPU for faster training and inference.

---

## ❓ FAQ

**Q: Why is my real-time prediction inaccurate?**  
A: Make sure you use the same feature extraction pipeline for both training and inference. Lighting and hand pose also affect results.

**Q: How do I add new ASL letters or gestures?**  
A: Add images to your dataset, extract features, retrain the model, and update the label encoder.

**Q: Can I use this for other sign languages?**  
A: Yes, with a suitable dataset and retraining.

---

## 📚 References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow](https://www.tensorflow.org/)
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## 👤 Author

- Developed by Prajwal Shrimali
- For questions, open an issue or contact via GitHub.
