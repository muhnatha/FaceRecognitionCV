# Face Recognition

This project is a workshop assignment for the Computer Vision course. It includes images of 7 individuals, with 10 pictures each, which are trained into the model pipeline. The algorithm used is Support Vector Classification (SVC). For the detailed steps, please refer to the `FaceRecog.ipynb` notebook.

# Prerequisites
Make sure you have Python 3.7 or newer installed.

# Usage

1. Clone repository

```bash
git clone https://github.com/your-username/face-recognition.git
cd face-recognition
```
2. Install packages
```bash
pip install -r requirements.txt
```

3. Add your images dataset

Add your face or a person face image into the dataset.

4. Run Jupter notebook FaceRecog.ipynb

Note : Sometimes the model does not detect face in an image that can lead to error, you need to change the delete or change the image.

5. Real Time Implementation

After you done adding some dataset and train the model. You can run the `realTImeFaceRecog.py`.

```bash
python3 realTImeFaceRecog.py
```