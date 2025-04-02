import cv2 
import os

# add according to your 
origin_paths = ["/home/muhnatha/code/Jupyter Notebook/FaceRecognitionCV/new_dataset/Muhammad_Natha_Ulinnuha"]
target_paths = ["/home/muhnatha/code/Jupyter Notebook/FaceRecognitionCV/dataset/images/Muhammad_Natha_Ulinnuha"]

for origin, target in zip(origin_paths, target_paths):
    os.makedirs(target, exist_ok=True)
    for filename in os.listdir(origin):
        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(origin, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Error loading {img_path}")
                continue

            resized_image = cv2.resize(image, (250,250))

            base_filename = os.path.splitext(filename)[0]
            output_filename = base_filename + ".jpg"
            output_path = os.path.join(target, output_filename)

            cv2.imwrite(output_path, resized_image)
            print(f"The images saved to {output_path}")