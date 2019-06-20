# face_recognition
Face recognition system created over the face_recognition library of @ageitgey

# encode_images.py

**Arguments:**
- -i --input : The input file directoy of NEW images
- -d --detection : Face detection method to be used. Can be "hog" or "cnn" (Default: cnn)

**What it does**
- Reads every image from the sub-folders of the image directory given as input.
- Encodes those images to a pickle file : encodings.pickle under face_data directory.
- Moves the images from the image directory to face_data folder
- Can be reused. The new images will be further appended to the exsting encodings.pickle file

**To run**
$python encode_images.py --input <new_image_dir> --detection "cnn"

# recognize_face.py

**Arguments:**
- -i --input : The input jpg images
- -d --detection : Face detection method to be used. Can be "hog" or "cnn" (Default: hog)

**What it does**
- Reads the image and creates its encoding
- Compares it with known encodings (known images)
- Gets corresponding details of the employee from "emp.xlsx" file in face_data
- Outputs image with bounding boxes over faces and their details.

**To run**
$python recognize_face.py --input <image.jpg> --detection "hog"
