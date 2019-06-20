import argparse
import pickle
import os
import numpy as np
import glob
import face_recognition
import cv2
from constants import FACE_DATA_PATH, ENCODINGS_PATH
import shutil



def encode_image(img_loc,detection):
    """
    Reads a image into RGB form and returns an encoding of faces in the picture.

    Input
    --------
    img_loc: str
        The path to the image we want to encode

    Output
    --------
    encodings: list
        The list of encodings of faces in the image
    boxes: list of tuples
        The list of tuples of locations of faces found in the image.
        (top,right,bottom,left)
    """

    # Reading the image
    img = cv2.imread(img_loc)

    # Convert images to rgb format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    # Gives (x,y) coordinates of faces in the image
    boxes = face_recognition.face_locations(img, model=detection)

    # Create encoding for the face
    encodings = face_recognition.face_encodings(img, boxes)

    return encodings, boxes


def move_image(img_loc,personName):

    image_name = os.path.basename(img_loc)
    new_image_loc = os.path.join(FACE_DATA_PATH, personName.title())

    if os.path.exists(new_image_loc) == False:
       os.mkdir(new_image_loc)

    # Moving from img_loc to new_image_loc
    shutil.move(img_loc, new_image_loc)

    return



def get_face_encodings(new_image_dir,detection):
    """
    Get encodings from an image directory
    Input
    ------
    new_image_dir: str
        The path to directory containing the subfolders of classes of NEW images
    detection : str
        the face detection method. Can be "hog" or "cnn"
    Returns
    --------
    known_encodings: list
        A list of known image encodings

    imgLabels: list
        The labels for the corresponding images (names in capital)
    """



    knownEncodings = []
    imgLabels = []

    for personName in os.listdir(new_image_dir):

        # Now we are accessing individual folders


        # Saving the person's directory name
        person_class_dir = os.path.join(str(new_image_dir),personName)

        # Running through images of that person
        for img_loc in glob.glob(person_class_dir + '/*.jpg'):

            # Get encoding for the image. Ignore boxes
            encodings, _ = encode_image(img_loc, detection)

            # Append these encodings to knownEncodings
            #WE ASSUME THERE IS ONE IMAGE OF THAT PERSON IN THE IMAGE
            try:
                knownEncodings.append(encodings[0])
                imgLabels.append(personName.title())
            except IndexError:
                print("Not able to locate at least one face in image :{}".format(img_loc))
                print("Aborting...")


            ## Moving the image from new_image_dir to face_data
            ## As soon as we encode a given image, we move it to our face_data folder

            # First making the face_data directory if not exists
            if os.path.exists(FACE_DATA_PATH) ==  False:
                os.mkdir(FACE_DATA_PATH)

            # Making new_image_loc in face_data
            move_image(img_loc,personName)


    return knownEncodings, imgLabels


def run():
    """
    Reads the images from new_images folder
    Moves the images from new_images to face_data folder after encoding
    Dumps the encoding files to face_data/encodings.pickle
    """

    # Read the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input directory of new images")
    ap.add_argument("-d", "--detection", type=str, default="cnn",
                    help="Detection method. can be either: 'hog' or 'cnn'. Default: 'cnn'")
    args = vars(ap.parse_args())

    #Making encodings directory


    #Getting encodings list
    print("[INFO] Reading new images")
    knownEncodings, imgLabels = get_face_encodings(args["input"],args["detection"])

    print("The images from {} have been encoded.".format(args["input"]))
    print("The encoded images are moved in {}".format(FACE_DATA_PATH))

    #Dumping encodings to encodings folder
    print("[INFO] Serializing encodings...")
    new_data = {"encodings": knownEncodings, "names": imgLabels}
    path = os.path.abspath(ENCODINGS_PATH)
    with open(path,"wb") as f:
        pickle.dump(new_data,f)
    print("Encodings of images saved in {}".format(ENCODINGS_PATH))

run()
