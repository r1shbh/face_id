import numpy as np
import pickle
import os
import cv2
import face_recognition
import argparse
from constants import FACE_DATA_PATH, ENCODINGS_PATH, EMPLOYEE_PATH
import pandas as pd

def load_data():
    """
    Loads data from the encodings pickle file.

    Returns
    --------
    data : dict
        Dictionary containing {"encodings": [....], "names": [....]}
    """

    # Loading data

    f = open(ENCODINGS_PATH, "rb")
    data = {"encodings": [], "names": []}

    while 1:
        try:
            load_data = pickle.load(f)
            data["encodings"].extend(load_data["encodings"])
            data["names"].extend(load_data["names"])
        except EOFError:
            break
    f.close()
    return data


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

def encodings_to_names(encodings,data):
    """
    Input
    -------
    encodings: list
        The list of encodings found in the given input image
    data:
        The data file of our images

    Returns
    --------
    names: list
         Gives list of names identified in the image
        For unknown person returns "Unknown"
    """

    # Initialize the list of names for each face detected
    names = []

    # Loop over facial embeddings
    if len(encodings) == 0:
        print("No face found in the image")
    else:
        for encoding in encodings:
            # Attempt to match each face in input image to our data
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # The matches contains a list of boolean values with each of the encoded images in our dataset
            # If there is any match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face was matched
                matchedIDxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # Loop over matched indexes
                # Maintain a count for number of votes for each name
                for i in matchedIDxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Determine the name with largest votes
                name = max(counts, key=counts.get)

            names.append(name)

    return names



def get_employee_details(name, df):
    """
    Input
    -------
    name : str
        The name of the person.
    df: pandas dataframe
        The dataframe to search for

    Returns
    -------
    emp_dict : dict
        The dict of employee details.
        If name is "Unknown", returns None
    """
    if name=="Unknown":
        return None
    else:
        emp_dict={}
        cond = df["Name"]==name
        for column in df.columns:
            emp_dict[column] = df.loc[cond,column]

        return emp_dict



def names_to_ids(names,file_loc):
    """
    Input
    -------
    names: list
        List of names found in images
    file_loc: str
        Path to the file containing the employee data

    Returns
    --------
    empIDxs : list
        List containing the employee details of people in images
    """
    #Read the file
    df = pd.read_excel(file_loc)

    #Initialize empIDxs
    empIDxs = []

    #Looping through each face in the image
    for name in names:
        emp_dict = get_employee_details(name,df)
        empIDxs.append(emp_dict)

    #Returning the empIDxs
    return empIDxs



def show_output_image(img_loc,empIDxs,boxes):
    """
    Outputs output image with employee details
    Input
    ------
    img_loc: str
        The path to the input image
    empIDxs: dict
        The details of the persons in the image
    boxes: list
        A list of tuples containing bounding boxes of faces
        (top,right,bottom,left)
    """
    assert(len(empIDxs)==len(boxes))

    #Looping over every face in the image
    for ((top,right,bottom,left),empID) in zip(boxes,empIDxs):
        #Draw bounding box over the face
        image = cv2.imread(img_loc)
        cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
        y = top - 15 if top - 15 > 15 else top + 15

        #Preparing what to write
        if empID == None:
            text = "Unknown face"
        else:
            text = ''
            for key in empID.keys():
                text += str(key)+': '+str(empID[key])+'\n'

        #Writing to image
        cv2.putText(image,text,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

        #Showing image
        cv2.imshow("Face Recognition Results", image)
        cv2.waitKey(0)


def run():
    #Get arguements
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input",required=True,
                    help="Input jpg image path")
    ap.add_argument("-d","--detection",default="hog",
                    help="Face detection method. 'hog' or 'cnn'. Default : 'hog'")

    args = vars(ap.parse_args())


    #Load data
    print("[INFO] Loading encodings")
    data = load_data()
    print("[INFO] Loaded face encodings of known faces.")


    #Get encoding and boxes of input image
    print("[INFO] Encoding input image")
    encodings, boxes = encode_image(args["input"],args["detection"])
    print("[INFO] Reading image done.")

    #Get names of persons in the image
    print("[INFO] Recognizing")
    names = encodings_to_names(encodings,data)

    #Get employee details from name
    empIDxs = names_to_ids(names,EMPLOYEE_PATH)

    #Returning the output image
    show_output_image(args["input"],empIDxs,boxes)


run()
