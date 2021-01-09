"""
Ain Shams University
Mechatronics & Automation Engineering
CSE 489: Machine Vision
Author: Ahmad Samy
Library for dataset labeling
"""

import cv2
import numpy as np
import sys

def label_dataset(num_of_imgs: int):
    """
    A function using selectROI to label stream of images and saves it as jpg image.
    :param num_of_imgs (int): Number of images to label
   """

    for i in range(num_of_imgs):
        image_name = "TestingImages/" + str(i) + ".jpg"
        img_raw = cv2.imread(image_name)
        cv2.imshow("IMAGE " + str(i), img_raw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Enter number of regions (N):")
        num_of_reg = int(input())
        print("Number of regions = "
              + str(num_of_reg) +
              ". \n Create rectangles and after each rectangle press ENTER."
                                                         " When done, press ESC to finish labeling this region. "
                                                         "Repeat for each region.")
        labels_arr = np.zeros((img_raw.shape[0], img_raw.shape[1]))
        for label_id in range(1, num_of_reg+1):
            print("Label Region #" + str(label_id))
            while True:
                boundingBoxes = cv2.selectROIs("LABEL REGION #" + str(label_id), img_raw)
                if not isinstance(boundingBoxes, tuple):
                    print("Region " + str(label_id) + " labeled")
                    for row in boundingBoxes:
                        labels_arr[row[1]:row[1] + row[3], row[0]:row[0] + row[2]] = label_id/num_of_reg
                    cv2.destroyAllWindows()
                    break
                else:
                    print("No Region Selected. Please select Region " + str(label_id))
        cv2.imshow("Labels Array", labels_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("TestingImages/" + str(i) + "_label.jpg", labels_arr*255)
        np.savez("TestingImages/" + str(i) + "_label", labels_arr)

def import_lab_image(image_path):
    """
    This function imports a previously labelled image from disk in format of RGBL.
    :param image_path: string containing the path of the image. Example: "/TestingImages/1.jpg"
    :return: labelled_image: numpy_arr containing RGBL of size (height, width, 4)
    """
    image_raw = cv2.imread(image_path)
    file = np.load(image_path[:-4] + "_label.npz")
    labels_arr = file['arr_0']
    labelled_image = np.zeros((image_raw.shape[0], image_raw.shape[1], image_raw.shape[2]+1))
    labelled_image[:, :, :3] = image_raw/255
    labelled_image[:, :, 3] = np.copy(labels_arr)
    print(labelled_image.shape)
    return labelled_image



