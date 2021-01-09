"""
Ain Shams University
Mechatronics & Automation Engineering
CSE 489: Machine Vision
Author: Ahmad Samy
Library for image segmentation using classical methods
"""
from algorithms import *


def main():
    num_of_imgs = 10
    for i in range(0, num_of_imgs):
        image_path = "TestingImages/" + str(i) + ".jpg"

        k_nearest_neighbor(image_path, k=5)
        cm_cluster(image_path, 20)
        bayes(image_path)
        svm(image_path)


if __name__ == '__main__':
    main()
