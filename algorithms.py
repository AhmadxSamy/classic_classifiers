"""
Ain Shams University
Mechatronics & Automation Engineering
CSE 489: Machine Vision
Author:     Ahmad Samy Shafiek  -   16P9051
            Ahmad Tarek Shams   -   16P8185
            Hadi Badr           -   16P8216
            Mohamed Ayman Salah -   16P8203
Library for image segmentation using classical methods
"""

import numpy as np
import cv2
import math
from sklearn import svm


def k_nearest_neighbor(image_path: str, k: int = 1):
    """
    K-Nearest Neighbour Algorithm for image segmentation. Using k=1 (which is default value), yields
    "Nearest-Neighbour" classfication algorithm. Output is saved on local disk.
    :param image_path: path to image.
    :param k: Number of nearest neighbours.
    """
    n_regions = 2   # NUMBER OF REGIONS
    if not k % 2:
        print("K cannot be even. K-Nearest Neighbour Algorithm Aborted.")
        return 0
    img_raw = cv2.imread(image_path)
    # cv2.imshow("IMAGE #" + image_path[14:-4] , img_raw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mean_arr = np.zeros((1, 4))     # ARRAY TO STORE MEAN RGB OF EACH BOX
    for label_id in range(0, n_regions):
        print("Label Region #" + str(label_id))
        while True:
            boundingBoxes = cv2.selectROIs("LABEL REGION #" + str(label_id), img_raw)
            if isinstance(boundingBoxes, tuple):    # MAKING SURE A AT LEAST ONE BOX WAS LABELED
                print("No Region Selected. Please select Region #" + str(label_id))
            else:
                print("Region " + str(label_id) + " labeled")
                cv2.destroyAllWindows()
                break

        for row in boundingBoxes:
            x1 = row[0]
            y1 = row[1]
            x2 = row[2]
            y2 = row[3]
            img_crop = img_raw[y1:y1 + y2, x1:x1 + x2]
            mean_vector = np.mean(img_crop, axis=(0, 1))
            mean_vector = np.append(mean_vector, label_id).reshape(1, 4)
            mean_arr = np.concatenate((mean_arr, mean_vector))
    mean_arr = np.delete(mean_arr, 0, 0)    # REMOVING FIRST ROW CONTAINING ZEROS DUE TO INITIALIZATION
    if k > 1:       # IF K>1, CAUSE CLASSIFICATION ALGORITHM TO PERFORM NN AND KNN
        k_list = [1, k]
    else:           # IF K=1, CAUSE ALGORITHM TO PERFORM NN ONLY
        k_list = 1

    for k_iter in k_list:
        segmented_img = np.zeros((img_raw.shape[0], img_raw.shape[1]))
        segmented_img_rgb = np.zeros_like(img_raw)
        c_rgb_sum = np.zeros((n_regions, 3))
        c_dist = np.zeros(n_regions)  # distribution of classes after classes
        c_rgb = np.zeros_like(c_rgb_sum)
        for row in range(0, img_raw.shape[0]):
            for col in range(0, img_raw.shape[1]):
                dist = np.linalg.norm((mean_arr[:, :3]-img_raw[row, col]), axis=1)  # DISTANCE TO FIRST RECTANGLE
                min_idx = dist.argsort()[:k_iter]    # RETRIEVES INDICES OF LOWEST K DISTANCES
                neighbor_labels = mean_arr[min_idx, 3]  # CREATES A VECTOR OF THE LABELS OF K LOWEST DISTANCES
                segmented_img[row, col] = \
                    np.bincount(neighbor_labels.astype(int)).argmax()     # MOST FREQUENT LABEL IS INSERTED
                for label_id in range(0, n_regions):
                    if segmented_img[row, col] == label_id:
                        c_rgb_sum[label_id] = c_rgb_sum[label_id] + img_raw[row, col]
                        c_dist[label_id] = c_dist[label_id] + 1

        for label_id in range(0, n_regions):
            c_rgb[label_id] = c_rgb_sum[label_id] / c_dist[label_id]

        for row in range(img_raw.shape[0]):
            for col in range(img_raw.shape[1]):
                for label_id in range(0, n_regions):
                    if segmented_img[row, col] == label_id:
                        segmented_img_rgb[row, col] = c_rgb[label_id]

        for label_id in range(0, n_regions):
            c_rgb[label_id] = c_rgb_sum[label_id] / c_dist[label_id]

        cv2.imshow("Original Image", img_raw)
        cv2.imshow("Segmented Image", segmented_img_rgb)
        cv2.imwrite(image_path[:-4] + "_" + str(k_iter) + "nn.jpg", segmented_img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cm_cluster(image_path: str, epochs_limit: int = 20):
    """
    C-means clustering classification algorithm for image segmentation. Output is saved on local disk.
    This implementation segments to two classes only.
    :param image_path: Path to image relative to .py file including image name.
    :param epochs_limit: Maximum number of iterations.
    """
    img_raw = cv2.imread(image_path)
    class_rgb = np.random.randint(size=(2,3), high=255, low=0)
    for n in range(epochs_limit):
        c0_dist = np.linalg.norm(img_raw - class_rgb[0], axis=2)
        c1_dist = np.linalg.norm(img_raw - class_rgb[1], axis=2)
        segmented_img = np.zeros((img_raw.shape[0], img_raw.shape[1]))
        c_sum = np.zeros((2, 3))
        class_dist = np.zeros(2)
        for row in range(img_raw.shape[0]):
            for col in range(img_raw.shape[1]):
                if c0_dist[row][col] < c1_dist[row][col]:
                    segmented_img[row][col] = 0
                    c_sum[0] = c_sum[0] + img_raw[row][col]
                    class_dist[0] = class_dist[0] + 1
                else:
                    segmented_img[row][col] = 255
                    c_sum[1] = c_sum[1] + img_raw[row][col]
                    class_dist[1] = class_dist[1] + 1
        class_rgb[0] = c_sum[0] / class_dist[0]
        class_rgb[1] = c_sum[1] / class_dist[1]
    segmented_img_rgb = np.zeros_like(img_raw)
    for row in range(img_raw.shape[0]):
        for col in range(img_raw.shape[1]):
            if segmented_img[row, col] == 0:
                segmented_img_rgb[row, col] = class_rgb[0]
            else:
                segmented_img_rgb[row, col] = class_rgb[1]
    cv2.imwrite(image_path[:-4] + "_cm_iter_" + str(epochs_limit) + ".jpg", segmented_img_rgb)


def bayes(image_path: str):
    """

    :param image_path:
    :return:
    """
    img_raw = cv2.imread(image_path)
    segmented_img_rgb = np.zeros_like(img_raw)
    bg_box = cv2.selectROI("SELECT BACKGROUND", img_raw)
    ob_box = cv2.selectROI("SELECT OBJECT", img_raw)
    cv2.destroyAllWindows()

    obj_cropped = img_raw[ob_box[1]: ob_box[1] + ob_box[3], ob_box[0]: ob_box[0] + ob_box[2], :]
    bg_cropped = img_raw[bg_box[1] : bg_box[1]+bg_box[3], bg_box[0] : bg_box[0]+ bg_box[2], :]


    obj_mean = np.mean(obj_cropped, axis=(0, 1))
    bg_mean = np.mean(bg_cropped, axis=(0, 1))

    # COVARIANCE CALCULATION FOR BACKGROUND
    sum_bg = 0
    for row in range(bg_cropped.shape[0]):
        for col in range(bg_cropped.shape[1]):
            delta = bg_cropped[row, col] - bg_mean
            delta = delta.reshape((3, 1))
            z = np.dot(delta, delta.T)
            sum_bg = sum_bg + z
    n = np.dot(bg_cropped.shape[0], bg_cropped.shape[1])
    sigma_bg = sum_bg / n
    sigma_bg_inv = np.linalg.inv(sigma_bg)
    sigma_bg_det = np.linalg.det(sigma_bg)

    # COVARIANCE CALCULATION FOR OBJECT
    sum_obj = 0
    for row in range(obj_cropped.shape[0]):
        for col in range(obj_cropped.shape[1]):
            delta = obj_cropped[row, col] - obj_mean
            delta = delta.reshape((3, 1))
            z = np.dot(delta, delta.T)
            sum_obj = sum_obj + z
    n = np.dot(obj_cropped.shape[0], obj_cropped.shape[1])
    sigma_obj = sum_obj / n
    sigma_obj_inv = np.linalg.inv(sigma_obj)
    sigma_obj_det = np.linalg.det(sigma_obj)

    P_obj = np.random.random(1)
    P_bg = 1 - P_obj
    # PIXEL PROBABILITY CALCULATION
    for row in range(img_raw.shape[0]):
        for col in range(img_raw.shape[1]):
            delta = img_raw[row][col][:] - bg_mean
            delta = delta.reshape((1, 3))
            z = ((-1.5 * np.log(2 * math.pi)) - (0.5 * np.log(sigma_bg_det)) - (
                        0.5 * np.dot(np.dot(delta, sigma_bg_inv), delta.T)))
            p_bg = pow(10, z)

            delta = img_raw[row, col] - obj_mean
            delta = delta.reshape(1, 3)
            z = ((-1.5 * np.log(2 * math.pi)) - (0.5 * np.log(sigma_obj_det)) - (
                        0.5 * np.dot(np.dot(delta, sigma_obj_inv), delta.T)))
            p_ob = pow(10, z)

            if P_obj * p_ob > P_bg * p_bg:
                segmented_img_rgb[row, col, :] = obj_mean
            else:
                segmented_img_rgb[row, col, :] = bg_mean

    cv2.imshow("RESULTS WITH BAYES CLASSIFIER", segmented_img_rgb)
    cv2.waitKey(0)
    cv2.imwrite(image_path[:-4] + "_bayes.jpg", segmented_img_rgb)
    cv2.destroyAllWindows()


def svm(image_path: str):
    image_raw = cv2.imread(image_path)
    bg_box = cv2.selectROI("SELECT BACKGROUND", image_raw)
    obj_box = cv2.selectROI("SELECT OBJECT", image_raw)
    segmented_img_rgb = np.zeros_like(image_raw)

    obj_cropped = image_raw[obj_box[1]: obj_box[1] + obj_box[3], obj_box[0]: obj_box[0] + obj_box[2], :]
    bg_cropped = image_raw[bg_box[1]: bg_box[1] + bg_box[3], bg_box[0]: bg_box[0] + bg_box[2], :]
    obj_flat = obj_cropped.reshape(-1, 3)
    bg_flat = bg_cropped.reshape(-1, 3)

    obj_mean = np.mean(obj_cropped, axis=(0, 1))
    bg_mean = np.mean(bg_cropped, axis=(0, 1))

    inputX_train = np.vstack((obj_flat, bg_flat))
    inputY_train = np.vstack((np.ones((len(obj_flat), 1)), np.zeros((len(bg_flat), 1)))).reshape(-1)

    model = svm.SVC()
    model.fit(inputX_train, inputY_train)

    img_flat = image_raw.reshape(-1, 3)
    segmented_img = model.predict(img_flat).reshape(image_raw.shape[0], image_raw.shape[1])

    for row in range(segmented_img.shape[0]):
        for col in range(segmented_img.shape[1]):
            if segmented_img[row, col] == 1:
                segmented_img_rgb[row, col] = obj_mean
            else:
                segmented_img_rgb[row, col] = bg_mean

    cv2.imshow("RESULTS Of SVM CLASSIFIER", segmented_img_rgb)
    cv2.imwrite((image_path[:,-4] + "_svm.jpg"), segmented_img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

