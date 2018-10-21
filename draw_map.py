import numpy as np
import cv2
from scipy.spatial.distance import cdist
import json
import os, os.path

scale = 2
height = 230*scale
width = 370*scale
unit = 55*scale
offset = 80*scale
#unit = 25*scale
#offset = 33*scale

# draw - (x, y)
# np - (height, width)

def create_tree(k, w, fix_r=20, fix_c=20):
    img = np.zeros((height, width), np.uint8)

    cv2.line(img, (0, int(unit/2)), (width, int(unit/2)), 255, unit)

    for i in range(0, k):
        x = int((width-k*unit*w)/(k+1) * (i+1) + unit*i*w + w*unit/2)

        cv2.line(img, (x, 0), (x, height), 255, w*unit)

    # cv2.imshow("tree_k{}_w{}.png".format(k, w), img)
    # cv2.waitKey(0)

    cv2.imwrite("fig/tree_k{}_w{}.png".format(k, w), img)
    build_map(img, "tree_k{}_w{}".format(k, w), fix_r, fix_c)

def create_ladder(k, w, fix_r, fix_c):
    img = np.zeros((height, width), np.uint8)

    cv2.line(img, (0, int(unit/2)), (width, int(unit/2)), 255, unit)
    cv2.line(img, (0, int(height-unit/2)), (width, int(height-unit/2)), 255, unit)

    for i in range(0, k):
        x = int((width-k*unit*w) / (k-1) * i + unit*i*w + w*unit/2)

        cv2.line(img, (x, 0), (x, height), 255, w*unit)

    # cv2.imshow("ladder_k{}_w{}.png".format(k, w), img)
    # cv2.waitKey(0)

    cv2.imwrite("fig/ladder_k{}_w{}.png".format(k, w), img)
    build_map(img, "ladder_k{}_w{}".format(k, w), fix_r, fix_c)

def create_window(k, w):
    img = np.zeros((height, width), np.uint8)

    cv2.line(img, (0, int(unit/2)), (width, int(unit/2)), 255, unit)
    cv2.line(img, (0, int(height-unit/2)), (width, int(height-unit/2)), 255, unit)
    cv2.line(img, (0, int(height/2)), (width, int(height/2)), 255, unit)

    for i in range(0, k):
        x = int((width-k*unit*w) / (k-1) * i + unit*i*w + w*unit/2)

        cv2.line(img, (x, 0), (x, height), 255, w*unit)

    # cv2.imshow("window_k{}_w{}.png".format(k, w), img)
    # cv2.waitKey(0)

    cv2.imwrite("fig/window_k{}_w{}.png".format(k, w), img)    
    build_map(img, "window_k{}_w{}".format(k, w))

def build_map(img, filename, fix_r=20, fix_c=20):
    centers = []
    for r in range(0, height//offset+1):
        for c in range(0, width//offset+1):
            if r*offset+fix_r >= height: continue
            if c*offset+fix_c >= width: continue
            centers.append((r*offset+fix_r, c*offset+fix_c))
    centers = np.array(centers)
    # print("center shape = ", centers.shape)

    data = []
    for r in range(0, height):
        for c in range(0, width):
            if img[r][c] == 255:
                data.append((r, c))
    data = np.array(data)
    # print("data shape = ", data.shape)

    rest_center = []
    useful_center = []
    for center in centers:
        r, c = center
        if img[r][c] == 0:
            rest_center.append(center)
            continue

        useful_center.append(center)
        temp = data - center
        # print(center)
        # print(data[0])
        # print(temp[0])
        # print(temp.shape)
        temp = np.power(temp[:, 0], 2) + np.power(temp[:, 1], 2)
        temp = np.power(temp, 0.5)
        result = data[temp<=unit] 
        # print(result.shape)
        data = data[temp>unit]
        # print("data shape = ", data.shape)

    rest_center = np.vstack(rest_center)
    while data.shape[0] != 0:
        dist = cdist(data, rest_center)
        # print(np.min(dist, axis=0).shape)
        x = np.argmin(np.min(dist, axis=0))
        y = np.argmin(np.min(dist, axis=1))
        # print(y, x)
        # print(dist[y, x])
        
        center = data[y]
        useful_center.append(center)
        temp = data - center
        temp = np.power(temp[:, 0], 2) + np.power(temp[:, 1], 2)
        temp = np.power(temp, 0.5)

        data = data[temp>unit]
        # print("data shape = ", data.shape)

    # connection
    useful_center = np.vstack(useful_center)
    print("useful_center.shape = ", useful_center.shape)
    connection = {i:[] for i, c in enumerate(useful_center)}
    mapping_dictionary = {
        (center[0], center[1]):i 
        for i, center in enumerate(useful_center)
    }
    mapping_dictionary_2 = {
        i:(center[1], center[0])
        for i, center in enumerate(useful_center)
    }
    reverse_mapping_dictionary = {
        str(i):(str(center[0]), str(center[1]))
        for i, center in enumerate(useful_center)
    }
    # print(mapping_dictionary)
    # print(reverse_mapping_dictionary)
    for center in useful_center:
        temp = useful_center - center
        temp = np.power(temp[:, 0], 2) + np.power(temp[:, 1], 2)
        temp = np.power(temp, 0.5)
        centers = useful_center[temp!=0]
        temp = temp[temp!=0]
        # print(center, end=" => ")
        # print(centers.shape)
        # print(temp.shape)
        possible_centers = centers[temp<=(unit*2)]
        # print(possible_centers.shape)

        # temp_img = np.copy(img)
        c_index = mapping_dictionary[(center[0], center[1])]
        for p_center in possible_centers:
            temp_img = np.copy(img)
            # print(p_center, end=" ")
            cv2.line(temp_img, (center[1], center[0]), (p_center[1], p_center[0]), 111, 1)
            temp_img[img==255] = 255
            if not np.any(temp_img==111):
                p_index = mapping_dictionary[(p_center[0], p_center[1])]
                connection[c_index].append(p_index)
                # print("o, ", end=" ")
        if not connection[c_index]:
            print("GGGGG")
        # print()

    with open("mapping/{}_mapping_dictionary.json".format(filename), 'w', encoding='utf-8') as outfile:
        json.dump(reverse_mapping_dictionary, outfile, indent=4)

    with open("state/{}_state.json".format(filename), 'w', encoding='utf-8') as outfile:
        json.dump(connection, outfile, indent=4)

    # draw image
    print("useful_center.shape = ", useful_center.shape)
    img2 = np.copy(img)
    for center in useful_center:
        cv2.circle(img2, (center[1], center[0]), unit, 150, -1)
    img2[img==0] = 0
    for center in useful_center:
        cv2.circle(img2, (center[1], center[0]), unit, 100, 1)
    
    for c1, c2_list in connection.items():
        p1 = mapping_dictionary_2[c1]
        for c2 in c2_list:
            cv2.line(img2, p1, mapping_dictionary_2[c2], 200, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for c1, c2_list in connection.items():
        p1 = mapping_dictionary_2[c1]
        cv2.circle(img2, p1, 2, 20, -1)
        cv2.putText(img2, str(c1), p1, font, 1, 220, 2)

    # cv2.imshow("img2", img2)
    cv2.imwrite("fig/{}_state_img.png".format(filename), img2)
    # cv2.waitKey(0)

def test():
    img = np.zeros((512, 512), np.uint8)

    cv2.line(img, (15, 20), (70, 50), 255, 5)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    cv2.imwrite("sample.png", img)

def main():
    # create_tree(2, 1, 50, 50)
    create_ladder(2, 1, 50, 50)
    # create_ladder(4, 2)
    # create_tree(4, 1)
    # create_ladder(4, 1)
    # create_window(4, 1)

if __name__ == "__main__":
    main()