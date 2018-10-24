import numpy as np
import cv2
from scipy.spatial.distance import cdist
import json
import os, os.path
from pprint import pprint

scale = 2
height = 230*scale
width = 370*scale
unit = 43*scale
offset = 40*scale


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
    build_map(img, "_tree_k{}_w{}".format(k, w), fix_r, fix_c)

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
    build_map(img, "_ladder_k{}_w{}".format(k, w), fix_r, fix_c)

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
    build_map(img, "window_k{}_w{}".format(k, w), 50, 50)

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
    # print(rest_center)
    if rest_center:
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
    center_ner = {_id:[] for cor, _id in mapping_dictionary.items()}
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
        center_ner[c_index].extend(mapping_dictionary[(p_center[0], p_center[1])] for p_center in possible_centers)

    count = {}
    number = 0
    for center_id, ner in center_ner.items():
        for pcenter_id in ner:
            # print("(c, p) = ({}, {})".format(center_id, pcenter_id))
            p_ner = center_ner[pcenter_id]
            img1 = np.zeros((height, width), np.uint8)
            img2 = np.zeros((height, width), np.uint8)
            cv2.circle(img1, mapping_dictionary_2[center_id], unit, 10, -1)
            cv2.circle(img2, mapping_dictionary_2[pcenter_id], unit, 10, -1)
            black_img = img1+img2
            black_img[black_img!=20] = 0
            black_img[black_img==20] = 255
            black_img[img==0] = 0

            # if center_id == 7 and pcenter_id == 20:
            #     cv2.imshow("black_img", black_img)
            #     cv2.imshow("img", img)
            #     cv2.waitKey(0)

            # black_set = list(set(center_ner.keys())-(set(ner)&set(p_ner) | set([center_id, pcenter_id])))
            # black_img = np.copy(img)
            # for black_index in black_set:
            #     x, y = mapping_dictionary_2[black_index]
            #     cv2.circle(black_img, (x, y), unit, 0, -1)

            shared_centers = list(set(ner) & set(p_ner))
            if not shared_centers and np.any(black_img!=0):
                connection[center_id].append(pcenter_id)
            # print(shared_centers)
            for shared_index in shared_centers:
                s_x, s_y = mapping_dictionary_2[shared_index]
                temp_img1 = np.copy(black_img)
                temp_img2 = np.copy(black_img)
                cv2.circle(temp_img1, (s_x, s_y), unit, 0, -1)
                cv2.circle(temp_img1, mapping_dictionary_2[center_id], unit, 10, -1)
                cv2.circle(temp_img1, (s_x, s_y), unit, 0, -1)

                cv2.circle(temp_img2, (s_x, s_y), unit, 0, -1)
                cv2.circle(temp_img2, mapping_dictionary_2[pcenter_id], unit, 10, -1)
                cv2.circle(temp_img2, (s_x, s_y), unit, 0, -1)
                temp_img = temp_img1 + temp_img2


                temp_img[img == 0] = 0
                temp_img[temp_img == 10] = 0
                temp_img[temp_img == 20] = 255
                temp_img[temp_img>=255] = 255

                temp_img1[img == 0] = 0
                temp_img2[img == 0] = 0
                temp_img1[temp_img1 != 0] = 255
                temp_img2[temp_img2 != 0] = 255
                temp_img1[temp_img1 != 255] = 0
                temp_img2[temp_img2 != 255] = 0

                #temp_img[temp_img == 20] = 0
                temp_img[temp_img != 255] =0
                output = cv2.connectedComponentsWithStats(temp_img, 8, cv2.CV_32S)
                output1 = cv2.connectedComponentsWithStats(temp_img1, 8, cv2.CV_32S)
                output2 = cv2.connectedComponentsWithStats(temp_img2, 8, cv2.CV_32S)
                # print(output)
                number = output[0]
                if number >=3:
                    for component in output[2]:
                        c_x, c_y = mapping_dictionary_2[center_id]
                        p_x, p_y = mapping_dictionary_2[pcenter_id]
                        if (not (component[0] <= c_x <= component[0] + component[2]) or not (
                                component[1] <= c_y <= component[1] + component[3])) and (
                                not (component[0] <= p_x <= component[0] + component[2]) or not (
                                component[1] <= p_y <= component[1] + component[3])):
                            number-=1
                number1 = output1[0]
                number2 = output2[0]

                if number1 >=3:
                    for component in output1[2]:
                        # print(len(component))
                        c_x, c_y = mapping_dictionary_2[center_id]
                        p_x, p_y = mapping_dictionary_2[pcenter_id]
                        # print(mapping_dictionary_2[center_id],mapping_dictionary_2[pcenter_id])
                        # print(component[0], component[0] + component[2], component[1], component[1] + component[3])

                        if (( c_x not in range(component[0], component[0] + component[2])) or (
                                c_y not in range(component[1], component[1] + component[3]))) and (( p_x not in range(component[0], component[0] + component[2])) or (
                                p_y not in range(component[1], component[1] + component[3]))):
                            number1 -= 1
                if number2 >=3:
                    for component in output2[2]:
                        c_x, c_y = mapping_dictionary_2[center_id]
                        p_x, p_y = mapping_dictionary_2[pcenter_id]
                        if (not (component[0] <= c_x <= component[0] + component[2]) or not (
                                component[1] <= c_y <= component[1] + component[3])) and  (not (component[0] <= p_x <= component[0] + component[2]) or not (
                                component[1] <= p_y <= component[1] + component[3])):
                            number2-=1
                # print('number = ',number)
                # count[number] = count.get(output[0], 0)+1
                # if center_id== 0 and pcenter_id ==9:
                #     cv2.imshow("temp_img", temp_img)
                #     cv2.waitKey(0)
                #     cv2.imshow("temp_img", temp_img1)
                #     cv2.waitKey(0)
                #     cv2.imshow("temp_img", temp_img2)
                #     cv2.waitKey(0)

                #
                # cv2.imshow("temp_img", temp_img2)
                # # cv2.imshow("img", img)
                # cv2.waitKey(0)
                # 1 cc
                if number == 1:
                    while pcenter_id in connection[center_id]:
                        connection[center_id].pop(connection[center_id].index(pcenter_id))
                    break
                if number == 2:
                    # print(number1, number2)
                    # print('ininnin')
                    if number1 == 2 and number2 == 2:
                        if pcenter_id not in connection[center_id]:
                            connection[center_id].append(pcenter_id)
                    else:
                        while pcenter_id in connection[center_id]:
                            connection[center_id].pop(connection[center_id].index(pcenter_id))
                        break
                # 2 cc
                elif number == 3:
                    if pcenter_id in  connection[center_id]:
                        connection[center_id].pop(connection[center_id].index(pcenter_id))
                    break
                # no connection
                else:
                    continue
                    print("connected GGG", output[0])
                    # cv2.imshow("test", temp_img)
                    # cv2.imwrite("test_{}.png".format(number), temp_img)
                    # number += 1
                    # cv2.waitKey(0)

                # if center_id == 0 and pcenter_id == 6:
                    # cv2.imshow("test", temp_img)
                    # cv2.waitKey(0)
        # quit()
        # for p_center in possible_centers:
        #     p_index = mapping_dictionary[(p_center[0], p_center[1])]
        #     print(p_center)
        #     print(center)
        #     temp_img = np.copy(img)
        #     # print(p_center, end=" ")
        #     for (x, y), center_id in mapping_dictionary.items():
        #         if center_id == c_index or center_id == p_index:
        #             continue
        #         cv2.circle(temp_img, (y, x), unit, 0, -1)

        #     cv2.imshow("test", temp_img)
        #     cv2.waitKey(0)

            # cv2.line(temp_img, (center[1], center[0]), (p_center[1], p_center[0]), 111, 1)
            # temp_img[img==255] = 255
            # if not np.any(temp_img==111):
            #     p_index = mapping_dictionary[(p_center[0], p_center[1])]
            #     connection[c_index].append(p_index)
            #     # print("o, ", end=" ")

        

        if not connection[center_id]:
            print("GGGGG")
        # print()
    # pprint(count)

    def remove():
        # a = []# [(1, 8), (4, 9)]
        a = []

        for i in range(len(a)):
            while a[i][0] in connection[a[i][1]]:
                connection[a[i][1]].pop(connection[a[i][1]].index(a[i][0]))
            while a[i][1] in connection[a[i][0]]:
                connection[a[i][0]].pop(connection[a[i][0]].index(a[i][1]))

    def add():
        a =[(1, 9)]
        for i in range(len(a)):
            connection[a[i][0]].append(a[i][1])
            connection[a[i][1]].append(a[i][0])

    # print(connection[5], connection[11])
    remove()
    add()

    # print(connection[5], connection[11])
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

    # create_ladder(4, 1, 50, 50)
    # create_ladder(4, 2)
    create_ladder(2, 2, 45, 50)
    # create_tree(1, 1, 65, 48)
    # create_ladder(4, 1)
    # create_window(3, 1)

if __name__ == "__main__":
    main()