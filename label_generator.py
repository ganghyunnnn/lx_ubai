import os
import cv2
import numpy as np
from tqdm import tqdm

# 경로 및 클래스별 RGB 값 정의
img_format, txt_format, json_format = 'png', 'txt', 'json'
base_path = 'data/all'
class_rgb = [[255, 0, 0], [0, 0, 255], [127, 255, 191], [238, 154, 0]]
n_class = len(class_rgb)
classes = {'0': 'trapezoid', '1': 'vert_long', '2': 'hori_long', '3': 'square'}

img_path = os.path.join(base_path, img_format)
txt_path = os.path.join(base_path, txt_format)
json_path = os.path.join(base_path, json_format)
files = os.listdir(img_path)

def convert_value(array, class_num):
    """
    :param array: 이미지 배열
    :param class_num: 클래스 수

    클래스별 2차원 배열로 변경
    0번 클래스부터 픽셀 값 1, 1번 클래스 2 ...
    """

    new_array = np.zeros([h, w]).astype(np.uint8)

    for i in range(h):
        for j in range(w):
            for k in range(1, n_class + 1):

                if class_num == k:
                    if all(array[i][j] == [class_rgb[k - 1][0], class_rgb[k - 1][1], class_rgb[k - 1][2]]):
                        new_array[i, j] = k
                    else:
                        new_array[i, j] = 0

    return new_array

def save_txt(path, f_name, n_class, points, norm=True):
    """
    :param path: txt 라벨 저장 경로
    :param f_name: txt 파일명 (확장자 제외)
    :param n_class: 클래스 번호 (0 ~ n-1)
    :param points: 포인트 좌표
    :return:

    YOLO txt format : n_class p1.x p1.y p2.x p2.y ... (\n)
    한 줄에 하나의 객체
    """

    txt_file = os.path.join(path, f_name).replace(img_format, txt_format)
    txt = open(txt_file, 'a')

    for i in range(len(points)):
        point_list = points[i].tolist()
        txt.write(str(n_class) + ' ')
        for j in range(len(point_list)):
            if norm:
                point_list[j][0][0] = point_list[j][0][0] / w
                point_list[j][0][1] = point_list[j][0][1] / h
            else:
                pass

            txt.write(str(point_list[j][0][0]).replace("'", "") + ' ')
            txt.write(str(point_list[j][0][1]).replace("'", "") + ' ')

        txt.write('\n')
    txt.close()
    
for i in tqdm(files, desc='converting files'):
    file = os.path.join(img_path, i)
    img_ = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # open tif
    h, w, c = np.shape(img_)  # img height & weight & channel

    if c >= 4:
        img_ = img_[:, :, 0:3]  # delete 4th band
    else:
        pass
    # img_ = cv2.imread(file, cv2.IMREAD_COLOR)  # open tif
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    label_1 = convert_value(img, 1)
    label_2 = convert_value(img, 2)
    label_3 = convert_value(img, 3)
    label_4 = convert_value(img, 4)


    # 꼭지점 추출
    # options: CHAIN_APPROX_NON, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS
    points_1, _ = cv2.findContours(label_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    points_2, _ = cv2.findContours(label_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    points_3, _ = cv2.findContours(label_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    points_4, _ = cv2.findContours(label_4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)


    txt_file = i.replace(img_format, txt_format)
    save_txt(txt_path, i, 0, points_1)
    save_txt(txt_path, i, 1, points_2)
    save_txt(txt_path, i, 2, points_3)
    save_txt(txt_path, i, 3, points_4)


    """
    # 시각화
    # img = img.copy()
    # points = points_1 + points_2 + points_3 + points_4 + points_5 + points_6
    # cv2.drawContours(img, points, -1, (0, 255, 0), 4)
    # cv2.imshow('draw contours', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 저장
    cv2.imwrite('../{}.jpg'.format(i),
                cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), points, -1, (0, 255, 0), 4))
    """
    
    