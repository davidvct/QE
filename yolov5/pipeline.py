import cv2
import detect_custom as det
import argparse
import os
from utils.plots import Annotator, colors

parser = argparse.ArgumentParser()
parser.add_argument('--string_img', type=str, default='../original_img', help='string image and the location')
args = parser.parse_args()


def singulation(img_src=args.string_img, weights='../weights/singulation.pt',
                data='./data/custom_data_singulation.yaml'):
    string_img = cv2.imread(img_src, 0)

    # take the left most units (2.5 panel size)
    cut_width = int(string_img.shape[1] / 28 * 2.5)
    panel_height = int(string_img.shape[0] / 2)

    x_start = 0
    singulated_imgs = []
    count = 0

    while count < 14:
        temp_img = string_img[:, x_start:(x_start + cut_width)]
        temp_im = "./data/temp_singulated_image.jpg"
        cv2.imwrite(temp_im, temp_img)
        singulated_detection = det.run(source=temp_im, weights=weights, data=data)

        singulated_detection = singulated_detection.tolist()

        # take only the first 4 leftmost panels
        singulated_detection = sorted(singulated_detection)[:4]

        if singulated_detection[0][1] > singulated_detection[1][1]:
            singulated_detection[0], singulated_detection[1] = singulated_detection[1], singulated_detection[0]

        if singulated_detection[2][1] > singulated_detection[3][1]:
            singulated_detection[2], singulated_detection[3] = singulated_detection[3], singulated_detection[2]

        for i in range(4):
            x1 = int(singulated_detection[i][0])
            x2 = int(singulated_detection[i][2])
            y1 = int(singulated_detection[i][1])
            y2 = int(singulated_detection[i][3])
            singulated_img = temp_img[y1:y2, x1:x2]
            save_name = "../singulated_img/" + "p" + "{0:0=2d}".format(count * 4 + i + 1) + ".jpg"

            cv2.imwrite(save_name, singulated_img)
            print(f"{save_name}...saved")

        # break

        x_start = x2 + 50

        count += 1


def defect_detect(img_src='../singulated_img/', weights='../weights/defect_detect.pt',
                  data='./data/custom_data_defect.yaml'):
    img_list = []
    for file in os.listdir(img_src):
        if file.endswith(".jpg") and not file.startswith("."):
            img_list.append(file)

    img_list = sorted(img_list)
    defect_panel = []
    k = 0

    for im in img_list:

        singulated_im = cv2.imread(img_src + im, 0)

        width = singulated_im.shape[1] // 2
        height = singulated_im.shape[0] // 2

        fp = "./data/"
        cv2.imwrite(fp + "temp_UL.jpg", singulated_im[0:height, 0:width])
        cv2.imwrite(fp + "temp_UR.jpg", singulated_im[0:height, width:2 * width])
        cv2.imwrite(fp + "temp_BL.jpg", singulated_im[height:2 * height, 0:width])
        cv2.imwrite(fp + "temp_BR.jpg", singulated_im[height:2 * height, width:2 * width])

        temp_ims = ['temp_UL.jpg', 'temp_UR.jpg', 'temp_BL.jpg', 'temp_BR.jpg']

        annotator = Annotator(singulated_im, line_width=3)

        defect_per_panel = []
        blob_count = 0
        crack_count = 0
        for img in temp_ims:
            defect_detection = det.run(source=fp + img, weights=weights, data=data)

            if defect_detection is None:
                defect_per_panel.append(defect_detection)
            else:
                df_list = defect_detection.tolist()

                for j in df_list:

                    if img == 'temp_UR.jpg':
                        j[0], j[2] = j[0] + width, j[2] + width
                    elif img == 'temp_BL.jpg':
                        j[1], j[3] = j[1] + height, j[3] + height
                    elif img == 'temp_BR.jpg':
                        j[0], j[1], j[2], j[3] = j[0] + width, j[1] + height, j[2] + width, j[3] + height

                    if j[-1] == 0:
                        blob_count += 1
                        annotator.box_label(j[:4], 'blob', color=colors(1, True))
                    else:
                        crack_count += 1
                        annotator.box_label(j[:4], 'crack', color=colors(3, True))

        if blob_count or crack_count:
            im0 = annotator.result()
            cv2.imwrite(img_src + im, im0)
            defect_panel.append([im[:3], blob_count, crack_count])

        print(f"{im} checked")

    print("b = blob")
    print("c = crack")
    print("[panel | b | c]")
    for j in defect_panel:
        print(j)

    print(f"Total defective panels (in a string): {len(defect_panel)} out of 56pcs")


singulation()
defect_detect()
