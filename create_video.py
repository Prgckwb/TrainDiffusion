import argparse
import os
import sys

import cv2
from tqdm import tqdm

OUTPUT_DIR = "videos"
IMAGE_PAR_DIR = "images"


def create_image(i, resolution, video_size):
    img_path = f"{IMAGE_PAR_DIR}/size_{resolution}/sequence"
    row, col = video_size
    w, h = (resolution * row, resolution * col)

    images1 = cv2.imread(f"{img_path}/0_{i}.png")
    for j in range(1, 5):
        img = cv2.imread(f"{img_path}/{j}_{i}.png")
        images1 = cv2.hconcat([images1, img])

    images2 = cv2.imread(f"{img_path}/5_{i}.png")
    for j in range(6, 10):
        img = cv2.imread(f"{img_path}/{j}_{i}.png")
        images2 = cv2.hconcat([images2, img])

    images3 = cv2.imread(f"{img_path}/10_{i}.png")
    for j in range(11, 15):
        img = cv2.imread(f"{img_path}/{j}_{i}.png")
        images3 = cv2.hconcat([images3, img])

    image = cv2.vconcat([images1, images2, images3])

    cv2.putText(image,
                text=f'{i}',
                org=(w // 2, h // 2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=3,
                lineType=cv2.LINE_4)

    return image


def create_video(resolution, video_size: tuple):
    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_name = f"{OUTPUT_DIR}/video_{resolution}.mp4"

    # output file name, encoder, fps, size(fit to image size)
    video = cv2.VideoWriter(video_name, fourcc, 90, video_size)

    if not video.isOpened():
        print("can't be opened")
        sys.exit()

    for i in tqdm(range(1000), desc="Creating video"):
        image = create_image(i, resolution, video_size)
        video.write(image)

    video.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--resolution", type=int, required=True)

    args = parser.parse_args()

    resolution = args.resolution
    print(f"Resolution: {str(resolution)}^2")
    col, row = (3, 5)
    video_size = (resolution * row, resolution * col)

    create_video(resolution, video_size)


if __name__ == '__main__':
    main()
