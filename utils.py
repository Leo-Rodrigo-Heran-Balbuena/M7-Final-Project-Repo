import cv2 as cv
import os

"""Utils for image processing in cv2"""


def generate_images_from_video(path=str, path_to_save=str):
    """ Generate images from a video file
    Parameters:
    :param path: path to video
    :param path_to_save: path to save images
    :return: none
    """
    if not path.endswith('.mp4'):
        print(f"File {path} does not end with .mp4")
        return                                                      # Return if the file path is not legal

    cam = cv.VideoCapture(path)                                     # Read the video file
    frame_number = 1

    file_name = os.path.splitext(os.path.basename(path))[0]         # Gets the name of the video files

    if not os.path.exists(path_to_save):
        print(f"Directory {path_to_save} does not exist")
        return

    path_to_save = os.path.join(path_to_save, file_name)
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)                                    # Create the directory if it does not already exist
        print(f"Directory {path_to_save} generated")

    while True:                                                      # Repeat process until video is completed
        ret, frame = cam.read()
        if ret:
            cv.imwrite(os.path.join(path_to_save, str(frame_number) + '.jpg'), frame)
            frame_number += 1
        else:
            break

    print(f"Saved images to {path_to_save}, {frame_number} Images saved")


def extract_objects(path=str, path_to_save=str):
    """ Extracts object from an image based on a generated ROI (Region of Interest)
    :param path: path to video file
    :param path_to_save: path to save images
    :return: none
    """

    if not os.path.exists(path):
        print(f"Directory : [{path_to_save}] does not exist")
        return

    if not os.path.exists(path_to_save):
        print(f"Directory : [{path_to_save}] does not exist")
        return

    path_to_save = os.path.join(path_to_save, os.path.basename(path.rstrip('/')))

    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)  # Create the directory if it does not already exist
        print(f"Directory : [{path_to_save}] generated")

    image_number = 1

    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
            # Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
            image = cv.imread(os.path.join(path, filename))
            original = image.copy()

            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (5, 5), 0)
            thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
            dilate = cv.dilate(thresh, kernel, iterations=1)

            # Find contours, obtain bounding box coordinates, and extract ROI
            cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                region_of_interest = original[y:y + h, x:x + w]

                cv.imwrite(os.path.join(path_to_save, f"{image_number}.jpg"), region_of_interest)
                image_number += 1

    print(f"Saved images to {path_to_save}, {image_number} Images saved")


# test
# generate_images_from_video("videos/brainstorm.mp4", "from_video")
extract_objects('from_video/brainstorm/', 'extracted_cards/')


