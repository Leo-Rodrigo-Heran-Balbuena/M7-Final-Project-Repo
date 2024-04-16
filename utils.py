import os
import cv2 as cv


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

            # Convert in gray color
            gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

            # Noise-reducing and edge-preserving filter
            gray = cv.bilateralFilter(gray, 11, 17, 17)

            # Edge extraction
            edge = cv.Canny(gray, 30, 200)

            # Find contours, obtain bounding box coordinates, and extract ROI
            _, cnts = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv.contourArea)

            for c in cnts[0]:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                region_of_interest = original[y:y + h, x:x + w]

                cv.imwrite(os.path.join(path_to_save, f"{image_number}.jpg"), region_of_interest)
                image_number += 1

    print(f"Saved images to {path_to_save}, {image_number} Images saved")


# test
# generate_images_from_video("videos/brainstorm.mp4", "from_video")
extract_objects('test', 'test/')


#
# def generate_images_from_video(path=str, path_to_save=str):
#     """ Generate images from a video file
#     Parameters:
#     :param path: path to video
#     :param path_to_save: path to save images
#     :return: none
#     """
#     if not path.endswith('.mp4'):
#         print(f"File {path} does not end with .mp4")
#         return  # Return if the file path is not legal
#
#     cam = cv.VideoCapture(path)  # Read the video file
#     frame_number = 1
#
#     file_name = os.path.splitext(os.path.basename(path))[0]  # Gets the name of the video files
#
#     savedir = os.path.join(path_to_save, file_name, 'images')
#
#     if not os.path.exists(os.path.join(path_to_save, file_name, 'images')):
#         savedir = os.path.join(path_to_save, file_name, 'images')
#         os.makedirs(savedir)
#
#     while True:  # Repeat process until video is completed
#         ret, frame = cam.read()
#         if ret:
#
#             cv.imwrite(os.path.join(savedir, str(frame_number) + '.jpg'), frame)
#             frame_number += 1
#         else:
#             break
#
#     print(f"Saved images to {path_to_save}, {frame_number} Images saved")
#
#
# def extract_objects(path=str, path_to_save=str, threshold=200):
#     """ Extracts object from an image based on a generated ROI (Region of Interest)
#     :param path: path to video file
#     :param path_to_save: path to save images
#     :return: none
#     """
#
#     if not os.path.exists(path):
#         print(f"Directory : [{path_to_save}] does not exist")
#         return
#
#     if not os.path.isdir(path_to_save):
#         os.makedirs(path_to_save)  # Create the directory if it does not already exist
#         print(f"Directory : [{path_to_save}] generated")
#
#     savedir = os.path.join(os.path.dirname(path), 'images')
#
#     if not os.path.exists(savedir):
#         os.makedirs(savedir)
#
#     image_number = 1
#
#     for filename in os.listdir(path):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
#             # Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
#             image = cv2.imread(os.path.join(path, filename))
#             original = image.copy()
#
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#             blur = cv2.GaussianBlur(gray, (5, 5), 0)
#             thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#             dilate = cv2.dilate(thresh, kernel, iterations=1)
#
#             # Find contours, obtain bounding box coordinates, and extract ROI
#             cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#             if len(cnts) > 0:
#                 c = cnts[0]
#                 x, y, w, h = cv2.boundingRect(c)
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
#                 region_of_interest = original[y:y + h, x:x + w]
#
#                 cv2.imwrite(os.path.join(savedir, f"{image_number}.jpg"), region_of_interest)
#             else:
#                 print(f"Error in {savedir} {image_number}")
#
#             image_number += 1
#
#     print(f"Saved images to {savedir}, {image_number} Images saved")


# extract_objects('test', 'test')
#
#
# def generate_binary_masks(path=str, path_to_save=str, threshold=100):
#     """ Generates binary masks from a folder of images
#
#     :param path: image folder path.
#     :param path_to_save: path to save the binary masks.
#     :param threshold: threshold for binary mask, the higher, the darker.
#     :return: none
#     """
#     if not os.path.exists(path):
#         print(f"Directory : [{path}] does not exist")
#         return
#
#     if not os.path.exists(path_to_save):
#         print(f"Directory : [{path_to_save}] does not exist")
#         return
#
#     if not os.path.isdir(path_to_save):
#         os.makedirs(path_to_save)  # Create the directory if it does not already exist
#         print(f"Directory : [{path_to_save}] generated")
#
#     image_number = 1
#
#     for filename in os.listdir(path):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
#             # Read the image
#             image = cv.imread(os.path.join(path, filename))
#
#             # Create a blank mask with the same dimensions as the input image
#             binary_mask = np.zeros_like(image)
#
#             # Convert the image to grayscale
#             gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#             # Apply thresholding to obtain binary mask
#             _, binary_mask_gray = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)
#
#             # Apply the binary mask to each channel
#             for i in range(3):  # for each channel (RGB)
#                 binary_mask[:, :, i] = binary_mask_gray
#
#             # Resize the binary mask to match the dimensions of the input image
#             binary_mask_resized = cv.resize(binary_mask, (image.shape[1], image.shape[0]))
#
#             print(f"Starting image size :{image.shape[1], image.shape[0]} New Image size :{binary_mask_resized.shape[1], binary_mask_resized.shape[0]}")
#             # Save the binary mask
#             cv.imwrite(os.path.join(path_to_save, f"{image_number}.jpg"), binary_mask_resized)
#             image_number += 1
#
#     print(f"Saved images to {path_to_save}, {image_number} Images saved")


# generate_binary_masks("data/brainstorm/images/", "data/brainstorm/masks/", 200)
# generate_binary_masks("data/counterspell/images/", "data/counterspell/masks/", 200)
# generate_binary_masks("data/bg_noise/images/", "data/bg_noise/masks/", 250)
# test
# generate_images_from_video("videos/brainstorm.mp4", "from_video")
# extract_objects('unprocessed', 'data/bg_noise/images/')