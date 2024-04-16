import cv2
import numpy as np
import os

def rotate_image(image, angle):
    """
    Rotate an image by a given angle.
    """
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))


def extract_card(video_path, output_folder, num_frames=100, resize_to=(120, 80)):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_frames):
        # Capture a frame from the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(frame_count))
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection using Canny
        edges = cv2.Canny(blurred, 30, 100)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Adjust the threshold values based on your card dimensions and aspect ratio
            if area > 1000 and 0.9 < aspect_ratio < 1.1:
                # Extract the card region
                card = frame[y:y+h, x:x+w]

                # Resize the extracted card
                card_resized = cv2.resize(card, resize_to)

                # Save the extracted card
                cv2.imwrite(os.path.join(output_folder, f'card_{extracted_count}.png'), card_resized)
                extracted_count += 1

                # Break the loop after extracting a card
                break

    cap.release()

    if extracted_count == 0:
        print("No cards extracted from the video. Adjust edge detection parameters or check video content.")
        return False
    else:
        print(f"{extracted_count} cards extracted successfully.")
        return True


def generate_synthetic_data_with_rotation_flip(input_folder, output_folder, num_images=100, canvas_size=(720, 1280), card_size=(120, 80)):
    card_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    if not card_files:
        print("No card images found in the input folder.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_images):
        # Randomly choose a card from the input folder
        card_file = np.random.choice(card_files)
        card_path = os.path.join(input_folder, card_file)

        # Load the card image
        card = cv2.imread(card_path)

        # Randomly rotate the card
        angle = np.random.randint(0, 360)
        rotated_card = rotate_image(card, angle)

        # Randomly flip the card horizontally
        if np.random.rand() > 0.5:
            rotated_card = cv2.flip(rotated_card, 1)

        # Randomly flip the card vertically
        if np.random.rand() > 0.5:
            rotated_card = cv2.flip(rotated_card, 0)

        # Resize rotated card to match card_size
        rotated_card_resized = cv2.resize(rotated_card, (card_size[1], card_size[0]))

        # Create blank canvas
        canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

        # Randomly position the card on the canvas
        card_x = np.random.randint(0, canvas_size[1] - card_size[1])
        card_y = np.random.randint(0, canvas_size[0] - card_size[0])
        canvas[card_y:card_y+card_size[0], card_x:card_x+card_size[1]] = rotated_card_resized

        # Save the generated image
        cv2.imwrite(os.path.join(output_folder, f'image_{i}.png'), canvas)
        print(f"Generated synthetic image {i+1}/{num_images}")

    print("Synthetic data generation completed.")



# Example usage:
video_path = 'videos/counterspell.mp4'
extracted_cards_folder = 'extracted_cards'
synthetic_data_folder = 'synthetic_data'

extract_card(video_path, extracted_cards_folder,100, resize_to=[120, 80])
# Extract cards from video
# generate_synthetic_data_with_rotation_flip(extracted_cards_folder, synthetic_data_folder, num_images=100)
