import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

PATH_MAIN = "data"

obj_dict = {1: {'folder': "counterspell", 'longest_min': 300, 'longest_max': 800},
            2: {'folder': "brainstorm", 'longest_min': 300, 'longest_max': 800}}

for k, _ in obj_dict.items():
    folder_name = obj_dict[k]['folder']

    files_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, 'images')))
    files_imgs = [os.path.join(PATH_MAIN, folder_name, 'images', f) for f in files_imgs]

    files_masks = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, 'masks')))
    files_masks = [os.path.join(PATH_MAIN, folder_name, 'masks', f) for f in files_masks]

    obj_dict[k]['images'] = files_imgs
    obj_dict[k]['masks'] = files_masks

files_bg_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, 'bg')))
files_bg_imgs = [os.path.join(PATH_MAIN, 'bg', f) for f in files_bg_imgs]

files_bg_noise_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "images")))
files_bg_noise_imgs = [os.path.join(PATH_MAIN, "bg_noise", "images", f) for f in files_bg_noise_imgs]

files_bg_noise_masks = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "masks")))
files_bg_noise_masks = [os.path.join(PATH_MAIN, "bg_noise", "masks", f) for f in files_bg_noise_masks]


def get_img_and_mask(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask_b = mask[:, :, 0] == 0  # This is boolean mask
    mask = mask_b.astype(np.uint8)  # This is binary mask

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    return img, mask




def resize_img(img, desired_max, desired_min=None):
    h, w = img.shape[0], img.shape[1]

    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))

    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new

    transform_resize = A.Compose([
        A.Sequential([
            A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
        ], p=1)
    ])

    transformed = transform_resize(image=img)
    img_r = transformed["image"]

    return img_r


def resize_transform_obj(img, mask, longest_min, longest_max, transforms=False):
    h, w = mask.shape[0], mask.shape[1]

    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))

    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new

    transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)

    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]

    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]

    return img_t, mask_t


transforms_bg_obj = A.Compose([
    A.RandomRotate90(p=1),
    A.ColorJitter(brightness=0.3,
                  contrast=0.3,
                  saturation=0.3,
                  hue=0.07,
                  always_apply=False,
                  p=1),
    A.Blur(blur_limit=(3, 15),
           always_apply=False,
           p=0.5)
])

transforms_obj = A.Compose([
    A.RandomRotate90(p=1),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                               contrast_limit=0.1,
                               brightness_by_max=True,
                               always_apply=False,
                               p=1)
])


def add_obj(img_comp, mask_comp, img, mask, x, y, idx):
    '''
    img_comp - composition of objects
    mask_comp - composition of objects` masks
    img - image of object
    mask - binary mask of object
    x, y - coordinates where center of img is placed
    Function returns img_comp in CV2 RGB format + mask_comp
    '''
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]

    h, w = img.shape[0], img.shape[1]

    x = x - int(w / 2)
    y = y - int(h / 2)

    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

    if x >= 0 and y >= 0:

        h_part = h - max(0,
                         y + h - h_comp)  # h_part - part of the image which gets into the frame of img_comp along y-axis
        w_part = w - max(0,
                         x + w - w_comp)  # w_part - part of the image which gets into the frame of img_comp along x-axis

        img_comp[y:y + h_part, x:x + w_part, :] = img_comp[y:y + h_part, x:x + w_part, :] * ~mask_rgb_b[0:h_part,
                                                                                             0:w_part, :] + (
                                                                                                                    img * mask_rgb_b)[
                                                                                                            0:h_part,
                                                                                                            0:w_part, :]
        mask_comp[y:y + h_part, x:x + w_part] = mask_comp[y:y + h_part, x:x + w_part] * ~mask_b[0:h_part, 0:w_part] + (
                                                                                                                              idx * mask_b)[
                                                                                                                      0:h_part,
                                                                                                                      0:w_part]
        mask_added = mask[0:h_part, 0:w_part]

    elif x < 0 and y < 0:

        h_part = h + y
        w_part = w + x

        img_comp[0:0 + h_part, 0:0 + w_part, :] = img_comp[0:0 + h_part, 0:0 + w_part, :] * ~mask_rgb_b[h - h_part:h,
                                                                                             w - w_part:w, :] + (
                                                                                                                        img * mask_rgb_b)[
                                                                                                                h - h_part:h,
                                                                                                                w - w_part:w,
                                                                                                                :]
        mask_comp[0:0 + h_part, 0:0 + w_part] = mask_comp[0:0 + h_part, 0:0 + w_part] * ~mask_b[h - h_part:h,
                                                                                         w - w_part:w] + (idx * mask_b)[
                                                                                                         h - h_part:h,
                                                                                                         w - w_part:w]
        mask_added = mask[h - h_part:h, w - w_part:w]

    elif x < 0 and y >= 0:

        h_part = h - max(0, y + h - h_comp)
        w_part = w + x

        img_comp[y:y + h_part, 0:0 + w_part, :] = img_comp[y:y + h_part, 0:0 + w_part, :] * ~mask_rgb_b[0:h_part,
                                                                                             w - w_part:w, :] + (
                                                                                                                        img * mask_rgb_b)[
                                                                                                                0:h_part,
                                                                                                                w - w_part:w,
                                                                                                                :]
        mask_comp[y:y + h_part, 0:0 + w_part] = mask_comp[y:y + h_part, 0:0 + w_part] * ~mask_b[0:h_part,
                                                                                         w - w_part:w] + (idx * mask_b)[
                                                                                                         0:h_part,
                                                                                                         w - w_part:w]
        mask_added = mask[0:h_part, w - w_part:w]

    elif x >= 0 and y < 0:

        h_part = h + y
        w_part = w - max(0, x + w - w_comp)

        img_comp[0:0 + h_part, x:x + w_part, :] = img_comp[0:0 + h_part, x:x + w_part, :] * ~mask_rgb_b[h - h_part:h,
                                                                                             0:w_part, :] + (
                                                                                                                    img * mask_rgb_b)[
                                                                                                            h - h_part:h,
                                                                                                            0:w_part, :]
        mask_comp[0:0 + h_part, x:x + w_part] = mask_comp[0:0 + h_part, x:x + w_part] * ~mask_b[h - h_part:h,
                                                                                         0:w_part] + (idx * mask_b)[
                                                                                                     h - h_part:h,
                                                                                                     0:w_part]
        mask_added = mask[h - h_part:h, 0:w_part]

    return img_comp, mask_comp, mask_added


def create_bg_with_noise(files_bg_imgs,
                         files_bg_noise_imgs,
                         files_bg_noise_masks,
                         bg_max=1920,
                         bg_min=1080,
                         max_objs_to_add=60,
                         longest_bg_noise_max=1000,
                         longest_bg_noise_min=200,
                         blank_bg=False):
    if blank_bg:
        img_comp_bg = np.ones((bg_min, bg_max, 3), dtype=np.uint8) * 255
        mask_comp_bg = np.zeros((bg_min, bg_max), dtype=np.uint8)
    else:
        idx = np.random.randint(len(files_bg_imgs))
        img_bg = cv2.imread(files_bg_imgs[idx])
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        img_comp_bg = resize_img(img_bg, bg_max, bg_min)
        mask_comp_bg = np.zeros((img_comp_bg.shape[0], img_comp_bg.shape[1]), dtype=np.uint8)

    for i in range(1, np.random.randint(max_objs_to_add) + 2):
        idx = np.random.randint(len(files_bg_noise_imgs))
        img, mask = get_img_and_mask(files_bg_noise_imgs[idx], files_bg_noise_masks[idx])
        x, y = np.random.randint(img_comp_bg.shape[1]), np.random.randint(img_comp_bg.shape[0])
        img_t, mask_t = resize_transform_obj(img, mask, longest_bg_noise_min, longest_bg_noise_max,
                                             transforms=transforms_bg_obj)
        img_comp_bg, _, _ = add_obj(img_comp_bg, mask_comp_bg, img_t, mask_t, x, y, i)

    return img_comp_bg


def check_areas(mask_comp, obj_areas, overlap_degree=0.3):
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
    masks = mask_comp == obj_ids[:, None, None]

    ok = True

    if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
        ok = False
        return ok

    for idx, mask in enumerate(masks):
        if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
            ok = False
            break

    return ok


def create_composition(img_comp_bg,
                       max_objs=15,
                       overlap_degree=0.2,
                       max_attempts_per_obj=10):
    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h, w), dtype=np.uint8)

    obj_areas = []
    labels_comp = []
    num_objs = np.random.randint(max_objs) + 2

    i = 1

    for _ in range(1, num_objs):

        obj_idx = np.random.randint(len(obj_dict)) + 1

        for _ in range(max_attempts_per_obj):

            imgs_number = len(obj_dict[obj_idx]['images'])
            idx = np.random.randint(imgs_number)
            img_path = obj_dict[obj_idx]['images'][idx]
            mask_path = obj_dict[obj_idx]['masks'][idx]
            img, mask = get_img_and_mask(img_path, mask_path)

            x, y = np.random.randint(w), np.random.randint(h)
            longest_min = obj_dict[obj_idx]['longest_min']
            longest_max = obj_dict[obj_idx]['longest_max']
            img, mask = resize_transform_obj(img,
                                             mask,
                                             longest_min,
                                             longest_max,
                                             transforms=transforms_obj)

            if i == 1:
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                obj_areas.append(np.count_nonzero(mask_added))
                labels_comp.append(obj_idx)
                i += 1
                break
            else:
                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                ok = check_areas(mask_comp, obj_areas, overlap_degree)
                if ok:
                    obj_areas.append(np.count_nonzero(mask_added))
                    labels_comp.append(obj_idx)
                    i += 1
                    break
                else:
                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()

    return img_comp, mask_comp, labels_comp, obj_areas


img_comp_bg = create_bg_with_noise(files_bg_imgs,
                                   files_bg_noise_imgs,
                                   files_bg_noise_masks,
                                   max_objs_to_add=20)

img_comp, mask_comp, labels_comp, obj_areas = create_composition(img_comp_bg,
                                                                 max_objs=15,
                                                                 overlap_degree=0.2,
                                                                 max_attempts_per_obj=10)
plt.figure(figsize=(40,40))
plt.imshow(mask_comp)
plt.show()

cv2.waitKey()