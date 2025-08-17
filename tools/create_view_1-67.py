import os
import cv2
import glob
import shutil
from tqdm import tqdm


def get_all_images_blendedmvs(root_dir):
    # data/BlendedMVS/raw/585a2a71b338a62ad50138dc/blended_images/00000000.jpg
    all_images = []
    all_views = os.listdir(root_dir)
    for view in tqdm(all_views):
        view_path = os.path.join(root_dir, view, 'blended_images')
        if os.path.isdir(view_path):
            imgs = glob.glob(os.path.join(view_path, '*.jpg'))
            if imgs:
                imgs = [im_path for im_path in imgs if os.path.isfile(im_path)]
                imgs = [im_path for im_path in imgs if "masked" not in im_path]
                all_images.extend(imgs)

    return all_images


def crop_and_resize_blendedmvs(impath, ratio=10/6):
    img = cv2.imread(impath)
    if img is None:
        print(f"Warning: Unable to read image {impath}")
        return None

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    new_w = int(w / ratio)
    new_h = int(h / ratio)
    cropped_img = img[center[1] - new_h // 2:center[1] + new_h // 2,
                      center[0] - new_w // 2:center[0] + new_w // 2]
    # Resize to original size
    if cropped_img.size == 0:
        print(f"Warning: Cropped image is empty for {impath}")
        return None
    resized_img = cv2.resize(
        cropped_img, (w, h), interpolation=cv2.INTER_CUBIC)

    return resized_img


def crop_only_blendedmvs(impath, ratio=10/6):
    img = cv2.imread(impath)
    if img is None:
        print(f"Warning: Unable to read image {impath}")
        return None

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    new_w = int(w / ratio)
    new_h = int(h / ratio)
    cropped_img = img[center[1] - new_h // 2:center[1] + new_h // 2,
                      center[0] - new_w // 2:center[0] + new_w // 2]

    return cropped_img


def main():
    blendedmvs_dir = './data/BlendedMVS/raw'
    all_images = get_all_images_blendedmvs(blendedmvs_dir)
    print(f"Total images found: {len(all_images)}")
    print(f"First 100 images: {all_images[:100]}")

    blendedmvs_167_dir = './data_BlendedMVS_1-67'
    os.system(f"rm -rf {blendedmvs_167_dir}")
    for impath in tqdm(all_images[:100]):
        # new_path = impath.replace(blendedmvs_dir, blendedmvs_167_dir)
        # new_dir = os.path.dirname(new_path)
        # os.makedirs(new_dir, exist_ok=True)
        # shutil.copy(impath, new_path)
        resized_img = crop_and_resize_blendedmvs(impath)
        if resized_img is not None:
            new_path = impath.replace(blendedmvs_dir, blendedmvs_167_dir)
            new_dir = os.path.dirname(new_path)
            os.makedirs(new_dir, exist_ok=True)
            cv2.imwrite(new_path, resized_img)

    # print(f"Processed and saved images to {blendedmvs_167_dir}")

    # blendedmvs_167_crop_only_dir = './data_BlendedMVS_1-67_crop_only'
    # for impath in tqdm(all_images[:100]):
    #     cropped_img = crop_only_blendedmvs(impath)
    #     if cropped_img is not None:
    #         new_path = impath.replace(blendedmvs_dir, blendedmvs_167_crop_only_dir)
    #         new_dir = os.path.dirname(new_path)
    #         os.makedirs(new_dir, exist_ok=True)
    #         cv2.imwrite(new_path, cropped_img)

    # print(
    #     f"Processed and saved cropped images to {blendedmvs_167_crop_only_dir}")


if __name__ == "__main__":
    main()
