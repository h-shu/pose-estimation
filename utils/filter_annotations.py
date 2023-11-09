import json
import cv2
from collections import Counter

def main():
    # Read in the annotation file.
    path = "annotations/person_keypoints_train2017.json"
    with open(path, 'r') as json_file:
        data = json.load(json_file)

    # Count the number of annotations per image.
    image_annotations = Counter()
    for annotation in data["annotations"]:
        image_annotations[annotation["image_id"]] += 1

    # Store the image ids that only have 1 annotation.
    ids = []
    for image_id in image_annotations:
        if image_annotations[image_id] == 1:
            ids.append(image_id)

    # Get image and their annotation.
    img_ann_dict = {}
    for image in data["images"]:
        if image["id"] in ids:
            img_ann_dict[image["id"]] = { "filename" : image["file_name"] }
    for annotation in data["annotations"]:
        if annotation["image_id"] in ids:
            img_ann_dict[annotation["image_id"]]["keypoints"] = annotation["keypoints"]

    # Verify that image and annotation is correct.
    img_ann_list = list(img_ann_dict.items())
    train_path = "train2017/"
    img_idx = 523
    img = cv2.imread(train_path + img_ann_list[img_idx][1]["filename"], cv2.IMREAD_COLOR)
    for i in range(0, 51, 3):
        x = int(img_ann_list[img_idx][1]["keypoints"][i])
        y = int(img_ann_list[img_idx][1]["keypoints"][i+1])
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)        
    cv2.imshow("Image", img)
    cv2.waitKey(0)

    # Write to filtered annotations.
    file = open("filtered_annotations/single_person_keypoints_train2017.txt", "w")
    for image_id in img_ann_dict:
        file.write(img_ann_dict[image_id]["filename"] + "\n")
        file.write(", ".join([str(x) for x in img_ann_dict[image_id]["keypoints"]]) + "\n")
    file.close()

if __name__ == "__main__":
    main()