import os

def main():
    # Read in file.
    images = []
    annotations = []
    idx = 0

    for line in open("filtered_annotations/single_person_keypoints_train2017.txt", "r"):
        if idx % 2 == 0:
            images.append(line.strip())
        else:
            annotations.append(line.strip())
        idx += 1

    # Delete images that have more than a single annotation.
    images_set = set(images)
    files = os.listdir("train2017")

    for filename in files:
        if filename not in images_set:
            os.remove("train2017/" + filename)

    # Check file count
    files = os.listdir("train2017")
    print(len(files))


if __name__ == "__main__":
    main()