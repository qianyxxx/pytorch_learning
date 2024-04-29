import os

# root_dir = "dataset/train"
# target_dir = "ants_image"
# img_path = os.listdir(os.path.join(root_dir, target_dir))
# label = target_dir.split('_')[0]
# output_dir = "ants_label"
#
# for i in img_path:
#     filename = i.split('.jpg')[0]
#     with open(os.path.join(root_dir, output_dir,"{}.txt".format(filename)), 'w') as f:
#         f.write(label)

root_dir = "dataset/train"
target_dirs = ["ants_image", "bees_image"]

for target_dir in target_dirs:
    img_path = os.listdir(os.path.join(root_dir, target_dir))
    label = target_dir.split('_')[0]
    output_dir = label + "_label"

    for i in img_path:
        filename = i.split('.jpg')[0]
        with open(os.path.join(root_dir, output_dir,"{}.txt".format(filename)), 'w') as f:
            f.write(label)