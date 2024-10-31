import os



def get_label_map(annotation_dir):
    label_map = {}  # Map class names to integer labels
    i = 0

    # Iterate through all JSON files in the annotation directory
    for file in os.listdir(annotation_dir):
        
        # Map the class name (file name without extension) to an integer
        label_map[os.path.splitext(file)[0]] = i
        i += 1

    return label_map