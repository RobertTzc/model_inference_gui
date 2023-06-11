import cv2
import os
def plot_results(image_dir,txt_dir,out_dir):
    image = cv2.imread(image_dir)
    image_name = os.path.basename(image_dir)
    os.makedirs(out_dir,exist_ok=True)
    with open(txt_dir,'r') as f:
        data = f.readlines()
    for line in data:
        line = line.replace('\n','').split(',')
        n = line[0]
        box = line[2:]+[line[1]]
        cv2.putText(image, n, (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.rectangle(image, (int(box[0]), int(
            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(out_dir,image_name),image)


