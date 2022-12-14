import sys
import cv2
import pandas as pd
import os

# eval_result_file = sys.argv[1]
# image_dir = sys.argv[2]
# output_dir = sys.argv[3]
# threshold = float(sys.argv[4])

eval_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_qrcode.txt"
output_dir = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\draw_eval"
threshold = 0.5
# eval_result_file = sys.argv[1]
# image_dir = sys.argv[2]
# output_dir = sys.argv[3]
# threshold = float(sys.argv[4])

eval_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_qrcode.txt"
output_dir = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\draw_eval"
threshold = 0.5

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

r = pd.read_csv(eval_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
r['x1'] = r['x1'].astype(int)
r['y1'] = r['y1'].astype(int)
r['x2'] = r['x2'].astype(int)
r['y2'] = r['y2'].astype(int)

for image_id, g in df_total.groupby('ImageID'):
    # image = cv2.imread(os.path.join(image_dir, image_id + ".jpg"))
    image = cv2.imread(image_id)
    head, tail = os.path.split(image_id)
    
    for row in g.itertuples():
        if row.Prob < threshold:
            continue
        cv2.rectangle(image, (row.x1, row.y1), (row.x2, row.y2), (255, 255, 0), 4)
        label = f"{row.Label}-{row.Prob:.2f}"

        label = f"{row.Label}-{row.Prob:.2f}"

        cv2.putText(image, label,
                    (row.x1 + 20, row.y1 + 40 ),
                    (row.x1 + 20, row.y1 + 40 ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    # cv2.imwrite(os.path.join(output_dir, image_id), image)
    cv2.imwrite(os.path.join(output_dir, tail), image)
print(f"Task Done. Processed {df_total.shape[0]} bounding boxes.")

