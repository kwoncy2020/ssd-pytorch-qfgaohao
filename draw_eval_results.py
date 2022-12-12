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

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


barcode_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_barcode.txt"
df_bar = pd.read_csv(barcode_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
df_bar['x1'] = df_bar['x1'].astype(int)
df_bar['y1'] = df_bar['y1'].astype(int)
df_bar['x2'] = df_bar['x2'].astype(int)
df_bar['y2'] = df_bar['y2'].astype(int)

df_bar['Label'] = df_bar['Prob']
df_bar.loc[:,'Label'] = 'barcode'


dmtx_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_dmtx.txt"
df_dmtx = pd.read_csv(dmtx_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
df_dmtx['x1'] = df_dmtx['x1'].astype(int)
df_dmtx['y1'] = df_dmtx['y1'].astype(int)
df_dmtx['x2'] = df_dmtx['x2'].astype(int)
df_dmtx['y2'] = df_dmtx['y2'].astype(int)

df_dmtx['Label'] = df_dmtx['Prob']
df_dmtx.loc[:,'Label'] = 'dmtx'


mpcode_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_mpcode.txt"
df_mpcode = pd.read_csv(mpcode_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
df_mpcode['x1'] = df_mpcode['x1'].astype(int)
df_mpcode['y1'] = df_mpcode['y1'].astype(int)
df_mpcode['x2'] = df_mpcode['x2'].astype(int)
df_mpcode['y2'] = df_mpcode['y2'].astype(int)

df_mpcode['Label'] = df_mpcode['Prob']
df_mpcode.loc[:,'Label'] = 'mpcode'


pdf417_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_pdf417.txt"
df_pdf417 = pd.read_csv(pdf417_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
df_pdf417['x1'] = df_pdf417['x1'].astype(int)
df_pdf417['y1'] = df_pdf417['y1'].astype(int)
df_pdf417['x2'] = df_pdf417['x2'].astype(int)
df_pdf417['y2'] = df_pdf417['y2'].astype(int)

df_pdf417['Label'] = df_pdf417['Prob']
df_pdf417.loc[:,'Label'] = 'pdf417'


qr_result_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\eval_results\det_test_qrcode.txt"
df_qr = pd.read_csv(qr_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
df_qr['x1'] = df_qr['x1'].astype(int)
df_qr['y1'] = df_qr['y1'].astype(int)
df_qr['x2'] = df_qr['x2'].astype(int)
df_qr['y2'] = df_qr['y2'].astype(int)

df_qr['Label'] = df_qr['Prob']
df_qr.loc[:,'Label'] = 'qrcode'


df_total = pd.concat([df_bar, df_dmtx, df_mpcode, df_pdf417, df_qr],ignore_index=True)
print(df_total.head(5))
# r = pd.read_csv(eval_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
# r['x1'] = r['x1'].astype(int)
# r['y1'] = r['y1'].astype(int)
# r['x2'] = r['x2'].astype(int)
# r['y2'] = r['y2'].astype(int)


for image_id, g in df_total.groupby('ImageID'):
    # image = cv2.imread(os.path.join(image_dir, image_id + ".jpg"))
    image = cv2.imread(image_id)
    head, tail = os.path.split(image_id)
    
    for row in g.itertuples():
        if row.Prob < threshold:
            continue
        cv2.rectangle(image, (row.x1, row.y1), (row.x2, row.y2), (255, 255, 0), 4)
        label = f"{row.Label}-{row.Prob:.2f}"

        cv2.putText(image, label,
                    (row.x1 + 20, row.y1 + 40 ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    # cv2.imwrite(os.path.join(output_dir, image_id), image)
    cv2.imwrite(os.path.join(output_dir, tail), image)
print(f"Task Done. Processed {df_total.shape[0]} bounding boxes.")

