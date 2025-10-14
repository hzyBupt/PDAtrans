from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse
import random
from collections import OrderedDict
def convert_AotraSeg24(dataset_path,task_id):

    task_name = "AotraSeg24"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(labelstr)
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    image_path = join(dataset_path, "images")
    label_path = join(dataset_path, "masks")

    image_names = os.listdir(image_path)
    train_files = []
    # 划分图像并将它们复制到目标文件夹
    for i in image_names:
        id = "CTA"+i[7:10]
        train_files.append(id)
        shutil.copy(join(image_path,i),    join(imagestr, id +"_0000.mha"))
        shutil.copy(join(label_path,i[:11]+"label.mha"), join(labelstr, id +".mha"))
      



    generate_dataset_json(out_base, {0: "CT"}, 
                          labels={
                            "background": 0,
                            "Zone 0": 1,
                            "Innominate": 2,
                            "Zone 1": 3,
                            "Left Common Carotid": 4,
                            "Zone 2": 5,
                            "Left Subclavian Artery": 6,

                            "Zone 3": 7,
                            "Zone 4": 8,
                            "Zone 5": 9,
                            "Zone 6": 10,
                            "Celiac Artery": 11,
                            "Zone 7": 12,
                            "SMA": 13,
                            "Zone": 8,
                            "Right Renal Artery": 15,
                            "Left Renal Artery": 16,
                            "Zone 9": 17,
                            "Zone 10R": 18,
                            "Zone 10L": 19,
                            "Right Internal Iliac Artery": 20,
                            "Left Internal Iliac Artery": 21,
                            "Zone 11R": 22,
                            "Zone 11L": 23,
                            
                            },
                          num_training_cases=len(train_files), 
                          file_ending='.mha',
                          dataset_name=task_name,
                          reference='https://aortaseg24.grand-challenge.org/',
                        #   overwrite_image_reader_writer='NibabelIOWithReorient',
                          )
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = [i for i in train_files]
    save_json(splits, join(out_base, "splits_final.json"), sort_keys=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/ai/data/data/aortaseg24')
    parser.add_argument('-d', required=False, type=int, default=241, help='nnU-Net Dataset ID, default: 230')
    args = parser.parse_args()
    convert_AotraSeg24(args.dataset_path, args.d)


       