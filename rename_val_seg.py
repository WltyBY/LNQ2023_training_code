import os

folder_path = "./val_prevnetv2_rebirth"
os.remove(os.path.join(folder_path, "dataset.json"))
os.remove(os.path.join(folder_path, "plans.json"))
os.remove(os.path.join(folder_path, "predict_from_raw_data_args.json"))
files_name = os.listdir(folder_path)
for file in files_name:
    file_path = os.path.join(folder_path, file)
    os.rename(file_path, os.path.join(folder_path, "lnq2023-val-{}-seg.nrrd".format(file[8:12])))
