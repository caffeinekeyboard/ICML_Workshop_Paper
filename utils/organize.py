import os
import random
import shutil

def split_dataset(finger_type, noise_level):
    
    SOURCE_DIR = f'/home/caffeinekeyboard/Codex/CVPR_2026_Biometrics_Workshop/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_{noise_level}'
    TRAIN_DIR = f'/home/caffeinekeyboard/Codex/CVPR_2026_Biometrics_Workshop/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_{noise_level}/train'
    TEST_DIR = f'/home/caffeinekeyboard/Codex/CVPR_2026_Biometrics_Workshop/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_{noise_level}/test'
    VAL_DIR = f'/home/caffeinekeyboard/Codex/CVPR_2026_Biometrics_Workshop/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_{noise_level}/val'

    for directory in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        os.makedirs(directory, exist_ok=True)

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')]
    total_files = len(all_files)
    print(f"Found {total_files} .png files in the source directory.")
    
    if total_files < 2100:
        print("Warning: You requested 2100 splits, but there are fewer files than that!")

    random.seed(42) 
    random.shuffle(all_files)
    train_files = all_files[:1500]
    val_files = all_files[1500:2000]
    test_files = all_files[2000:2100]

    def copy_files(file_list, destination_dir, set_name):
        print(f"Copying {len(file_list)} files to {set_name}...")
        
        for file_name in file_list:
            src_path = os.path.join(SOURCE_DIR, file_name)
            dst_path = os.path.join(destination_dir, file_name)
            shutil.move(src_path, dst_path)

    copy_files(train_files, TRAIN_DIR, "Training Set")
    copy_files(test_files, TEST_DIR, "Testing Set")
    copy_files(val_files, VAL_DIR, "Validation Set")
    print("\nDataset successfully split and organized!")

if __name__ == "__main__":
    for finger_type in ['Arch_Only', 'Double_Loop_Only', 'Left_Loop_Only', 'Natural', 'Right_Loop_Only', 'Tented_Arch_Only', 'Whorl_Only']:
        for noise_level in [0, 5, 10, 15, 20]:
            split_dataset(finger_type, noise_level)