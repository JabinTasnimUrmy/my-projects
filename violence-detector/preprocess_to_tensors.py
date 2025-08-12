

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def preprocess_dataset(source_base_dir, target_base_dir, sequence_length=16):
    """
    Reads clips of JPG frames from the source directory, converts them to tensors,
    and saves each clip as a single .pt file in the target directory.
    """
    # Basic transform to resize and convert images to tensors.
    # We do not apply random augmentations here.
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to a slightly larger size before cropping in training
        transforms.CenterCrop((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loop over 'train' and 'val' directories
    for split in ['train', 'val']:
        source_split_dir = os.path.join(source_base_dir, split)
        target_split_dir = os.path.join(target_base_dir, split)

        if not os.path.isdir(source_split_dir):
            print(f"Source directory not found: {source_split_dir}")
            continue

        # Loop over 'Fight' and 'NonFight'
        for class_name in ['Fight', 'NonFight']:
            source_class_dir = os.path.join(source_split_dir, class_name)
            target_class_dir = os.path.join(target_split_dir, class_name)

            if not os.path.isdir(source_class_dir):
                print(f"Source directory not found: {source_class_dir}")
                continue

            # Create the target directory if it doesn't exist
            os.makedirs(target_class_dir, exist_ok=True)
            
            print(f"\nProcessing {split}/{class_name}...")
            
            # Loop through video IDs
            for vid_id in tqdm(os.listdir(source_class_dir), desc=f"  {class_name} videos"):
                vid_path = os.path.join(source_class_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue

                # Loop through clips in each video
                for clip_folder in os.listdir(vid_path):
                    clip_path = os.path.join(vid_path, clip_folder)
                    if not os.path.isdir(clip_path):
                        continue

                    frames = sorted([
                        os.path.join(clip_path, f)
                        for f in os.listdir(clip_path)
                        if f.lower().endswith(('.jpg', '.png'))
                    ])

                    if len(frames) >= sequence_length:
                        frame_paths = frames[:sequence_length]
                        
                        clip_tensors = []
                        for fp in frame_paths:
                            img = Image.open(fp).convert('RGB')
                            img_tensor = transform(img)
                            clip_tensors.append(img_tensor)
                        
                        # Stack tensors into a single (T, C, H, W) clip tensor
                        final_clip_tensor = torch.stack(clip_tensors, dim=0)

                        # Save the entire clip as one file
                        target_filename = f"{vid_id}_{clip_folder}.pt"
                        target_filepath = os.path.join(target_class_dir, target_filename)
                        torch.save(final_clip_tensor, target_filepath)

if __name__ == "__main__":
    SOURCE_DIR = "preprocessed_clips"
    TARGET_DIR = "preprocessed_tensors"
    print(f"Starting preprocessing from '{SOURCE_DIR}' to '{TARGET_DIR}'...")
    preprocess_dataset(SOURCE_DIR, TARGET_DIR)
    print("\nPreprocessing complete!")
    print(f"Your new dataset is ready in the '{TARGET_DIR}' folder.")