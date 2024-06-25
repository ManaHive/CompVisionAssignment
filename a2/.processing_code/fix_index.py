import os

item_name = "syringe"
input_dir = f"../.processed_images/{item_name}_processed/"

files = os.listdir(input_dir)

index = 1
for file in files:
    file_extension = os.path.splitext(file)[1]
    new_file_str = os.path.join(input_dir, item_name + str(index).zfill(3) + file_extension)
    
    # Construct the full path for the old file
    old_file_path = os.path.join(input_dir, file)
    
    # Rename the file
    os.rename(old_file_path, new_file_str)
    print(f"Renamed {old_file_path} to {new_file_str}")
    
    # Increment the index
    index += 1

print("Renaming complete!")