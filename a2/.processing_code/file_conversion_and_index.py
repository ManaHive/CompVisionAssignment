from PIL import Image
import os
import shutil

# HOW TO USE:
# Enter name of item into item_name variable
# Run script, rename input_dir as needed
# Images get converted to PNG and indexed
item_name = "mobile_phone"
input_dir = f"../{item_name}/"
output_dir = f"../.processed_images/{item_name}_processed/"
invalid_dir = "invalid/"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create invalid directory if it doesn't exist
if not os.path.exists(invalid_dir):
    os.makedirs(invalid_dir)

# Get all files in the input directory
files = os.listdir(input_dir)

# Iterate through each file in the input directory
index = 262
for file in files:
    new_file_str = item_name + str(index) + ".png"
    # Get the full path of the file
    image_path = os.path.join(input_dir, file)

    # Check if the file is an image
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".webp"):
        # Open the image
        image_path = os.path.join(input_dir, file)
        image = Image.open(image_path)

        # Convert the image format (e.g., from JPEG to PNG) + index the file
        output_path = os.path.join(output_dir, new_file_str)
        image.save(output_path, format="PNG")

        print(f"Converted {file} to PNG")
        index = index + 1
    elif file.endswith(".png"):
        # Copy the PNG file to the output directory with the new name
        output_path = os.path.join(output_dir, new_file_str)
        shutil.copyfile(image_path, output_path)
        print(f"Copied {file} to {new_file_str}")
        index += 1
    else:
        # Move invalid files to the invalid directory
        invalid_path = os.path.join(invalid_dir, file)
        shutil.move(image_path, invalid_path)
        print(f"Invalid file moved to invalid directory: {file}")

print("Conversion complete!")
print(f"Index = {index}")