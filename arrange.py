import json
import os
import glob


# Ensure the REAL and FAKE directories exist
real_dir = r"REAL"
fake_dir = r"FAKE"

if not os.path.exists(real_dir):
    os.makedirs(real_dir)

if not os.path.exists(fake_dir):
    os.makedirs(fake_dir)

t = 0

real = []
fake = []
l =  glob.glob("train_sample_videos/*.json")
for i in l:
    with open(i, "r") as f:
        x = json.load(f)

    for file in x:
        if x[file]["label"] == "REAL":
            real.append(file)
        else:
            fake.append(file)

print("Real: ", real)
print("Fake: ", fake)

with open("REAL.txt", "w") as f:
    for i in real:
        f.write(i + "\n")

with open("FAKE.txt", "w") as f:
    for i in fake:
        f.write(i + "\n")   

# for file in x:
#     try:
#         if x[file]["label"] == "REAL":
#             os.rename(
#                 f"C:\\Users\\vaibh\\OneDrive\\Desktop\\deepfake_project\\train_sample_videos\\{file}",
#                 f"{real_dir}\\{file}",
#             )
#     except Exception as e:
#         print(f"Error moving REAL video {file}: {e}")

#     try:
#         if x[file]["label"] == "FAKE":
#             os.rename(
#                 f"C:\\Users\\vaibh\\OneDrive\\Desktop\\deepfake_project\\train_sample_videos\\{file}",  # Corrected path
#                 f"{fake_dir}\\{file}",
#             )
#     except Exception as e:
#         print(f"Error moving FAKE video {file}: {e}")
