import subprocess
import gdown
import os



# Download DTU demo set
demo_scans = ["scan3"]
url = 'https://drive.google.com/uc?id=1tp_EPBT1SIgtaUn9ycQs39RrKVEUa4M3'
output = 'datasets/DTU_demo.zip'
if os.path.exists(output):
    print("zip file aready exists")
else:
    gdown.download(url, output, quiet=False)
subprocess.call(f"mkdir -p datasets/DTU/Rectified", shell=True)
subprocess.call("unzip datasets/DTU_demo.zip -d datasets", shell=True)
if not os.path.exists("datasets/DTU/Cameras"):
    subprocess.call(f"mv datasets/DTU_demo/Cameras datasets/DTU/Cameras", shell=True)
for scan in demo_scans:
    if os.path.exists(f"datasets/DTU/Rectified/{scan}"):
        print(f"{scan} folder already exists")
    else:
        subprocess.call(f"mv datasets/DTU_demo/Rectified/{scan} datasets/DTU/Rectified/{scan}", shell=True)
        



# Download Tanks & Temples demo set
demo_scans = ["Ignatius", "Meetingroom"]
url = 'https://drive.google.com/uc?id=1PGRuWwgAPYmArYCAPYusqwNY18XY9-Gt'
output = 'datasets/TNT_demo.zip'
if os.path.exists(output):
    print("zip file aready exists")
else:
    gdown.download(url, output, quiet=False)
subprocess.call(f"mkdir -p datasets/TanksAndTemples/training_input", shell=True)
subprocess.call("unzip datasets/TNT_demo.zip -d datasets", shell=True)
for scan in demo_scans:
    if os.path.exists(f"datasets/TanksAndTemples/training_input/{scan}"):
        print(f"{scan} folder already exists")
    else:
        subprocess.call(f"mv datasets/TNT_demo/{scan} datasets/TanksAndTemples/training_input/{scan}", shell=True)
