import os

#directory = "mushrooms"
directory = "ImageAnomalyDetection"
for root, dirs, files in os.walk(directory):
    print(f"{root}: {len([name for name in os.listdir(root)])}")
 
 