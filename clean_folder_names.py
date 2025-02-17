import os


# for folder_name in os.listdir("mushrooms"):
#     new_name = folder_name.replace(" ", "-")
#     os.rename(f"mushrooms/{folder_name}", f"mushrooms/{new_name}")
   
    
for folder_name in os.listdir("ImageAnomalyDetection"):
    new_name = folder_name.replace(" ", "-")
    os.rename(f"ImageAnomalyDetection/{folder_name}", f"ImageAnomalyDetection/{new_name}")