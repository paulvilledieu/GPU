import matplotlib.pyplot as plt
import sys
import os
from image_to_txt import image_to_txt, txt_to_image
import cv2

if len(sys.argv) < 2:
    print("Please provide a version (--cpu or --gpu)")
    exit(1)

txt_path = "txt/"
out_txt_path = "out_txt/"
out_jpg_path = "out_jpg/"
jpg_path = "jpg/"

if sys.argv[1] == "--cpu":
    print("Compiling morpho for cpu...")
    os.system("g++ src/morpho_cpu.cpp src/image_processor.cpp -o morpho_cpu")
    inputs_jpg = os.listdir(jpg_path)
    


    inputs_with_shapes = []
    for f in inputs_jpg:
        filename = os.path.join(txt_path, f.split('.')[0] + ".txt")
        shape = cv2.imread(os.path.join(jpg_path, f), 0).shape
        image_to_txt(os.path.join(jpg_path, f), filename)
        inputs_with_shapes.append((f, shape[0], shape[1]))
    
    
    for f, x, y in inputs_with_shapes:
        for s in ["erosion", "dilation"]:
            txt_file = f.split(".")[0] + "_" + s + ".txt"
            in_filename = os.path.join(txt_path, f.split(".")[0] + ".txt")
            out_filename = os.path.join(out_txt_path, txt_file)
            print("./morpho_cpu " + s + " " + in_filename + " " + str(x) + " " + str(y) + " " + out_filename)
            os.system("./morpho_cpu " + s + " " + in_filename + " " + str(x) + " " + str(y) + " " + out_filename)
            txt_to_image(out_filename, os.path.join(out_jpg_path, txt_file.split(".")[0] + ".jpg"))

if "--display" in sys.argv:
    rows = len(inputs_with_shapes)
    results_in = list(map(lambda x: os.path.join(jpg_path, x), os.listdir(jpg_path)))
    results_out = list(map(lambda x: os.path.join(out_jpg_path, x), os.listdir(jpg_path)))
    fig, axs = plt.subplots(rows, 3)
    for i in range(rows):
        erosion = results_out[i].split(".")[0] + "_erosion.jpg"
        dilation = results_out[i].split(".")[0] + "_dilation.jpg"
        print(erosion)
        axs[i, 0].imshow(cv2.imread(results_in[i], 0))
        axs[i, 1].imshow(cv2.imread(erosion, 0))
        axs[i, 2].imshow(cv2.imread(dilation, 0))

    plt.show()
    

