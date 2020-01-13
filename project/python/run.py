import matplotlib.pyplot as plt
import sys
import os
from image_to_txt import image_to_txt, txt_to_image
import cv2

if len(sys.argv) < 3:
    print("Please provide a version (--cpu or --gpu) and a structuring element type for cpu.")
    exit(1)

txt_path = "txt/"
out_txt_path = "out_txt/"
out_jpg_path = "out_jpg/"
jpg_path = "jpg/"

if sys.argv[1] == "--cpu":
    exe_name = "morpho_cpu"
    print("Compiling morpho for cpu...")
    os.system("g++ src/morpho_cpu.cpp src/image_processor.cpp -o morpho_cpu")
    bench_files = ["bench/bench_cpu_dilation.csv"]
else:
    exe_name = "morpho_gpu"
    print("Compiling morpho for gpu...")
    os.system("nvcc src/morpho.cpp src/image_processor.cpp src/render.cu -o morpho_gpu")
    bench_files = ["bench/bench_gpu_basic.csv", "bench/bench_gpu_sep.csv", "bench/bench_gpu_sep_shared.csv"] 
    
for f in bench_files:
    print("Emptying " + f + "...")
    open(f, 'w').close()

inputs_with_shapes = []
inputs_jpg = sorted(os.listdir(jpg_path))
    
for f in inputs_jpg:
    filename = os.path.join(txt_path, f.split('.')[0] + ".txt")
    shape = cv2.imread(os.path.join(jpg_path, f), 0).shape
    image_to_txt(os.path.join(jpg_path, f), filename)
    inputs_with_shapes.append((f, shape[0], shape[1]))
    
    
for f, x, y in inputs_with_shapes:
    if sys.argv[1] == "--cpu":
        modes = ["erosion", "dilation"]
    else:
        modes = ["erosion", "dilation", "erosion_sep", "dilation_sep", "erosion_sep_shared", "dilation_sep_shared"]
    for s in modes:
        txt_file = f.split(".")[0] + "_" + s + ".txt"
        in_filename = os.path.join(txt_path, f.split(".")[0] + ".txt")
        out_filename = os.path.join(out_txt_path, txt_file)
        if (sys.argv[1] == "--cpu"):
            t = sys.argv[2][2:]
            exe_command = "./" + exe_name + " " + s + " " + in_filename + " " + str(x) + " " + str(y) + " " + t + " 5 " + out_filename
        else:
            exe_command = "./" + exe_name + " " + s + " " + in_filename + " " + str(x) + " " + str(y) + " 5 " + out_filename
            
        os.system(exe_command)
        txt_to_image(out_filename, os.path.join(out_jpg_path, txt_file.split(".")[0] + ".jpg"))

if "--display" in sys.argv:
    rows = len(inputs_with_shapes)
    results_in = sorted(list(map(lambda x: os.path.join(jpg_path, x), os.listdir(jpg_path))))
    results_out = sorted(list(map(lambda x: os.path.join(out_jpg_path, x), os.listdir(jpg_path))))
    fig, axs = plt.subplots(rows, 3)
    for i in range(rows):
        erosion = results_out[i].split(".")[0] + "_erosion.jpg"
        dilation = results_out[i].split(".")[0] + "_dilation.jpg"
        
        axs[i, 0].imshow(cv2.imread(results_in[i], 0))
        axs[i, 0].set_title('Input ' + results_in[i].split("/")[1])
        axs[i, 0].axis('off')

        axs[i, 1].imshow(cv2.imread(erosion, 0))
        axs[i, 1].set_title('Erosion ' + results_in[i].split("/")[1])
        axs[i, 1].axis('off')

        axs[i, 2].imshow(cv2.imread(dilation, 0))
        axs[i, 2].set_title('Dilation ' + results_in[i].split("/")[1])
        axs[i, 2].axis('off')

    plt.show()
    

