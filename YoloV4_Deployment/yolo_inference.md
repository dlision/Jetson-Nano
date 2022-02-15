# YoloV4 Inference on Jetson Nano

You only look once (YOLO) is a state-of-the-art model that has been used extensively used for object detection in many industrial applications. In this tutorial, we will showcase the use of YoloV4 for object detection on NVIDIA's Jetson Nano through the Triton Inference Server.



Note that this tutorial assumes you have Jetpack 4.6 installed on your Nvidia Jetson Nano device.  You can download the relevant JetPack from the official site [here]([JetPack SDK | NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack#:~:text=NVIDIA%20JetPack%20SDK%20is%20the,to%2Dend%20accelerated%20AI%20applications.&text=It%20also%20includes%20samples%2C%20documentation,analytics%20and%20Isaac%20for%20robotics.))



## Docker

Firstly, we need to download the machine learning container for Jetson and JetPack. Specifically, we use the l4t-ml docker image containing the most commonly used ML and data science frameworks.



For installation, run:

```bash
sudo docker pull nvcr.io/nvidia/l4t-ml:r32.6.1-py3
```



## YoloV4

**Acknowledgements:** Next, we make use of this [Yolov4 model for TensorRT](https://github.com/isarsoft/yolov4-triton-tensorrt). This repository shows how to run this model on the Triton Inference Server for x86 architecture. We will use this as our foundation and showcase running this model and calling inference on Jetson Nano instead.

```bash
sudo git clone https://github.com/isarsoft/yolov4-triton-tensorrt.git
```



## Installing cmake

The main difference with running this model on the jetson device is the building process it requires. To that end, we must install and use cmake inside of the docker image we pulled earlier.



In the parent directory of the folder cloned above, we execute the following command to run our docker:

```bash
sudo docker run -it  -v $(pwd)/yolov4-triton-tensorrt:/yolov4-triton-tensorrt  nvcr.io/nvidia/l4t-ml:r32.6.1-py3
```

And then install cmake inside this docker:

```bash
wget https://cmake.org/files/v3.21/cmake-3.21.0.tar.gz
tar -xf cmake-3.21.0.tar.gz
 cd cmake-3.21.0
./configure && make install
```



## Compilation/Building

The compilation steps using cmake may be executed as follows:

```bash
cd /yolov4-triton-tensorrt
mkdir build
cd build
cmake ..
make
```

This will generate an executable file **main** and a library file **liblayerplugin.so**. The library file contains all unsupported TensorRT layers whilst the executable will build us an optimized engine. 



## Weights

Next, we download the model weights from the [relevant repository](https://github.com/isarsoft/yolov4-triton-tensorrt). These weights are provided in the public google drive link [here]([YOLOv4 Weights (PUBLIC) – Google Drive](https://drive.google.com/drive/folders/1YUDVgEefnk2HENpGMwq599Yj45i_7-iL)).



Next, copy the weights inside the docker using the relevant container id. To obtain the running container id, we run the following command, and make sure to copy the id corresponding to our current docker container.

```bash
sudo docker ps
```

Next, we copy the weights into this docker

```bash
sudo docker ps <path to weights> <container id> :/yolov4-triton-tensorrt/yolov4.wts
```



## TensorRT engine

Now, we run the following command to generate a serialized TensorRT engine

```bash
./main
```

This generates the `yolov4.engine` file, which, together with our `liblayerplugin.so` file will allow us to deploy to the Triton Inference Server.



## Deployment

We need to create a model repository file structure containing our model config file. To do this, we follow the steps below mentioned:

1. Create the required directory structure and cd into this path
   
   ```bash
   mkdir -p [path-to-cwd]/model/yolov4/1
   cd [path-to-cwd]/model/yolov4/1
   ```

2. Copy the engine file into this newly created folder, and rename it for our use
   
   ```bash
   sudo cp [path-to-basefolder]/yolov4-triton-tensorrt/build/yolov4.engine ./
   sudo mv yolov4.engine model.plan
   ```



Next, we download the triton server to deploy our model on using this [link](https://github.com/triton-inference-server/server/releases/download/v2.17.0/tritonserver2.17.0-jetpack4.6.tgz) and unzip it. For our use case, we downloaded and extracted the server in the default: `/home/nano/downloads` folder.



    3. Set Path and Run the Triton Server

```bash
# To explicitly allow the triton server to find this library
export LD_PRELOAD = <Path to libplayerplugin.so>
```

```bash
[path_to_triton_server]/tritonserver2.17.0-jetpack4.6/bin/tritonserver --backend-directory=/tritonserver2.17.0-jetpack4.6/backends/ --model-repository=path to the model
```



## Inference

This repo also contains the `client.py` file taken from the original link, to run inference using our model:

```bash
python client.py -o [output_path/image_result.jpg] image [input_path/image.jpg]
```



## Acknowledgements

The foundational code is taken from the github repository [here](https://github.com/isarsoft/yolov4-triton-tensorrt). This repo showcases how to deploy a YOLO model on Triton Server for the x86 architecture. 

We make use of this foundational code to show how to deploy the same model instead on Jetson Nano.


