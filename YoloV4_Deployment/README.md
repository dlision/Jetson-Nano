# YoloV4 Inference on Jetson Nano

You only look once (YOLO) is a state-of-the-art model that has been used extensively used for object detection in many industrial applications. In this tutorial, we will showcase the use of YoloV4 for object detection on NVIDIA's Jetson Nano through the Triton Inference Server.

Note that this tutorial assumes you have Jetpack 4.6 installed on your Nvidia Jetson Nano device.  You can download the relevant JetPack from the official site [JetPack SDK | NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack#:~:text=NVIDIA%20JetPack%20SDK%20is%20the,to%2Dend%20accelerated%20AI%20applications.&text=It%20also%20includes%20samples%2C%20documentation,analytics%20and%20Isaac%20for%20robotics.)

![car.gif](images/car.gif)

## Docker

Firstly, we need to download the machine learning container for Jetson and JetPack. Specifically, we use the l4t-ml docker image containing the most commonly used ML and data science frameworks.

For installation, run:

```bash
sudo docker pull nvcr.io/nvidia/l4t-ml:r32.6.1-py3
```

## YoloV4

**Acknowledgements:** Next, we make use of this [Yolov4 model for TensorRT](https://github.com/isarsoft/yolov4-triton-tensorrt) which shows how to run this model on the Triton Inference Server for x86 architecture. We will use this as our foundation and showcase running the model and calling inference on Jetson Nano instead.

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

This will generate an executable file **main** and a library file **liblayerplugin.so**. The library file contains all unsupported TensorRT layers whilst the executable helps build an optimized engine. 

## Weights

The file directory structure that we will use to store our model weights and config files is as follows, as presented in this repository as well:

```bash
├───models
│   └───yolo
│       │   config.pbtxt
│       │
│       └───1
│               weights.sh
```

Model weights may be downloaded from the public drive link [here](https://www.google.com/url?q=https://drive.google.com/file/d/1JCbl0x-9PAXvapIPmqnW16pf89xSmgCa/view?usp%3Dsharing&sa=D&source=hangouts&ust=1645030637805000&usg=AOvVaw0tOlRdJFc3Ie8lbCvaJ6WI) and stored in the `\yolo\1` folder.

Alternatively, you may choose to run the shell script file already saved in this folder (and presented in this repository) instead, to download these weights, by running:

```bash
bash weights.sh
```

inside the `\models\yolo\1` folder.

Next, copy the weights inside the docker using the relevant container id. To obtain the running container id, we run the following command, and make sure to copy the id corresponding to our current docker container.

```bash
sudo docker ps
```

Next, we copy the weights into this docker

```bash
sudo docker ps <path to weights> <container id> :/yolov4-triton-tensorrt/yolov4.wts
```

## Deployment

With the required model directory structure in place and model weights stored, we now download the triton server to deploy our model on using this [link](https://github.com/triton-inference-server/server/releases/download/v2.17.0/tritonserver2.17.0-jetpack4.6.tgz) and unzip it. For our use case, we downloaded and extracted the server in the default: `/home/nano/downloads` folder.

 

To set the path and run Triton Server:

```bash
# To explicitly allow the triton server to find this library
export LD_PRELOAD=<Path to libplayerplugin.so>
```

```bash
[path_to_triton_server]/tritonserver2.17.0-jetpack4.6/bin/tritonserver --backend-directory=/tritonserver2.17.0-jetpack4.6/backends/ --model-repository=path to the model
```

## Inference

This repo also contains the `client.py` file taken from the original link, to run a sample inference using our model:

```bash
python client.py -o [output_path/image_result.jpg] image [input_path/image.jpg]
python client.py -o [output_path/video_result.mp4] video [input_path/video.mp4]
```

In our demo, we use:

```bash
python client.py -o ./car_result.mp4 video /data/car.mp4
```

## Acknowledgements

The foundational code is taken from the github repository [here](https://github.com/isarsoft/yolov4-triton-tensorrt) which showcases how to deploy a YOLO model on Triton Server for the x86 architecture. 

We make use of this foundational code to show how to deploy the same model instead on Jetson Nano.
