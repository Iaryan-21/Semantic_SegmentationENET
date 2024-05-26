# SEMANTIC SEGMENTATION USING EFFICIENT NET

## EFFICIENT NET:

EfficientNet is a convolutional Neural Network architecture which aims to achieve state of the art accuracy on image classfication and segmentation task while simulatneously being computaionally efficient. It achieves efficiency by using compound scaling method that scales the depth, width and resolution of the network in a balanced way. This approach allows to achieve better performance by balancing model size and computational cost.

## EfficientNet (ENet) Architecture

This README provides an overview of the EfficientNet (ENet) architecture, detailing the layers and output sizes for an example input of 512x512.

### Model Architecture

| Name            | Type           | Output Size      |
|-----------------|----------------|------------------|
| initial         | 16x            | 256x256          |
| bottleneck1.0  | downsampling   | 64x128x128       |
| 4×bottleneck1.x| -              | 64x128x128       |
| bottleneck2.0  | downsampling   | 128x64x64        |
| bottleneck2.1  | -              | 128x64x64        |
| bottleneck2.2  | dilated 2      | 128x64x64        |
| bottleneck2.3  | asymmetric 5   | 128x64x64        |
| bottleneck2.4  | dilated 4      | 128x64x64        |
| bottleneck2.5  | -              | 128x64x64        |
| bottleneck2.6  | dilated 8      | 128x64x64        |
| bottleneck2.7  | asymmetric 5   | 128x64x64        |
| bottleneck2.8  | dilated 16     | 128x64x64        |
| Repeat section 2 without bottleneck2.0 | - | 128x64x64 |
| bottleneck4.0  | upsampling     | 64x128x128       |
| bottleneck4.1  | -              | 64x128x128       |
| bottleneck4.2  | -              | 64x128x128       |
| bottleneck5.0  | upsampling     | 16x256x256       |
| bottleneck5.1  | -              | 16x256x256       |
| fullconv        | -              | Cx512x512        |

## DATASET : CITYSCAPES

This large-scale dataset contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames.
Details and download are available at: [www.cityscapes-dataset.com](www.cityscapes-dataset.com)

More details about the dataset can be found at: [https://github.com/mcordts/cityscapesScripts](https://github.com/mcordts/cityscapesScripts)

## LIBRARIES TO BE INSTALLED

1. [PyTorch](https://pytorch.org/)
2. [Numpy](https://numpy.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [PIL](https://pillow.readthedocs.io/)

## INSTRUCTIONS TO RUN

1. The python file is in the notebook format. One can either download and open it in Google Colab directly to run or can open in the local host.

## License

The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
