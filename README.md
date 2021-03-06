# Implementation of Neural Networks for Nuclei segmentation

Implementation of CNN, U-Net, Mask RCNN and Resnet 101 for Nuclei segmentation

The idea of this project is to spot nuclei to speed up curing process for every disease. The aim is to create an algorithm to automate nucleus detection so that researchers can identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work. I tackled this problem by using the following segmentation models – CNN, U-Net, Mask RCNN and Resnet 101. 

For the first steps of understanding the problem I needed to get a domain knowledge of what cells and nuclei are, understanding what segmentation is and what are the various models used for it. I chose these three models to understand and compare how the three varied from each other and in my experiments, I found U-Net to have the one of the best accuracy of 83% and it could segment the nuclei properly. Mask RCNN was the best performing model and Resnet which is pretrained on the coco dataset was very resource exhaustive and generated only a few test images as there were 20000 training steps.

## Data source: https://www.kaggle.com/c/data-science-bowl-2018/data

The dataset contains many segmented nuclei images. Images were acquired under a variety of conditions and vary in the cell type, magnification and imaging modality. The dataset is split into train and test images. There are 670 train samples and around 4000 test samples. This dataset split shows there are more test samples than train ones. 

## Models

-- CNN, U-Net codes are available in Final_blog.ipynb notebook

-- Final_blog.ipynb notebook also contains architecture explanantion and Exploratory data analysis of nuclei dataset

-- Mask RCNN folder has python files and instructions to run mask rcnn are given in instructions.txt

-- Resnet 101 folder has python files and instructions to run resnet code are given in instructions.txt

-- Results folder has the output from Mask RCNN

-- Research Paper folder has description of various models and visualizations

-- Screenshots folder holds images and screenshots of training and architecture






