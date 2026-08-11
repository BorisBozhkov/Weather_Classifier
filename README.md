## Classifying weather on the photo

## Goal
Goal (product): Develop a model to classify weather from image

Goal (study): Get a practical experience in Computer Vision, data augmentation, fine-tuning, ensembling, TTA

## Data
Data was provided by Sber company and loaded into Google Cloud. Data contains training folder (labeled) and test folder (not labeled) 

## Process
Data was loaded to local Google Colab folder and unzipped. Then "train" data had been split to train and validation (20% of all data) data. Next was created a transform which contained several data augmentations which suit for our problem (Rotation, HorizontalFlip, ColorJitter, GaussianBlur). Images were resized to 384*384 pixels. Next we created a dataset and dataloader for training and validation (no augmentations) data. During the training we incorporated lr_scheduler. For the problem we used ensemble of 4 different models: efficientnet_b4, efficientnet_v2, ConvNextTiny and Swin_v2_Tiny. Firstly, we froze all weights. Secondly, we unfroze the last three layers of efficientnet_b4, two last layers of efficientnet_v2 and ConvNextTiny and last one layer of Swin_v2_Tiny. And after all we changed the heads of models to provide correct number of outputs (3 for our problem). During several tests, evaluating each model on validation data, we selected the most suitable dropouts and numbers of epochs for models. After training, we have written custom dataset class for test data and created 4 different transforms (base, horizontal mirroring, light ColorJitter, horizontal mirroring+light ColorJitter) for future TTA. After averaging predictions from each model on test data with each transform we submitted our .csv file. Macro-F1 score on test data was 0.93986. We took 15th place (from >75), so our solution was in the top 20% of the competition. All the training and validation were performed on GoogleColab's T4 GPU.

## Stack
Python 3.10+, PyTorch, torchvision, torchinfo, scikit-learn, pandas, numpy, matplotlib, tqdm, Pillow

## Run
pip install -r requirements.txt 
python AIChallenge_Weather.py