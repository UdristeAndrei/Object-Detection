Object Detection using the COCO dataset.

The project's main purpose is to use a pre-trained model to infer a portion of the data from the COCO dataset and train a model on the same dataset, to see how the two of them will compare. 

To run the code the following steps are required:
* Installing the required packages using the requirements.txt file
* Running the process_data.py function, which splits the data into training(70%), validation(20%), and test(10%) datasets. The second function of the module is to transform the annotations from the COCO format into the Yolo format, more specifically from a ".json" format into a ".txt" format.
* The visualize_data.py is an optional function used to see some of the images, and their annotations from the different datasets.
* train.py is used to train the YoloV8m model.
* and to run the inference on either the pre-trained or the trained models the function inference.py is used. To change the model that is used for inference, just change the model that is loaded at the beginning of the file. The results can be found in the results folder, for each of the models that has been tested.
* To run the code, firstly you run the process_data.py module, with the right path to the data. Then an optional step would be to train a model by running the train.py module. The last step would be to run the inference on the test data by running the inference.py module, here you can also select either a pre-trained model or the one you trained in the previous step.


 Interpretation of results

The results can be found in the results folder, where each model that has been trained has its folder. Inside the folders can be found a few images which have the highest IoU value (red bbox represents the IoU area, the blue one represents the labeled bbox, and the green one represents the prediction made by the model).
Besides the images there can be found a statistics.csv file, which contains the main statistics for each class, like accuracy per class, average IoU values, the total number of labels of that class, and the number of labels that have been correctly predicted per class. Those are the global accuracies of the models:

* Pre-trained model: 0.81
* Model trained with random initial weights: 0.01
* Model trained with pre-trained initial weights: 0.28
* Model trained with pre-trained initial weights and dropout rate of 0.3: 0.28

It can be observed that the model with the best accuracy is the pre-trained model, the next best is the model trained with pre-trained weights, and the worst-performing model is the one trained with random initial weights. Those results are expected since the pre-trained model has been trained on the same dataset. One unexpected thing would be that there is no difference between the model trained with pre-trained weight without dropout and the one with dropout. One possible explanation could be that the dropout hasn't been applied as expected, or the saves have been corrupted, and the same data has been saved twice. 

 Improve accuracy

Data Augmentation:
* The first method to improve the accuracy of the model would be to increase the amount of data that is used to train the model. And to improve this step even further we can use techniques from active learning to select the images that would have the biggest impact on our model instead of randomly selecting images.
* The second way to increase the number of images that we have available would be to apply different transformation techniques on the images and their annotations, for example: rotations, sliding, zooming, etc

Model improvement:
* The first method to improve the model that is being used (YoloV8m) would be to start the training with the original pre-trained weights and start training from that point onward. But this method has one downside, and that would be overfitting, which can already be seen in the results of training. This is happening because the weights have been already trained beforehand, and the number of training images is quite limited.
* But to mitigate this problem there are two techniques that could be used, the first one was also implemented in my original training, referring here to the dropout method, to try and create new connections in the Neural Network. The second method that could be used is early stopping, where the training could be stopped before the model starts overfitting.
* Another method to improve the accuracy of the model would be hyperparameter tuning, where different sets of hyperparameters are used to see which one would produce the best results. For this, we could do a random search, grid search, bayesian optimization, or an evolutionary approach.
* The last method to improve the accuracy of the model would be to try and use a different model, for example, RCNN.
