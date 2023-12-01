Object Detection using the COCO dataset.

The main purpose of the project is to use a pre-trained model to infer a portion of the data from the COCO dataset and to also train a model on the same dataset, to see how the two of them will compare. 

To run the code the following steps are required:
* Installing the required packages using the requirements.txt file
* Running the process_data.py function, which splits the data into training(70%), validation(20%), and test(10%) datasets. The second function of the module is to transform the annotations from the COCO format into the Yolo format, more specifically from a ".json" format into a ".txt" format.
* The visualize_data.py is an optional function that is used to see some of the images, and their annotations from the different datasets.
* train.py is used to train the YoloV8m model.
* and to run the inference on either the pre-trained or the trained models the function inference.py is used. To change the model that is used for inference, just change the model that is loaded at the beginning of the file. The results can be found in the results folder, for each of the models that has been tested.

