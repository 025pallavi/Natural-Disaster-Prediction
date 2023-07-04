# Natural Disaster Prediction Model using Deep Learning

The Natural Disaster Prediction Model leverages deep learning and Convolutional Neural Networks (CNNs) to predict and identify natural disasters (Worked specifically for wildfire as of now). Additional data sources such as satellite imagery, weather data are integrated to capture a comprehensive view of the disasters. Our project offers disaster preparedness, and efficient response, addressing critical societal challenges.

## Problem Statement

The problem statement involves developing AI algorithms and models that utilize environmental data to predict and identify potential natural disasters, providing early warnings and assisting in disaster preparedness and response efforts.

## Environment Setup

1. Install Python (version 3.6 or higher) on your system.

2. Clone this repository to your local machine using the following command:
   git clone https://github.com/your-username/natural-disaster-prediction.git

3. Navigate to the project directory:
   
   cd natural-disaster-prediction

4. Install the required Python packages by running:
   
   pip install -r requirements.txt

5. Download the dataset for training and testing the model. The dataset should be organized into three directories: train, valid, and test, each containing separate folders for different classes of natural disasters.

## Running the Code

1. Before running the code, make sure you have set up the dataset as described above.

2. Open the main.py file and adjust the configuration parameters according to your requirements (e.g., image shape, batch size, number of epochs).

3. Execute the following command to start the training process:
   
   python main.py
   
   This will train the model using the provided dataset and save the best model checkpoints.

4. To evaluate the trained model on the test dataset, run the following command:

   python evaluate.py

   This will generate predictions and calculate evaluation metrics (accuracy, AUC) for the test dataset.
   
## Key Features and Functionality

• Technologies Used: Deep learning (CNNs), TensorFlow, Keras, Python.

• New Components: CNN model architecture and data preprocessing.

• Scaling Parameters: Data sizes (number of images) and QPS estimates (prediction speed).

• Rollout Strategy: Data collection, model development, training, evaluation, and deployment/integration.

• Information Security/Privacy Concerns: Protecting data through measures like anonymization, encryption, access controls, and secure transmission.

## Future Enhancements

In the future, we plan to incorporate the following enhancements with different image datasets of hurricanes, floods, and earthquakes to predict and identify various potential natural disasters.

• Integration with real-time satellite data to provide live updates and adjust predictions accordingly.

• Create a user interface with interactive visualizations.
