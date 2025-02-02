Diabetes Prediction API
This project uses machine learning to predict whether a patient has diabetes based on a set of health-related features. It uses a Logistic Regression model and is deployed as a FastAPI service.

Steps to Reproduce
Follow these steps to run the project locally:

1. Install Dependencies
First, ensure you have pip installed on your machine. Create a virtual environment and install all required dependencies by running the following command:

bash

pip install -r requirements.txt
Make sure that the requirements.txt file contains all the necessary libraries like scikit-learn, fastapi, joblib, pandas, and uvicorn.

Example of requirements.txt:
nginx
fastapi
uvicorn
scikit-learn
pandas
joblib
2. Run Data Preprocessing and Model Training
Before starting the FastAPI server, you need to preprocess the data and train the machine learning model.

Run the training script to preprocess the data, train the model, and save the model:

bash
python diabetes_predict.ipynb
This will:

Preprocess the dataset (handling missing values and scaling).
Train the Logistic Regression model.
Save the trained model and scaler as diabetes_model.pkl and scaler.pkl.
3. Start the FastAPI Server
After training the model, start the FastAPI server to expose the prediction API.

Run the following command:


uvicorn app:app --reload
This will start the FastAPI server at http://127.0.0.1:8000.

4. Test the API with Sample Input
Once the server is running, you can test the API by sending a POST request with sample input data. You can use Postman, cURL, or any HTTP client to test it.

Here's an example of the JSON input you can send to the /predict endpoint:

json
{
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 80,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.5,
    "age": 35
}
5. Check Model Predictions and Adjust Accordingly
The model will return a prediction (either 0 or 1), indicating whether the patient is predicted to have diabetes (1) or not (0).

Example response:

json
{
    "prediction": 1
}
You can adjust the input parameters based on your requirements and re-test the predictions.

Troubleshooting
If you encounter errors like missing dependencies, ensure all packages are installed from requirements.txt.
If the FastAPI server is not running, verify that uvicorn is installed and the app:app is pointing to the correct FastAPI instance in your project.
For issues related to model predictions, check the model training script (train_model.py) and verify that the dataset is properly preprocessed.
Conclusion
This setup should allow you to reproduce the diabetes prediction model locally and make predictions using the FastAPI endpoint. Let me know if you have any questions or run into any issues!