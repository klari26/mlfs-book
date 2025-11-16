# Rome Air Quality Prediction - PM2.5

This repository contains the source code for a machine learning pipeline that predicts PM2.5 in Rome. The system retrieves data from four distinct sensors across the city, engineers features, trains individual models for each sensor, and runs a daily batch inference pipeline to generate forecasts.


## mlfs-book

O'Reilly book - Building Machine Learning Systems with a feature store: batch, real-time, and LLMs


## Features

* **Historical Data Backfilling:** Loads past air quality records from local CSV files and organizes them into a feature group for analysis
* **Daily Feature Pipeline:** Collects daily data from AQICN (air quality) and Open-Meteo (weather), then engineers new features automatically
* **Sensor-Specific Training:** Trains an individual XGBoost regression model for each of the 4 sensors 
* **Batch Inference:** Runs a daily pipeline to generate 6-day air quality forecasts for every sensor
* **Prediction Monitoring:** Stores predictions in a monitoring feature group and generates hindcast plots to compare predictions against actual values

 **Note on Sensor 3:** To ensure consistency and balance the record count across all sensors, dataset for sensor 3 was augmented with readings from a nearby sensor. 

## Technology Stack

* **Feature Store / Model Registry:** [Hopsworks](https://hopsworks.ai)
* **Model Training:** XGBoost, Scikit-learn
* **Data Manipulation:** Pandas
* **AI Assistant:** Gradio, LangChain, OpenAI
* **Core:** Python


## Notebook Workflow

This project guides the process from data preparation to model deployment and monitoring. The first three notebooks are executed separately for each sensor. This ensures that the unique characteristics of each sensor’s location are captured, while still following the same underlying logic across all sensors.

### 1. `1_air_quality_feature_backfill_SENSORX.ipynb`

* **Purpose:** Backfill historical data
* **Actions:**
    * Reads a local CSV file containing historical PM2.5 data
    * Fetches data for 13 weather features
    * Creates lag features (lag_1, lag_2, lag_3)
    * Connects to Hopsworks and inserts this historical data into the `air_quality_lagged_sensorx` feature group

### 2. `2_air_quality_feature_pipeline_SENSORX.ipynb`

* **Purpose:** Serves as the daily operational pipeline for feature engineering
* **Actions:**
    * Retrieves the latest PM2.5 data for sensor 1 from the AQICN API
    * Fetches the 7-day weather forecast from the Open-Meteo API
    * Engineers features and inserts the new daily data into their respective feature groups in Hopsworks

### 3. `3_air_quality_training_pipeline_SENSORX.ipynb`

* **Purpose:** Trains the prediction model
* **Actions:**
    * Creates a Feature View from Hopsworks
    * Joins air quality and weather data
    * Creates a time-series training/test split
    * Trains an XGBoost Regressor model on the data
    * Evaluates the model (MSE, R2) and saves the trained model to the Hopsworks Model Registry

### 4. `4_air_quality_batch_inference.ipynb`

* **Purpose:** Generates the final predictions for all sensors
* **Actions:**
    * Connects to Hopsworks and downloads the trained XGBoost models for all sensors 
    * Fetches the latest 6-day weather forecast data from the `weather_sensor` feature group
    * Performs auto-regressive inference: It predicts Day 1 and uses that prediction as a feature (lag) to predict Day 2, and so on for all 6 days
    * Saves the resulting forecast and hindcast plots
    * Uploads the predictions to a monitoring feature group (`aq_predictions_lagged_sx`)

### 5. `5_function_calling.ipynb`

* **Purpose:** Launches a user-facing AI assistant
* **Actions:**
    * Loads the models, LLM (Hermes or OpenAI), and feature views
    * Defines functions for the LLM to call
    * Launches a Gradio web interface where a user can ask questions in plain text or by voice to get air quality predictions


## How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/klari26/mlfs-book
    cd mlfs-book
    ```

2.  **Install Dependencies**
    ```bash
    conda create -n aq python==3.10
    conda activate aq
    conda install twofish

    pip install uv invoke python-dotenv
    uv pip install -r requirements.txt

    # Start the editor to run the code 
    jupyter notebook
    ```

3.  **Set Up Hopsworks**
    * Create a free account on [Hopsworks.ai](https://hopsworks.ai)
    * Create a new project
    * Create an API key in Hopsworks
    * Save this key as a *secret* 

4.  **Set Up Sensor Data & Secrets**
    * This project requires the location and aqcin url for each different sensor. This information should be stored as a *secret*. These secrets are created in the Hopsworks project's Secret Store. They should be JSON strings containing info like:
        ```json
        {
          "country": "italy",
          "city": "rome",
          "street": "ViaClelia",
          "aqicn_url": " https://api.waqi.info/feed/A98194/",
          "latitude": 41.886,
          "longitude": 12.536
        }
        ```
    
5.  **Run the Notebooks**
    * **Step 1:** Run `1_air_quality_feature_backfill_SENSORX.ipynb` to create the feature store
    * **Step 2:** Run `2_air_quality_feature_pipeline_SENSORX.ipynb` to ensure the daily pipeline works
    * **Step 3:** Run `3_air_quality_training_pipeline_SENSORX.ipynb` to train the models and save them to the registry
    * **Step 4:** Run `4_air_quality_batch_inference.ipynb` to generate the first batch of predictions
    * **Step 5:** Run `5_function_calling.ipynb` to start the interactive Gradio assistant

**Note:** Step 1-3 must be done for each sensor. Step 2 and 4 need to be scheduled to run daily.

