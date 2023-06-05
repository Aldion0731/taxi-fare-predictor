### Taxi Fare Predictor

This project uses a `Keras` model trained on the NYC Taxi Fare dataset, to predict taxi fares based on various features present in the dataset. Instead of using traditional methods like reading the data into Pandas and passing it to the `Keras` model, this project leverages the power of TensorFlow's `tf.data` API to load data directly from a CSV file.
___

#### Dataset

The NYC Taxi Fare dataset contains information about taxi rides in New York City. It includes features like pickup and drop off locations, passenger count, and hour of day. The dataset is provided in a CSV file format and the various splits used can be seen in the project's `data` directory.

___


#### Solution Approach

The project takes advantage of the `tf.data` API. This is a powerful tool in TensorFlow for building efficient input pipelines. By using `tf.data` to directly load data from the CSV file, we eliminate the need to read the data into Pandas or any other intermediate data manipulation library. This approach offers several advantages:

- Efficiency: 
    - Loading data directly with `tf.data` allows for parallelized and optimized data preprocessing, making the data pipeline more efficient. It minimizes memory usage and enables training on larger datasets without running into memory constraints.

- Scalability: 
    - The `tf.data` API seamlessly integrates with TensorFlow's distributed training capabilities. This means that the taxi fare predictor can be easily scaled to train on distributed systems without major code modifications, enabling efficient training on large clusters or cloud environments.

- Flexibility: 
    - `tf.data`allows us to easily apply a wide range of transformations and preprocessing operations directly within the model pipeline. This allows for easy feature engineering without the need for complex data manipulation steps beforehand. In this project, we take advantage of these capabilities to add preprocessing layers to the model, simplifying the feature engineering process. An overview of this can be seen from the model architecture in the next section.

___

#### Model Architecture

![model-architecture](/outputs/engineered_model.png)

___

#### To use this project:

- Clone the repository

```bash
git@github.com:Aldion0731/taxi-fare-predictor.git
```

- Install Dependencies

```bash
pipenv sync
```

- Navigate to the notebook at `src/notebooks/taxi_fare_prediction.ipynb`, and run the code.

#### Results

- The image below show the performance of the model on the train and validation sets during a training run of 20 epochs

![results](/outputs/results.png)

- From the graph above, we can see that with minimal training and even without hyperparameter tuning (see next section), the model generalizes well to the validation set. This generalization also extends to the test set. A summary of the Root Mean Square Error (RMSE) values for each split is given below.

```
train    5.147889
valid    5.001075
test     4.846545
```
___

### Further Improvements

The results obtained may be significantly improved by hyperparameter tuning and the employment of additional feature engineering and feature selection techniques. 
Since one of the immediate objectives of the project was to demonstrate the effectiveness of feature engineering within the model pipeline, no hyperparameter tuning was done at this stage.