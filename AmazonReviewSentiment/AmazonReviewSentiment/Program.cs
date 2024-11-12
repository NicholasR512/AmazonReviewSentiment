using Microsoft.ML;
using Microsoft.ML.Data;
using AmazonReviewSentiment;
using static Microsoft.ML.DataOperationsCatalog;
string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "reviews-again.txt");
// See https://aka.ms/new-console-template for more information
MLContext mlContext = new MLContext();
TrainTestData splitDataView = LoadData(mlContext);
TrainTestData LoadData(MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<AmazonReviewData>(_dataPath, hasHeader: false);
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(AmazonReviewData.AmazonReviewText)).Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model;
}

Evaluate(mlContext, model, splitDataView.TestSet);

void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

UseModelWithSingleItem(mlContext, model);

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<AmazonReviewData, AmazonReviewPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<AmazonReviewData, AmazonReviewPrediction>(model);

    AmazonReviewData sampleStatement = new AmazonReviewData
    {
        AmazonReviewText = "This was a very bad steak"
    };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.AmazonReviewText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

UseModelWithBatchItems(mlContext, model);
void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<AmazonReviewData> sentiments = new[]
{
        new AmazonReviewData
        {
            AmazonReviewText = "This was a horrible meal"
        },
        new AmazonReviewData
        {
            AmazonReviewText = "I love this spaghetti."
        }
    };

    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    IDataView predictions = model.Transform(batchComments);

    // Use model to predict whether comment data is Positive (1) or Negative (0).
    IEnumerable<AmazonReviewPrediction> predictedResults = mlContext.Data.CreateEnumerable<AmazonReviewPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    foreach (AmazonReviewPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.AmazonReviewText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("=============== End of predictions ===============");

}