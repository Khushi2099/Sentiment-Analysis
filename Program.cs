using System;
using Microsoft.ML;

class Program
{
    static void Main()
    {
        // Step 1: Create ML Context
        var mlContext = new MLContext();

        // Step 2: Load Sample Data
        var data = new[]
        {
            new SentimentData { Text = "I love this product!", Label = true },
            new SentimentData { Text = "This is the worst experience ever.", Label = false },
            new SentimentData { Text = "The service was excellent!", Label = true },
            new SentimentData { Text = "I hate waiting in long lines.", Label = false }
        };

        var trainingData = mlContext.Data.LoadFromEnumerable(data);

        // Step 3: Build Training Pipeline
        var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Text", outputColumnName: "Features")
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

        // Step 4: Train the Model
        var model = pipeline.Fit(trainingData);

        // Step 5: Use the Model for Predictions
        var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        // Step 6: Test the Model
        Console.WriteLine("Enter a sentence to analyze sentiment:");
        string? userInput = Console.ReadLine();
        if (!string.IsNullOrEmpty(userInput))
        {
            var result = predictionEngine.Predict(new SentimentData { Text = userInput });
            Console.WriteLine($"Prediction: {(result.Prediction ? "Positive" : "Negative")}");
        }
        else
        {
            Console.WriteLine("Invalid input. Please enter some text.");
        }

    }
}
