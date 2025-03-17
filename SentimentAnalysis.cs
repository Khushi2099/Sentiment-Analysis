using Microsoft.ML.Data;

public class SentimentData
{
    [LoadColumn(0)]
    public string? Text { get; set; }   // Input text

    [LoadColumn(1)]
    public bool Label { get; set; }    // True = Positive, False = Negative
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}
