using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AmazonReviewSentiment
{
    public class AmazonReviewData
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool AmazonReview;

        [LoadColumn(1)]
        public string? AmazonReviewTitle;

        [LoadColumn(2)]
        public string? AmazonReviewText;

       
    }

    public class AmazonReviewPrediction : AmazonReviewData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
