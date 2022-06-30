import pandas as pd


def main():
    metrics = ["accuracy", "precision", "recall", "f1"]

    model_filenames = {
        "model.onnx": "default",
        "model_quantized.onnx": "quantized",
        "model_optimized.onnx": "optimized",
    }
    results = {
        model: {metric: None for metric in metrics}
        for model in model_filenames.values()
    }

    from latency.pilot.distilbert_base_sst import evaluate_quantized

    for model_filename in model_filenames:
        result = evaluate_quantized.main(model_filename)
        for metric in metrics:
            results[model_filenames[model_filename]][metric] = result[metric]

    from latency.pilot.distilbert_base_sst import evaluate_infinity

    infinity_results = evaluate_infinity.main()
    results["infinity"] = {metric: infinity_results[metric] for metric in metrics}

    df = pd.DataFrame.from_dict(results)
    print(df.head())
    df.to_csv("latency/out/distilbert_base_sst.csv", index=True)


if __name__ == "__main__":
    main()
