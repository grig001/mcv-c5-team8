import evaluate

def compute_metrics(predictions, references):
    # Load the metrics
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    # Compute each metric
    res_b1 = bleu.compute(predictions=predictions, references=references, max_order=1)
    res_b2 = bleu.compute(predictions=predictions, references=references, max_order=2)
    res_r = rouge.compute(predictions=predictions, references=references)
    res_m = meteor.compute(predictions=predictions, references=references)

    # Prepare results and convert to standard Python floats
    results = {
        'BLEU-1': float(res_b1['bleu']),
        'BLEU-2': float(res_b2['bleu']),
        'ROUGE-L': float(res_r['rougeL']),
        'METEOR': float(res_m['meteor'])
    }

    return results


# Example Usage
if __name__ == "__main__":
    reference = [["A child is running in the campus"]]
    prediction = ["A child campus"]
    metrics = compute_metrics(predictions=prediction, references=reference)
    print(metrics)
