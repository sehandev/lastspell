import numpy

def calculate_precision_recall(target, predict):
    '''
    < Function >
    "calculate_precision_recall" gets lastspell model's result.
    It calculates the result's precision and recall.

    < Parameter >
    target : array of answer. "predict" will be simillar with "target."
    predict : array of predictions. lastspell model predicts answers and saves the predictions in this.

    < Return >
    precision : float data of precision.
    recall : float data of recall.
    '''
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(target)):
        if predict[i] == 1:
            if target[i] == 1:
                tp += 1
            elif target[i] == 0:
                fp += 1
        elif predict[i] == 0:
            if target[i] == 1:
                fn += 1
            elif target[i] == 0:
                tn += 1
                
    print("tp:" + str(tp))
    print("tn:" + str(tn))
    print("fp:" + str(fp))
    print("fn:" + str(fn))
    
    if (tp + fp) != 0 and (tp + fn) != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("precision : " + str(precision))
        print("recall : " + str(recall))
    else:
        precision = 0
        recall = 0
        print("precision and recall make some ERROR")

    return precision, recall

