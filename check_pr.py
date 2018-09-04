import numpy

def sehan_precision_recall(target, score):
    # target : [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, ...]
    # score : [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, ...]

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(target)):
        if score[i] == 1:
            if target[i] == 1:
                tp += 1
            elif target[i] == 0:
                fp += 1
        elif score[i] == 0:
            if target[i] == 1:
                fn += 1
            elif target[i] == 0:
                tn += 1
#    print("tp : " + str(tp))
#    print("tn : " + str(tn))
#    print("fp : " + str(fp))
#    print("fn : " + str(fn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("\n\n===========================")
    print("precision : " + str(precision))
    print("recall : " + str(recall))

    return precision, recall

