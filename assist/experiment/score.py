from collections import defaultdict
from pathlib import Path


def score(decoded, references):
    '''score the performance

    args:
        decoded: the decoded tasks as a dictionary
        references: the reference tasks as a dictionary
    returns:
        - the scores as a tuple (precision, recal f1)
        - a dictionary with scores per label
    '''

    # The number of true positives
    correct = {}
    # The number of positives = true positives + false positives
    positives = {}
    # The number of references = true positives + false negatives
    labels = {}

    acc_count = 0
    acc = 0        

    # Count the number of correct arguments in the correct tasks
    for uttid, reference in references.items():
        prediction = decoded[uttid]

        # Update positives
        if prediction.name not in positives:
            positives[prediction.name] = [0, {}]
        positives[prediction.name][0] += 1
        for arg_name, arg_value in prediction.args.items():
            if arg_name not in positives[prediction.name][1]:
                positives[prediction.name][1][arg_name] = {}
            if arg_value not in positives[prediction.name][1][arg_name]:
                positives[prediction.name][1][arg_name][arg_value] = 0
            positives[prediction.name][1][arg_name][arg_value] += 1

        #update labels
        if reference.name not in labels:
            labels[reference.name] = [0, {}]
        labels[reference.name][0] += 1
        for arg_name, arg_value in reference.args.items():
            if arg_name not in labels[reference.name][1]:
                labels[reference.name][1][arg_name] = {}
            if arg_value not in labels[reference.name][1][arg_name]:
                labels[reference.name][1][arg_name][arg_value] = 0
            labels[reference.name][1][arg_name][arg_value] += 1

        #update correct correct
        if reference.name not in correct:
            correct[reference.name] = [0, {}]
        if reference.name == prediction.name:
            correct[reference.name][0] += 1
            for arg_name, arg_value in reference.args.items():
                if arg_name not in correct[reference.name][1]:
                    correct[reference.name][1][arg_name] = {}
                if arg_value not in correct[reference.name][1][arg_name]:
                    correct[reference.name][1][arg_name][arg_value] = 0
                if arg_name in prediction.args and arg_value == prediction.args[arg_name]:
                    correct[reference.name][1][arg_name][arg_value] += 1
                    acc_count += 1
            if acc_count == len(reference.args):
                acc += 1
            acc_count = 0

    #collect the scores
    numpositives = 0
    numlabels = 0
    numcorrect = 0
    numitems = 0
    macroprec = 0
    macrorecall = 0
    macrof1 = 0
    scores = {}
    for t in labels:

        if t not in positives:
            positives[t] = [0, {}]

        #udate global scores
        numlabels += labels[t][0]
        numcorrect += correct[t][0]
        numpositives += positives[t][0]

        scores[t] = [comp_score(correct[t][0], labels[t][0], positives[t][0]),
                     {}]
        numitems += 1
        macroprec += scores[t][0][0]
        macrorecall += scores[t][0][1]
        macrof1 += scores[t][0][2]

        for arg_name in labels[t][1]:

            scores[t][1][arg_name] = {}

            if arg_name not in positives[t][1]:
                positives[t][1][arg_name] = {}
            if arg_name not in correct[t][1]:
                correct[t][1][arg_name] = {}

            for val in labels[t][1][arg_name]:
                if val not in positives[t][1][arg_name]:
                    positives[t][1][arg_name][val] = 0
                if val not in correct[t][1][arg_name]:
                    correct[t][1][arg_name][val] = 0

                #update global scores
                numlabels += labels[t][1][arg_name][val]
                numcorrect += correct[t][1][arg_name][val]
                numpositives += positives[t][1][arg_name][val]

                scores[t][1][arg_name][val] = comp_score(
                    correct[t][1][arg_name][val],
                    labels[t][1][arg_name][val],
                    positives[t][1][arg_name][val])

                numitems += 1
                macroprec += scores[t][1][arg_name][val][0]
                macrorecall += scores[t][1][arg_name][val][1]
                macrof1 += scores[t][1][arg_name][val][2]

    s = comp_score(numcorrect, numlabels, numpositives)

    if numitems:
        macroprec /= numitems
        macrorecall /= numitems
        macrof1 /= numitems
        
    print(f"Accuracy: {acc}/{len(references)} = {acc/len(references):.5f}")
    acc /= len(references)

    metrics = {
        "precision": s[0],
        "recal": s[1],
        "f1": s[2],
        "macro precision": macroprec,
        "macro recal": macrorecall,
        "macro f1": macrof1,
        "accuracy": acc
    }

    return metrics, scores

def write_scores(scores, savedir):
    '''write the scores to a readable file'''

    labels = {
        "label_f1": "Label f1 scores",
        "label_recal": "Label recal",
        "label_precision": "Label precision",
        "label_labelcount": "Label reference positive count",
        "label_positives": "Label detected positive count",
        "label_truepositive": "Label true positive count",
    }

    for index, (filename, title) in enumerate(labels.items()):
        with open(Path(savedir, filename), "w") as f:
            f.write(f"{title}\n")
            write_index(scores, index, "%f", f)


def write_index(scores, index, fmt, fid):
    '''write a part of the scores'''

    for t in scores:
        fid.write(('\n%s: ' + fmt) % (t, scores[t][0][index]))
        for arg_name in scores[t][1]:
            fid.write('\n\t%s:' % arg_name)
            for val in scores[t][1][arg_name]:
                fid.write(
                    ('\n\t\t%s: ' + fmt) %
                    (val, scores[t][1][arg_name][val][index]))


def comp_score(correct, labels, positives):
    '''compute scores'''

    if labels:
        recal = correct/labels
    else:
        recal = 1

    if positives:
        precision = correct/positives
    else:
        precision = 1

    if precision + recal:
        f1 = 2*precision*recal/(precision + recal)
    else:
        f1 = 0

    return precision, recal, f1, labels, positives, correct
