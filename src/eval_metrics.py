import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def scores(results, truths):

    test_preds = results.cpu().detach()
    test_truth = truths.cpu().detach()
    
    predictions = []
    for i in range(len(test_preds)):
        x =np.argmax(test_preds[i])
        predictions.append(x)
        
    test_preds = predictions
    
    test_preds_i = np.array(test_preds)
    test_truth_i = np.array(test_truth)
    f1ma = f1_score(test_truth_i, test_preds_i, average='macro')
    f1mi = f1_score(test_truth_i, test_preds_i, average='micro')
    pre_ma = precision_score(test_truth_i, test_preds_i, average='macro')
    pre_mi = precision_score(test_truth_i, test_preds_i, average='micro')
    re_ma = recall_score(test_truth_i, test_preds_i, average='macro')
    re_mi = recall_score(test_truth_i, test_preds_i, average='micro')
    
    print('F1-Score Macro = {:5.3f}'.format(f1ma))
    print('F1-Score Micro = {:5.3f}'.format(f1mi))
    print('-------------------------')
    print('Precision Macro = {:5.3f}'.format(pre_ma))
    print('Precision Micro = {:5.3f}'.format(pre_mi))
    print('-------------------------')
    print('Recall Macro = {:5.3f}'.format(re_ma))
    print('Recall Micro = {:5.3f}'.format(re_mi))
    print('-------------------------')


