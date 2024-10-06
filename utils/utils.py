from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np


class Recorder_auto:

    def __init__(self, early_step, start_count=5):
        self.best_metrics = {'metric': 0}  
        self.latest_metrics = {'metric': 0}  
        self.best_epoch = 0  
        self.current_epoch = 0  
        self.early_stop_limit = early_step  
        self.initial_epoch = start_count  

    def add(self, metrics):
        self.latest_metrics = metrics  
        self.current_epoch += 1  
        print("curent", self.latest_metrics)  
        return self.judge()

    def judge(self):
        
        if max(self.latest_metrics['metric'], self.best_metrics['metric']) == self.latest_metrics['metric'] or self.current_epoch == self.initial_epoch:
            self.best_metrics = self.latest_metrics
            self.best_epoch = self.current_epoch
            self.showfinal()
            return 'save'
        
        self.showfinal()

        if (self.current_epoch - self.best_epoch >= self.early_stop_limit) and (self.current_epoch >= self.initial_epoch):
            return 'esc'
        return 'continue'

    def showfinal(self):
        print("Max", self.best_metrics)  


class Recorder:

    def __init__(self, early_step):
        self.best_performance = {'metric': 0}  
        self.latest_performance = {'metric': 0}  
        self.best_epoch = 0  
        self.current_epoch = 0  
        self.early_stop_limit = early_step  

    def add(self, performance):
        self.latest_performance = performance
        self.current_epoch += 1
        print("curent", self.latest_performance)  
        return self.judge()

    def judge(self):
      
        if max(self.latest_performance['metric'], self.best_performance['metric']) == self.latest_performance['metric']:
            self.best_performance = self.latest_performance
            self.best_epoch = self.current_epoch
            self.showfinal()
            return 'save'

        self.showfinal()

     
        if self.current_epoch - self.best_epoch >= self.early_stop_limit:
            return 'esc'
        return 'continue'

    def showfinal(self):
        print("Max", self.best_performance)  



def metrics(y_true, y_pred):
    results = {}  

    results['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    results['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    
    y_pred_class = (np.sign(np.array(y_pred) - 0.5) + 1) // 2  
    
    results['metric'] = f1_score(y_true, y_pred_class, average='macro')  
    results['f1_neg'], results['f1_pos'] = f1_score(y_true, y_pred_class, average=None) 
    results['recall'] = recall_score(y_true, y_pred_class, average='macro')  
    results['recall_neg'], results['recall_pos'] = recall_score(y_true, y_pred_class, average=None)  
    results['precision'] = precision_score(y_true, y_pred_class, average='macro')  
    results['precision_neg'], results['precision_pos'] = precision_score(y_true, y_pred_class, average=None) 
    results['acc'] = accuracy_score(y_true, y_pred_class)  

    return results



def data2gpu(batch, use_cuda):
    
    return {
        'content': batch[0].cuda() if use_cuda else batch[0],
        'content_masks': batch[1].cuda() if use_cuda else batch[1],
        'event': batch[2].cuda() if use_cuda else batch[2],
        'event_masks': batch[3].cuda() if use_cuda else batch[3],
        'label': batch[4].cuda() if use_cuda else batch[4],
        'domain': batch[5].cuda() if use_cuda else batch[5],
        'kernel_features': batch[-3].cuda() if use_cuda else batch[-3],
        'soc_feat': batch[-2].cuda() if use_cuda else batch[-2],
        'soc_feat_masks': batch[-1].cuda() if use_cuda else batch[-1]
    }


class Averager:

    def __init__(self):
        self.total = 0  
        self.count = 0  

    def add(self, value):
        # 使用更直接的公式替代逐步累积
        self.total += value
        self.count += 1

    def item(self):
        return self.total / self.count if self.count > 0 else 0 
