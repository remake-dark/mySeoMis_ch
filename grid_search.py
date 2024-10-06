import logging
import os
import json

from models.bigru import Trainer as BiGRUTrainer
from models.bert import Trainer as BertTrainer
from models.eann import Trainer as EANNTrainer
from models.mdfend import Trainer as MDFENDTrainer
from models.flant5 import Trainer as T5Trainer
from models.bigruseomis import Trainer as BiGRU_SEOMISTrainer
from models.bertseomis import Trainer as BERT_SEOMISTrainer
from models.eannseomis import Trainer as EANN_SEOMISTrainer
from models.mdfendseomis import Trainer as MDFEND_SEOMISTrainer
from models.roberta import Trainer as RobertaTrainer
from models.robertasentobl import Trainer as Roberta_SENTOBLTrainer
from models.flant5seomis import Trainer as T5SEOMISTrainer

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump
class Run():
    def __init__(self,
                 config
                 ):
        self.config = config
    

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)  
        
        train_param = {
            'lr': [self.config['lr']] * 1,
        }
        print(train_param)
        param = train_param
        best_param = []
        json_path = './logs/json/' + self.config['model_name'] + str(self.config['aug_prob']) + '.json'
        json_result = []
        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                self.config['lr'] = v
                if self.config['model_name'] == 'eann':
                    trainer = EANNTrainer(self.config)
                elif self.config['model_name'] == 'bigru':
                    trainer = BiGRUTrainer(self.config)
                elif self.config['model_name'] == 'mdfend':
                    trainer = MDFENDTrainer(self.config)
                elif self.config['model_name'] == 'bert':
                    trainer = BertTrainer(self.config)
                elif self.config['model_name'] == 'roberta':
                    trainer = RobertaTrainer(self.config)
                elif self.config['model_name'] == 'flant5':
                    trainer = T5Trainer(self.config)
                elif self.config['model_name'] == 'roberta_sentobl':
                    trainer = Roberta_SENTOBLTrainer(self.config)
                elif 'bigru_seo' in self.config['model_name']:
                    trainer = BiGRU_SEOMISTrainer(self.config)
                elif 'bert_seo' in self.config['model_name']:
                    trainer = BERT_SEOMISTrainer(self.config)
                elif 'eann_seo' in self.config['model_name']:
                    trainer = EANN_SEOMISTrainer(self.config)
                elif 'mdfend_seo' in self.config['model_name']:
                    trainer = MDFEND_SEOMISTrainer(self.config)
                elif 'flant5_seo' in self.config['model_name']:
                    trainer = T5_SEOMISTrainer(self.config)
                    
                metrics, model_path = trainer.train(logger)
                json_result.append(metrics)
                if metrics['metric'] > best_metric['metric']:
                    best_metric['metric'] = metrics['metric']
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('--------------------------------------\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)
