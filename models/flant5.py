import os
import torch
import tqdm
import string
from .layers import *
from sklearn.metrics import *
from transformers import T5EncoderModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class FlanT5forMisinfo(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(FlanT5forMisinfo, self).__init__()
        self.flan_t5 = T5EncoderModel.from_pretrained('/mySeoSent/mySeoSent_ch/mySeoSent_ch/mt5/damo/nlp_mt5_zero-shot-augment_chinese-base').requires_grad_(False)

        for name, param in self.flan_t5.named_parameters():
            if name.startswith("encoder.layer.11"): #or name.startswith("encoder.layer.10"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.mlp = MLP(emb_dim, mlp_dims, dropout)
        self.attention = MaskAttention(emb_dim)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        t5_feature = self.flan_t5(inputs, attention_mask = masks)[0]
        t5_feature, _ = self.attention(t5_feature, masks)
        output = self.mlp(t5_feature)
        bias_pred = torch.sigmoid(output.squeeze(1))
        return bias_pred


class Trainer():
    def __init__(self, config):
        self.config = config
        
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)

    def train(self, logger=None):
        if(logger):
            logger.info('start training......')
        self.model = FlanT5forMisinfo(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'])
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'], tokenizer_type='flant5')

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'], self.config['batchsize'], shuffle=True, use_endef=False, aug_prob=self.config['aug_prob'], tokenizer_type='flant5')
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save'or epoch == (self.config['epoch']-1):
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_flant5.pkl'))  # Change filename to reflect the model change
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_flant5.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'], tokenizer_type='flant5')
        future_results = self.test(test_future_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_flant5.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
        
        return metrics(label, pred)