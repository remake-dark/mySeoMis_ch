import os
import torch
import tqdm
import numpy as np
import string
from .layers import *
from sklearn.metrics import *
from transformers import T5EncoderModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class FlanT5SEOMISModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(FlanT5SEOMISModel, self).__init__()
        self.flan_t5 = T5EncoderModel.from_pretrained('mt5/damo/nlp_mt5_zero-shot-augment_chinese-base').requires_grad_(False)
        for name, param in self.flan_t5.named_parameters():
            if name.startswith("encoder.layer.11"):# or name.startswith("encoder.layer.10") : \
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.first_trainable_layer = 11
        self.last_trainable_layer = 12

        #self.embedding = self.flan_t5.embeddings

        #Some Extra Args for social event percenption
        self.num_expert = 4
        self.kernel_dim = 22
        self.soc_mlp = MLP(2*emb_dim + 2*self.kernel_dim, mlp_dims, dropout)
        #end
        

        self.mlp = MLP(emb_dim, mlp_dims, dropout)
        self.mlp_w_soc = MLP(emb_dim + 2*self.kernel_dim, mlp_dims, dropout)
        self.attention = MaskAttention(emb_dim)
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp)
        #add moe strcture to event effect model.
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
            #expert.append(torch.nn.Sequential(cnn_extractor(feature_kernel, emb_dim), MLP(mlp_input_shape, mlp_dims, dropout)))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.num_expert),
                                      nn.Softmax(dim = 1))
        self.attention_gate = MaskAttention(emb_dim)
    
    
    def forward(self, **kwargs):

        inputs = kwargs['content'] 
        masks = kwargs['content_masks']
        
        #only focus on social event preception     May Be only in use in Training
        soc_feat_sem = kwargs['soc_feat'] # size (batch, 2, 768)
        soc_feat_masks = kwargs['soc_feat_masks'] #size(batch, 2)
        kernel_features = kwargs['kernel_features'] #size (batch, 2*kernel_dim)
        mask_extra_column = torch.ones((masks.shape[0], 2), dtype=torch.long, device=masks.device)
        expanded_masks = torch.cat((masks, mask_extra_column), dim=1)
        
        if self.training == True:
            soc_prob = self.soc_feat_effect_mlp_only(soc_feat_sem, kernel_features)
        else:
            soc_prob = self.soc_feat_effect_mlp_only(soc_feat_sem, kernel_features)
        #social event preception end
        
        t5_feature = self.flan_t5(inputs, attention_mask = masks)[0]
        t5_feature = torch.cat((t5_feature, soc_feat_sem), dim=1)
        #print('将进酒')
        #print(bert_feature.shape)
        #print(soc_feat_sem.shape)
        #print(expanded_masks.shape)
        feature, _ = self.attention(t5_feature, expanded_masks)
        #print(feature.shape)
        feature_1 = torch.cat((feature, kernel_features), dim=1)
        bias_pred = 0.9*self.mlp(feature).squeeze(1) + 0.1 * self.mlp_w_soc(feature_1).squeeze(1)
        #bias_pred = self.mlp_w_soc(feature).squeeze(1)
        #bias_pred = 0.94*self.mlp(feature).squeeze(1) + 0.03*

        
        event = kwargs['event']
        masks = kwargs['event_masks']
        event_feature = self.flan_t5(event, attention_mask = masks)[0]
        #entity_prob = self.entity_net(entity_feature).squeeze(1)
        gate_feature, _ = self.attention_gate(event_feature, masks)
        #forward MoE Micro event bia effect
        shared_event_feature = 0
        #if self.training == True:
            #idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
            #domain_embedding = self.domain_embedder(idxs).squeeze(1)
        #else:
            #batchsize = inputs.size(0)
            #domain_embedding = self.domain_embedder(torch.LongTensor(range(8)).cuda()).squeeze(1).mean(dim = 0, keepdim = True).expand(batchsize, self.emb_dim)

        #gate_input = torch.cat([domain_embedding, gate_input_feature], dim = -1)
        gate_value = self.gate(gate_feature)
        #print(entity_feature.shape)
        if self.training == True:
            for i in range(self.num_expert):
                tmp_event_feature = self.expert[i](event_feature)
                shared_event_feature += (tmp_event_feature * gate_value[:, i].unsqueeze(1))#(tmp_entity_feature * gate_value[:, i].unsqueeze(1))
            event_feature= 0.5 * self.entity_convs(event_feature)+ 0.5 * shared_event_feature
        else:
            gate_inference = torch.ones_like(gate_value) * (1 / self.num_expert)
            
            for i in range(self.num_expert):
                tmp_event_feature = self.expert[i](event_feature)
                shared_event_feature += (tmp_event_feature * gate_value[:, i].unsqueeze(1))
            event_feature= 0.8 * self.entity_convs(event_feature)+ 0.2 * shared_event_feature
        
        event_prob = self.entity_mlp(event_feature).squeeze(1)
        #entity_prob = 0.5 * entity_prob + 0.5 * shared_entity_feature

        
        
        return torch.sigmoid(0.85 * bias_pred + 0.16 * event_prob - 0.01 * soc_prob), torch.sigmoid(event_prob), torch.sigmoid(soc_prob),torch.sigmoid(bias_pred)

    def find_first_trainable_layer(self):
        for i, layer in enumerate(self.flan_t5.encoder.layer):
            for param in layer.parameters():
                if param.requires_grad:
                    return i  
        return None

    def encode_cluster_vec_feat (self, soc_feat, soc_feat_masks):
        soc_feat_sem = soc_feat[:,:, -768:]
        for i in range(self.first_trainable_layer, self.last_trainable_layer):
            layer = self.flan_t5.encoder.layer[i]
            soc_feat_sem = layer(soc_feat_sem, soc_feat_masks)[0]
        return soc_feat_sem

    def soc_feat_effect_mlp_only(self, soc_feat_sem, kernel_features):
        combined_soc_feat = torch.cat([soc_feat_sem[:,0,:], soc_feat_sem[:,1,:] ,kernel_features], dim = 1)
        soc_prob = self.soc_mlp(combined_soc_feat).squeeze(1)
        return soc_prob

class Trainer():
    def __init__(self,
                 config
                 ):
        self.config = config
        
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = FlanT5SEOMISModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'])
        model_weights_for_post_train = torch.load('param_model/flant5/parameter_flant5.pkl')
        self.model.load_state_dict(model_weights_for_post_train, strict=False)
        
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'], tokenizer_type='flant5')

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'], self.config['batchsize'], shuffle=True, use_endef=True, aug_prob=self.config['aug_prob'], tokenizer_type='flant5')
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred, entity_pred, soc_pred, _ = self.model(**batch_data)
                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float()) + 0.01 * loss_fn(soc_pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save' or epoch == (self.config['epoch']-1):
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_t5seomis.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_t5seomis.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'], tokenizer_type='flant5')
        future_results, future_pred = self.test_last(test_future_loader)
        pred_array = np.array(future_pred)
        #path = '/mySeoSent/mySeoSent_ch/pred.npy'  # 替换为你的具体路径
        #np.save(path, pred_array)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_t5seomis.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                _, __, ___, batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
        
        return metrics(label, pred)

    def test_last(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                _, __, ___, batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
        
        return metrics(label, pred), pred