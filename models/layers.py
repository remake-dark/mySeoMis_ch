import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class MLP_botnec(nn.Module):
    def __init__(self, input_dim, hidden_dim=192, dropout = 0.05):
        super().__init__()
        # 压缩层，减少到瓶颈维度
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 激活层
        self.activation = nn.ReLU()
        # 扩展层，恢复到原始维度
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.dropout(h)
        h = self.activation(self.fc2(h))
        # 加入残差连接
        return x + h


class MLP_soc(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        self.mlp = torch.nn.Sequential(*layers)
        
        if output_layer:
            self.output_layer = torch.nn.Linear(input_dim, 1)
            self.projection = torch.nn.Linear(self.input_dim, 1)
        else:
            self.output_layer = None

    def forward(self, x):
        identity = x
        x = self.mlp(x)
        if self.output_layer:
            x = self.output_layer(x)
            residual = self.projection(identity) 
            x = x + residual
        return x



class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        # print("inputs: ", inputs.shape)     #(128, 170, 768)
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        # print("scores: ", scores.shape)     #(128, 170)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        # print("scores: ", scores.shape)     #(128, 1, 170)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # print("outputs: ", outputs.shape)   #(128, 768)

        return outputs, scores

class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn



class SinglePassCluster():
    """ Single-Pass Clustering
    """
    def __init__(self, stopWords_path="", my_stopwords=None,
                 max_df=0.5, max_features=1000,
                 simi_threshold=0.5, cluster_name='', res_save_dir='', res_save_path="./cluster_res_ori.json"):
        self.simi_thr = simi_threshold
        self.cluster_center_vec = []
        self.cluster_vec_memory = []
        self.idx_2_text = {}
        self.cluster_2_idx = {}
        self.res_path = res_save_path
        self.cluster_name = cluster_name
        self.res_save_dir = res_save_dir
    
    def load_SBERT_embeddings(self, embedding_path):
        print('load in SBERT embeddings...')
        with open(embedding_path, 'rb') as handle:
            pkl_data = pickle.load(handle)

        for i, val in pkl_data.items():
            print('id:{} embedding shape:{} embedding type:{}'.format(i, val.shape, type(val)))
            break

        np_data = []
        for i, val in pkl_data.items():
            np_data.append(val)
        np_data = np.array(np_data)
        print('np_data shape:{} np_data type:{}'.format(np_data.shape, type(np_data)))

        return np_data

    def cosion_simi(self, vec):
        simi = cosine_similarity(np.array([vec]), np.array(self.cluster_center_vec))
        max_idx = np.argmax(simi, axis=1)[0]
        max_val = simi[0][max_idx]
        return max_val, max_idx

    def single_pass(self, text_path, embedding_type, embedding_path=''):        
        if embedding_type == 'SBERT':
            SBERT_embeddings = self.load_SBERT_embeddings(embedding_path)
            text_embeddings = SBERT_embeddings
        else:
            print('No match embedding type!')
            exit()
    
        # Start loop
        for idx, vec in tqdm(enumerate(text_embeddings)):
            # Init the first cluster
            if not self.cluster_center_vec:
                self.cluster_center_vec.append(vec)
                self.cluster_vec_memory.append([vec])
                self.cluster_2_idx[0] = [idx]
            # Clustering
            else:
                max_simi, max_idx = self.cosion_simi(vec)
                if max_simi >= self.simi_thr:
                    self.cluster_2_idx[max_idx].append(idx)
                    
                    # Update 
                    self.cluster_vec_memory[max_idx].append(vec)
                    self.cluster_center_vec[max_idx] = np.mean(self.cluster_vec_memory[max_idx], axis=0)
                else:
                    self.cluster_center_vec.append(vec)
                    self.cluster_2_idx[len(self.cluster_2_idx)] = [idx]

                    self.cluster_vec_memory.append([vec])

        with open(os.path.join(self.res_save_dir, self.cluster_name+'_ori.json'), "w", encoding="utf-8") as f:
            json.dump(self.cluster_2_idx, f, ensure_ascii=False)

        cluster_view_pool = res_process(self.cluster_2_idx, text_path)
        pd.DataFrame(cluster_view_pool).to_json(os.path.join(self.res_save_dir, self.cluster_name+'_view.json'), indent=2, force_ascii=False, orient='records')
        pd.DataFrame(cluster_res_pool).to_json(os.path.join(self.res_save_dir, self.cluster_name+'_res.json'), indent=2, force_ascii=False, orient='records')


def res_process(cluster_res, online_path):
    """ Process The Results Into Analyzable Format
    """
    online_df = pd.read_json(online_path)
    #cluster_view_pool, cluster_res_pool = [], []
    cluster_view_pool = []
    for cluster_id, cluster_list in tqdm(cluster_res.items()):
        count = len(cluster_list)
        #view_contents, res_contents = [], []
        view_contents = []
        for id in cluster_list:
            cur_data = online_df.iloc[id]
            #id = cur_data['id']
            #m = cur_data['month']
            #s = cur_data['season']
            #y = cur_data['year']
            #y_m = str(cur_data['year']) + '_' + str(cur_data['month'])
            #y_s = str(cur_data['year']) + '_' + str(cur_data['month'])
            #label = cur_data['label']
            content = cur_data['content']
            cluster_label = cluster_id

            #res_contents.append({
                #'id': id,
                #'year': y,
                #'season': s,
                #'month': m,
                #'year-season': y_s,
                #'year-month': y_m,
                #'label': label,
                #'cluster_label': cluster_label,
                #'content': content
            #})
            view_contents.append(content)
        cur_view_cluster = {
            'cluster_id': cluster_id,
            'count': count,
            'contents': view_contents,
        }
        #cur_res_cluster = {
            #'cluster_id': cluster_id,
            #'count': count,
            #'contents': res_contents
        #}
        cluster_view_pool.append(cur_view_cluster)
        #cluster_res_pool.append(cur_res_cluster)

    return cluster_view_pool#, cluster_res_pool