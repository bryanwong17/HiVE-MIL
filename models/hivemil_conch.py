import json
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dataset_model import get_class_names
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
_tokenizer = _Tokenizer()

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        # Get dtype from one of the model's parameters
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        out = OrderedDict()
        for i, (k,v) in enumerate(prompts.items()):
            x = v + self.positional_embedding.type(self.dtype)
            x  = x.permute(1, 0, 2)  
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  
            x = self.ln_final(x)
            
            x = x[torch.arange(x.shape[0]), tokenized_prompts[k].argmax(dim=-1)]
            x = x @ self.text_projection  
            out[k] = x
        return out

class PromptLearner(nn.Module):
    def __init__(self, class_names, conch_model, llm_descriptions,
                 n_ctx=16, class_specific_token=False, class_token_position='end'):
        super().__init__()
        n_cls = len(class_names)
        n_ctx = n_ctx
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]

        # Get the tokenizer
        self.tokenizer = get_tokenizer()
        
        if class_specific_token:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors) 

        tokenized_prompts = OrderedDict()
        embedding = OrderedDict()

        for i, (k, v) in enumerate(llm_descriptions.items()):
            prompts = [f"{prompt_prefix} {term} : {explanation}" for term, explanation in v.items()]
            tokenized_prompts[k] = tokenize(self.tokenizer, prompts)
            with torch.no_grad():
                embedding[k] = conch_model.text.token_embedding(tokenized_prompts[k]).type(dtype)
       
        self.token_prefix = OrderedDict()
        self.token_suffix = OrderedDict()
        for i, (k, v) in enumerate(embedding.items()):
            self.token_prefix[k] = embedding[k][:, :1, :]
            self.token_suffix[k] = embedding[k][:, 1+n_ctx:, :]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # Use the tokenizer's encode method for getting lengths
        self.name_lens = [len(self.tokenizer.encode(name, 
                                                   max_length=127,
                                                   truncation=True)) 
                         for name in class_names]
        self.class_token_position = class_token_position

    def forward(self,device):
        ctx = self.ctx

        prompts = OrderedDict()
        for index, (k, v) in enumerate(self.tokenized_prompts.items()):
            prefix = self.token_prefix[k].to(device)
            suffix = self.token_suffix[k].to(device)

            if ctx.dim() == 2:
                ctx_k = ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1)  
            else:
                ctx_k = ctx[index].unsqueeze(0).expand(prefix.shape[0], -1, -1)  

            if self.class_token_position == 'end':
                prompts[k] = torch.cat(
                    [
                        prefix,  
                        ctx_k,     
                        suffix,  
                    ],
                    dim=1,
                )
            elif self.class_token_position == 'middle':
                single_desc_prompts = []
                half_n_ctx = self.n_ctx // 2
                for i in range(len(self.name_lens[k])):  
                    name_len = self.name_lens[k][i]
                    prefix_i = prefix[i:i+1,:,:]
                    descriptor_i = suffix[i:i+1, :name_len, :]
                    suffix_wo_desc_i = suffix[i:i+1, name_len:, :]
                    ctx_half_i_1 = ctx_k[i:i+1, :half_n_ctx, :]
                    ctx_half_i_2 = ctx_k[i:i+1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,     
                            ctx_half_i_1,  
                            descriptor_i,      
                            ctx_half_i_2,  
                            suffix_wo_desc_i,     
                        ],
                        dim=1,
                    )
                    single_desc_prompts.append(prompt)
                prompts[k] = torch.cat(single_desc_prompts, dim=0)

            elif self.class_token_position == 'front':
                single_desc_prompts = []
                for i in range(len(self.name_lens[k])):
                    name_len = self.name_lens[k][i]
                    prefix_i = prefix[i : i + 1, :, :]
                    descriptor_i = suffix[i : i + 1, :name_len, :]
                    suffix_wo_desc_i = suffix[i : i + 1, name_len:, :]
                    ctx_k_i = ctx_k[i : i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  
                            descriptor_i,   
                            ctx_k_i,     
                            suffix_wo_desc_i,  
                        ],
                        dim=1,
                    )
                    single_desc_prompts.append(prompt)
                prompts[k] = torch.cat(prompts, dim=0)

            else:
                raise ValueError

        return prompts  


class HierarchicalAggregator(nn.Module):
    def __init__(self, dim, hier_edge_types=None, num_heads=2):
        super().__init__()
        self.dim = dim
        self.scale = dim ** (-0.5)

        # Modality-Scale Attention (MHA EDGE)
        self.scale_embeddings = nn.Embedding(2, dim)
        self.mha = nn.MultiheadAttention(dim, num_heads)
        self.q_projs = nn.ModuleDict({et: nn.Linear(dim, dim) for et in hier_edge_types})
        self.k_projs = nn.ModuleDict({et: nn.Linear(dim, dim) for et in hier_edge_types})

    def forward(self, query, key, edge_type=None, query_scale=None, key_scale=None):
       
        q = self.q_projs[edge_type](query + self.scale_embeddings(query_scale))
        k = self.k_projs[edge_type](key + self.scale_embeddings(key_scale))
        out, _ = self.mha(q,k,k)

        return query + out

class HierHeteroGraphLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        edge_types = [
            ('x5text', 'self', 'x5text'),
            ('x20text', 'self', 'x20text'),
            ('x5image', 'self', 'x5image'),
            ('x20image', 'self', 'x20image'),
            ('x20image', 'x20_intra', 'x20text'),
            ('x20text', 'x20_intra', 'x20image'),
            ('x5image', 'x5_intra', 'x5text'),
            ('x5text', 'x5_intra', 'x5image')
        ]
        
        # Intra-scale aggregator
        self.convs = HeteroConv({
            (src, rel, dst): SAGEConv(in_size, out_size, aggr='mean')
            for (src, rel, dst) in edge_types if 'hier' not in rel
        }, aggr='sum')
    
        # Hierarchical aggregator
        self.attn = HierarchicalAggregator(dim=in_size, hier_edge_types=['hier_x20tox5_image', 'hier_x5tox20_image', 'hier_x20tox5_text', 'hier_x5tox20_text'])

    def forward(self, g, h, shape_data=None):
        new_h = {ntype: h[ntype].clone() for ntype in h}

        edge_index_dict = {etype: g[etype]['edge_index'] for etype in g.edge_types if 'hier' not in etype[1]}

        h_updated = self.convs(h, edge_index_dict)
        for edge_type in g.edge_types:
            src_type, rel_type, dst_type = edge_type
            if 'hier' in rel_type:
                src_feat = h[src_type]
                dst_feat = h[dst_type]

                if shape_data:
                    src_feat += shape_data.get(src_type, torch.zeros_like(src_feat))
                    dst_feat += shape_data.get(dst_type, torch.zeros_like(dst_feat))

                device = new_h[src_type].device 
            
                if src_type == 'x20image' or src_type == 'x20text':
                    new_h[dst_type] += self.attn(dst_feat, src_feat, query_scale=torch.tensor(0, device=device), key_scale=torch.tensor(1, device=device), edge_type=edge_type[1])
                elif src_type == 'x5image' or src_type == 'x5text':
                    new_h[dst_type] += self.attn(dst_feat, src_feat, query_scale=torch.tensor(1, device=device), key_scale=torch.tensor(0, device=device), edge_type=edge_type[1])

        for ntype in h_updated:
            new_h[ntype] += h_updated[ntype]

        return new_h


class HierHeteroGNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate=0.5):
        super().__init__()
        self.layer1 = HierHeteroGraphLayer(in_size, hidden_size)
        self.layer2 = HierHeteroGraphLayer(hidden_size, out_size)
        self.dropout = nn.Dropout(p=dropout_rate)  

    def forward(self, g, x_dict, shape_data=None):
      
        h = self.layer1(g, x_dict, shape_data)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.layer2(g, h, shape_data)
        h = {k: F.normalize(v, p=2, dim=-1) for k, v in h.items()}

        return h['x5text'], h['x5image'], h['x20text'], h['x20image']

class CustomCLIP(nn.Module):
    def __init__(self, class_names, conch_model, llm_descriptions, num_context_tokens=16, class_specific_token=False, 
                 class_token_position='end', low_mag='5x', num_low_mag_texts=4, high_mag='20x', num_high_mag_subtexts=3, filter_alpha=0.5):
        super(CustomCLIP, self).__init__()
        
        # freeze
        for p in conch_model.parameters():
            p.requires_grad = False

        self.num_classes = len(class_names)
        self.filter_alpha = filter_alpha
        self.dtype = next(conch_model.parameters()).dtype
        self.dim = 512
        self.graph_prompt_learner = HierHeteroGNN(self.dim, self.dim, self.dim)
        self.prompt_learner = PromptLearner(class_names, conch_model.float(), llm_descriptions,
                                            n_ctx=num_context_tokens, class_specific_token=class_specific_token,
                                            class_token_position=class_token_position)
        self.text_encoder = TextEncoder(conch_model.float())
        self.logit_scale = conch_model.logit_scale

        high, low = int(high_mag.rstrip('x')), int(low_mag.rstrip('x'))
        self.mag_ratio = int((high / low) ** 2) # high: 20x, low: 5x, mag_ratio: 16

        self.num_low_mag_texts = num_low_mag_texts
        self.num_high_mag_subtexts = num_high_mag_subtexts

    def get_tokenized_prompts(self, mag):
        return {class_name: prompts[mag] for class_name, prompts in self.prompt_learner.tokenized_prompts.items()}

    def process_similarity(self, text_features, image_features, filter_alpha):
        sim_matrix = image_features @ text_features.T * self.logit_scale.exp()

        sim_min = sim_matrix.amin(dim=0, keepdim=True)
        sim_max = sim_matrix.amax(dim=0, keepdim=True)
        sim_matrix = (sim_matrix - sim_min) / (sim_max - sim_min + 1e-8)

        mean_col = sim_matrix.mean(dim=0, keepdim=True)
        std_col = sim_matrix.std(dim=0, keepdim=True)
        threshold_col = mean_col + filter_alpha * std_col 

        sim_matrix = torch.where(sim_matrix >= threshold_col, sim_matrix, torch.zeros_like(sim_matrix))
        image_mask = (sim_matrix.mean(dim=1) > 0).to(torch.bool)

        return sim_matrix, image_mask
    
    # anchor: high text
    def hierarchical_text_contrastive_loss(self, low_text, high_text, num_classes):
        device = low_text.device
        num_samples = low_text.shape[0]
        num_high_per_low = high_text.shape[0] // num_samples

        anchor = high_text.view(num_samples, num_high_per_low, -1).mean(dim=1)
        sim_pos = F.cosine_similarity(anchor, low_text, dim=-1)

        class_size = num_samples // num_classes
        class_labels = torch.arange(num_samples, device=device) // class_size

        sim_matrix = anchor @ anchor.T
        mask = (class_labels[:, None] != class_labels[None, :]).float()
        sim_neg = sim_matrix * mask
        sim_neg = sim_neg / (mask.sum(dim=1, keepdim=True) + 1e-8)
        sim_neg = sim_neg.diag()

        loss = -torch.log(torch.sigmoid(sim_pos - sim_neg)).mean()
        return loss

    def forward(self, x5_image_features, x20_image_features):

        # Text Encoding
        text_features = self.text_encoder(self.prompt_learner(x5_image_features.device), self.prompt_learner.tokenized_prompts)
        text_features_ = torch.stack(list(text_features.values()), dim=0)

        x5_text_features = text_features_[:, :self.num_low_mag_texts, :].reshape(-1, self.dim)
        x20_text_features = text_features_[:, self.num_low_mag_texts:, :].reshape(-1, self.dim)

        # Text Guided Dynamic Filtering
        
        # Stage 1: Low magnification filtering
        x5_sim_matrix, image_mask = self.process_similarity(x5_text_features, x5_image_features, self.filter_alpha)

        # Stage 2: High magnification refinement
        x20_image_features = x20_image_features.reshape(-1, self.mag_ratio, self.dim)
        x20_image_features = x20_image_features[image_mask]
        x20_image_features = x20_image_features.reshape(-1, self.dim)
        x20_sim_matrix, _ = self.process_similarity(x20_text_features, x20_image_features, self.filter_alpha)
      
        all_features = {
            'x5text': x5_text_features, 'x5image': x5_image_features,
            'x20text': x20_text_features, 'x20image': x20_image_features
        }

        # Hierarchical Heterogeneous Graph Construction
        graph = self.build_hier_hetero_graph(x5_sim_matrix, x20_sim_matrix)

        # Hierarchical Heterogeneous Graph Learning
        x5_text_embeddings, x5_image_embeddings, x20_text_embeddings, x20_image_embeddings = self.graph_prompt_learner(graph, all_features)
    
        # Logits Computation (High Magnification)
        logits_high = x20_image_embeddings @ x20_text_embeddings.reshape(-1, self.dim).t()
        k = min(100, logits_high.size(0))
        logits_high = self.logit_scale * torch.topk(logits_high, k, dim=0)[0].mean(0)
        logits_high = logits_high.reshape(self.num_classes, -1).mean(1).squeeze()

        # Logits Computation (Low Magnification)
        logits_low = x5_image_embeddings @ x5_text_embeddings.t()
        k = min(2, logits_low.size(0))
        logits_low = self.logit_scale * torch.topk(logits_low, k, dim=0)[0].mean(0)
        logits_low = logits_low.reshape(self.num_classes, -1).mean(1).squeeze()

        # Final Logits Computation
        logits = logits_low + logits_high

        # Class-Wise Hierarchical Text Contrastive Loss
        loss = self.hierarchical_text_contrastive_loss(x5_text_embeddings, x20_text_embeddings, self.num_classes)
       
        return logits, loss

    def build_hier_hetero_graph(self, sim_matrix_x5, sim_matrix_x20):
        device = sim_matrix_x5.device
        num_x5_image, num_x5_text = sim_matrix_x5.shape
        num_x20_image, num_x20_text = sim_matrix_x20.shape

        data = HeteroData()
        data['x5image'].num_nodes = num_x5_image
        data['x5text'].num_nodes = num_x5_text
        data['x20image'].num_nodes = num_x20_image
        data['x20text'].num_nodes = num_x20_text

        def create_intra_scale_edges(sim_matrix, src_type, rel_type, dst_type):
        
            sim_matrix = sim_matrix.clone()
            sparse_sim = sim_matrix.to_sparse()
            data[(src_type, rel_type, dst_type)].edge_index = sparse_sim.indices()

        # intra-scale edges from text-guided dynamic filtering
        create_intra_scale_edges(sim_matrix_x5, 'x5image', 'x5_intra', 'x5text')
        create_intra_scale_edges(sim_matrix_x20, 'x20image', 'x20_intra', 'x20text')
        create_intra_scale_edges(sim_matrix_x5.T, 'x5text', 'x5_intra', 'x5image')
        create_intra_scale_edges(sim_matrix_x20.T, 'x20text', 'x20_intra', 'x20image')

        def create_hierarchical_edges(src_ids, stride, dst_ids):
            num_src = src_ids.size(0)
            indices = torch.arange(num_src, device=device)
            group_ids = torch.div(indices, stride, rounding_mode='floor')
            valid_mask = group_ids < dst_ids.size(0)
            return src_ids[valid_mask], dst_ids[group_ids[valid_mask]]

        x5_image_ids = torch.arange(num_x5_image, device=device)
        x5_text_ids = torch.arange(num_x5_text, device=device)
        x20_image_ids = torch.arange(num_x20_image, device=device)
        x20_text_ids = torch.arange(num_x20_text, device=device)
        
        # hierarchical edges
        img_src, img_dst = create_hierarchical_edges(x20_image_ids, self.mag_ratio, x5_image_ids)
        if img_src.numel() > 0:
            data[('x20image', 'hier_x20tox5_image', 'x5image')].edge_index = torch.stack([img_src, img_dst], dim=0)
            data[('x5image', 'hier_x5tox20_image', 'x20image')].edge_index = torch.stack([img_dst, img_src], dim=0)

        text_src, text_dst = create_hierarchical_edges(x20_text_ids, self.num_high_mag_subtexts, x5_text_ids)
        if text_src.numel() > 0:
            data[('x20text', 'hier_x20tox5_text', 'x5text')].edge_index = torch.stack([text_src, text_dst], dim=0)
            data[('x5text', 'hier_x5tox20_text', 'x20text')].edge_index = torch.stack([text_dst, text_src], dim=0)

        return data


class HiVE_MIL(nn.Module):
    def __init__(self, args, dataset_name, feature_extractor_name, text_prompt_dir, LLM='gpt4o', num_context_tokens=16, class_specific_token=False, class_token_position='end',
                low_mag='5x', num_low_mag_texts=4, high_mag='20x', num_high_mag_subtexts=3, filter_alpha=0.5):
        super().__init__()
    
        self.class_names = get_class_names(dataset_name)

        self.args = args

        conch_model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="conch/pytorch_model.bin")
        _ = conch_model.eval()

        self.loss_ce = nn.CrossEntropyLoss()

        text_prompt_file = Path(text_prompt_dir, 'HiVE_MIL', LLM, f'{dataset_name}.json')
        with open(text_prompt_file, 'r') as f:
            llm_descriptions = json.load(f)

        self.custom_vlm = CustomCLIP(
            class_names=self.class_names,
            conch_model=conch_model,
            llm_descriptions=llm_descriptions,
            num_context_tokens=num_context_tokens,
            class_specific_token=class_specific_token,
            class_token_position=class_token_position,
            low_mag=low_mag,
            num_low_mag_texts=num_low_mag_texts,
            high_mag=high_mag,
            num_high_mag_subtexts=num_high_mag_subtexts,
            filter_alpha=filter_alpha
        )

        for name, param in self.custom_vlm.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.custom_vlm.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        
        # print(f"Parameters to be updated: {enabled}")
        # print(f"Trainable parameters: {sum(p.numel() for p in self.custom_vlm.parameters() if p.requires_grad)}")
    
    def forward(self, x_5, A, x_20, B, label):

        logits, LOSS_ = self.custom_vlm(x_5, x_20)
        pred = logits.unsqueeze(0)
        loss = self.loss_ce(pred, label) + self.args.contrastive_lambda * LOSS_
        Y_prob = F.softmax(pred, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        
        return Y_prob, Y_hat, loss