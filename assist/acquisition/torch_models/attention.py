import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from assist.acquisition.torch_models.base import BaseClassifier
from assist.acquisition.torch_models.utils import SequenceDataset, pad_and_sort_batch, get_attention_mask


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop,) * 2 if isinstance(drop, (float, int)) else drop

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ClassAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        B, N, C = x.shape
        
        x_cls, x = x[:, 0], x[:, 1:]
        N -= 1
        
        q = self.q(x_cls).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls  


class LayerScaleBlockCA(nn.Module):

    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Attention_block=ClassAttention,
                 mlp_block=MLP,
                 init_values=1e-4):
        
        super(LayerScaleBlockCA, self).__init__()
        
        self.attn = Attention_block(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls,x),dim=1)
        x_cls = x_cls + self.gamma_1 * self.attn(self.norm1(u))
        x_cls = x_cls + self.gamma_2 * self.mlp(self.norm2(x_cls))
        return x_cls


class ClassAttnDecoder(nn.Module):
    
    def __init__(self, input_dim, 
                 embed_dim, 
                 num_classes, 
                 n_heads, 
                 num_layers=3, 
                 cls_layers=1, 
                 maxlen=512, 
                 dropout=0.):
        
        super(ClassAttnDecoder, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.pos_embedding = nn.Parameter(torch.zeros(1, maxlen, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, dropout=dropout, activation="gelu"),
            num_layers, norm=nn.Identity()  #nn.LayerNorm(embed_dim)
        )
        
        self.cls_transformer = nn.ModuleList([
            LayerScaleBlockCA(embed_dim, 1, norm_layer=nn.Identity)
            for _ in range(cls_layers)
        ])
        
        # self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")
         
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embedding, std=.02)
        
    def forward(self, features, lengths, labels=None):
        features = self.forward_features(features)
        logits = self.classifier(features)
        
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits
    
        return logits
    
    def forward_features(self, x):

        B, L, _ = x.size()
        
        x = self.embed(x)
        x += self.pos_embedding[:, :L]
        x = self.pos_dropout(x)
        x = self.transformer(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        for layer in self.cls_transformer:
            cls_tokens = layer(x, cls_tokens)
            
        x = torch.cat([cls_tokens, x], dim=1)
        # x = self.norm(x)
        return x[:, 0]

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class AttentiveDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, d_model=None, num_heads=8, dropout=0.1, bias=True):
        super(AttentiveDecoder, self).__init__()
        
        if d_model and input_dim != d_model:
            self.embedding = nn.Linear(input_dim, d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model or input_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, inputs, input_lengths, labels=None, return_attention=False):

        if hasattr(self, "embedding"):
            inputs = self.embedding(inputs)

        keys = inputs.transpose(0, 1).contiguous()
        key_padding_mask = get_attention_mask(input_lengths).to(keys.device)
        linear_combination, energy  = self.attention(
            keys, keys, keys, key_padding_mask=key_padding_mask)

        logits = self.output_layer(self.dropout(linear_combination.sum(0)))

        if labels is None and not return_attention:
            return logits

        output = tuple()

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output += (loss,)

        output += (logits,)

        if return_attention:
            output += (energy,)

        return output

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class AttentiveRecurrentDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, d_model, rnn_type="lstm", num_layers=1, rnn_dropout=0.1, num_heads=8, attn_dropout=0.1, attn_bias=True):
        super(AttentiveRecurrentDecoder, self).__init__()
        RNNClass = getattr(nn, rnn_type.upper())
        self.encoder = RNNClass(
            input_dim,
            d_model,
            num_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=attn_bias
        )

        # self.dropout = nn.Dropout(rnn_dropout)
        self.output_layer = nn.Linear(d_model * 2, output_dim)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, inputs, input_lengths, labels=None, return_attention=False):
        outputs, hidden = self.encoder(inputs)

        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)

        query = hidden.unsqueeze(1).transpose(0, 1).contiguous()
        key = outputs.transpose(0, 1).contiguous()
        linear_combination, energy  = self.attention(query, key, key)
        linear_combination = linear_combination.squeeze(0)
        logits = self.output_layer(linear_combination)

        if labels is None and not return_attention:
            return logits

        output = tuple()
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output += (loss,)

        output += (logits,)

        if return_attention:
            output += (energy,)

        return output

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class Classifier(BaseClassifier):

    def build(self):
        self.device = torch.device(self.config["device"])

        if self.config["name"] == "att":

            model = AttentiveDecoder(
                input_dim=int(self.config["input_dim"]),
                output_dim=self.n_classes,
                d_model=int(self.config["hidden_dim"]),
                num_heads=int(self.config.get("num_heads", 4)),
                dropout=float(self.config["dropout"]),
                bias=True
            )

        elif self.config["name"] == "att_rnn":

            model = AttentiveRecurrentDecoder(
                input_dim=int(self.config["input_dim"]),
                output_dim=self.n_classes,
                d_model=int(self.config["hidden_dim"]),
                rnn_type=self.config.get("rnn_type", "lstm"),
                num_layers=int(self.config["num_layers"]),
                rnn_dropout=float(self.config["dropout"]),
                num_heads=int(self.config.get("num_heads", 4)),
                attn_dropout=float(self.config["dropout"])
            )
            
        elif self.config["name"] == "cls_att":
            
            model = ClassAttnDecoder(
                input_dim=int(self.config["input_dim"]),
                embed_dim=int(self.config["hidden_dim"]),
                num_classes=self.n_classes,
                n_heads=int(self.config.get("num_heads", 4)),
                num_layers=int(self.config["num_layers"]),
                maxlen=512,
                dropout=float(self.config["dropout"])
            )

        else:
            raise NotImplementedError(f"Unknown model: {self.config['name']}")

        return model.to(self.device)

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(
            dataset,
            shuffle=is_train,
            batch_size=self.batch_size,
            collate_fn=pad_and_sort_batch
        )

    def prepare_inputs(self, features, labels=None):
        return SequenceDataset(features, labels=labels),
