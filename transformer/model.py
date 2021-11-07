from pytorch_transformers import BertModel, BertConfig,BertTokenizer
# from transformers import BertModel,BertConfig,BertTokenizer
import torch
import torch.nn as nn
class TextNet(nn.Module):
    def __init__(self,code_length):
        super(TextNet,self).__init__()
        modelConfig = BertConfig.from_pretrained('bert-base-chinese')
        self.textExtractor = BertModel.from_pretrained('bert-base-chinese',config = modelConfig)
        # modelConfig = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        # self.textExtractor = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large',config = modelConfig)
        '''
        万一上不去外网用这个方法：
        modelConfig = BertConfig.from_pretrained('bert-base-uncased-config.json')
        self.tetExtractor = BertModel.from_pretrained('bert-base-uncased-pytorch_model.bin',config = modelConfig)
        '''
        embedding_dim = self.textExtractor.config.hidden_size
        self.fc = nn.Linear(embedding_dim,code_length)
        self.tanh = torch.nn.Tanh()
    def forward(self,tokens,segments,input_masks):
        output = self.textExtractor(tokens,token_type_ids = segments,attention_mask = input_masks)
        text_embeddings = output[0]
        return text_embeddings