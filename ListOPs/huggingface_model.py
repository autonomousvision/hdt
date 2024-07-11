# IDEA: Make a single lightning class that will work for all models that are defined by huggingface's "...forSequenceClassification" models

import torch
from pytorch_lightning import LightningModule
from transformers import BertConfig
import transformers
import Config
import my_modeling_bert
from attention_mask_variable_listops import batch_get_attn_mask, batch_get_attn_mask_red, load_token_ids, batch_get_attn_mask_green
from transformers import LongformerConfig, LongformerForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
import os

class huggingface_model(LightningModule):
    attention_mask = None
    max_position_embeddings = 2048

    def __init__(self, model_name, log_dir, **kwargs):
        super().__init__()
        self.model_name = model_name
        if model_name == 'BERT':
            config = BertConfig(vocab_size=Config.VOCAB_SIZE, 
                            max_position_embeddings=self.max_position_embeddings, 
                            intermediate_size=kwargs['intermediate_size'], 
                            hidden_size=kwargs['hidden_size'], 
                            num_attention_heads=kwargs['n_heads'], 
                            num_hidden_layers=kwargs['n_layers'], 
                            num_labels=10)
            self.representation_model = transformers.BertForSequenceClassification(config) # original BERT implementation
        elif model_name == 'HDT':
            config = BertConfig(vocab_size=Config.VOCAB_SIZE, 
                            max_position_embeddings=self.max_position_embeddings, 
                            intermediate_size=kwargs['intermediate_size'], 
                            hidden_size=kwargs['hidden_size'], 
                            num_attention_heads=kwargs['n_heads'], 
                            num_hidden_layers=kwargs['n_layers'], 
                            num_labels=10)
            self.attn_color = kwargs['attn_color']
            self.representation_model = my_modeling_bert.BertForSequenceClassification(config) # my BERT implementation with hierarchical attention mask
        elif model_name == 'Longformer':
            configuration = LongformerConfig(    
                max_position_embeddings = self.max_position_embeddings,   
                intermediate_size = kwargs['intermediate_size'], # 3072 
                hidden_size = kwargs['hidden_size'], # 786
                num_attention_heads = kwargs['n_heads'], # 12
                num_hidden_layers = kwargs['n_layers'], # 12
                num_labels = 10,
# -------------------   Longformer specific parameters below   -------------------
                attention_window = kwargs['attention_window'], # local attention window
                sep_token_id = 99, # undefinded for ListOPs?
                pad_token_id = 4,
                bos_token_id = 99, # undefined for ListOPs?
                eos_token_id = 99, # undefined for ListOPs?
                vocab_size = 100, #  30522
                hidden_act = "gelu",
                hidden_dropout_prob = 0.1,
                attention_probs_dropout_prob = 0.1,
                # Adds a learned embedding for the token_type_ids that can be given as a parameter during the forward pass
                # ListOPs doesnt have token type IDs, so it doesn't really matter what value I specify here. Default was 2
                type_vocab_size = 1, 
                initializer_range = 0.02,
                layer_norm_eps = 1e-12,
                onnx_export = False,
                )
            self.representation_model = LongformerForSequenceClassification(configuration)
        elif model_name == 'HAT':
            from modeling_hat import HATForSequenceClassification, HATConfig
            if kwargs['n_layers'] != 12:
                raise Exception("define encoder layout for n_layers!= 12")
            configuration = HATConfig(
                max_position_embeddings=self.max_position_embeddings, # 512
                intermediate_size=kwargs['intermediate_size'], # 3072
                hidden_size = kwargs['hidden_size'], # 786
                num_attention_heads = kwargs['n_heads'], # 12
                num_hidden_layers = kwargs['n_layers'], # 12
                num_labels=10,
# -------------------   HAT specific parameters below   ------------------
                vocab_size=30522,
                max_sentences=64,
                max_sentence_size=128,
                max_sentence_length=128,# had to add this
                model_max_length=8192,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                type_vocab_size=1, # 2
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                position_embedding_type="absolute",
                # encoder_layout=None,
                use_cache=True,
                classifier_dropout=None,
                encoder_layout= {"0": {"sentence_encoder": True, "document_encoder":  False},
"1": {"sentence_encoder": True, "document_encoder":  False},
"2": {"sentence_encoder": True, "document_encoder":  False},
"3": {"sentence_encoder": True, "document_encoder":  False},
"4": {"sentence_encoder": True, "document_encoder":  False},
"5": {"sentence_encoder": True, "document_encoder":  False},
"6": {"sentence_encoder": True, "document_encoder":  False},
"7": {"sentence_encoder": True, "document_encoder":  False},
"8": {"sentence_encoder": False, "document_encoder":  True},
"9": {"sentence_encoder": False, "document_encoder":  True},
"10": {"sentence_encoder": False, "document_encoder":  True},
"11": {"sentence_encoder": False, "document_encoder":  True}
}
            )
            self.representation_model = HATForSequenceClassification(configuration)
        else:
            raise ValueError(f"Model {model_name} not recognized")
        
        self.summary_writer = SummaryWriter(log_dir=os.path.join('runs', log_dir))
        
        # Define the loss function
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.charset_ids = load_token_ids()

    def forward(self, input_ids, attention_mask, labels):
        # input ids are a tensor of shape (batch_size, sequence_length)
        outputs = self.representation_model(input_ids, attention_mask=attention_mask) 
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        if self.model_name == 'HDT':
            if self.attn_color == 'green':
                attention_mask = batch_get_attn_mask_green(batch['input_ids'], self.charset_ids)
            elif self.attn_color == 'red':
                attention_mask = batch_get_attn_mask_red(batch['input_ids'], self.charset_ids)
            else:
                attention_mask = batch_get_attn_mask(batch['input_ids'], self.charset_ids)
        else:
            attention_mask = batch['attention_mask']
        labels = batch['label']
        loss = self(input_ids, attention_mask, labels)
        outputs = self(input_ids, attention_mask, labels)
        logits = outputs.logits
        loss = self.ce_loss(logits, labels)
        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.summary_writer.add_scalar('train_loss', loss, self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        if self.model_name == 'HDT':
            if self.attn_color == 'green':
                attention_mask = batch_get_attn_mask_green(batch['input_ids'], self.charset_ids)
            elif self.attn_color == 'red':
                attention_mask = batch_get_attn_mask_red(batch['input_ids'], self.charset_ids)
            else:
                attention_mask = batch_get_attn_mask(batch['input_ids'], self.charset_ids)
        else:
            attention_mask = batch['attention_mask']
        labels = batch['label']
        loss = self(input_ids, attention_mask, labels)
        outputs = self(input_ids, attention_mask, labels)
        logits = outputs.logits
        preds = logits.argmax(dim=1)
        accuracy = torch.eq(preds, labels).sum().item() / len(labels)
        # Log the loss
        self.log('validation_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.summary_writer.add_scalar('validation_accuracy', accuracy, self.global_step)
        return loss
    
    @property
    def val_check_interval(self):
        return 0.2
   
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        if self.model_name == 'HDT':
            if self.attn_color == 'green':
                attention_mask = batch_get_attn_mask_green(batch['input_ids'], self.charset_ids)
            elif self.attn_color == 'red':
                attention_mask = batch_get_attn_mask_red(batch['input_ids'], self.charset_ids)
            else:
                attention_mask = batch_get_attn_mask(batch['input_ids'], self.charset_ids)
        else:
            attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask, labels)
        logits = outputs.logits
        preds = logits.argmax(dim=1)

        # Calculate the accuracy
        accuracy = torch.eq(preds, labels).sum().item() / len(labels)
        self.log(f"test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.LEARNING_RATE)
        if Config.LEARNING_RATE_SCHEDULER == 'fixed':
            return optimizer
        if Config.LEARNING_RATE_SCHEDULER == 'cosine':
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                                     num_warmup_steps=self.n_steps*0.1, 
                                                                     num_training_steps=self.n_steps*Config.EPOCHS,
                                                                     num_cycles=0.5)
            return optimizer, scheduler
        return optimizer # just return the optimizer if no scheduler is specified. that's ok, somehow



class BERT(huggingface_model):
    def __init__(self, **kwargs):
        super().__init__('BERT', 'runs', **kwargs)

class HDT(huggingface_model):
    def __init__(self, **kwargs):
        super().__init__('HDT', 'runs', **kwargs)

class Longformer(huggingface_model):
    def __init__(self, **kwargs):
        super().__init__('Longformer', 'runs', **kwargs)

class HAT(huggingface_model):
    def __init__(self, **kwargs):
        super().__init__('HAT', 'runs', **kwargs)