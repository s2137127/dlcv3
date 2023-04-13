from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('../hw3_data/caption_tokenizer.json')
class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr_backbone = 1e-4
        self.lr = 1e-5

        # Epochs
        self.epochs = 70
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True
        
        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 8
        self.num_workers = 8
        self.checkpoint = './checkpoint_58.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        # self.vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        self.vocab_size =30522
        self.enc_layers = 5
        self.dec_layers = 4
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        # self.dir = '../coco'
        self.limit = -1