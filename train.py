import tensorflow as tf
from tensorflow.keras import layers as tfl

import transformers
from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import json

with open('config.json', 'r') as f:
    config = json.load(f)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def load_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordPiece(unk_token='<UNK>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=['<SOS>', '<EOS>', '<PAD>', '<UNK>'])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

def get_ds(config):
    ds = load_dataset('opus100', f'{config["lang_src"]}-{config["lang_tgt"]}')
    
    tokenizer_src = load_tokenizer(config, ds, 'ar')
    tokenizer_tgt = load_tokenizer(config, ds, 'en')
    