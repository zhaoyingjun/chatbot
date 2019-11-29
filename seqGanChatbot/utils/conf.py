__author__ = 'liuyuemaicha'
import os


class disc_config(object):
    batch_size = 64
    lr = 0.2
    lr_decay = 0.9
    vocab_size = 2500
    embed_dim = 64
    steps_per_checkpoint = 20
    # hidden_neural_size = 128
    num_layers = 2
    train_dir = './disc_data/'
    name_model = "disc_model"
    tensorboard_dir = "./tensorboard/disc_log/"
    name_loss = "disc_loss"
    max_len = 50
    piece_size = batch_size * steps_per_checkpoint
    piece_dir = "./disc_data/batch_piece/"
    # query_len = 0
    valid_num = 100
    init_scale = 0.1
    num_class = 2
    keep_prob = 0.5
    # num_epoch = 60
    # max_decay_epoch = 30
    max_grad_norm = 5
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


class gen_config(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 64
    emb_dim = 64
    num_layers = 2
    vocab_size = 2500
    train_dir = "./gen_data/"
    name_model = "st_model"
    tensorboard_dir = "./tensorboard/gen_log/"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 100000
    steps_per_checkpoint = 20
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]


class GSTConfig(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 256
    emb_dim = 1024
    num_layers = 2
    vocab_size = 2500
    train_dir = "./gst_data/"
    name_model = "st_model"
    tensorboard_dir = "./tensorboard/gst_log/"
    name_loss = "gst_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]


class source_data_utils(object):
    train_enc = "./gen_data/train.query"
    train_dec = "./gen_data/train.answer"
    test_enc = "./gen_data/test.query"
    test_dec = "./gen_data/test.answer"
    resource_data = "./data_source/conv.txt"

    TEST_SIZE = 400
    e = "E"
    m = "M"
