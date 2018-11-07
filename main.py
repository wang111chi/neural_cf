#!/usr/bin python3
# coding=utf-8

import math
from time import time
import argparse

import tensorflow as tf

from evaluation import evaluate_model
from neural_mf import NeuMF
from load_data import LoadData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', default='data/ml-1m/',
                        help='Input data path.')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')

    parser.add_argument('--num_factors', type=int, default=32,
                        help='Embedding size.')

    parser.add_argument('--reg_id_embedding', nargs='?', default=0.0, type=int,
                        help="Regularization for user and item embeddings.")

    parser.add_argument('--reg_others', nargs='?', default=0.0, type=float,
                        help="Regularization for general variables.")

    parser.add_argument('--init_stddev', nargs='?', default=0.1, type=float,
                        help="Init stddev value for variables.")

    parser.add_argument('--num_neg_inst', type=int, default=4,
                        help='Number of negative instances to pair with a '
                             'positive instance.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')

    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')

    parser.add_argument('--loss', default='log_loss',
                        help='type of loss function.')

    parser.add_argument('--eta', type=float, default=0.1,
                        help='eta of adadelta')

    parser.add_argument('--topk', type=int, default=10,
                        help='Evaluate the top k items.')

    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer "
                             "is the concatenation of user and item embeddings. "
                             "So layers[0]/2 is the embedding size.")

    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")

    return parser.parse_args()


def single_run(args, dataset):
    model = NeuMF(args, dataset.num_users, dataset.num_items)
    model.build_model()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    t1 = time()
    ahit, andcg = evaluate_model(sess, model, dataset, args.topk)
    best_hr, best_ndcg, best_iter = ahit, andcg, -1
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (ahit, andcg, time() - t1))

    tf.summary.scalar('loss', best_hr)
    summaryMerged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graph_tensorboard', sess.graph)
    for epoch in range(args.epochs):
        t1 = time()
        train_users, train_items, train_labels, num_inst = \
            dataset.make_training_instances(args.num_neg_inst)
        # print(train_labels[0:20])
        loss_per_epoch, error_per_epoch = 0, 0
        for ite in range((num_inst - 1) // args.batch_size + 1):
            start_idx = ite * args.batch_size
            end_idx = min((ite + 1) * args.batch_size, num_inst)
            cur_user_indices = train_users[start_idx: end_idx]
            cur_item_indices = train_items[start_idx: end_idx]
            cur_label = train_labels[start_idx: end_idx]

            _, loss, error = sess.run(
                [model.train_step, model.loss, model.raw_error],
                {model.user_indices: cur_user_indices,
                 model.item_indices: cur_item_indices,
                 model.ratings: cur_label})
            loss_per_epoch += loss
            error_per_epoch += error

        summary = sess.run(summaryMerged)
        writer.add_summary(summary, epoch)
        error_per_epoch /= num_inst

        t2 = time()
        if epoch % args.verbose == 0:
            ahit, andcg = evaluate_model(sess, model, dataset, args.topk)
            print('epoch %d\t[%.1f s]: HR= %.4f\tNDCG= %.4f\tloss= %.4f\terror'
                  '= %.4f\t[%.1f s]' % (
                      epoch, t2 - t1, ahit, andcg, loss_per_epoch,
                      error_per_epoch, time() - t2))
            if ahit > best_hr:
                best_hr = ahit
                best_iter = epoch

            if andcg > best_ndcg:
                best_ndcg = andcg

    print("End. Best Epoch %d:  HR = %.4f, NDCG = %.4f. " % (
        best_iter, best_hr, best_ndcg))


if __name__ == '__main__':
    args = parse_args()

    print("runtime arguments: %s" % (args))

    dataset = LoadData(args.path)

    print(dataset.num_users, dataset.num_items)

    single_run(args, dataset)

