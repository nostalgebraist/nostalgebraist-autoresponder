import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time


class AccumulatingOptimizer(object):
    def __init__(self, opt, var_list):
        self.opt = opt
        self.var_list = var_list
        self.accum_vars = {
            tv: tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
            for tv in var_list
        }
        self.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars.values()]
        updates.append(self.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def compute_gradients(self, loss):
        grads = self.opt.compute_gradients(loss, self.var_list)
        updates = [self.accum_vars[v].assign_add(g) for (g, v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(1.0))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(self):
        grads = [(g / self.count_loss, v) for (v, g) in self.accum_vars.items()]
        with tf.control_dependencies([self.opt.apply_gradients(grads)]):
            return self.total_loss / self.count_loss

    def get_name(self):
        return self.opt.get_name()


class GradientAccumulator:
    def __init__(self, var_list, noise_scale_estimates=False, grad_clip=1):
        self.var_list = var_list
        self.noise_scale_estimates = noise_scale_estimates
        self.grad_clip = grad_clip

        self.accum_vars = {
            tv: tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
            for tv in var_list
        }
        self.accum_gnorm = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars.values()]
        updates.append(self.accum_gnorm.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def add_gradients_without_loss(self, grads):
        updates = [self.accum_vars[v].assign_add(g) for (g, v) in grads]
        if self.noise_scale_estimates:
            gn = tf.global_norm([g for g, v in grads]) ** 2
            updates.append(self.accum_gnorm.assign_add(gn))
            updates.append(
                self.count_loss.assign_add(1.0)
            )  # need this for gn avg later
        with tf.control_dependencies(updates):
            return tf.no_op()

    def add_gradients(self, loss, grads):
        updates = [self.accum_vars[v].assign_add(g) for (g, v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(1.0))
        if self.noise_scale_estimates:
            gn = tf.global_norm([g for g, v in grads]) ** 2
            updates.append(self.accum_gnorm.assign_add(gn))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(self, opt, **kwargs):
        grads = [(g / self.count_loss, v) for (v, g) in self.accum_vars.items()]
        gs_clipped, _ = tf.clip_by_global_norm([g for g, v in grads], self.grad_clip)

        grads_for_apply = [(g, v) for g, (_, v) in zip(gs_clipped, grads)]

        if self.noise_scale_estimates:
            gn_big = tf.global_norm([g for g, v in grads]) ** 2
            gn_small = self.accum_gnorm / self.count_loss
        with tf.control_dependencies([opt.apply_gradients(grads_for_apply, **kwargs)]):
            if self.noise_scale_estimates:
                return self.total_loss / self.count_loss, gn_big, gn_small
            return self.total_loss / self.count_loss
