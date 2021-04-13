import functools

import tensorflow as tf


def get_gram(x):
  ba, hi, wi, ch = [i.value for i in x.get_shape()]
  feature = tf.reshape(x, [ba, int(hi * wi), ch])
  feature_T = tf.transpose(feature, [0, 2, 1])
  gram = tf.matmul(feature_T, feature)
  size = 1 / (hi * wi * ch)
  return gram * size

def Percept_loss(fai_out, fai_gt, fai_comp, layers):
  out_gt = []
  compt_gt = []
  for layer in layers:
    out_gt.append(tf.reduce_mean(tf.abs(fai_out[layer] - fai_gt[layer])))
    compt_gt.append(tf.reduce_mean(tf.abs(fai_comp[layer] - fai_gt[layer])))
  out_gt_loss = functools.reduce(tf.add, out_gt)
  compt_gt_loss = functools.reduce(tf.add, compt_gt)
  return out_gt_loss + compt_gt_loss

def Style_loss_out(fai_out, fai_gt, layers):
  styleloss = []
  for layer in layers:
    gram_out = get_gram(fai_out[layer])
    gram_gt = get_gram(fai_gt[layer])
    styleloss.append(tf.reduce_mean(tf.abs(gram_out - gram_gt)))
  style_out_loss = functools.reduce(tf.add, styleloss)
  return style_out_loss

def Style_loss_comp(fai_comp, fai_gt, layers):
  styleloss = []
  for layer in layers:
    gram_comp = get_gram(fai_comp[layer])
    gram_gt = get_gram(fai_gt[layer])
    styleloss.append(tf.reduce_mean(tf.abs(tf.subtract(gram_comp, gram_gt))))
  style_comp_loss = functools.reduce(tf.add, styleloss)
  return style_comp_loss


def get_total_loss(I_out, I_gt, fai_out, fai_gt, fai_comp, layers,
                   I_comp):
  percept_loss = Percept_loss(fai_out, fai_gt, fai_comp, layers)
  style_loss_out = Style_loss_out(fai_out, fai_gt, layers)
  style_loss_comp = Style_loss_comp(fai_comp, fai_gt, layers)
  all_loss = 0.05 * percept_loss + 120 * (style_loss_out + style_loss_comp)
  return all_loss