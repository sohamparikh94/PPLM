#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

from IPython import embed


import argparse
import json
import spacy
from operator import add
from wordfreq import word_frequency
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange, tqdm
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

from pplm_classification_head import ClassificationHead
from pplm_regression_head import RegressionHead1, RegressionHead2

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
PPLM_REGRESS = 4
SMALL_CONST = 1e-15
BIG_CONST = 1e10

ONE_HOT = 0
WORD_FREQ = 1
WORD_FREQ_LOG = 2
WORD_FREQ_AVGLOG = 3
WORD_FREQ_EXP = 4
WORD_FREQ_SQ = 5

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
    'kitchen': "./paper_code/wordlists/kitchen.txt",
    'minimal1': "./paper_code/wordlists/minimal1.txt",
    'minimal2': "./paper_code/wordlists/minimal2.txt",
    'minimal3': "./paper_code/wordlists/minimal3.txt"
}



PROMPTS_MAP = {
    'legal': './100_prompts/legal.txt',
    'kitchen': './100_prompts/kitchen.txt',
    'technology': './100_prompts/technology.txt',
    'science': './100_prompts/science.txt',
    'agnostic': './100_prompts/domain_agnostic_prompts.txt'
}


DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}

nlp_light = spacy.load('en', disable=['parser', 'tagger', 'ner'])


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        bows_vectors=None,
        classifier=None,
        regressor=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        kl_bow = False,
        device='cuda',
        verbosity_level=REGULAR
):
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            for bow in bows_vectors:
                bow_logits = torch.mm(probs, torch.t(bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
                if(kl_bow):
                    single_bow = torch.zeros_like(unpert_probs).to(device)
                    rows, columns = np.where(bow.cpu() > 0)
                    for x in range(len(rows)):
                        single_bow[0][columns[x]] = bow[rows[x]][columns[x]]
                    single_bow = single_bow/torch.sum(single_bow)
                    single_bow += SMALL_CONST
                    bow_kl_loss = (single_bow * (single_bow / unpert_probs).log()).sum()
                    loss += bow_kl_loss
                    loss_list.append(bow_kl_loss)

            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=inputs_embeds
                )
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            prediction = classifier(new_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))

            label = torch.tensor(prediction.shape[0] * [class_label],
                                 device=device,
                                 dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        if loss_type == PPLM_REGRESS:
            mse_loss = torch.nn.MSELoss()
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                input_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=input_embeds
                )
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            prediction = regressor(new_accumulated_hidden/ (curr_length + 1 + horizon_length))
            regress_loss = mse_loss(prediction, torch.tensor([[100]], 
                                    device=device, 
                                    dtype=torch.float))
            loss += regress_loss
            loss_list.append(regress_loss)


        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_regressor(
        name: Optional[str],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = RegressionHead1(
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    return classifier




def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer, omit_file=None) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        if(omit_file):
            with open(omit_file, "r") as f:
                omit_words = f.read().strip().split("\n")

            new_words = list()
            for word in words:
                if(word not in omit_words):
                    new_words.append(word)
            bow_indices.append(
                [tokenizer.encode(word.strip(),
                                  add_prefix_space=True,
                                  add_special_tokens=False)
                 for word in new_words])
        else:
            bow_indices.append(
                [tokenizer.encode(word.strip(),
                                  add_prefix_space=True,
                                  add_special_tokens=False)
                 for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    
    return one_hot_bows_vectors


def build_bows_freq_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        word_freq = [1/word_frequency(tokenizer.decode(idx), lang='en') for idx in single_bow]
        sum_ = np.sum(word_freq)
        word_freq = torch.tensor([[(x/sum_)*len(word_freq)] for x in word_freq]).to(device)
        bow = bow.scatter_(1, single_bow, word_freq)
        bows_vectors.append(bow)

    return bows_vectors


def build_bows_logfreq_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        word_freq = [np.log(1/word_frequency(tokenizer.decode(idx), lang='en')) for idx in single_bow]
        sum_ = np.sum(word_freq)
        word_freq = torch.tensor([[(x/sum_)*len(word_freq)] for x in word_freq]).to(device)
        bow = bow.scatter_(1, single_bow, word_freq)
        bows_vectors.append(bow)

    return bows_vectors


def build_bows_avglogfreq_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        word_freq = [(1/word_frequency(tokenizer.decode(idx), lang='en') + np.log(1/word_frequency(tokenizer.decode(idx), lang='en')))/2 for idx in single_bow]
        sum_ = np.sum(word_freq)
        word_freq = torch.tensor([[(x/sum_)*len(word_freq)] for x in word_freq]).to(device)
        bow = bow.scatter_(1, single_bow, word_freq)
        bows_vectors.append(bow)

    return bows_vectors


def build_bows_expfreq_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        word_freq = [np.exp(1/word_frequency(tokenizer.decode(idx), lang='en')) for idx in single_bow]
        sum_ = np.sum(word_freq)
        word_freq = torch.tensor([[(x/sum_)*len(word_freq)] for x in word_freq]).to(device)
        bow = bow.scatter_(1, single_bow, word_freq)
        bows_vectors.append(bow)

    return bows_vectors

def build_bows_sqfreq_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        word_freq = [np.square(1/word_frequency(tokenizer.decode(idx), lang='en')) for idx in single_bow]
        sum_ = np.sum(word_freq)
        word_freq = torch.tensor([[(x/sum_)*len(word_freq)] for x in word_freq]).to(device)
        bow = bow.scatter_(1, single_bow, word_freq)
        bows_vectors.append(bow)

    return bows_vectors


def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        bow_type=0,
        discrim=None,
        regress=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        kl_bow=False,
        verbosity_level=REGULAR,
        omit_file=None,
        omit_seq=False,
        **kwargs
):
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    regressor = get_regressor(
        regress,
        device)

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer, omit_file)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    elif regressor is not None:
        loss_type = PPLM_REGRESS

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        if(omit_seq):
            if bag_of_words:
                bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                                    tokenizer, omit_file)
            with open(omit_file) as f:
                omit_words = f.read().strip().split('\n')
            with open(BAG_OF_WORDS_ARCHIVE_MAP[bag_of_words.split(';')[0]]):
                bow_words = f.read().strip().split('\n')
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            bow_type=bow_type,
            classifier=classifier,
            regressor=regressor,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            kl_bow=kl_bow,
            verbosity_level=verbosity_level,
        )
        generated_text = tokenizer.decode(pert_gen_tok_text[0][1:])
        pert_gen_tok_texts.append(generated_text)
        if(omit_seq):
            for token in nlp_light(generated_text):
                if(token.text in bow_words and token.text not in omit_words):
                    omit_words.append(token.text)
            new_words = list()
            for word in bow_words:
                if(word not in omit_words):
                    new_words.append(word)
            tokenized_words = [tokenizer.encode(word.strip(),
                            add_prefix_space=True,
                            add_special_tokens=False)
                     for word in new_words]
            if(len(tokenized_words)):
                with open(omit_file, 'w') as f:
                    for word in omit_words:
                        f.write(word)
                        f.write('\n')
            else:
                with open(omit_file, 'w') as f:
                    f.write('')


        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        bow_type=0,
        classifier=None,
        regressor=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        kl_bow=False,
        verbosity_level=REGULAR
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    if(bow_type == ONE_HOT):
        bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
                                                      device)
    elif(bow_type == WORD_FREQ):
        bows_vectors = build_bows_freq_vectors(bow_indices, tokenizer, device)
    elif(bow_type == WORD_FREQ_LOG):
        bows_vectors = build_bows_logfreq_vectors(bow_indices, tokenizer, device)
    elif(bow_type == WORD_FREQ_AVGLOG):
        bows_vectors = build_bows_avglogfreq_vectors(bow_indices, tokenizer, device)
    elif(bow_type == WORD_FREQ_EXP):
        bows_vectors = build_bows_expfreq_vectors(bow_indices, tokenizer, device)
    elif(bow_type == WORD_FREQ_SQ):
        bows_vectors = build_bows_sqfreq_vectors(bow_indices, tokenizer, device)
    

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    bows_vectors=bows_vectors,
                    classifier=classifier,
                    regressor=regressor,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    kl_bow=kl_bow,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy()
                )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta


def run_pplm_example(
        pretrained_model="gpt2-medium",
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        bow_type=0,
        discrim=None,
        regress=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        kl_bow=False,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular',
        multiple_prompts=False,
        prompts_file=None,
        output_file=None,
        omit_file=None,
        omit_seq=False
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if regress == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if regress is not None:

        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[regress][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if(multiple_prompts):

        generated_texts = []
        all_prompts = list()
        if(prompts_file in PROMPTS_MAP):
            filepath = PROMPTS_MAP[prompts_file]
        else:
            filepath = prompts_file
        with open(filepath) as f:
            for line in f:
                if(line.strip()):
                    all_prompts.append(line.strip())
        for prompt in tqdm(all_prompts):
            if(uncond):
                tokenized_cond_text = tokenizer.encode(
                    [tokenizer.bos_token], add_special_tokens=False
                )
            else:
                raw_text = prompt
                tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + raw_text,
                    add_special_tokens=False)
            unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
                model=model,
                tokenizer=tokenizer,
                context=tokenized_cond_text,
                device=device,
                num_samples=num_samples,
                bag_of_words=bag_of_words,
                bow_type=bow_type,
                discrim=discrim,
                regress=regress,
                class_label=class_label,
                length=length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl_scale,
                kl_bow=kl_bow,
                verbosity_level=verbosity_level,
                omit_file=omit_file
            )
            with open(output_file, 'a') as f:

                for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
                    generated_texts.append(pert_gen_tok_text)
                    curr_text = tokenizer.decode(pert_gen_tok_text[0][1:])
                    f.write(curr_text)
                    f.write('\n-------------------------------------------\n')


        decoded_texts = list()
        for text in generated_texts:
            decoded_texts.append(tokenizer.decode(text[0][1:]))

        return decoded_texts

    elif uncond:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + raw_text,
            add_special_tokens=False
        )
    if(verbosity_level >= REGULAR):
        print("= Prefix of sentence =")
        print(tokenizer.decode(tokenized_cond_text))
        print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        bow_type=bow_type,
        discrim=discrim,
        regress=regress,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        kl_bow=kl_bow,
        verbosity_level=verbosity_level,
        omit_file=omit_file,
        omit_seq=omit_seq
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    if verbosity_level >= REGULAR:
        print("=" * 80)
    if(verbosity_level >= REGULAR):
        print("= Unperturbed generated text =")
        print(unpert_gen_text)
        print()

    generated_texts = []

    bow_word_ids = set()
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer, omit_file)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ''
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += '{}{}{}'.format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id])
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

            if(verbosity_level >= REGULAR):
                print("= Perturbed generated text {} =".format(i + 1))
                print(pert_gen_text)
                print()
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(pert_gen_tok_text)

    # decoded_texts = list()

    # for text in generated_texts:
    #     decoded_texts.append(tokenizer.decode(text[0][1:]))


    return generated_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument("--bow_type", type=int, default=0, choices=(0,1,2,3,4,5))
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument(
        "--regress",
        "-R",
        type=str,
        default=None,
        choices=("generic"),
        help="Regressor to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--kl_bow", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument("--multiple_prompts", action="store_true",
                        help="if using prompts from a file")
    parser.add_argument("--prompts_file", type=str, 
                        help="type of prompts to be used or the filepath for custom prompts")
    parser.add_argument("--output_file", type=str, 
                        help="file where you want to write the generated sequences")
    parser.add_argument("--omit_file", type=str, 
                        help="file where you want to keep the words to be ommitted")
    parser.add_argument("--omit_seq", action="store_true",
                        help="if omitting words after every sequence")

    args = parser.parse_args()
    run_pplm_example(**vars(args))
