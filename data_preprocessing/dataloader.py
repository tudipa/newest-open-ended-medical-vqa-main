import pickle
import sys

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class medvqaDataset(Dataset):
    def __init__(self, path, split='train', like_test=False, prefix_length=2, model_type='gpt2'):
        super().__init__()
        data_path = path + split + '.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        sys.stdout.flush()
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.img_ids = data['img_ids']
        self.img_prefixes = data['img_prefix']
        self.questions = data['questions']
        self.answers = data['answers']
        self.img_paths = data['img_path']

        self.max_seqs_len = data['max_seqs_len']
        self.labels = data['class_ids']
        self.train_setting = True if (split != 'test' and like_test is False) else False
        self.prefix_len = prefix_length

    def __len__(self):
        return len(self.answers)

    def pad_sequences(self, index):
        m = [
            torch.tensor(self.tokenizer.encode('question: ')),
            torch.tensor(self.tokenizer.encode(' context:')),
            torch.tensor(self.tokenizer.encode('answer ')),
            torch.tensor(self.tokenizer.encode('<|endoftext|>')),
        ]
        # Attention mask: 1 for real tokens, 0 for padding.
        m_mask = [
            torch.ones(len(m[0])),
            torch.ones(len(m[1])),
            torch.ones(len(m[2])),
            torch.ones(len(m[3])),
        ]

        if self.train_setting:
            q = torch.tensor(self.tokenizer.encode(self.questions[index]))
            a = torch.tensor(self.tokenizer.encode(str(self.answers[index])))

            q, q_mask, leftover_tokens = self.make_padding(self.max_seqs_len[0], q, question=True)
            q_len = m[0].size(0) + q.size(0) + m[1].size(0)

            a, a_mask, _ = self.make_padding(self.max_seqs_len[1], a, leftover_tokens=leftover_tokens)
            if len((a == 0).nonzero()) != 0:
                pad_start = (a == 0).nonzero()[0]
            else:
                pad_start = []

            a = torch.cat((a, m[3])) if len(pad_start) == 0 else torch.cat((a[:pad_start], m[3], a[pad_start:]))
            q = torch.cat((m[0], q, m[1], torch.ones(self.prefix_len), m[2], a))

            q_mask = torch.cat((
                m_mask[0],
                q_mask,
                m_mask[1],
                torch.ones(self.prefix_len),
                m_mask[2],
                a_mask,
                m_mask[3],
            ))
            return q, q_mask, q_len

        # test setting
        q = torch.tensor(self.tokenizer.encode(self.questions[index]))
        q, q_mask, _ = self.make_padding_test_setting(self.max_seqs_len[0], q)
        q_len = m[0].size(0) + q.size(0) + m[1].size(0)
        q = torch.cat((m[0], q, m[1], torch.ones(self.prefix_len), m[2]))

        # Keep mask aligned with tokens for inference paths that may use it.
        q_mask = torch.cat((m_mask[0], q_mask, m_mask[1], torch.ones(self.prefix_len), m_mask[2]))
        return q, q_mask, q_len

    def make_padding(self, max_len, tokens, question=False, leftover_tokens=0):
        padding = max_len - tokens.size(0)
        if padding > 0:
            if question:
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0))
            else:
                mask = torch.cat((torch.ones(tokens.size(0)), torch.zeros(padding + leftover_tokens)))
                tokens = torch.cat((tokens, torch.zeros(padding + leftover_tokens)))

        elif padding == 0:
            if question:
                mask = torch.ones(tokens.size(0))
            else:
                mask = torch.cat((torch.ones(tokens.size(0)), torch.zeros(leftover_tokens)))
                tokens = torch.cat((tokens, torch.zeros(leftover_tokens)))

        else:  # padding < 0
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len)
            else:
                tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
                mask = torch.cat((torch.ones(max_len), torch.zeros(leftover_tokens)))

        return tokens, mask, leftover_tokens

    def make_padding_test_setting(self, max_len, tokens, do_padding=False):
        padding = max_len - tokens.size(0)
        padding_len = 0
        if padding > 0:
            if do_padding:
                mask = torch.cat((torch.ones(tokens.size(0)), torch.zeros(padding)))
                tokens = torch.cat((tokens, torch.zeros(padding)))
                padding_len = padding
            else:
                mask = torch.ones(tokens.size(0))
        elif padding == 0:
            mask = torch.ones(max_len)
        else:  # padding < 0
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
        return tokens, mask, padding_len

    def __getitem__(self, index):
        prefix = self.img_prefixes[self.img_ids[index]]
        tokens, mask, q_len = self.pad_sequences(index)
        return prefix, self.labels[index], tokens, mask, q_len