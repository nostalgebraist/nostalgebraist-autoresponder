import glob
import numpy as np
import os
import tensorflow as tf
import tqdm


def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    accum_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text = fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
                accum_text += raw_text

            if accum_text and len(accum_text) >= combine:
                tokens = np.stack(enc.encode(accum_text))
                token_chunks.append(tokens)
                accum_text = ''

    if accum_text:
        tokens = np.stack(enc.encode(accum_text))
        token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

def contbyte(b):
  n = ord(b)
  # https://en.wikipedia.org/wiki/UTF-8#Description
  return n & (1 << 7) and not n & (1 << 6)

def nextchar(f):
  b = f.read(1)
  # skip leading continuation bytes
  while b and contbyte(b):
    b = f.read(1)
  # append up to three continuation bytes
  for _ in range(3):
    c = f.read(1)
    if c and contbyte(c):
      b += c
    elif c:
      # not a continuation byte; back up one character
      f.seek(-1, 1)
      break
  return b.decode('utf-8')

def nextchars(f, n):
  s = ""
  for i in range(n):
    c = nextchar(f)
    if c is None:
      break
    s += c
  if len(s) > 0:
    return s

def grab_tokens(f, enc, n):
  n += 4
  count = n
  line = nextchars(f, count)
  if not line:
    return [], None
  tokens = enc.encode(line)
  while len(tokens) < n:
    count *= 2
    l = nextchars(f, count)
    if not l:
      break
    line += l
    tokens = enc.encode(line)
  # skip the first couple tokens in case we started in the middle of a
  # word (which is the likely case for a random seek anywhere in the
  # dataset).
  return tokens[3:], line

class TextSampler(object):
  def __init__(self, fp, enc, seed=None, verbose=False):
    if isinstance(fp, str):
      fp = open(fp, 'rb')
    self.fp = fp
    fp.seek(0, 2)
    self.total_size = fp.tell()
    self.rs = np.random.RandomState(seed=seed)
    self.enc = enc
    self.verbose = verbose

  def sample(self, length):
    attempts = 0
    while True:
      attempts += 1
      if attempts > 10:
        print('Could not sample from dataset; too small?')
        return None
      index = self.rs.randint(0, self.total_size)
      self.fp.seek(index, 0)
      tokens, line = grab_tokens(self.fp, self.enc, length)
      if len(tokens) >= length:
        if self.verbose:
          line = self.enc.decode(tokens)
          print(repr(line))
        return tokens[0:length]

def random_indices(n, rs=np.random):
    for ix in rs.choice(range(n), n, replace=False):
        yield ix

def infinite_random_dataset_indices(data_size, sample_len, rs=np.random, verbose=False):
    while True:
        offset = rs.randint(low=0, high=sample_len)
        max_ix = data_size // sample_len
        generation = random_indices(max_ix, rs=rs)
        if verbose:
            print("reset")
            print(f"offset: {offset}")
        for ix in generation:
            yield sample_len*ix + offset

class RobSampler:
    def __init__(self, dataset, sample_length, verbose=False, seed=None):
        self.dataset = dataset
        self.sample_length = sample_length
        self.total_size = len(self.dataset)
        self.verbose = verbose

        self.rs = np.random.RandomState(seed=seed)
        self.sample_generator = infinite_random_dataset_indices(
            data_size=self.total_size, sample_len=self.sample_length, rs=self.rs,
            verbose=self.verbose
        )

        self.dataset_with_wrap = np.concatenate([self.dataset, self.dataset[:self.sample_length]])

    def _get_n_indices(self, n: int):
        results = []
        while len(results) < n:
            results.append(next(self.sample_generator))
        return results

    def sample_n(self, n: int):
        ixs = self._get_n_indices(n)
        results = [self.dataset_with_wrap[ix:ix+self.sample_length] for ix in ixs]
        return results

    def sample(self):
        return self.sample_n(1)[0]

class RobSamplerInterface:
    def __init__(self, chunks, seed=None, verbose=False):
        self.samplers = {}
        self.dataset = np.concatenate(chunks)  # don't care about chunks
        self.total_size = len(self.dataset)
        self.seed = seed
        self.verbose = verbose

    def sample(self, length):
        int_len = int(length)
        if int_len not in self.samplers:
            self.samplers[int_len] = RobSampler(dataset=self.dataset,
                                                sample_length=length,
                                                seed=self.seed,
                                                verbose=self.verbose)
        return self.samplers[int_len].sample()
