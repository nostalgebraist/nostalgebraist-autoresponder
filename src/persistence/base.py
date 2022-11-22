import os
import jsonlines
import shutil

from tqdm.auto import tqdm


class SelfArchivingJsonlStore:
    def __init__(
        self,
        name: str,
        max_entries_hot: int,
        archival_min_batch_size: int,
        directory='data',
        loads=None,
        dumps=None,
    ):
        self.name = name
        self.directory = directory
        self.max_entries_hot = max_entries_hot
        self.archival_min_batch_size = archival_min_batch_size
        self.loads = loads
        self.dumps = dumps

        self.path, self.path_coldstore, self.path_backup = self.format_paths(self.directory, self.name)

        self.data = None
        self.count_entries()

    @staticmethod
    def format_paths(directory, name):
        path = os.path.join(directory, name + '.jsonl')
        path_coldstore = os.path.join(directory, name + '__coldstore.jsonl')
        path_backup = os.path.join(directory, name + '__backup.jsonl')
        return path, path_coldstore, path_backup

    @staticmethod
    def create(data: list,
               name: str,
               max_entries_hot: int,
               archival_min_batch_size: int,
               directory='data',
               loads=None,
               dumps=None,
        ):
        path, path_coldstore, _ = SelfArchivingJsonlStore.format_paths(directory, name)

        with jsonlines.open(path, 'w', dumps=dumps) as f:
            for entry in tqdm(data, mininterval=0.25):
                f.write(entry)

        with open(path_coldstore, 'w') as f:
            pass

        return SelfArchivingJsonlStore(
            name=name,
            max_entries_hot=max_entries_hot,
            archival_min_batch_size=archival_min_batch_size,
            directory=directory,
            loads=loads,
            dumps=dumps,
        )

    @property
    def data_available(self):
        return self.data is not None

    def count_entries(self):
        n_entries = 0
        with open(self.path) as f:
            for line in f:
                n_entries += 1
        self.n_entries = n_entries

    def load_data(self):
        with jsonlines.open(self.path, loads=self.loads) as f:
            self.data = list(f)

    def write_entry(self, entry, do_backup=True):
        with jsonlines.open(self.path, mode='a', dumps=self.dumps) as f:
            f.write(entry)

        self.n_entries += 1

        if self.data_available:
            self.data.append(entry)

        self.maybe_archive(do_backup=do_backup)

    def needs_archive(self):
        return self.n_entries > (self.max_entries_hot + self.archival_min_batch_size)

    def maybe_archive(self, do_backup=True):
        if not self.needs_archive():
            return

        with open(self.path) as f:
            lines = f.readlines()  # includes \n at end of each line

        batch_size = len(lines) - self.max_entries_hot

        batch, rest = lines[:batch_size], lines[batch_size:]

        with open(self.path_coldstore, 'a') as f:
            f.writelines(batch)  # does not add \ns

        if do_backup:
            shutil.copy(self.path, self.path_backup)

        with open(self.path, 'w') as f:
            f.writelines(rest)  # does not add \n

        self.n_entries -= len(batch)


class CallbackDict(dict):
    def set_callback(self, fn):
        self.callback = fn

    def _do_callback(self, mapping):
        for key, value in mapping.items():
            if key not in self or self.get(key) != value:
                self.callback(key, value)

    def __setitem__(self, key, value):
        self._do_callback({key: value})
        super().__setitem__(key, value)

    def update(self, other, other2=None):
        if other2 is not None:
            raise NotImplementedError

        self._do_callback(other)
        super().update(other)

    def copy(self):
        copy = super().copy()
        return CallbackDict.from_dict(copy, self.callback)

    def __delitem__(*args, **kwargs):
        raise NotImplementedError

    def pop(*args, **kwargs):
        raise NotImplementedError

    def popitem(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict, callback):
        cd = CallbackDict(d)
        cd.set_callback(callback)
        return cd
