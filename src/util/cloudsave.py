import time
import pickle

import smart_open.gcs
from smart_open import open


CLOUDSAVE_BUCKET = "nost-us"


def resilient_pickle_save(obj, path, backup_path):
    tries = 0
    done = False

    while not done:
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            done = True
        except smart_open.gcs.UploadFailedError:
            # TODO: delete the object in gcs programmatically here.
            # as is, it will probably fail endlessly b/c it's partway through a multipart upload
            # however, at least that keeps the obj in memory until i can manually delete it
            sleeptime = 2**tries
            print(f"gcs upload failed for {path}. {tries} tries so far.  sleeping {sleeptime}s")
            if tries == 0:
                print(f"saving backup to {backup_path}s")
                with open(backup_path, "wb") as f:
                    pickle.dump(obj, f)
            time.sleep(sleeptime)
            tries += 1


# TODO: maybe DRY (but also be safe)
def resilient_pickle_load(path):
    tries = 0
    done = False

    while not done:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            done = True
        except smart_open.gcs.UploadFailedError:
            sleeptime = 2**tries
            print(f"gcs download failed for {path}. {tries} tries so far.  sleeping {sleeptime}s")
            time.sleep(sleeptime)
            tries += 1
    return obj
