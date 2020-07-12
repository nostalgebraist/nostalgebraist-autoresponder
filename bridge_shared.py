import hashlib


def bridge_service_unique_id(url, data):
    unique_string = url + str({k: data[k] for k in sorted(data.keys())})
    hashed = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
    return hashed
