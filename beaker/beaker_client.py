import os
import re
import requests
from tqdm import tqdm
import math
from urllib.request import urlopen

# TODO:
# Sanitize base address, add https if needed.
# Authenticate all requests with token if one is provided.
# Clean up exceptions (raise_for_status calls).
# Support name resolution.

_id_pattern = re.compile(r'^\w\w_[a-z0-9]{12}$')


def download_from_url(url, dst, headers=None, chunksize=1024):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(requests.head(url).headers['Content-Length'])
    req = requests.get(url, headers=headers, stream=True)
    with(open(dst, 'ab')) as f:
        pbar = tqdm(req.iter_content(chunk_size=chunksize),
                    unit='B',
                    unit_scale=True,
                    total=int(file_size),
                    desc=dst)
        for chunk in pbar:
            if chunk:
                f.write(chunk)
    return req.status_code


class Client:
    def __init__(self, address, token=None):
        self.address = address
        self.token = token
        if not token:
            self.token = os.environ.get('BEAKER_TOKEN')

    def dataset(self, ref):
        return Dataset(self, self._canonicalize(ref))

    def experiment(self, ref):
        return Experiment(self, self._canonicalize(ref))

    def group(self, ref):
        return Group(self, self._canonicalize(ref))

    def task(self, id):
        return Task(self, id)

    def who_am_i(self):
        if not self.token:
            raise 'User is not authenticated'

        headers = {'Authorization': 'Bearer ' + self.token}
        resp = requests.get('/'.join((self.address, 'api/v3/auth/whoami')), headers=headers)
        if resp.status_code >= 400:
            resp.raise_for_status()
        u = resp.json()
        return User(id=u['id'], name=u['name'], display_name=u['displayName'])

    def _canonicalize(self, ref) -> str:
        # If this looks like an ID, assume that it is.
        if _id_pattern.match(ref):
            return ref

        # Ensure name references are fully qualified.
        parts = ref.split('/')
        if len(parts) > 2:
            raise 'Invalid reference: ' + ref
        if len(parts) > 1:
            return ref
        return '{}/{}'.format(self.who_am_i().name, ref)


class Dataset:
    def __init__(self, client, ref):
        self.client = client
        self.ref = ref
        self._is_file = None

    def __repr__(self):
        return self.ref

    @property
    def is_file(self):
        if self._is_file is None:
            resp = requests.get('/'.join((self._baseurl, 'manifest')))
            if resp.status_code >= 400:
                resp.raise_for_status()
            self._is_file = resp.json().get('single_file', False)
        return self._is_file

    def file(self, path):
        return File(self, path)

    @property
    def files(self):
        resp = requests.get('/'.join((self._baseurl, 'manifest')))
        if resp.status_code >= 400:
            resp.raise_for_status()
        self._is_file = resp.json().get('single_file', False)
        return [
            File(self, f['file'], size=f['size'], updated=f['time_last_modified'])
            for f in resp.json()['files']
        ]

    @property
    def _baseurl(self):
        return '/'.join((self.client.address, 'api/v3/datasets', self.ref))


class Experiment:
    def __init__(self, client, ref):
        self.client = client
        self.ref = ref

    def __repr__(self):
        return self.ref

    def tasks(self):
        resp = requests.get(self._baseurl)
        if resp.status_code >= 400:
            resp.raise_for_status()

        result = []
        for task in resp.json()['nodes']:
            result.append(Task(self.client, task['task_id']))
        return result

    @property
    def _baseurl(self):
        return '/'.join((self.client.address, 'api/v3/experiments', self.ref))


class File:
    def __init__(self, dataset, path, size=None, updated=None):
        self.dataset = dataset
        self.path = path
        self.size = size
        self.updated = updated
        self.headers = {'Authorization': 'Bearer ' + self.dataset.client.token}

    def __repr__(self):
        return self.path

    def download(self, output_dir) -> bytes:
        url = '/'.join((self.dataset._baseurl, 'files', self.path))
        basename = os.path.basename(self.path)
        dst = os.path.join(output_dir, basename)
        status = download_from_url(url, dst=dst, headers=self.headers)
        return status


class Group:
    def __init__(self, client, ref):
        self.client = client
        self.ref = ref

    def __repr__(self):
        return self.ref

    def experiments(self):
        resp = requests.get('/'.join((self._baseurl, 'experiments')))
        if resp.status_code >= 400:
            resp.raise_for_status()

        result = []
        for exp_id in resp.json():
            result.append(Experiment(self.client, exp_id))
        return result

    @property
    def _baseurl(self):
        return '/'.join((self.client.address, 'api/v3/groups', self.ref))


class Task:
    def __init__(self, client, id):
        self.client = client
        self.id = id
        self._result = None

    def __repr__(self):
        return self.id

    @property
    def result(self):
        if not self._result:
            self._update()
        return self._result

    @property
    def _baseurl(self):
        return '/'.join((self.client.address, 'api/v3/tasks', self.id))

    def _update(self):
        resp = requests.get(self._baseurl)
        if resp.status_code >= 400:
            resp.raise_for_status()
        body = resp.json()
        self._result = Dataset(self.client, body['result_id'])


class User:
    def __init__(self, id, name, display_name):
        self.id = id
        self.name = name
        self.display_name = display_name
