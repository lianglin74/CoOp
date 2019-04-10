import os.path as op
from azure.storage.blob import BlockBlobService
from azure.storage.common.storageclient import logger
from .qd_common import load_from_yaml_file
from .qd_common import cmd_run
import logging
from tqdm import tqdm
logger.propagate = False

def create_cloud_storage(x):
    c = CloudStorage(load_from_yaml_file('./aux_data/configs/{}blob_account.yaml'.format(x)))
    return c

def azcopy_upload(src, dest_url, dest_key):
    cmd = ['azcopy', '--source',
            src,
            '--destination',
            dest_url,
            '--exclude-older',
            '--dest-key',
            "{}".format(dest_key),
            '--quiet',
            '--parallel-level',
            '32']
    from qd.process_tsv import hash_sha1
    resume_file = '/tmp/azure.' + hash_sha1([src, dest_url]) + '.jnl'
    cmd.append('--resume')
    cmd.append(resume_file)
    if op.isdir(src):
        cmd.append('--recursive')
    cmd_run(cmd, shell=True)

def blob_download_all_qdoutput(prefix):
    c = create_cloud_storage('vig')
    all_blob_name = list(c.list_blob_names(prefix))
    def parse_blob_root_folder(b):
        b = b.replace(prefix, '')
        if b.startswith('/'):
            b = b[1:]
        while '/' in b:
            b = op.dirname(b)
        return b
    all_full_expid = set([parse_blob_root_folder(b) for b in all_blob_name])
    for full_expid in all_full_expid:
        logging.info(full_expid)
        src_path = op.join(prefix, full_expid)
        target_folder = op.join('output', full_expid)
        blob_download_qdoutput(src_path, target_folder)

def blob_download_qdoutput(src_path, target_folder):
    c = create_cloud_storage('vig')
    def is_in_snapshot(b):
        return op.basename(op.dirname(b)) == 'snapshot'
    all_blob_name = list(c.list_blob_names(src_path))
    in_snapshot_blobs = [b for b in all_blob_name if is_in_snapshot(b)]
    not_in_snapshot_blobs = [b for b in all_blob_name if not is_in_snapshot(b)]
    try:
        not_in_snapshot_blobs.remove(src_path)
    except:
        pass
    try:
        not_in_snapshot_blobs.remove(src_path + '/snapshot')
    except:
        pass
    need_download_blobs = []
    need_download_blobs.extend(not_in_snapshot_blobs)
    from qd.qd_common import parse_iteration
    iters = [parse_iteration(f) for f in in_snapshot_blobs]
    if len(iters) > 0:
        max_iters = max(iters)
        need_download_blobs.extend([f for f, i in zip(in_snapshot_blobs, iters) if i ==
                max_iters])
    for f in tqdm(need_download_blobs):
        target_f = f.replace(src_path, target_folder)
        if not op.isfile(target_f):
            c.download_to_path(f, target_f)

class CloudStorage(object):
    def __init__(self, config=None):
        if config is None:
            config_file = 'aux_data/configs/azure_blob_account.yaml'
            config = load_from_yaml_file(config_file)
        account_name = config['account_name']
        account_key = config.get('account_key')
        self.sas_token = config.get('sas_token')
        self.container_name = config['container_name']

        self.block_blob_service = BlockBlobService(account_name=account_name,
                account_key=account_key, sas_token=self.sas_token)

        self.account_name = account_name
        self.account_key = account_key

    def list_blob_names(self, prefix):
        return self.block_blob_service.list_blob_names(self.container_name,
                prefix=prefix)

    def upload_stream(self, s, name, force=False):
        if not force and self.block_blob_service.exists(self.container_name,
                name):
            return self.block_blob_service.make_blob_url(
                    self.container_name,
                    name)
        else:
            self.block_blob_service.create_blob_from_stream(
                    self.container_name,
                    name,
                    s)
            return self.block_blob_service.make_blob_url(
                    self.container_name,
                    name)

    def upload_folder(self, folder, target_prefix):
        def remove_tailing(x):
            if x.endswith('/') or x.endswith('\\'):
                x = x[:-1]
            return x
        folder = remove_tailing(folder)
        target_prefix = remove_tailing(target_prefix)
        import os
        for root, dirs, files in os.walk(folder):
            for f in files:
                src_file = op.join(root, f)
                assert src_file.startswith(folder)
                target_file = src_file.replace(folder, target_prefix)
                self.upload_file(src_file, target_file)
            for d in dirs:
                self.upload_folder(op.join(root, d),
                        op.join(target_prefix, d))

    def upload_file(self, src_file, target_file):
        logging.info('uploading {} to {}'.format(src_file, target_file))
        #import time
        #start_time = time.time()
        bar = [None]
        last = [0]
        def upload_callback(curr, total):
            if total < 1024 ** 3:
                return
            if bar[0] is None:
                bar[0] = tqdm(total=total, unit_scale=True)
            bar[0].update(curr - last[0])
            last[0] = curr

        self.block_blob_service.create_blob_from_path(
                self.container_name,
                target_file,
                src_file,
                max_connections=8,
                progress_callback=upload_callback)

    def az_upload(self, src_dir, dest_dir):
        dest_url = op.join('https://{}.blob.core.windows.net'.format(self.account_name),
                self.container_name, dest_dir)
        if self.account_key:
            azcopy_upload(src_dir, dest_url, self.account_key)
        else:
            raise Exception

    def az_upload2(self, src_dir, dest_dir):
        assert self.sas_token
        cmd = []
        cmd.append(op.expanduser('~/code/azcopy/azcopy'))
        cmd.append('cp')
        cmd.append(op.realpath(src_dir))
        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        url = op.join(url, self.container_name, dest_dir)
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)
        if op.isdir(src_dir):
            cmd.append('--recursive')
        cmd_run(cmd)
        return data_url, url

    def download_to_path(self, blob_name, local_path):
        from qd.qd_common import ensure_directory
        ensure_directory(op.dirname(local_path))
        self.block_blob_service.get_blob_to_path(self.container_name,
                blob_name, local_path)

    def download_to_stream(self, blob_name, s):
        self.block_blob_service.get_blob_to_stream(self.container_name,
                blob_name, s)

    def exists(self, path):
        return self.block_blob_service.exists(
                self.container_name, path)

