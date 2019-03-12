import os.path as op
from azure.storage.blob import BlockBlobService
from azure.storage.common.storageclient import logger
from qd_common import load_from_yaml_file
import logging
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
    from qd.qd_common import cmd_run
    cmd_run(cmd, shell=True)

class CloudStorage(object):
    def __init__(self, config=None):
        if config is None:
            config_file = 'aux_data/configs/azure_blob_account.yaml'
            config = load_from_yaml_file(config_file)
        account_name = config['account_name']
        account_key = config.get('account_key')
        sas_token = config.get('sas_token')
        self.container_name = config['container_name']

        self.block_blob_service = BlockBlobService(account_name=account_name,
                account_key=account_key, sas_token=sas_token)

        self.account_name = account_name
        self.account_key = account_key

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
            import tqdm
            if bar[0] is None:
                bar[0] = tqdm.tqdm(total=total, unit_scale=True)
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

    def download_to_path(self, blob_name, local_path):
        self.block_blob_service.get_blob_to_path(self.container_name,
                blob_name, local_path)

    def download_to_stream(self, blob_name, s):
        self.block_blob_service.get_blob_to_stream(self.container_name,
                blob_name, s)

    def exists(self, path):
        return self.block_blob_service.exists(
                self.container_name, path)

