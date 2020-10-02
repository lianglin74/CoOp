import sys
import os.path as op
from azure.storage.blob import BlockBlobService
from azure.storage.common.storageclient import logger
import glob
from qd.qd_common import load_from_yaml_file
from qd.qd_common import cmd_run
from qd.qd_common import parse_iteration
import logging
from tqdm import tqdm
import os
from qd.qd_common import ensure_directory
logger.propagate = False

def create_cloud_storage(x=None, config_file=None, config=None):
    if config is not None:
        return CloudStorage(config)
    if config_file is None:
        config_file = './aux_data/configs/{}blob_account.yaml'.format(x)
    config = load_from_yaml_file(config_file)
    c = CloudStorage(config)
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
    from qd.qd_common import hash_sha1
    resume_file = '/tmp/azure.' + hash_sha1([src, dest_url]) + '.jnl'
    cmd.append('--resume')
    cmd.append(resume_file)
    if op.isdir(src):
        cmd.append('--recursive')
    cmd_run(cmd, shell=True)

def blob_upload_qdoutput(src_path, dest_path, client):
    to_copy = get_to_copy_file_for_qdoutput(src_path, dest_path)
    for f, d in to_copy:
        client.az_upload2(f, d)

def get_to_copy_file_for_qdoutput(src_path, dest_path):
    # upload all the files under src_path
    to_copy = []
    all_src_file = glob.glob(op.join(src_path, '*'))
    exclude_suffix = ['txt', 'zip']
    for f in all_src_file:
        if op.isfile(f) and not any(f.endswith(s) for s in exclude_suffix):
            to_copy.append((f, op.join(dest_path, op.basename(f))))

    # for the model and the tested files, only upload the best
    all_src_file = glob.glob(op.join(src_path, 'snapshot',
        'model_iter_*.caffemodel'))
    all_src_file.extend(glob.glob(op.join(src_path, 'snapshot',
        'model_iter_*.pt')))
    all_iter = [parse_iteration(f) for f in all_src_file]
    max_iters = max(all_iter)
    need_copy_files = [f for f, i in zip(all_src_file, all_iter) if i == max_iters]
    dest_snapshot = op.join(dest_path, 'snapshot')
    for f in need_copy_files:
        to_copy.append((f, op.join(dest_snapshot, op.basename(f))))
        if f.endswith('.caffemodel'):
            f = f.replace('.caffemodel', '.solverstate')
            to_copy.append((f, op.join(dest_snapshot, op.basename(f))))
    return to_copy

def blob_upload(src, dst, c=None):
    if c is None:
        c = create_cloud_storage('vig')
    c.az_upload2(src, dst)

def get_root_all_full_expid(full_expid_prefix, all_blob_name):
    # full_expid_prefix can be the folder of full_expid; or with some prefix so
    # that we can filter
    if all(b.startswith(full_expid_prefix + '/') for b in all_blob_name):
        root = full_expid_prefix
    else:
        root = op.dirname(full_expid_prefix)
    all_full_expid = set(b[len(root) + 1:].split('/')[0] for b in
            all_blob_name if b.startswith(root))
    return root, all_full_expid

def blob_download_all_qdoutput(prefix, c=None, out_folder='output',
        latest_only=True):
    # e.g. prefix = jianfw/work/qd_output/TaxVehicleV1_1_with_bb_e2e_faster_rcnn_R_50_FPN_1x_M_BS8_MaxIter20e_LR0.01
    # out_folder = 'output'
    if c is None:
        c = 'vig'
    c = create_cloud_storage(c)
    from datetime import datetime, timedelta
    creation_time_larger_than = datetime.utcnow() - timedelta(days=14)
    all_blob_name = list(c.list_blob_names(
        prefix,
        creation_time_larger_than=creation_time_larger_than))
    root, all_full_expid = get_root_all_full_expid(prefix, all_blob_name)
    for full_expid in all_full_expid:
        logging.info(full_expid)
        src_path = op.join(root, full_expid)
        target_folder = op.join(out_folder, full_expid)
        c.blob_download_qdoutput(
            src_path, target_folder,
            latest_only=latest_only,
            creation_time_larger_than=creation_time_larger_than,
        )

def get_azcopy():
    # this is v10
    azcopy = op.expanduser('~/code/azcopy/azcopy')
    if not op.isfile(azcopy):
        azcopy = 'azcopy'
    return azcopy

def get_leaf_names(all_fname):
    # build the tree first
    from ete3 import Tree
    root = Tree()
    for fname in all_fname:
        components = fname.split('/')
        curr = root
        for com in components:
            currs = [c for c in curr.children if c.name == com]
            if len(currs) == 0:
                curr = curr.add_child(name=com)
            else:
                assert len(currs) == 1
                curr = currs[0]
    result = []
    for node in root.iter_leaves():
        ans = [s.name for s in node.get_ancestors()[:-1]]
        ans.insert(0, node.name)
        result.append('/'.join([a for a in ans[::-1]]))
    return result

def blob_download_qdoutput(src_path, target_folder):
    c = create_cloud_storage('vig')
    c.blob_download_qdoutput(src_path, target_folder)

class CloudStorage(object):
    def __init__(self, config=None):
        if config is None:
            config_file = 'aux_data/configs/azure_blob_account.yaml'
            config = load_from_yaml_file(config_file)
        account_name = config['account_name']
        account_key = config.get('account_key')
        self.sas_token = config.get('sas_token')
        self.container_name = config['container_name']
        self.account_name = account_name
        self.account_key = account_key
        self._block_blob_service = None

    @property
    def block_blob_service(self):
        if self._block_blob_service is None:
            self._block_blob_service = BlockBlobService(
                    account_name=self.account_name,
                    account_key=self.account_key,
                    sas_token=self.sas_token)
        return self._block_blob_service

    def list_blob_names(self, prefix=None, creation_time_larger_than=None):
        if creation_time_larger_than is not None:
            creation_time_larger_than = creation_time_larger_than.timestamp()
            return (b.name for b in self.block_blob_service.list_blobs(self.container_name,
                                                                       prefix=prefix)
                    if b.properties.creation_time.timestamp() > creation_time_larger_than)
        else:
            return self.block_blob_service.list_blob_names(self.container_name, prefix=prefix)

    def upload_stream(self, s, name, force=False):
        if not force and self.block_blob_service.exists(self.container_name,
                name):
            return self.block_blob_service.make_blob_url(
                    self.container_name,
                    name)
        else:
            if sys.version_info.major == 3 and type(s) is bytes:
                self.block_blob_service.create_blob_from_bytes(
                        self.container_name,
                        name,
                        s)
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
        if target_file.startswith('/'):
            logging.info('remove strarting slash for {}'.format(target_file))
            target_file = target_file[1:]
        self.block_blob_service.create_blob_from_path(
                self.container_name,
                target_file,
                src_file,
                max_connections=8,
                progress_callback=upload_callback)

    def az_upload(self, src_dir, dest_dir):
        # this is using the old version of azcopy. prefer to use az_upload2
        dest_url = op.join('https://{}.blob.core.windows.net'.format(self.account_name),
                self.container_name, dest_dir)
        if self.account_key:
            azcopy_upload(src_dir, dest_url, self.account_key)
        else:
            raise Exception

    def az_sync(self, src_dir, dest_dir):
        assert self.sas_token
        cmd = []
        cmd.append(get_azcopy())
        cmd.append('sync')
        cmd.append(op.realpath(src_dir))
        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        if dest_dir.startswith('/'):
            dest_dir = dest_dir[1:]
        url = op.join(url, self.container_name, dest_dir)
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)
        if op.isdir(src_dir):
            cmd.append('--recursive')
        cmd_run(cmd)
        return data_url, url

    def az_upload2(self, src_dir, dest_dir, sync=False):
        assert self.sas_token
        cmd = []
        cmd.append(get_azcopy())
        if sync:
            cmd.append('sync')
        else:
            cmd.append('cp')
        cmd.append(op.realpath(src_dir))
        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        if dest_dir.startswith('/'):
            dest_dir = dest_dir[1:]
        url = op.join(url, self.container_name, dest_dir)
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)
        if op.isdir(src_dir):
            cmd.append('--recursive')
        cmd_run(cmd)
        return data_url, url

    def az_download_all(self, remote_path, local_path):
        all_blob_name = list(self.list_blob_names(remote_path))
        all_blob_name = get_leaf_names(all_blob_name)
        for blob_name in all_blob_name:
            target_file = blob_name.replace(remote_path, local_path)
            if not op.isfile(target_file):
                self.az_download(blob_name, target_file,
                        sync=False)
    def is_folder(self, remote_path):
        is_folder = False
        for x in self.list_blob_names(remote_path + '/'):
            is_folder = True
            break
        return is_folder

    def az_download_each(self, remote_path, local_path):
        # if it is a folder, we will download each file individually
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        if remote_path.endswith('/'):
            remote_path = remote_path[:-1]
        is_folder = self.is_folder(remote_path)
        if is_folder:
            all_remote_file = self.list_blob_names(remote_path + '/')
            all_local_file = [op.join(local_path, r[len(remote_path) + 1:])
                              for r in all_remote_file]
        else:
            all_remote_file = [remote_path]
            all_local_file = [local_path]
        for r, l in zip(all_remote_file, all_local_file):
            self.az_download(r, l, sync=True)

    def az_download(self,
                    remote_path,
                    local_path,
                    sync=True,
                    is_folder=None,
                    tmp_first=True,
                    ):
        if is_folder is not None:
            logging.warn('no need to specify is_folder. deprecating')
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        if remote_path.endswith('/'):
            remote_path = remote_path[:-1]
        is_folder = self.is_folder(remote_path)
        if is_folder:
            if op.isdir(local_path) and tmp_first:
                if len(os.listdir(local_path)) > 0:
                    logging.error('ignore to download from {} to {}'
                                  ' since destination is not empty'.format(
                                      remote_path,
                                      local_path,
                                  ))
                    return
                from qd.qd_common import ensure_remove_dir
                ensure_remove_dir(local_path)
        else:
            if sync:
                if tmp_first:
                    sync = False
                elif not op.isfile(local_path):
                    sync = False
        ensure_directory(op.dirname(local_path))
        origin_local_path = local_path
        if tmp_first:
            local_path = local_path + '.tmp'
        ensure_directory(op.dirname(local_path))
        assert self.sas_token
        cmd = []
        cmd.append(op.expanduser('~/code/azcopy/azcopy'))
        if sync:
            cmd.append('sync')
        else:
            cmd.append('cp')
        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        url = '/'.join([url, self.container_name, remote_path])
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)
        cmd.append(op.realpath(local_path))
        if is_folder:
            cmd.append('--recursive')
            if sync:
                # azcopy's requirement
                ensure_directory(local_path)
        cmd_run(cmd)
        if tmp_first:
            os.rename(local_path, origin_local_path)
        return data_url, url

    def download_to_path(self, blob_name, local_path,
            max_connections=2):
        dir_path = op.dirname(local_path)
        from qd.qd_common import get_file_size
        if op.isfile(dir_path) and get_file_size(dir_path) == 0:
            os.remove(dir_path)
        ensure_directory(dir_path)
        tmp_local_path = local_path + '.tmp'
        pbar = {}
        def progress_callback(curr, total):
            if len(pbar) == 0:
                pbar['tqdm'] = tqdm(total=total, unit_scale=True)
                pbar['last'] = 0
                pbar['count'] = 0
            pbar['count'] += 1
            if pbar['count'] > 100:
                pbar['tqdm'].update(curr - pbar['last'])
                pbar['last'] = curr
                pbar['count'] = 0
        self.block_blob_service.get_blob_to_path(self.container_name,
                blob_name, tmp_local_path,
                progress_callback=progress_callback,
                max_connections=max_connections)
        os.rename(tmp_local_path, local_path)

    def download_to_stream(self, blob_name, s):
        self.block_blob_service.get_blob_to_stream(self.container_name,
                blob_name, s)

    def exists(self, path):
        return self.block_blob_service.exists(
                self.container_name, path)

    def blob_download_qdoutput(self, src_path, target_folder, latest_only=True,
                               creation_time_larger_than=None):
        def is_in_snapshot(b):
            return op.basename(op.dirname(b)) == 'snapshot'
        all_blob_name = list(self.list_blob_names(
            src_path,
            creation_time_larger_than))
        all_blob_name = get_leaf_names(all_blob_name)
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
        iters = [parse_iteration(f) for f in in_snapshot_blobs]
        if len(iters) > 0 and latest_only:
            max_iters = max(iters)
            need_download_blobs.extend([f for f, i in zip(in_snapshot_blobs, iters) if i ==
                    max_iters])
        need_download_blobs.extend([f for f, i in zip(in_snapshot_blobs, iters) if
                i == -2])
        to_remove = []
        for i, b1 in enumerate(need_download_blobs):
            for b2 in need_download_blobs:
                if b1 != b2 and b2.startswith(b1) and b2.startswith(b1 + '/'):
                    to_remove.append(b1)
                    break
        for t in to_remove:
            need_download_blobs.remove(t)
        need_download_blobs = [t for t in need_download_blobs
            if not t.endswith('.tmp')]
        f_target_f = [(f, f.replace(src_path, target_folder)) for f in
            need_download_blobs]
        f_target_f = [(f, target_f) for f, target_f in f_target_f if not op.isfile(target_f)]


        for f, target_f in tqdm(f_target_f):
            if not op.isfile(target_f):
                if len(f) > 0:
                    logging.info('download {} to {}'.format(f, target_f))
                    try:
                        self.az_download(f, target_f, sync=False)
                    except:
                        pass
                    #self.download_to_path(f, target_f)

if __name__ == '__main__':
    from qd.qd_common import init_logging
    from pprint import pformat
    from qd.qd_common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
