import os.path as op
from qd.qd_common import retry_agent, cmd_run
from qd.cloud_storage import create_cloud_storage
import logging


def get_philly_fs():
    return op.expanduser('~/code/philly/philly-fs.bash')

def infer_type(vc, cluster):
    all_prem = [('resrchvc', 'gcr'),
            ('pnrsy', 'rr1')]
    all_ap = [('input', 'philly-prod-cy4')]
    all_azure = [
            ('input', 'sc2'),
            ('input', 'eu2'),
            ('input', 'wu1'),
            ('pnrsy', 'eu1'),
            ('pnrsy', 'sc1'),
            ('pnrsy', 'et1')]
    if any(v == vc and c == cluster for v, c in all_prem):
        return 'prem'
    elif any(v == vc and c == cluster for v, c in all_ap):
        return 'ap'
    elif any(v == vc and c == cluster for v, c in all_azure):
        return 'azure'
    assert False


def philly_input_run(cmd, return_output=False):
    i = 0
    while True:
        try:
            working_dir = op.expanduser('~/code/philly/philly-fs-ap-v5')
            logging.info('working dir: {}'.format(working_dir))
            result = cmd_run(cmd, env={'PHILLY_USER': 'jianfw',
                'PHILLY_VC': 'input',
                'PHILLY_CLUSTER_HDFS_HOST': '131.253.41.35',
                'PHILLY_CLUSTER_HDFS_PORT':
                '81/nn/http/hnn.philly-prod-cy4.cy4.ap.gbl/50070'},
                working_dir=working_dir,
                return_output=return_output)
            logging.info('succeed: {}'.format(' '.join(cmd)))
            return result
        except Exception:
            logging.info('fails: try {}-th time'.format(i))
            i = i + 1
            import time
            time.sleep(5)

def philly_run(sub_cmd, vc, cluster, return_output=False, extra_env={}):
    '''
    sub_cmd = ['-ls', 'path']
    '''
    t = infer_type(vc, cluster)
    if t == 'prem' or t == 'ap':
        disk_type = 'hdfs'
    else:
        disk_type = 'gfs'
    folder_prefix = '{}://{}/{}/'.format(disk_type, cluster, vc)
    if sub_cmd[0] == '-mkdir' or \
            sub_cmd[0] == '-ls':
        assert len(sub_cmd) == 2
        sub_cmd[1] = '{}{}'.format(folder_prefix, sub_cmd[1])
    elif sub_cmd[0] == '-cp':
        '''
        sub_cmd[-1] is the index of which one is in philly
        '''
        sub_cmd[sub_cmd[-1]] = '{}{}'.format(folder_prefix,
                sub_cmd[sub_cmd[-1]])
        del sub_cmd[-1]
    else:
        assert False
    if t == 'prem' or t == 'azure':
        philly_cmd = get_philly_fs()
        cmd = []
        cmd.append(philly_cmd)
        cmd.extend(sub_cmd)
        env = {'PHILLY_VC': vc}
        env.update(extra_env)
        return retry_agent(cmd_run, cmd,
                return_output=return_output, env=env)
    elif t == 'ap':
        cmd = []
        cmd.append('./philly-fs')
        cmd.extend(sub_cmd)
        output = philly_input_run(cmd, return_output)
        if output:
            logging.info(output)
        return output

def philly_mkdir(dest_dir, vc, cluster):
    sub_cmd = ['-mkdir', dest_dir]
    philly_run(sub_cmd, vc, cluster)

def upload_through_blob(src_dir, dest_dir, vc, cluster):
    assert len(dest_dir) > 0 and dest_dir[0] != '/' and dest_dir[0] != '\\'
    account = 'vig'
    dest_url = op.join('https://{}.blob.core.windows.net/data',
            account,
            dest_dir,
            op.basename(src_dir))

    c = create_cloud_storage(account)
    dest_url, _ = c.az_upload2(src_dir, op.join(dest_dir, op.basename(src_dir)))

    env = {'AZURE_STORAGE_ACCESS_KEY': c.account_key}

    sub_cmd = ['-cp', '-r', dest_url, op.join(dest_dir, op.basename(src_dir)), 3]
    philly_run(sub_cmd, vc, cluster, extra_env=env)


def philly_upload_dir(src_dir, dest_dir, vc='input', cluster='philly-prod-cy4',
        blob=True):
    philly_mkdir(dest_dir, vc, cluster)
    t = infer_type(vc, cluster)
    if t == 'prem' or t == 'ap':
        disk_type = 'hdfs'
    else:
        disk_type = 'gfs'
    src_dir = op.realpath(src_dir)
    folder_prefix = '{}://{}/{}/'.format(disk_type, cluster, vc)
    if t == 'prem' or t == 'azure':
        if not blob:
            philly_cmd = op.expanduser('~/code/philly/philly-fs.bash')
            cmd = []
            cmd.append(philly_cmd)
            cmd.append('-cp')
            cmd.append('-r')
            cmd.append(src_dir)
            cmd.append('{}{}'.format(folder_prefix, dest_dir))
            retry_agent(cmd_run, cmd, env={'PHILLY_VC': vc})
        else:
            upload_through_blob(src_dir, dest_dir, vc, cluster)
    elif t == 'ap':
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-cp')
        cmd.append('-r')
        cmd.append(src_dir)
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        philly_input_run(cmd)

