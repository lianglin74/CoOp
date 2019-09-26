from pprint import pformat
import subprocess as sp
import sys
import psutil
import re
import logging
from qd.qd_common import cmd_run, remote_run, collect_process_info
from qd.qd_common import init_logging
from qd.qd_common import try_once


@try_once
def try_get_gpu_info():
    return get_gpu_info()

def get_gpu_info():
    r = cmd_run(['nvidia-smi'], return_output=True)
    ps = parse_gpu_usage(r)
    return ps

@try_once
def try_get_nvidia_smi_out():
    return get_nvidia_smi_out()

def get_nvidia_smi_out():
    return cmd_run(['nvidia-smi'], return_output=True)

def gpu_available(all_resource):
    if all(all(g == -1 for g in gpus) for ssh_info, gpus in all_resource):
        return True

    machines = {}
    for ssh_info, _ in all_resource:
        if ssh_info == {}:
            machines['_'] = {}
        elif ssh_info['ip'] not in machines:
            machines[ssh_info['ip']] = ssh_info
    machine_usage = {}
    for ip in machines:
        m = machines[ip]
        if m == {}:
            r = cmd_run(['nvidia-smi'], return_output=True)
            ps = parse_gpu_usage(r)
            machine_usage['_'] = ps
        else:
            try:
                r = remote_run('nvidia-smi', ssh_info=m, return_output=True)
            except Exception as e:
                logging.info('{}'.format(e))
                return False
            ps = parse_gpu_usage(r)
            machine_usage[m['ip']] = ps

    avail = True
    for ssh_info, gpus in all_resource:
        if ssh_info == {}:
            usage = machine_usage['_']
        else:
            usage = machine_usage[ssh_info['ip']]
        if all(g == -1 for g in gpus):
            continue
        if any(g >= len(usage) or usage[g][0] > 4000 or usage[g][2] > 20  for g in gpus):
            avail = False

    return avail

def terminate_my_gpu_process():
    ps = check_gpu_process()
    for process_id in ps:
        if 'jianfw' in ps[process_id]['username']:
            p = psutil.Process(process_id)
            p.terminate()

def parse_gpu_usage_dict(result):
    '''
    replace parse_gpu_usage
    '''
    used = []
    p = '^\|.* ([0-9]*)MiB \/ *([0-9]*)MiB *\| *([0-9]*)\%.*Default \|$'
    for line in result.split('\n'):
        line = line.strip()
        r = re.match(p, line)
        if r != None:
            u = [int(g) for g in r.groups()]
            names = ['mem_used', 'mem_total', 'gpu_util']
            used.append({n: v for n, v in zip(names, u)})
    return used

def parse_gpu_usage(result):
    '''
    prefer to use parse_gpu_usage_dict
    '''
    used = []
    p = '^\|.* ([0-9]*)MiB \/ *([0-9]*)MiB *\| *([0-9]*)\%.*Default \|$'
    for line in result.split('\n'):
        line = line.strip()
        r = re.match(p, line)
        if r != None:
            u = [int(g) for g in r.groups()]
            used.append(u)
    return used

def check_gpu_process():
    result = sp.check_output(['nvidia-smi']).decode()
    ps = parse_nvidia_smi(result)
    psinfo = collect_process_info()
    for key in ps:
        if key not in psinfo:
            continue
        for k in psinfo[key]:
            ps[key][k] = psinfo[key][k]
    logging.info(pformat(ps))
    return ps

def parse_nvidia_smi(result):
    ps = {}
    for line in result.split('\n'):
        r = re.match('^\| *([0-9]+) *([0-9]+).*\|$', line)
        if r != None:
            gpu = int(r.groups()[0])
            pid = int(r.groups()[1])
            ps[pid] = {}
            ps[pid]['gpu'] = gpu
    return ps

def test_usage():
    result = cmd_run(['nvidia-smi'], return_output=True)
    logging.info(parse_gpu_usage(result))

if __name__ == '__main__':
    init_logging()
    if len(sys.argv) >= 2 and sys.argv[1] == 'kill':
        terminate_my_gpu_process()
    check_gpu_process()

