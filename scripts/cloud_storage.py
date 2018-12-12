from azure.storage.blob import BlockBlobService
from azure.storage.common.storageclient import logger
from qd_common import load_from_yaml_file
logger.propagate = False

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

    def upload(self, file_name):
        assert False, 'use upload_stream'
        block_blob_service.create_blob_from_path(self.container_name,
                blob_name, file_name)

    def upload_stream(self, s, name):
        if self.block_blob_service.exists(self.container_name,
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

    def download(self, blob_name, local_path):
        self.block_blob_service.get_blob_to_path(self.container_name,
                blob_name, local_path)
