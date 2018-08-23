from azure.storage.blob import BlockBlobService
from azure.storage.common.storageclient import logger
from qd_common import load_from_yaml_file
logger.propagate = False

class CloudStorage(object):
    def __init__(self):
        config_file = 'aux_data/configs/azure_blob_account.yaml'
        config = load_from_yaml_file(config_file)
        account_name = config['account_name']
        account_key = config['account_key']
        self.container_name = 'detectortrain'

        self.block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

    def upload(self, file_name):
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

