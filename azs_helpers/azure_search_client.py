import math
import json
import time
from pprint import pprint
from pathlib import Path
import os
import warnings
import requests

from .l2r_helper import grouper


class azure_search_client:
    def __init__(self, service_name, endpoint, api_version, api_key, index_name):
        self.service_name = service_name
        self.endpoint = endpoint.strip('/') + '/'
        self.api_version = api_version
        self.api_key = api_key
        self.index_name = index_name

    @classmethod
    def from_json(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        instance = cls(
            config["service_name"],
            config["endpoint"],
            config["api_version"],
            config["api_key"],
            config["index_name"],
        )
        instance.test_service(config_path)
        return instance

    @property
    def datasource_name(self):
        return self.index_name + "-datasource" 

    @property
    def indexer_name(self):
        return self.index_name + "-indexer" 

    @property
    def headers(self):
        return {"Content-Type": "application/json", "api-key": self.api_key}

    @property
    def index_api_uri(self):
        return (
            self.endpoint
            + "indexes?api-version="
            + self.api_version
            + "&service="
            + self.service_name
        )
    
    @property
    def index_doc_count_api_uri(self):
        return (
            self.endpoint
            + "indexes/"
            + self.index_name
            + "/docs/$count?api-version="
            + self.api_version
            + "&service="
            + self.service_name
        )

    @property
    def search_api_uri(self):
        return (
            self.endpoint
            + "indexes/"
            + self.index_name
            + "/docs/search?api-version="
            + self.api_version
            + "&service="
            + self.service_name
            + "&featuresMode=enabled"
        )

    @property
    def datasource_api_uri(self):
        return (
            self.endpoint
            + "datasources?api-version="
            + self.api_version
            + "&service="
            + self.service_name
        )
    
    @property
    def indexer_api_uri(self):
        return (
            self.endpoint
            + "indexers?api-version="
            + self.api_version
            + "&service="
            + self.service_name
        )
    
    def test_service(self, config_path):
        try:
            response = requests.get(self.index_api_uri + "&$select=name", headers=self.headers, verify=False)
        except:
            raise Exception(f'The provide Azure Search service metadata is invalid. Please verify the information provided in the config file located at {config_path}.')
        
        if response.status_code != 200:
            raise Exception(f'We failed to connect to the Azure Search Service {self.service_name} with http status {response.status_code}. \n Please verify the information provided in the config file located at {config_path}.')

    def index_documents_count(self):
        response = requests.get(
            self.index_doc_count_api_uri, headers=self.headers, verify=False
        )
        response.encoding = 'utf-8-sig'
        return int(response.text)

    def get_indexes(self):
        response = requests.get(self.index_api_uri, headers=self.headers, verify=False)
        print(response.json())
        return response.json()

    def search(self, body, verbose=False):
        for retry_count in range(5):
            response = requests.post(
                self.search_api_uri, headers=self.headers, json=body, verify=False
            )

            if response.status_code == 200:
                return response.json()["value"]
            else:
                if verbose:
                    print(
                        f"Search request failed with status: {response.status_code}. Sleeping 100ms. Retrying... Retry count so far {retry_count}"
                    )
                time.sleep(0.1)

        print(f"Search request failed with status: {response.status_code} after {retry_count} retries.")


    def resource_exist(self, api_uri, resource_name):
        response = requests.get(
            api_uri + "&$select=name", headers=self.headers, verify=False
        )

        if response.status_code != 200:
            print(f"Failed to connect to search service '{self.service_name}'. Response code '{response.status_code}'")
            print(response.text)
            return False
        else:
            matching_resource = [resource['name'] for resource in response.json()['value'] if resource['name'] == resource_name]
            return len(matching_resource) == 1

    def create_index(self):
        if self.resource_exist(self.index_api_uri, self.index_name):
            print(f"Index {self.index_name} already exists. Skipping creation.")
            return

        print(f"Index {self.index_name} does not exist in service. Creating.")
            
        script_dir = Path(os.path.dirname(__file__))
        full_path = script_dir / "index_schema" / "docs-multilingual-20200217.json"
        with open(full_path, 'r') as f:
            schema_body = json.load(f)
            index_schema = {}
            index_schema.update(schema_body)
            # Overwrite existing name in index schema with user-defined one.
            index_schema["name"] = self.index_name

        response = requests.post(
            self.index_api_uri,
            headers=self.headers,
            json=index_schema,
            verify=False
        )
    
    def create_datasource(self):
        if self.resource_exist(self.datasource_api_uri, self.datasource_name):
            print(f"Datasource {self.datasource_name} already exists. Skipping creation.")
            return

        datasource = {
            "name" : self.datasource_name,
            "type" : "azureblob",
            "credentials" : { "connectionString" : "ContainerSharedAccessUri=https://azsmsftdocs.blob.core.windows.net/docs-english-20200217?st=2020-04-29T23%3A11%3A03Z&se=2040-01-01T23%3A11%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=gPzw2oM3RbdHOlWa06V1j0Tn4qKLCPchMUAfb2Vs4vg%3D" },
            "container" : { "name" : "docs-english-20200217" }
        } 

        response = requests.post(
            self.datasource_api_uri,
            headers=self.headers,
            json=datasource,
            verify=False
        )

        if response.status_code > 299:
            print(f"Failed to create datasource '{self.datasource_name}' in search service '{self.service_name}'. Status:'{response.status_code}' Error: {response.content}")
        else:
            print(f"Data source '{self.datasource_name}' created succesfully.")

    def create_indexer(self):
        if self.resource_exist(self.indexer_api_uri, self.indexer_name):
            print(f"Indexer {self.indexer_name} already exists. Skipping creation.")
            return

        print(f"Indexer {self.indexer_name} does not exist in service. Creating.")

        indexer = {
            "name" : self.indexer_name,
            "dataSourceName" : self.datasource_name,
            "targetIndexName" : self.index_name,
            "parameters" : { "configuration" : { "parsingMode" : "json" } }
        }

        response = requests.post(
            self.indexer_api_uri,
            headers=self.headers,
            json=indexer,
            verify=False
        )

        if response.status_code > 299:
            print(f"Failed to create indexer '{self.indexer_name} in search service '{self.service_name}'. Status:'{response.status_code}' Error: {response.content}")
        else:
            print(f"Indexer '{self.indexer_name}' created succesfully.")

    def wait_for_indexer_completion(self, expected_document_count):
        if not self.resource_exist(self.indexer_api_uri, self.indexer_name):
            return

        current_count = 0
        while current_count < expected_document_count:
            current_count = self.index_documents_count()
            print(f"Ingested {current_count} out of {expected_document_count} documents. Waiting 10 seconds.")
            time.sleep(10)
        print("Completed indexing.") 

    def ingest_documents_from_blob_storage(self, expected_document_count):
        actual_document_count = self.index_documents_count()

        if self.index_documents_count() < expected_document_count: 
            print(f"Index {self.index_name} contains only {actual_document_count} out of {expected_document_count} documents. Uploading documents.")
            
            # this step will create a new data source in your service to connect to our public
            # Azure Storage blob container which contains the documents 
            self.create_datasource()
            self.create_indexer()
            self.wait_for_indexer_completion(expected_document_count)
                
        else:
            print(f"Index {self.index_name} contains all {actual_document_count} documents. Skipping ingesting documents.")