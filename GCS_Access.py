# Imports the Google Cloud client library
import os
from google.cloud import storage
from google.cloud import bigquery


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "Unilever-aaf0d3b1ca2c.json"

google_API_key = 'AIzaSyAwTBbT2qaX0OWJ4VAOnYpMTIUWtACjwH4'


def create_bucket(bucket_name):
    # Instantiates a client
    storage_client = storage.Client()

    # Creates the new bucket
    bucket = storage_client.create_bucket(bucket_name)

    print('Bucket {} created.'.format(bucket.name))


def get_bucket(bucket_name):
    client = storage.Client.from_service_account_json(
        'Unilever-aaf0d3b1ca2c.json')
    print ' ... get bucket ', bucket_name
    # bucket = client.get_bucket(bucket_name)
    buckets = list(client.list_buckets())
    print(buckets)
    print '... done '


def run_quickstart():
    # [START bigquery_quickstart]
    # Imports the Google Cloud client library

    # Instantiates a client
    bigquery_client = bigquery.Client()

    # The name for the new dataset
    dataset_id = 'jcl_test'

    # Prepares a reference to the new dataset
    dataset_ref = bigquery_client.dataset(dataset_id)
    dataset = bigquery.Dataset(dataset_ref)

    # Creates the new dataset
    dataset = bigquery_client.create_dataset(dataset)

    print('Dataset {} created.'.format(dataset.dataset_id))
    # [END bigquery_quickstart]


if __name__ == '__main__':
    print 'GDS Access started ..'
    # get_bucket('work_jcl')
    run_quickstart()