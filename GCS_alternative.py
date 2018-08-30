import logging
import os
import cloudstorage as gcs
import webapp2

from google.appengine.api import app_identity

if __name__ == '__main__':
    print 'Google Cloud Storage ...'