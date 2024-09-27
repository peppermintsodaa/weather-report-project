# code ref: https://stackoverflow.com/a/21957017
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys

class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*') # this is to allow accessing files to load datasets
        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    # bind to custom header to avoid using localhost
    test(CORSRequestHandler, HTTPServer, bind='127.0.0.1', port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)