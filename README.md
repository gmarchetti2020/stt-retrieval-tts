# Retrieval Chatbot
This code implements a sample retrieval chatbot using Google speech-to-text, translation API and Haystack.
The retrieved text is then read back using the client's native text-to-speech engine (for mac in this case, but you can use any other).
Note that:
- We assume that the document store is ElasticSearch. For testing purposes, you can run a local containerized instance
  ```
  docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.2
  docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
  ```
  In testing mode, the document store will be cleaned out at each run and documents in a given directory will be re-uploaded.
- The code retrieves 4 documents and infers two answers out of those. You may want to change that, e.g. to return only the most likely answer.
- Haystack requires pytorch. To save setup time, you can run on a deep learning machine in Google cloud.
- The requirements.txt contains the prerequisites. Install with `pip install -r requirements.txt`
- You will have to change the parameters in the python code so they point to GCP project, source and target languages of your choice.
- You will need to provide application credentials to access the APIs. For more details see https://cloud.google.com/docs/authentication   

I am providing a couple of pdf files to build a simple document base.
To test, simply run:
`python3 main.py`