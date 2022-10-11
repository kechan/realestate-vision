from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable

import json, gc

from pathlib import Path
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class ESListingsClient:
  def __init__(self, es_ip: str, es_port: int):
    self.es_ip = es_ip
    self.es_port = es_port

    self.es_client = Elasticsearch([f'http://{self.es_ip}:{self.es_port}/'])

    self.index_name = 'rlp_listing_current'

    self.today_str = datetime.today().date().strftime("%Y_%m_%d")

    self.cache_dir = Path.home()/'.elastic_search_cache'
    self.cache_dir.mkdir(exist_ok=True)
  
  def info(self):
    return Elasticsearch.info(self.es_client)

  def count(self) -> int:
    return self.es_client.count(index=self.index_name, body={"query":{"match_all": {}},})['count']

  def get_jumpIds(self) -> List[str]:
    payload = []

    res = self.es_client.search(index=self.index_name, body={"_source": ["jumpId"],
                                                  "query":
                                                  {"match_all": {}},
                                                  'size': 1000
                                                  },
                                                  
                          scroll='500m'
                          )

    scroll_id = res['_scroll_id']

    payload_per_page = [r['_source'] for r in res['hits']['hits']]
    payload += payload_per_page
    # print(scroll_id)

    k = 0
    while len(payload_per_page) != 0:
      
      res = self.es_client.scroll(scroll_id=scroll_id, scroll='1s')
      scroll_id = res['_scroll_id']
      
      payload_per_page = [r['_source'] for r in res['hits']['hits']]
      payload += payload_per_page
      
      # track progress
      print(".", end='')
      if k % 10 == 0: print(k, end='')
      if k % 100 == 0: print("")
      k += 1
      
      if k % 10 == 0: # save every 10th and clear payload
        with open(f'{self.cache_dir}/jumpId_{self.today_str}_{k}.json', 'w') as f:
          json.dump(payload, f)
          payload = []
          gc.collect()

    if len(payload) > 0:     #remainder
      with open(f'{self.cache_dir}/jumpId_{self.today_str}_{k}.json', 'w') as f:
        json.dump(payload, f)
        payload = []
        gc.collect()

    jsons = self.cache_dir.lf(f'jumpId_{self.today_str}*')

    payloads = []
    for f in jsons:
      with open(f, 'r') as fp:
        payload = json.load(fp)
      payloads += payload

    jumpIds = [p['jumpId'] for p in payloads if 'jumpId' in p.keys()]

    return jumpIds

  def get_payload(self, jumpIds: List[str], columns: List[str]) -> Dict:
    res = self.es_client.search(index=self.index_name, body={
      "_source": columns, 
      "query" : {"terms" : {"jumpId": jumpIds,}}},
                                                  
      size=500
    )

    payload = [r['_source'] for r in res['hits']['hits']]

    return payload