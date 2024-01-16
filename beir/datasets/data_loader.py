from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv

logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder: str = None,         # This is the path to the data folder on the disk
                 prefix: str = None,                    # Not sure what this is for
                 corpus_file: str = "corpus.jsonl",     #   #    #   #   #   #   #
                 query_file: str = "queries.jsonl",     #       Pre-defined      #  
                 qrels_folder: str = "qrels",           #          names
                 qrels_file: str = "",                 # qrel files not known / depends on test/train/dev split
                 split_index: int = 10):
        
        # Instantiate the three objects as empty dictionaries
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.k = split_index
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        # sets the file paths; can handle cases where the folder isn't required because the path is passed directly
        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file
    
    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        # Returns tuple of:
        #    corpus = { "query_id" : {"text" : string, "title" : string} }
        #    queries = { "doc_id" : "string" }
        #    qrels =  { "query_id" : { "doc_id" : int(score) } }
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
       
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _load_corpus(self):
        
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line_num, line in enumerate(tqdm(fIn, total=num_lines)):
                line = json.loads(line)
                doc_id = line.get("_id")
                if doc_id in self.queries:
                    self.corpus[doc_id] = {
                    "text": line.get("text"),
                    "title": line.get("title")}
                else:
                    pass
    
    def _load_queries(self):
        k = self.k
        with open(self.query_file, encoding='utf8') as fIn:
            for line_num, line in fIn:
                if line_num >= k: # Check if we have reached the 1st-k limit
                    break
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score