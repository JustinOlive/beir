import torch.multiprocessing as multiprocessing
import logging
from torch import Tensor
logger = logging.getLogger(__name__)

class SentenceTransformerModel:
    def __init__(self, embed_model,  sep = " ",   **kwargs):
        self.sep = sep
        self.q_model = embed_model
        self.doc_model = self.q_model

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = multiprocessing.get_context('spawn')
        input_queue, output_queue = ctx.Queue(), ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(target = SentenceTransformer._encode_multi_process_worker,
                            args = (process_id, device_name,  self.doc_model, input_queue, output_queue),
                            daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool['output']
        [output_queue.get() for _ in range(len(pool['processes']))]
        return self.doc_model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        updated_queries = [f"{query}" for query in queries] # query:

        return self.q_model.encode(updated_queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]

        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)

    ## Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str], batch_size: int = 8, chunk_id: int = None, **kwargs):
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]

        if chunk_id is not None and chunk_id >= len(pool['processes']):
            output_queue = pool['output']
            output_queue.get()

        input_queue = pool['input']
        input_queue.put([chunk_id, batch_size, sentences])