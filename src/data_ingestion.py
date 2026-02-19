import os
import json
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path_or_links: str) -> List[Dict[str, Any]]:
        pass

class DataIngestorJson(DataIngestor):
    def ingest(self, file_path_or_link):
        try:
            logger.info("Ingesting JSON data from: %s", file_path_or_link)
            with open(file_path_or_link, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("Successfully ingested data from %s (items=%s)", file_path_or_link,
                        (len(data) if hasattr(data, '__len__') else type(data)))
            return data
        except Exception as e:
            logger.exception("Failed to ingest JSON data from %s: %s", file_path_or_link, e)
            raise

class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path):
        try:
            logger.info("Ingesting CSV data from: %s", file_path)
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
            logger.info("Successfully ingested data from %s (rows=%d)", file_path, len(data))
            return data
        except Exception as e:
            logger.exception("Failed to ingest CSV data from %s: %s", file_path, e)
            raise

