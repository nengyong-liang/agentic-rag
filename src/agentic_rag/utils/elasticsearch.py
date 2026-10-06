from typing import List, Dict
import os
import logging

from agentic_rag.config import settings

logger = logging.getLogger(__name__)

# 尝试导入 Elasticsearch
elasticsearch_available = False
try:
    from elasticsearch import Elasticsearch
    elasticsearch_available = True
    logger.info("elasticsearch 已加载")
except ImportError:
    logger.warning("elasticsearch 未安装，关键词检索功能将不可用")


class ElasticsearchService:
    """Elasticsearch 关键词检索服务"""
    
    def __init__(self):
        self.host = settings.ES_HOST
        self.port = settings.ES_PORT
        self.index = settings.ES_INDEX
        self.available = elasticsearch_available
        
        if self.available:
            try:
                self.client = Elasticsearch([f"http://{self.host}:{self.port}"])
                self.create_index()
            except Exception as e:
                logger.error(f"初始化 Elasticsearch 失败：{e}")
                self.available = False
    
    def create_index(self):
        """创建索引"""
        if not self.available:
            logger.error("Elasticsearch 功能不可用，因为 elasticsearch 未安装")
            return

        try:
            if self.client.indices.exists(index=self.index):
                logger.info(f"索引 {self.index} 已存在")
                return

            index_config = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "document_id": {"type": "long"},
                        "chunk_index": {"type": "integer"},
                        "content": {
                            "type": "text"
                        },
                        "metadata": {"type": "object"},
                        "title": {"type": "text"},
                        "section": {"type": "keyword"}
                    }
                }
            }

            self.client.indices.create(index=self.index, body=index_config)
            logger.info(f"创建索引 {self.index} 成功")
        except Exception as e:
            logger.error(f"创建索引失败：{e}")
            self.available = False
    
    def insert_chunks(self, chunks: List[Dict]):
        """批量插入文档切片"""
        if not self.available:
            logger.error("Elasticsearch 功能不可用，因为 elasticsearch 未安装")
            return
        
        try:
            actions = []
            for chunk in chunks:
                action = {
                    "_index": self.index,
                    "_id": f"{chunk['document_id']}_{chunk['chunk_index']}",
                    "_source": {
                        "document_id": chunk['document_id'],
                        "chunk_index": chunk['chunk_index'],
                        "content": chunk['content'],
                        "metadata": chunk.get('metadata', {}),
                        "title": chunk.get('metadata', {}).get('title', ''),
                        "section": chunk.get('metadata', {}).get('section', '')
                    }
                }
                actions.append(action)
            
            from elasticsearch import helpers
            helpers.bulk(self.client, actions)
            
            logger.info(f"插入 {len(actions)} 个切片到 ES 成功")
        except Exception as e:
            logger.error(f"插入 ES 失败：{e}")
            raise
    
    def search(self, query: str, top_k: int = 10, filter_dict: Dict = None):
        """关键词搜索"""
        if not self.available:
            logger.error("Elasticsearch 功能不可用，因为 elasticsearch 未安装")
            return []
        
        try:
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "title^3", "section"],
                                    "operator": "or"
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": ["document_id", "chunk_index", "content", "metadata"]
            }
            
            if filter_dict:
                filter_clauses = []
                for key, value in filter_dict.items():
                    filter_clauses.append({"term": {key: value}})
                if filter_clauses:
                    search_body["query"]["bool"]["filter"] = filter_clauses
            
            response = self.client.search(index=self.index, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "document_id": hit["_source"]["document_id"],
                    "chunk_index": hit["_source"]["chunk_index"],
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"]
                })
            
            return results
        except Exception as e:
            logger.error(f"ES 搜索失败：{e}")
            return []
    
    def delete_by_document_id(self, document_id: int):
        """根据文档 ID 删除"""
        if not self.available:
            logger.error("Elasticsearch 功能不可用，因为 elasticsearch 未安装")
            return
        
        try:
            query = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                }
            }
            self.client.delete_by_query(index=self.index, body=query)
            logger.info(f"删除 ES 中文档 {document_id} 的记录成功")
        except Exception as e:
            logger.error(f"删除 ES 记录失败：{e}")
            raise


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    es_service = ElasticsearchService()
else:
    es_service = None
