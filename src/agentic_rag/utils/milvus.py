from typing import List, Dict
import os
import logging

from agentic_rag.config import settings
from agentic_rag.utils.embedding import embedding_service

logger = logging.getLogger(__name__)

# 尝试导入 pymilvus
pymilvus_available = False
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    pymilvus_available = True
    logger.info("pymilvus 已加载")
except ImportError:
    logger.warning("pymilvus 未安装，Milvus 向量数据库功能将不可用")


class MilvusService:
    """Milvus 向量数据库服务"""
    
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION # 集合名称
        self.available = pymilvus_available
        
        if self.available:
            try:
                self.connect()
                self.create_collection()
            except Exception as e:
                logger.error(f"初始化 Milvus 失败：{e}")
                self.available = False
    
    def connect(self):
        """连接到 Milvus"""
        if not self.available:
            logger.error("Milvus 功能不可用，因为 pymilvus 未安装")
            return
        
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"成功连接到 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败：{e}")
            raise
    
    def create_collection(self):
        """创建集合"""
        if not self.available:
            logger.error("Milvus 功能不可用，因为 pymilvus 未安装")
            return
        
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在")
                return
            
            # 定义集合字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="document_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            
            schema = CollectionSchema(fields=fields, description="文档切片向量集合")
            collection = Collection(name=self.collection_name, schema=schema)
            
            # ⚠️ 低配版说明：
            #    架构图设计为 HNSW 索引（图算法，检索精度高但内存占用大）。
            #    实际使用 IVF_FLAT（倒排文件 + 暴力搜索），在无 GPU 环境下
            #    索引构建快、内存开销低，16GB 内存机器可稳定运行。
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
            
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            
            logger.info(f"创建集合 {self.collection_name} 成功")
        except Exception as e:
            logger.error(f"创建集合失败：{e}")
            raise
    
    def insert_chunks(self, chunks: List[Dict]):
        """插入文档切片"""
        if not self.available:
            logger.error("Milvus 功能不可用，因为 pymilvus 未安装")
            return None
        
        try:
            collection = Collection(self.collection_name)
            
            entities = []
            for chunk in chunks:
                entity = {
                    "id": f"{chunk['document_id']}_{chunk['chunk_index']}",
                    "document_id": chunk['document_id'],
                    "chunk_index": chunk['chunk_index'],
                    "content": chunk['content'],
                    "metadata": chunk.get('metadata', {}),
                    "embedding": embedding_service.embed_text(chunk['content'])
                }
                entities.append(entity)
            
            data = [
                [e["id"] for e in entities],
                [e["document_id"] for e in entities],
                [e["chunk_index"] for e in entities],
                [e["content"] for e in entities],
                [e["metadata"] for e in entities],
                [e["embedding"] for e in entities]
            ]
            
            mr = collection.insert(data)
            collection.flush()
            
            logger.info(f"插入 {len(entities)} 个切片成功")
            return mr
        except Exception as e:
            logger.error(f"插入切片失败：{e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 10, filter_expr: str = None):
        """向量相似度搜索"""
        if not self.available: # 如果Milvus不可用
            logger.error("Milvus 功能不可用，因为 pymilvus 未安装")
            return [] # 返回空值
        
        try:
            collection = Collection(self.collection_name) # 获取集合
            
            query_vector = embedding_service.embed_text(query) # 对查询进行嵌入
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}# 设置搜索参数
            
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "document_id", "chunk_index", "content", "metadata"]
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("id"),
                        "document_id": hit.entity.get("document_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"向量搜索失败：{e}")
            return []
    
    def delete_by_document_id(self, document_id: int):
        """根据文档 ID 删除切片"""
        if not self.available:
            logger.error("Milvus 功能不可用，因为 pymilvus 未安装")
            return
        
        try:
            collection = Collection(self.collection_name)
            expr = f"document_id == {document_id}"
            collection.delete(expr)
            collection.flush()
            logger.info(f"删除文档 {document_id} 的切片成功")
        except Exception as e:
            logger.error(f"删除切片失败：{e}")
            raise


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    milvus_service = MilvusService()
else:
    milvus_service = None
