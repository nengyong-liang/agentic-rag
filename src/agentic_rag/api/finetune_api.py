"""
微调服务 API 端点
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import uuid
import json
from pathlib import Path

router = APIRouter(prefix="/api/v1/finetune", tags=["微调服务"])

# 任务状态存储
tasks = {}


class FinetuneTask(BaseModel):
    """微调任务"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    message: str
    domain: Optional[str] = "default"
    model_name: Optional[str] = None


class FinetuneRequest(BaseModel):
    """微调请求"""
    domain: Optional[str] = "default"
    model_name: Optional[str] = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    num_train_epochs: Optional[int] = 2
    learning_rate: Optional[float] = 2e-4


@router.get("/")
async def get_finetune_status():
    """获取微调服务状态"""
    return {
        "status": "ok",
        "service": "finetune",
        "version": "1.0.0"
    }


@router.post("/upload")
async def upload_training_data(file: UploadFile = File(...), domain: str = "default"):
    """
    上传训练数据
    
    Args:
        file: 训练数据文件（JSON/Excel/CSV）
        domain: 领域标识（hr/it/finance）
    """
    try:
        # 生成任务 ID
        task_id = str(uuid.uuid4())
        
        # 保存文件
        finetune_dir = Path(__file__).parent.parent / "finetune"
        data_dir = finetune_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = data_dir / f"{task_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 创建任务记录
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "数据上传完成，等待训练",
            "domain": domain,
            "data_file": str(file_path)
        }
        
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "数据上传成功，请调用 /train 接口开始训练"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败：{str(e)}")


@router.post("/train")
async def start_training(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """
    开始微调训练
    
    Args:
        request: 微调请求
    """
    try:
        # 生成任务 ID
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "训练任务已创建",
            "domain": request.domain,
            "model_name": request.model_name,
            "config": {
                "lora_r": request.lora_r,
                "lora_alpha": request.lora_alpha,
                "num_train_epochs": request.num_train_epochs,
                "learning_rate": request.learning_rate
            }
        }
        
        # 后台训练任务
        def run_training():
            try:
                import sys
                from pathlib import Path
                
                # 添加 finetune 模块路径
                finetune_dir = Path(__file__).parent.parent / "finetune"
                sys.path.insert(0, str(finetune_dir))
                
                from config import FinetuneConfig
                from data_preprocessor import DataPreprocessor
                from train_lora import LoRATrainer
                
                # 更新配置
                config = FinetuneConfig()
                config.model_name = request.model_name
                config.lora_r = request.lora_r
                config.lora_alpha = request.lora_alpha
                config.num_train_epochs = request.num_train_epochs
                config.learning_rate = request.learning_rate
                
                # 更新任务状态
                tasks[task_id]["status"] = "running"
                tasks[task_id]["progress"] = 10
                tasks[task_id]["message"] = "加载模型..."
                
                # 加载模型
                trainer = LoRATrainer(config)
                trainer.load_model_and_tokenizer()
                
                tasks[task_id]["progress"] = 30
                tasks[task_id]["message"] = "配置 LoRA..."
                
                trainer.setup_lora()
                
                tasks[task_id]["progress"] = 40
                tasks[task_id]["message"] = "开始训练..."
                
                # 开始训练
                trainer.train()
                
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["progress"] = 100
                tasks[task_id]["message"] = "训练完成"
            
            except Exception as e:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["message"] = f"训练失败：{str(e)}"
                import traceback
                logger.error(f"训练失败：{traceback.format_exc()}")
        
        # 启动后台任务
        background_tasks.add_task(run_training)
        
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "训练任务已启动"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练失败：{str(e)}")


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]


@router.get("/tasks")
async def list_tasks():
    """获取所有任务列表"""
    return {"tasks": list(tasks.values())}


@router.post("/inference")
async def inference(query: str, domain: str = "default"):
    """
    使用微调模型推理
    
    Args:
        query: 用户问题
        domain: 领域标识
    """
    try:
        import sys
        from pathlib import Path
        
        # 添加 finetune 模块路径
        finetune_dir = Path(__file__).parent.parent / "finetune"
        sys.path.insert(0, str(finetune_dir))
        
        from integration import generate_with_finetuned_model
        
        # 生成回答
        answer = generate_with_finetuned_model(query, domain, use_finetuned=True)
        
        return {
            "query": query,
            "answer": answer,
            "model": "finetuned",
            "domain": domain
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败：{str(e)}")


@router.get("/models")
async def list_models():
    """获取可用模型列表"""
    try:
        import sys
        from pathlib import Path
        
        # 添加 finetune 模块路径
        finetune_dir = Path(__file__).parent.parent / "finetune"
        sys.path.insert(0, str(finetune_dir))
        
        from integration import model_manager
        
        models = model_manager.list_available_models()
        
        return {
            "models": models
        }
    
    except Exception as e:
        return {
            "models": [],
            "error": str(e)
        }


@router.post("/switch-model")
async def switch_model(domain: str):
    """
    切换到指定领域的模型
    
    Args:
        domain: 领域标识
    """
    try:
        import sys
        from pathlib import Path
        
        # 添加 finetune 模块路径
        finetune_dir = Path(__file__).parent.parent / "finetune"
        sys.path.insert(0, str(finetune_dir))
        
        from integration import model_manager
        
        model_manager.switch_model(domain)
        
        return {
            "status": "ok",
            "message": f"已切换到领域模型：{domain}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换模型失败：{str(e)}")
