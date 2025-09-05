import os
import gc
import logging
import time
import uuid
import multiprocessing as mp
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import json
import signal
import sys
from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io
import base64

# 设置环境变量
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/.cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConcurrentRequest:
    request_id: str
    image_data: str  # base64编码的图像数据
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int = 0

def gpu_worker_process(gpu_id: int, request_queue: mp.Queue, result_queue: mp.Queue):
    """GPU工作进程 - 基于稳定的单GPU版本"""
    try:
        # 设置进程专用的CUDA环境
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        import torch
        from PIL import Image
        import io
        import base64
        from modelscope import QwenImageEditPipeline
        
        # 配置进程专用日志
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[GPU-{gpu_id}] %(asctime)s - %(levelname)s - %(message)s'
        )
        worker_logger = logging.getLogger(f'GPU-{gpu_id}')
        
        pipeline = None
        request_count = 0
        
        def initialize_pipeline():
            nonlocal pipeline
            try:
                worker_logger.info(f"Initializing pipeline on GPU {gpu_id}")
                
                # 确保CUDA可用
                if not torch.cuda.is_available():
                    worker_logger.error("CUDA is not available")
                    return False
                
                worker_logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
                
                # 清理显存
                torch.cuda.empty_cache()
                
                # 加载模型 - 完全按照您的工作代码
                pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
                pipeline.to(torch.bfloat16)
                pipeline.to("cuda")
                pipeline.set_progress_bar_config(disable=True)
                
                # 简单测试确保模型工作正常
                test_image = Image.new('RGB', (64, 64), color='white')
                with torch.inference_mode():
                    test_inputs = {
                        "image": test_image,
                        "prompt": "test",
                        "negative_prompt": "",
                        "num_inference_steps": 1,
                        "true_cfg_scale": 1.0,
                        "generator": torch.manual_seed(0)
                    }
                    _ = pipeline(**test_inputs)
                
                worker_logger.info(f"Pipeline initialized successfully on GPU {gpu_id}")
                return True
                
            except Exception as e:
                worker_logger.error(f"Pipeline initialization failed: {e}")
                worker_logger.error(traceback.format_exc())
                return False
        
        # 初始化pipeline
        if not initialize_pipeline():
            worker_logger.error(f"GPU {gpu_id} worker initialization failed")
            result_queue.put({
                "worker_id": gpu_id,
                "status": "initialization_failed",
                "error": "Pipeline initialization failed"
            })
            return
        
        # 发送就绪信号
        result_queue.put({
            "worker_id": gpu_id,
            "status": "worker_ready",
            "message": f"GPU {gpu_id} worker is ready for requests"
        })
        
        worker_logger.info(f"GPU {gpu_id} worker ready for requests")
        
        # 主处理循环
        while True:
            try:
                # 等待请求
                try:
                    request_data = request_queue.get(timeout=5)
                except:
                    continue  # 超时继续等待
                
                if request_data is None:  # 关闭信号
                    worker_logger.info(f"GPU {gpu_id} received shutdown signal")
                    break
                
                # 处理请求
                request = ConcurrentRequest(**request_data)
                request_count += 1
                worker_logger.info(f"Processing request {request.request_id} (#{request_count}) on GPU {gpu_id}")
                
                start_time = time.time()
                
                try:
                    # 解码图像
                    image_bytes = base64.b64decode(request.image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # 限制图像尺寸以确保稳定性
                    max_size = 768
                    original_size = image.size
                    if max(image.size) > max_size:
                        ratio = max_size / max(image.size)
                        new_size = tuple(int(dim * ratio) for dim in image.size)
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                        worker_logger.info(f"Resized image from {original_size} to {new_size}")
                    
                    # 准备输入 - 完全按照您的原始代码
                    inputs = {
                        "image": image,
                        "prompt": request.prompt,
                        "generator": torch.manual_seed(request.seed),
                        "true_cfg_scale": request.true_cfg_scale,
                        "negative_prompt": request.negative_prompt,
                        "num_inference_steps": request.num_inference_steps,
                    }
                    
                    # 执行推理
                    with torch.inference_mode():
                        output = pipeline(**inputs)
                        result_image = output.images[0]
                    
                    # 保存结果
                    output_path = f"outputs/{request.request_id}.png"
                    os.makedirs("outputs", exist_ok=True)
                    result_image.save(output_path)
                    
                    processing_time = time.time() - start_time
                    
                    # 返回成功结果
                    result = {
                        "request_id": request.request_id,
                        "status": "completed",
                        "output_path": output_path,
                        "processing_time": processing_time,
                        "gpu_id": gpu_id,
                        "completed_at": time.time(),
                        "request_count": request_count
                    }
                    
                    result_queue.put(result)
                    worker_logger.info(f"Request {request.request_id} completed in {processing_time:.2f}s")
                    
                except Exception as e:
                    worker_logger.error(f"Error processing request {request.request_id}: {e}")
                    worker_logger.error(traceback.format_exc())
                    
                    # 返回错误结果
                    error_result = {
                        "request_id": request.request_id,
                        "status": "error",
                        "error": str(e),
                        "gpu_id": gpu_id,
                        "error_type": type(e).__name__
                    }
                    result_queue.put(error_result)
                
                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                worker_logger.error(f"Worker loop error: {e}")
                worker_logger.error(traceback.format_exc())
                continue
        
        worker_logger.info(f"GPU {gpu_id} worker shutting down after processing {request_count} requests")
        
    except Exception as e:
        print(f"[GPU-{gpu_id}] Critical worker error: {e}")
        print(f"[GPU-{gpu_id}] {traceback.format_exc()}")

class MultiGPUConcurrentManager:
    """多GPU并发管理器"""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.processes = {}
        self.request_queues = {}
        self.result_queue = mp.Queue()
        self.result_store = {}
        self.is_running = True
        self.worker_status = {}
        self.request_counter = 0
        self.last_assigned_gpu = 0  # 用于轮询分配
        
        # 启动所有GPU工作进程
        self._start_all_workers()
        
        # 启动结果收集线程
        import threading
        self.result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.result_thread.start()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_workers, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Multi-GPU concurrent manager started with {len(gpu_ids)} GPUs")
    
    def _start_all_workers(self):
        """启动所有GPU工作进程"""
        logger.info(f"Starting {len(self.gpu_ids)} GPU workers...")
        
        for gpu_id in self.gpu_ids:
            try:
                request_queue = mp.Queue()
                
                process = mp.Process(
                    target=gpu_worker_process,
                    args=(gpu_id, request_queue, self.result_queue),
                    name=f"GPU-{gpu_id}"
                )
                process.start()
                
                self.processes[gpu_id] = process
                self.request_queues[gpu_id] = request_queue
                self.worker_status[gpu_id] = "starting"
                
                logger.info(f"Started GPU {gpu_id} worker process (PID: {process.pid})")
                
                # 错开启动时间避免资源竞争
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Failed to start GPU {gpu_id} worker: {e}")
                self.worker_status[gpu_id] = "failed"
        
        logger.info("All GPU workers started, waiting for ready signals...")
    
    def _collect_results(self):
        """收集处理结果和工作器状态"""
        while self.is_running:
            try:
                result = self.result_queue.get(timeout=1)
                
                # 处理工作器状态消息
                if "worker_id" in result:
                    gpu_id = result["worker_id"]
                    status = result["status"]
                    self.worker_status[gpu_id] = status
                    logger.info(f"GPU {gpu_id} status: {status}")
                    
                    if status == "worker_ready":
                        logger.info(f"GPU {gpu_id} is ready for requests")
                    elif status == "initialization_failed":
                        logger.error(f"GPU {gpu_id} initialization failed: {result.get('error', 'Unknown error')}")
                    
                    continue
                
                # 处理请求结果
                request_id = result["request_id"]
                self.result_store[request_id] = result
                
                if result.get("status") == "completed":
                    logger.info(f"Request {request_id} completed in {result.get('processing_time', 0):.2f}s on GPU {result.get('gpu_id')}")
                else:
                    logger.warning(f"Request {request_id} failed: {result.get('error', 'Unknown error')}")
                
            except:
                continue
    
    def _monitor_workers(self):
        """监控工作器健康状态"""
        while self.is_running:
            try:
                # 检查进程状态
                for gpu_id, process in self.processes.items():
                    current_status = self.worker_status.get(gpu_id, "unknown")
                    
                    if not process.is_alive() and current_status not in ["failed", "stopped"]:
                        logger.error(f"GPU {gpu_id} worker process died unexpectedly")
                        self.worker_status[gpu_id] = "died"
                
                time.sleep(15)  # 每15秒检查一次
                
            except Exception as e:
                logger.error(f"Worker monitoring error: {e}")
    
    def _get_next_available_gpu(self) -> Optional[int]:
        """使用轮询方式获取下一个可用的GPU"""
        ready_gpus = [
            gpu_id for gpu_id, process in self.processes.items()
            if process.is_alive() and self.worker_status.get(gpu_id) == "worker_ready"
        ]
        
        if not ready_gpus:
            return None
        
        # 轮询分配，确保负载均衡
        ready_gpus.sort()  # 确保顺序一致
        
        # 找到下一个GPU
        current_index = -1
        if self.last_assigned_gpu in ready_gpus:
            current_index = ready_gpus.index(self.last_assigned_gpu)
        
        next_index = (current_index + 1) % len(ready_gpus)
        next_gpu = ready_gpus[next_index]
        self.last_assigned_gpu = next_gpu
        
        return next_gpu
    
    def submit_request(self, request: ConcurrentRequest) -> str:
        """提交请求到可用的GPU"""
        gpu_id = self._get_next_available_gpu()
        if gpu_id is None:
            ready_count = sum(1 for status in self.worker_status.values() if status == "worker_ready")
            alive_count = sum(1 for p in self.processes.values() if p.is_alive())
            raise RuntimeError(f"No available GPU workers (Ready: {ready_count}, Alive: {alive_count}, Total: {len(self.processes)})")
        
        self.request_counter += 1
        
        # 提交请求到选定的GPU
        request_dict = asdict(request)
        self.request_queues[gpu_id].put(request_dict)
        
        # 记录请求状态
        self.result_store[request.request_id] = {
            "status": "queued",
            "submitted_at": time.time(),
            "assigned_gpu": gpu_id,
            "request_number": self.request_counter
        }
        
        logger.info(f"Request #{self.request_counter} ({request.request_id[:8]}) assigned to GPU {gpu_id}")
        return request.request_id
    
    def get_result(self, request_id: str) -> Optional[dict]:
        """获取请求结果"""
        return self.result_store.get(request_id)
    
    def get_status(self) -> dict:
        """获取系统状态"""
        alive_processes = sum(1 for p in self.processes.values() if p.is_alive())
        ready_workers = sum(1 for status in self.worker_status.values() if status == "worker_ready")
        
        # 统计请求状态
        completed_requests = sum(1 for r in self.result_store.values() if r.get("status") == "completed")
        failed_requests = sum(1 for r in self.result_store.values() if r.get("status") == "error")
        queued_requests = sum(1 for r in self.result_store.values() if r.get("status") == "queued")
        
        return {
            "total_gpus": len(self.processes),
            "alive_processes": alive_processes,
            "ready_workers": ready_workers,
            "total_requests": self.request_counter,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "queued_requests": queued_requests,
            "worker_status": self.worker_status.copy(),
            "processes_info": {
                gpu_id: {
                    "is_alive": process.is_alive(),
                    "pid": process.pid if process.is_alive() else None,
                    "status": self.worker_status.get(gpu_id, "unknown")
                }
                for gpu_id, process in self.processes.items()
            }
        }
    
    def wait_for_ready(self, timeout: int = 120) -> bool:
        """等待所有工作器准备就绪"""
        logger.info(f"Waiting for GPU workers to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            ready_count = sum(1 for status in self.worker_status.values() if status == "worker_ready")
            failed_count = sum(1 for status in self.worker_status.values() if status in ["failed", "initialization_failed"])
            
            logger.info(f"Workers status - Ready: {ready_count}/{len(self.gpu_ids)}, Failed: {failed_count}")
            
            if ready_count > 0:  # 至少有一个工作器准备好了
                logger.info(f"Service ready with {ready_count} GPU workers")
                return True
            
            if failed_count == len(self.gpu_ids):  # 所有工作器都失败了
                logger.error("All GPU workers failed to initialize")
                return False
            
            time.sleep(2)
        
        logger.warning(f"Timeout waiting for workers, {sum(1 for status in self.worker_status.values() if status == 'worker_ready')} ready")
        return False
    
    def shutdown(self):
        """关闭所有工作进程"""
        logger.info("Shutting down multi-GPU concurrent manager...")
        self.is_running = False
        
        # 发送关闭信号
        for gpu_id, queue in self.request_queues.items():
            try:
                queue.put(None)
                self.worker_status[gpu_id] = "stopping"
            except:
                pass
        
        # 等待进程结束
        for gpu_id, process in self.processes.items():
            try:
                logger.info(f"Waiting for GPU {gpu_id} worker to shutdown...")
                process.join(timeout=20)
                if process.is_alive():
                    logger.warning(f"Force terminating GPU {gpu_id} worker")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                self.worker_status[gpu_id] = "stopped"
            except Exception as e:
                logger.error(f"Error stopping GPU {gpu_id} worker: {e}")
        
        logger.info("Multi-GPU concurrent manager shutdown complete")

# 全局管理器
concurrent_manager: Optional[MultiGPUConcurrentManager] = None

# 信号处理
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    if concurrent_manager:
        concurrent_manager.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# FastAPI应用
app = FastAPI(title="Multi-GPU Concurrent Image Edit Service", version="5.0.0")

@app.on_event("startup")
async def startup_event():
    global concurrent_manager
    
    logger.info("Starting multi-GPU concurrent image editing service...")
    
    # 检查CUDA
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    # 获取GPU列表
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_ids_str = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    if gpu_ids_str:
        gpu_ids = [int(x) for x in gpu_ids_str.split(",")]
    else:
        # 使用所有可用的GPU
        gpu_ids = available_gpus
    
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Will use GPUs: {gpu_ids}")
    
    # 创建并发管理器
    concurrent_manager = MultiGPUConcurrentManager(gpu_ids)
    
    # 等待工作器准备就绪
    if not concurrent_manager.wait_for_ready():
        raise RuntimeError("Failed to initialize GPU workers")
    
    logger.info("Multi-GPU concurrent service started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    if concurrent_manager:
        concurrent_manager.shutdown()

@app.post("/edit")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(50),
    true_cfg_scale: float = Form(4.0),
    seed: int = Form(0)
):
    try:
        # 验证输入
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # 读取图像
        max_file_size = 10 * 1024 * 1024  # 10MB
        image_data = await image.read()
        if len(image_data) > max_file_size:
            raise HTTPException(status_code=413, detail="Image too large")
        
        # 验证图像格式
        try:
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            logger.info(f"Received image: {pil_image.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # 创建请求
        request_id = str(uuid.uuid4())
        
        # 编码图像数据
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        request = ConcurrentRequest(
            request_id=request_id,
            image_data=image_b64,
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip(),
            num_inference_steps=max(10, min(num_inference_steps, 100)),
            true_cfg_scale=max(1.0, min(true_cfg_scale, 10.0)),
            seed=seed
        )
        
        # 提交请求
        concurrent_manager.submit_request(request)
        
        status = concurrent_manager.get_status()
        
        return {
            "request_id": request_id,
            "status": "submitted",
            "message": "Request submitted successfully",
            "ready_workers": status["ready_workers"],
            "total_requests": status["total_requests"],
            "estimated_position": status["queued_requests"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in edit_image: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{request_id}")
async def get_request_status(request_id: str):
    result = concurrent_manager.get_result(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Request not found")
    return result

@app.get("/result/{request_id}")
async def get_result_image(request_id: str):
    result = concurrent_manager.get_result(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Request not found")
    
    if result["status"] != "completed":
        return JSONResponse(content=result)
    
    output_path = result["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(output_path, media_type="image/png", filename=f"{request_id}.png")

@app.get("/system/status")
async def get_system_status():
    return concurrent_manager.get_status()

@app.get("/health")
async def health_check():
    try:
        status = concurrent_manager.get_status()
        is_healthy = status["ready_workers"] > 0
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": time.time(),
            "ready_workers": status["ready_workers"],
            "total_gpus": status["total_gpus"],
            "total_requests": status["total_requests"],
            "completed_requests": status["completed_requests"],
            "failed_requests": status["failed_requests"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1
    )