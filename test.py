import requests
import time
import json
from PIL import Image, ImageDraw
import concurrent.futures
import threading
from pathlib import Path

class ConcurrentTestClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 60
    
    def create_test_image(self, filename, content="TEST"):
        """创建测试图像"""
        image = Image.new('RGB', (512, 512), color='lightblue')
        draw = ImageDraw.Draw(image)
        
        # 画一个简单图案
        draw.rectangle([100, 100, 400, 400], fill='white', outline='black', width=3)
        draw.ellipse([150, 150, 350, 350], fill='yellow', outline='red', width=2)
        draw.text((220, 250), content, fill='black')
        
        image.save(filename)
        return filename
    
    def get_system_status(self):
        """获取系统状态"""
        try:
            response = self.session.get(f"{self.base_url}/system/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get system status: {e}")
            return None
    
    def submit_request(self, image_path, prompt, **kwargs):
        """提交请求"""
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}
                data = {
                    "prompt": prompt,
                    "negative_prompt": kwargs.get("negative_prompt", ""),
                    "num_inference_steps": kwargs.get("num_inference_steps", 50),
                    "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
                    "seed": kwargs.get("seed", 0)
                }
                
                response = self.session.post(f"{self.base_url}/edit", files=files, data=data)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Failed to submit request: {e}")
            return None
    
    def wait_for_result(self, request_id, timeout=300):
        """等待结果"""
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/status/{request_id}")
                response.raise_for_status()
                status = response.json()
                
                if status.get("status") != last_status:
                    print(f"Request {request_id[:8]}: {status.get('status')}")
                    last_status = status.get("status")
                
                if status.get("status") == "completed":
                    return status
                elif status.get("status") == "error":
                    print(f"Request {request_id[:8]} error: {status.get('error')}")
                    return status
                
            except Exception as e:
                print(f"Status check error: {e}")
            
            time.sleep(2)
        
        print(f"Request {request_id[:8]} timed out")
        return None
    
    def download_result(self, request_id, output_path):
        """下载结果"""
        try:
            response = self.session.get(f"{self.base_url}/result/{request_id}")
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"Failed to download result: {e}")
            return False

def test_concurrent_processing():
    """测试并发处理能力"""
    client = ConcurrentTestClient()
    
    print("=== Multi-GPU Concurrent Test ===")
    
    # 检查系统状态
    status = client.get_system_status()
    if not status:
        print("Cannot get system status")
        return
    
    print(f"System Status:")
    print(f"  Total GPUs: {status['total_gpus']}")
    print(f"  Ready Workers: {status['ready_workers']}")
    print(f"  Alive Processes: {status['alive_processes']}")
    
    if status["ready_workers"] == 0:
        print("No ready workers available")
        return
    
    # 准备多个测试请求
    num_requests = min(status["ready_workers"] * 2, 16)  # 每个GPU 2个请求，最多16个
    test_prompts = [
        "Change the colors to purple and gold with magical sparkles",
        "Transform into a sunset scene with warm orange and pink colors", 
        "Make it look like a winter landscape with snow and ice",
        "Convert to a vibrant spring garden with flowers and butterflies",
        "Create a futuristic cyberpunk style with neon lights",
        "Transform into an underwater scene with fish and coral",
        "Make it look like an autumn forest with falling leaves",
        "Convert to a space scene with stars and nebula",
        "Create a medieval castle scene with dragons",
        "Transform into a tropical beach with palm trees",
        "Make it look like a steampunk machine",
        "Convert to a fantasy forest with magical creatures",
        "Create a post-apocalyptic wasteland scene",
        "Transform into a Japanese zen garden",
        "Make it look like a colorful abstract painting",
        "Convert to a retro 80s style with bright colors"
    ]
    
    # 创建测试图像
    test_images = []
    for i in range(num_requests):
        filename = f"concurrent_test_{i}.png"
        client.create_test_image(filename, f"TEST{i}")
        test_images.append(filename)
    
    print(f"\nSubmitting {num_requests} concurrent requests...")
    
    # 同时提交所有请求
    submitted_requests = []
    start_submit_time = time.time()
    
    def submit_single_request(i):
        result = client.submit_request(
            test_images[i],
            test_prompts[i % len(test_prompts)],
            num_inference_steps=30,  # 减少步数加快测试
            seed=i * 42
        )
        if result:
            print(f"Request {i+1} submitted: {result['request_id'][:8]}")
            return {
                "index": i,
                "request_id": result['request_id'],
                "submitted_at": time.time()
            }
        return None
    
    # 使用线程池并发提交请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        submit_futures = [executor.submit(submit_single_request, i) for i in range(num_requests)]
        
        for future in concurrent.futures.as_completed(submit_futures):
            result = future.result()
            if result:
                submitted_requests.append(result)
    
    submit_time = time.time() - start_submit_time
    print(f"All {len(submitted_requests)} requests submitted in {submit_time:.2f}s")
    
    # 等待所有结果
    print("\nWaiting for all results...")
    
    def wait_for_single_result(req_info):
        request_id = req_info["request_id"]
        index = req_info["index"]
        
        result = client.wait_for_result(request_id)
        
        if result and result.get("status") == "completed":
            # 下载结果
            output_file = f"concurrent_result_{index}_{request_id[:8]}.png"
            if client.download_result(request_id, output_file):
                return {
                    "success": True,
                    "index": index,
                    "request_id": request_id[:8],
                    "processing_time": result['processing_time'],
                    "gpu_id": result['gpu_id'],
                    "output_file": output_file,
                    "total_time": time.time() - req_info["submitted_at"]
                }
        
        return {
            "success": False,
            "index": index,
            "request_id": request_id[:8]
        }
    
    # 并发等待所有结果
    start_wait_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        wait_futures = [executor.submit(wait_for_single_result, req_info) for req_info in submitted_requests]
        
        for future in concurrent.futures.as_completed(wait_futures):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                print(f"✓ Request {result['index']+1} completed: {result['processing_time']:.2f}s processing, {result['total_time']:.2f}s total (GPU {result['gpu_id']})")
            else:
                print(f"✗ Request {result['index']+1} failed")
    
    total_time = time.time() - start_submit_time
    successful_results = [r for r in results if r["success"]]
    
    # 统计结果
    print(f"\n=== Concurrent Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {len(successful_results)}")
    print(f"Failed requests: {num_requests - len(successful_results)}")
    print(f"Total test time: {total_time:.2f}s")
    
    if successful_results:
        avg_processing_time = sum(r["processing_time"] for r in successful_results) / len(successful_results)
        avg_total_time = sum(r["total_time"] for r in successful_results) / len(successful_results)
        
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Average total time: {avg_total_time:.2f}s")
        
        # GPU使用统计
        gpu_usage = {}
        for r in successful_results:
            gpu_id = r["gpu_id"]
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
        
        print("GPU usage distribution:", dict(sorted(gpu_usage.items())))
        
        # 计算并发效率
        sequential_time = sum(r["processing_time"] for r in successful_results)
        concurrent_efficiency = sequential_time / total_time
        print(f"Concurrency efficiency: {concurrent_efficiency:.2f}x")
    
    # 最终系统状态
    final_status = client.get_system_status()
    if final_status:
        print(f"\nFinal system status:")
        print(f"  Completed requests: {final_status['completed_requests']}")
        print(f"  Failed requests: {final_status['failed_requests']}")
        print(f"  Ready workers: {final_status['ready_workers']}")
    
    # 清理测试文件
    for filename in test_images:
        try:
            Path(filename).unlink()
        except:
            pass
    
    print("\nConcurrent test completed!")

if __name__ == "__main__":
    test_concurrent_processing()