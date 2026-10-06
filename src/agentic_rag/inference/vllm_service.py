"""
vLLM 推理服务优化模块

==================== PagedAttention 原理 ====================

传统推理方式的显存问题：
    传统 LLM 推理为每个序列预分配固定大小的 KV Cache（最大序列长度 × 层数 × 头维度），
    导致大量显存浪费。实际推理中，短序列的 KV Cache 大量空间闲置，显存利用率仅约 20%。

PagedAttention 核心思想：
    将 KV Cache 划分为固定大小的 block（页），按需分配，类似操作系统的虚拟内存分页机制。
    - 每个序列的 KV Cache 由一组 block 组成，block 之间不需要物理连续
    - 新 token 生成时，仅分配新的 block，不预分配全部空间
    - block 大小（block_size）通常为 16，即每个 block 存储 16 个 token 的 KV 向量

关键概念：
    - block_size: 每个 block 存储的 token 数量（默认 16）
    - num_gpu_blocks: GPU 上可用的 block 总数，决定了最大并发序列数
    - swap_space: CPU 内存中用于换出的空间，当 GPU block 不足时将冷数据换出到 CPU
    - block_table: 每个序列的 block 映射表，记录逻辑 block → 物理 block 的对应关系

PagedAttention 效果：
    - 显存利用率：从 ~20% 提升到 ~90%
    - 吞吐量：提升 2-4x（同等显存下可服务更多并发请求）
    - 支持 KV Cache 跨请求共享（Prefix Caching），相同前缀的请求可复用 KV Cache

==================== Continuous Batching ====================

传统 Static Batching：
    等待 batch 中所有序列完成生成后，才处理下一批请求。短序列完成后空等长序列，
    导致 GPU 利用率低。

Continuous Batching（迭代级调度）：
    每次迭代（生成一个 token）后，检查是否有序列完成。完成的序列立即移出 batch，
    新请求立即加入，无需等待整个 batch 完成。效果：GPU 利用率接近 100%，延迟降低。

==================== 量化对比 ====================

| 方法   | 显存占用  | 推理速度  | 质量损失  | 适用场景              |
|--------|----------|----------|----------|----------------------|
| FP16   | 1x（基准）| 基准      | 无       | 精度敏感场景          |
| INT8   | ~0.5x    | ~1.5-2x  | 极小      | 通用部署              |
| INT4   | ~0.25x   | ~2-3x    | 较小      | 资源受限场景          |
| GPTQ   | ~0.25x   | ~2-3x    | 小        | 离线量化，精度保持好   |
| AWQ    | ~0.25x   | ~2-3x    | 小        | 在线量化，激活感知    |

==================== Speculative Decoding ====================

原理：使用小模型（draft model）快速生成候选 token，大模型（target model）并行验证。
    - 小模型自回归生成 k 个候选 token
    - 大模型一次前向传播验证所有候选 token
    - 接受的 token 直接输出，拒绝的 token 从拒绝位置重新生成

效果：在保证输出质量完全一致的前提下，推理速度提升 2-3x（接受率高时）。
"""

import os
import time
import json
import logging
import subprocess
import signal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 检测 vLLM 是否可用
_VLLM_AVAILABLE = False
try:
    import vllm  # noqa: F401
    _VLLM_AVAILABLE = True
except ImportError:
    pass


@dataclass
class InferenceConfig:
    """vLLM 推理服务配置"""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    quantization: Optional[str] = None  # None / "awq" / "gptq" / "squeezellm"
    dtype: str = "auto"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True


@dataclass
class QuantizationConfig:
    """量化对比基准测试配置"""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization_methods: List[str] = field(
        default_factory=lambda: ["fp16", "int8", "int4", "awq"]
    )
    test_prompts: List[str] = field(default_factory=list)
    max_tokens: int = 128
    num_requests: int = 10


@dataclass
class BenchmarkResult:
    """单次量化基准测试结果"""

    method: str
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    model_size_mb: float = 0.0
    quality_score: Optional[float] = None  # 与 FP16 对比的质量评分


class VLLMService:
    """vLLM 推理服务管理器

    支持：
    1. 本地部署 vLLM 推理服务
    2. OpenAI 兼容 API 调用
    3. 量化模型加载
    4. 性能监控

    当 vLLM 未安装时自动切换到 Mock 模式，返回模拟数据。
    """

    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self._process: Optional[subprocess.Popen] = None
        self._mock_mode = not _VLLM_AVAILABLE

        if self._mock_mode:
            logger.warning(
                "[VLLMService] vLLM 未安装，使用 Mock 模式。"
                "安装方法：pip install vllm"
            )
        else:
            logger.info(f"[VLLMService] vLLM 可用，模型：{self.config.model_name}")

    def start_server(self) -> bool:
        """启动 vLLM 服务（子进程）

        通过 subprocess 启动 vLLM 的 OpenAI 兼容 API 服务器。
        Mock 模式下模拟启动成功。

        Returns:
            是否启动成功
        """
        if self._mock_mode:
            logger.info("[VLLMService] Mock 模式：模拟启动 vLLM 服务")
            return True

        if self.is_running():
            logger.warning("[VLLMService] 服务已在运行中")
            return True

        try:
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.config.model_name,
                "--host", self.config.host,
                "--port", str(self.config.port),
                "--tensor-parallel-size", str(self.config.tensor_parallel_size),
                "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
                "--max-model-len", str(self.config.max_model_len),
                "--dtype", self.config.dtype,
            ]

            if self.config.quantization:
                cmd.extend(["--quantization", self.config.quantization])

            if self.config.enable_prefix_caching:
                cmd.append("--enable-prefix-caching")

            if self.config.enable_chunked_prefill:
                cmd.append("--enable-chunked-prefill")

            logger.info(f"[VLLMService] 启动命令：{' '.join(cmd)}")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 等待服务启动（简化检测，实际应轮询 /health 端点）
            time.sleep(5)

            if self._process.poll() is not None:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                logger.error(f"[VLLMService] 服务启动失败：{stderr_output[:500]}")
                return False

            logger.info(
                f"[VLLMService] 服务已启动：http://{self.config.host}:{self.config.port}"
            )
            return True

        except Exception as e:
            logger.error(f"[VLLMService] 启动异常：{e}")
            return False

    def stop_server(self) -> bool:
        """停止 vLLM 服务

        向子进程发送终止信号，优雅关闭服务。

        Returns:
            是否停止成功
        """
        if self._mock_mode:
            logger.info("[VLLMService] Mock 模式：模拟停止服务")
            return True

        if not self.is_running():
            logger.warning("[VLLMService] 服务未在运行")
            return True

        try:
            # 优雅终止
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # 强制终止
                self._process.kill()
                self._process.wait(timeout=5)

            logger.info("[VLLMService] 服务已停止")
            self._process = None
            return True

        except Exception as e:
            logger.error(f"[VLLMService] 停止异常：{e}")
            return False

    def is_running(self) -> bool:
        """检查 vLLM 服务是否在运行

        Returns:
            服务是否运行中
        """
        if self._mock_mode:
            return True

        if self._process is None:
            return False

        return self._process.poll() is None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict:
        """生成文本

        通过 OpenAI 兼容 API 调用 vLLM 服务生成文本。
        Mock 模式下返回预设回答。

        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            生成结果字典，包含 text、usage、model 等字段
        """
        if self._mock_mode:
            return self._mock_generate(prompt, max_tokens, temperature)

        if not self.is_running():
            logger.error("[VLLMService] 服务未运行，无法生成")
            return {"text": "", "usage": {}, "model": "", "error": "服务未运行"}

        try:
            import requests

            url = f"http://{self.config.host}:{self.config.port}/v1/completions"
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                logger.error(f"[VLLMService] 生成失败：HTTP {response.status_code}")
                return {"text": "", "usage": {}, "model": "", "error": f"HTTP {response.status_code}"}

            result = response.json()
            choices = result.get("choices", [])
            text = choices[0].get("text", "") if choices else ""

            return {
                "text": text,
                "usage": result.get("usage", {}),
                "model": result.get("model", self.config.model_name),
                "latency_ms": round(elapsed_ms, 2),
            }

        except Exception as e:
            logger.error(f"[VLLMService] 生成异常：{e}")
            return {"text": "", "usage": {}, "model": "", "error": str(e)}

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
    ) -> List[Dict]:
        """批量生成文本

        利用 vLLM 的 Continuous Batching 能力，高效处理多个请求。
        Mock 模式下逐个模拟生成。

        Args:
            prompts: 输入提示词列表
            max_tokens: 最大生成 token 数

        Returns:
            生成结果列表
        """
        if self._mock_mode:
            return [
                self._mock_generate(p, max_tokens, 0.7)
                for p in prompts
            ]

        if not self.is_running():
            logger.error("[VLLMService] 服务未运行，无法批量生成")
            return [{"text": "", "usage": {}, "model": "", "error": "服务未运行"}] * len(prompts)

        try:
            import requests

            url = f"http://{self.config.host}:{self.config.port}/v1/completions"
            results = []

            start_time = time.time()

            # vLLM 内部自动做 Continuous Batching，这里发送多个请求
            for prompt in prompts:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                }
                response = requests.post(url, json=payload, timeout=120)

                if response.status_code != 200:
                    results.append({"text": "", "usage": {}, "model": "", "error": f"HTTP {response.status_code}"})
                    continue

                result = response.json()
                choices = result.get("choices", [])
                text = choices[0].get("text", "") if choices else ""
                results.append({
                    "text": text,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", self.config.model_name),
                })

            total_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[VLLMService] 批量生成完成：{len(prompts)} 条，总耗时 {total_ms:.0f}ms"
            )

            return results

        except Exception as e:
            logger.error(f"[VLLMService] 批量生成异常：{e}")
            return [{"text": "", "usage": {}, "model": "", "error": str(e)}] * len(prompts)

    def get_model_info(self) -> Dict:
        """获取模型信息

        通过 /v1/models 端点获取当前加载的模型信息。
        Mock 模式下返回配置中的模型信息。

        Returns:
            模型信息字典
        """
        if self._mock_mode:
            return {
                "model": self.config.model_name,
                "max_model_len": self.config.max_model_len,
                "quantization": self.config.quantization or "fp16",
                "dtype": self.config.dtype,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "prefix_caching": self.config.enable_prefix_caching,
                "chunked_prefill": self.config.enable_chunked_prefill,
                "mock_mode": True,
            }

        try:
            import requests

            url = f"http://{self.config.host}:{self.config.port}/v1/models"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

            result = response.json()
            data = result.get("data", [])
            if data:
                model_info = data[0]
                model_info.update({
                    "max_model_len": self.config.max_model_len,
                    "quantization": self.config.quantization or "fp16",
                    "prefix_caching": self.config.enable_prefix_caching,
                    "chunked_prefill": self.config.enable_chunked_prefill,
                })
                return model_info

            return {"model": self.config.model_name, "mock_mode": False}

        except Exception as e:
            logger.error(f"[VLLMService] 获取模型信息异常：{e}")
            return {"error": str(e)}

    def get_server_stats(self) -> Dict:
        """获取服务统计信息

        返回服务运行状态、请求统计等。
        Mock 模式下返回模拟统计数据。

        Returns:
            服务统计字典
        """
        if self._mock_mode:
            return {
                "status": "running",
                "model": self.config.model_name,
                "mock_mode": True,
                "uptime_seconds": 0,
                "num_requests": 0,
                "avg_latency_ms": 0,
                "gpu_memory_used_mb": 0,
                "gpu_memory_total_mb": 0,
            }

        if not self.is_running():
            return {"status": "stopped", "mock_mode": False}

        try:
            import requests

            # vLLM 的 /stats 端点（如果可用）
            url = f"http://{self.config.host}:{self.config.port}/stats"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return response.json()

            # 降级：返回基本信息
            return {
                "status": "running",
                "model": self.config.model_name,
                "mock_mode": False,
            }

        except Exception as e:
            return {"status": "running", "error": str(e), "mock_mode": False}

    def _mock_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict:
        """Mock 模式下的模拟生成

        根据输入 prompt 返回预设回答，模拟真实推理延迟。

        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            模拟生成结果字典
        """
        # 模拟推理延迟（50-150ms）
        mock_latency = 0.05 + (hash(prompt) % 100) / 1000
        time.sleep(mock_latency)

        # 根据输入内容返回合理的模拟回答
        mock_text = (
            f"这是一个基于 vLLM 的模拟回答。"
            f"在 Mock 模式下，vLLM 服务未实际启动。"
            f"输入提示词长度：{len(prompt)} 字符。"
            f"如需实际推理，请安装 vLLM 并确保 GPU 可用。"
        )

        # 模拟 token 数量
        mock_tokens = min(max_tokens, len(mock_text) // 2)

        return {
            "text": mock_text,
            "usage": {
                "prompt_tokens": len(prompt) // 2,
                "completion_tokens": mock_tokens,
                "total_tokens": len(prompt) // 2 + mock_tokens,
            },
            "model": self.config.model_name,
            "latency_ms": round(mock_latency * 1000, 2),
            "mock_mode": True,
        }


class QuantizationBenchmark:
    """量化对比基准测试

    对比不同量化方法（FP16/INT8/INT4/AWQ/GPTQ）的推理性能：
    - 延迟：平均/P50/P99
    - 吞吐量：tokens/sec
    - 显存占用
    - 模型大小
    - 质量评分（与 FP16 对比）

    当 vLLM 未安装时使用 Mock 模式，返回合理的模拟数据。
    """

    # Mock 模式下各量化方法的基准数据
    _MOCK_BENCHMARK_DATA = {
        "fp16": {
            "avg_latency_ms": 100.0,
            "p50_latency_ms": 95.0,
            "p99_latency_ms": 150.0,
            "throughput_tokens_per_sec": 500.0,
            "memory_usage_mb": 1024.0,
            "model_size_mb": 1024.0,
            "quality_score": 1.0,
        },
        "int8": {
            "avg_latency_ms": 60.0,
            "p50_latency_ms": 55.0,
            "p99_latency_ms": 90.0,
            "throughput_tokens_per_sec": 800.0,
            "memory_usage_mb": 550.0,
            "model_size_mb": 512.0,
            "quality_score": 0.98,
        },
        "int4": {
            "avg_latency_ms": 40.0,
            "p50_latency_ms": 35.0,
            "p99_latency_ms": 70.0,
            "throughput_tokens_per_sec": 1200.0,
            "memory_usage_mb": 320.0,
            "model_size_mb": 256.0,
            "quality_score": 0.95,
        },
        "awq": {
            "avg_latency_ms": 50.0,
            "p50_latency_ms": 45.0,
            "p99_latency_ms": 80.0,
            "throughput_tokens_per_sec": 1000.0,
            "memory_usage_mb": 350.0,
            "model_size_mb": 280.0,
            "quality_score": 0.97,
        },
        "gptq": {
            "avg_latency_ms": 55.0,
            "p50_latency_ms": 50.0,
            "p99_latency_ms": 85.0,
            "throughput_tokens_per_sec": 950.0,
            "memory_usage_mb": 340.0,
            "model_size_mb": 270.0,
            "quality_score": 0.96,
        },
    }

    # 默认测试 prompt
    _DEFAULT_TEST_PROMPTS = [
        "请解释什么是 PagedAttention，它如何优化 LLM 推理？",
        "vLLM 的 Continuous Batching 机制与传统 Static Batching 有何区别？",
        "比较 INT4 量化和 AWQ 量化的优缺点。",
        "Speculative Decoding 如何在不损失质量的前提下加速推理？",
        "解释 KV Cache 的分页管理原理。",
    ]

    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self._mock_mode = not _VLLM_AVAILABLE

        if not self.config.test_prompts:
            self.config.test_prompts = self._DEFAULT_TEST_PROMPTS

        if self._mock_mode:
            logger.warning(
                "[QuantizationBenchmark] vLLM 未安装，使用 Mock 模式返回模拟数据"
            )

    def run_benchmark(self) -> List[BenchmarkResult]:
        """运行完整基准测试

        对配置中所有量化方法逐一测试，返回结果列表。

        Returns:
            各量化方法的 BenchmarkResult 列表
        """
        results = []
        logger.info(
            f"[QuantizationBenchmark] 开始基准测试，方法：{self.config.quantization_methods}"
        )

        for method in self.config.quantization_methods:
            logger.info(f"[QuantizationBenchmark] 测试量化方法：{method}")
            result = self.run_single(method)
            results.append(result)

        logger.info(f"[QuantizationBenchmark] 基准测试完成，共 {len(results)} 个方法")
        return results

    def run_single(self, method: str) -> BenchmarkResult:
        """单方法基准测试

        对指定量化方法运行 num_requests 次推理，统计延迟、吞吐量等指标。
        Mock 模式下返回预设模拟数据。

        Args:
            method: 量化方法名称（fp16/int8/int4/awq/gptq）

        Returns:
            该方法的 BenchmarkResult
        """
        if self._mock_mode:
            return self._mock_run_single(method)

        # 实际 vLLM 测试流程
        quantization_param = None if method == "fp16" else method

        config = InferenceConfig(
            model_name=self.config.model_name,
            quantization=quantization_param,
        )
        service = VLLMService(config)

        if not service.start_server():
            logger.error(f"[QuantizationBenchmark] 无法启动 {method} 服务")
            return BenchmarkResult(
                method=method,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p99_latency_ms=0,
                throughput_tokens_per_sec=0,
                memory_usage_mb=0,
                model_size_mb=0,
                quality_score=None,
            )

        try:
            latencies = []
            total_tokens = 0

            for i in range(self.config.num_requests):
                prompt_idx = i % len(self.config.test_prompts)
                prompt = self.config.test_prompts[prompt_idx]

                result = service.generate(prompt, max_tokens=self.config.max_tokens)

                if "error" in result and result.get("error"):
                    logger.warning(f"[QuantizationBenchmark] 请求 {i} 失败：{result['error']}")
                    continue

                latency = result.get("latency_ms", 0)
                latencies.append(latency)
                total_tokens += result.get("usage", {}).get("completion_tokens", 0)

            if not latencies:
                return BenchmarkResult(
                    method=method,
                    avg_latency_ms=0,
                    p50_latency_ms=0,
                    p99_latency_ms=0,
                    throughput_tokens_per_sec=0,
                    memory_usage_mb=0,
                    model_size_mb=0,
                )

            sorted_latencies = sorted(latencies)
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            p50 = self._percentile(sorted_latencies, 50)
            p99 = self._percentile(sorted_latencies, 99)

            total_time_sec = sum(sorted_latencies) / 1000
            throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0

            stats = service.get_server_stats()
            memory_mb = stats.get("gpu_memory_used_mb", 0)

            model_size_mb = self._estimate_model_size(method)

            return BenchmarkResult(
                method=method,
                avg_latency_ms=round(avg_latency, 2),
                p50_latency_ms=round(p50, 2),
                p99_latency_ms=round(p99, 2),
                throughput_tokens_per_sec=round(throughput, 2),
                memory_usage_mb=round(memory_mb, 2),
                model_size_mb=round(model_size_mb, 2),
            )

        finally:
            service.stop_server()

    def compare_results(self, results: List[BenchmarkResult]) -> Dict:
        """对比分析各量化方法结果

        以 FP16 为基准，计算各方法的相对性能指标。

        Args:
            results: BenchmarkResult 列表

        Returns:
            对比分析字典，包含各方法相对于 FP16 的性能比
        """
        if not results:
            return {}

        # 找到 FP16 基准
        fp16_result = None
        for r in results:
            if r.method == "fp16":
                fp16_result = r
                break

        comparison = {
            "baseline": "fp16",
            "methods": {},
        }

        for r in results:
            method_data = {
                "avg_latency_ms": r.avg_latency_ms,
                "p50_latency_ms": r.p50_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                "memory_usage_mb": r.memory_usage_mb,
                "model_size_mb": r.model_size_mb,
                "quality_score": r.quality_score,
            }

            if fp16_result and r.method != "fp16":
                # 计算相对 FP16 的比值
                if fp16_result.avg_latency_ms > 0:
                    method_data["latency_ratio"] = round(
                        r.avg_latency_ms / fp16_result.avg_latency_ms, 3
                    )
                if fp16_result.throughput_tokens_per_sec > 0:
                    method_data["throughput_ratio"] = round(
                        r.throughput_tokens_per_sec / fp16_result.throughput_tokens_per_sec, 3
                    )
                if fp16_result.memory_usage_mb > 0:
                    method_data["memory_ratio"] = round(
                        r.memory_usage_mb / fp16_result.memory_usage_mb, 3
                    )
                if fp16_result.model_size_mb > 0:
                    method_data["size_ratio"] = round(
                        r.model_size_mb / fp16_result.model_size_mb, 3
                    )

            comparison["methods"][r.method] = method_data

        return comparison

    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """生成可读的基准测试报告

        Args:
            results: BenchmarkResult 列表

        Returns:
            格式化的报告字符串
        """
        comparison = self.compare_results(results)
        if not comparison:
            return "无测试结果"

        lines = []
        lines.append("=" * 70)
        lines.append("vLLM 量化对比基准测试报告")
        lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"模型：{self.config.model_name}")
        lines.append(f"测试请求数：{self.config.num_requests}")
        lines.append(f"最大 Token 数：{self.config.max_tokens}")
        lines.append("=" * 70)
        lines.append("")

        # 表头
        header = f"{'方法':<8} {'延迟(ms)':<12} {'P50(ms)':<10} {'P99(ms)':<10} {'吞吐(t/s)':<12} {'显存(MB)':<10} {'模型(MB)':<10} {'质量':<6}"
        lines.append(header)
        lines.append("-" * 70)

        for r in results:
            quality = f"{r.quality_score:.2f}" if r.quality_score is not None else "N/A"
            line = (
                f"{r.method:<8} "
                f"{r.avg_latency_ms:<12.1f} "
                f"{r.p50_latency_ms:<10.1f} "
                f"{r.p99_latency_ms:<10.1f} "
                f"{r.throughput_tokens_per_sec:<12.1f} "
                f"{r.memory_usage_mb:<10.1f} "
                f"{r.model_size_mb:<10.1f} "
                f"{quality:<6}"
            )
            lines.append(line)

        lines.append("")
        lines.append("-" * 70)
        lines.append("相对 FP16 基准对比：")
        lines.append("")

        methods = comparison.get("methods", {})
        for method_name, data in methods.items():
            if method_name == "fp16":
                continue
            latency_ratio = data.get("latency_ratio", "N/A")
            throughput_ratio = data.get("throughput_ratio", "N/A")
            memory_ratio = data.get("memory_ratio", "N/A")
            size_ratio = data.get("size_ratio", "N/A")

            lines.append(
                f"  {method_name.upper():<6} → "
                f"延迟 ×{latency_ratio}, "
                f"吞吐 ×{throughput_ratio}, "
                f"显存 ×{memory_ratio}, "
                f"模型大小 ×{size_ratio}"
            )

        lines.append("")
        lines.append("=" * 70)

        if self._mock_mode:
            lines.append("⚠️ 以上数据为 Mock 模式模拟数据，仅供参考")
            lines.append("安装 vLLM 后可运行实际基准测试：pip install vllm")

        return "\n".join(lines)

    def _mock_run_single(self, method: str) -> BenchmarkResult:
        """Mock 模式下的单方法测试

        返回预设的模拟基准数据，数据基于典型量化性能比例。

        Args:
            method: 量化方法名称

        Returns:
            模拟的 BenchmarkResult
        """
        mock_data = self._MOCK_BENCHMARK_DATA.get(method, self._MOCK_BENCHMARK_DATA["fp16"])

        # 添加随机波动（±5%），使多次测试结果不完全相同
        import random
        jitter = 0.95 + random.random() * 0.1

        return BenchmarkResult(
            method=method,
            avg_latency_ms=round(mock_data["avg_latency_ms"] * jitter, 2),
            p50_latency_ms=round(mock_data["p50_latency_ms"] * jitter, 2),
            p99_latency_ms=round(mock_data["p99_latency_ms"] * jitter, 2),
            throughput_tokens_per_sec=round(mock_data["throughput_tokens_per_sec"] / jitter, 2),
            memory_usage_mb=round(mock_data["memory_usage_mb"] * jitter, 2),
            model_size_mb=mock_data["model_size_mb"],
            quality_score=mock_data["quality_score"],
        )

    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        """计算百分位数

        Args:
            sorted_values: 已排序的数值列表
            p: 百分位（0-100）

        Returns:
            百分位数值
        """
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p / 100.0
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[f]
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    @staticmethod
    def _estimate_model_size(method: str) -> float:
        """估算模型大小（MB）

        基于 Qwen2.5-0.5B 模型估算不同量化方法下的模型大小。

        Args:
            method: 量化方法

        Returns:
            估算的模型大小（MB）
        """
        # Qwen2.5-0.5B FP16 约 1GB
        base_size_mb = 1024.0
        size_ratios = {
            "fp16": 1.0,
            "int8": 0.5,
            "int4": 0.25,
            "awq": 0.27,
            "gptq": 0.26,
            "squeezellm": 0.25,
        }
        ratio = size_ratios.get(method, 1.0)
        return base_size_mb * ratio
