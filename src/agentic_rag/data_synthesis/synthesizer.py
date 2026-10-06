"""
数据合成管线模块

核心概念：
- Self-Instruct：用 LLM 自动生成指令数据，从种子指令出发，通过 LLM 扩展生成大量多样化指令
- Evol-Instruct：指令进化，简单→复杂→高难度，逐步提升指令质量和难度
- 质量过滤：多样性去重（ROUGE-L）+ 质量评分 + 规则过滤，确保数据质量
- 偏好数据生成：chosen/rejected 对自动构建（用于 DPO/RLHF 训练）
- 数据飞轮：合成数据→训练→评测→迭代，持续提升模型能力

流程：Self-Instruct → Evol-Instruct → 质量过滤 → 偏好数据生成 → 输出

参考论文：
- Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang et al., 2023)
- WizardLM: Empowering Large Language Models to Follow Complex Instructions (Xu et al., 2023)
- Direct Preference Optimization (Rafailov et al., 2023)
"""

import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 默认种子指令（RAG 场景）
# ============================================================================

DEFAULT_SEED_INSTRUCTIONS: List[str] = [
    "查询员工年假规定",
    "对比不同配件的价格和库存",
    "如何提交报销申请",
    "维修工单的创建流程是什么",
    "查询某员工的在职状态和基本信息",
    "低库存配件有哪些需要补货",
    "病假需要提供什么证明材料",
    "公司加班费如何计算",
    "如何办理离职手续",
    "差旅费报销标准和流程",
]


# ============================================================================
# Self-Instruct 数据生成器
# ============================================================================

@dataclass
class SelfInstructConfig:
    """Self-Instruct 配置

    Args:
        seed_instructions: 种子指令列表
        num_to_generate: 目标生成数量
        temperature: LLM 生成温度
        filter_similar_threshold: ROUGE-L 相似度阈值，超过此值视为重复
        max_instruction_length: 指令最大长度（字符数）
    """
    seed_instructions: List[str] = field(default_factory=lambda: DEFAULT_SEED_INSTRUCTIONS.copy())
    num_to_generate: int = 50
    temperature: float = 0.7
    filter_similar_threshold: float = 0.8
    max_instruction_length: int = 200


class SelfInstructGenerator:
    """Self-Instruct 数据生成器

    从种子指令出发，利用 LLM 扩展生成大量多样化指令数据。
    当 LLM 不可用时，使用模板/mock 生成。

    核心流程：
    1. 从种子池中随机采样若干种子指令
    2. 构造 Prompt 让 LLM 生成新指令
    3. 过滤低质量/重复指令
    4. 为通过过滤的指令生成回答
    """

    def __init__(self, config: SelfInstructConfig = None, llm_service=None):
        """初始化

        Args:
            config: Self-Instruct 配置，为 None 时使用默认配置
            llm_service: LLM 服务实例，需提供 generate(prompt: str) -> str 接口；
                         为 None 时使用 mock 生成
        """
        self.config = config or SelfInstructConfig()
        self.llm_service = llm_service

    def generate_instructions(self) -> List[Dict]:
        """生成指令数据

        从种子指令出发，生成多样化指令并为其构造回答。

        Returns:
            指令数据列表，每项包含 {"instruction": str, "response": str, "source": str}
        """
        seeds = self.config.seed_instructions
        results: List[Dict] = []
        generated_instructions: List[str] = list(seeds)  # 初始池包含种子

        attempts = 0
        max_attempts = self.config.num_to_generate * 3  # 防止无限循环

        while len(results) < self.config.num_to_generate and attempts < max_attempts:
            attempts += 1

            # 从种子池中随机采样 2-3 条作为参考
            sample_size = min(random.randint(2, 3), len(seeds))
            sample_seeds = random.sample(seeds, sample_size)

            # 生成新指令
            new_instruction = self._generate_single_instruction(sample_seeds)

            if not new_instruction or not new_instruction.strip():
                continue

            new_instruction = new_instruction.strip()

            # 长度过滤
            if len(new_instruction) > self.config.max_instruction_length:
                continue

            # 多样性过滤：与已有指令的 ROUGE-L 相似度
            if self._is_too_similar(new_instruction, generated_instructions):
                continue

            # 为指令生成回答
            response = self.generate_response(new_instruction)

            results.append({
                "instruction": new_instruction,
                "response": response,
                "source": "self_instruct",
            })
            generated_instructions.append(new_instruction)

        logger.info(f"[Self-Instruct] 生成完成：目标 {self.config.num_to_generate}，实际 {len(results)} 条")
        return results

    def generate_response(self, instruction: str) -> str:
        """为指令生成回答

        Args:
            instruction: 用户指令

        Returns:
            生成的回答文本
        """
        if self.llm_service is not None:
            try:
                prompt = (
                    f"请针对以下问题给出专业、准确、带引用来源的回答：\n"
                    f"问题：{instruction}\n"
                    f"要求：回答需包含具体信息，并标注来源（如文档名、页码等）。"
                )
                response = self.llm_service.generate(prompt)
                if response and response.strip():
                    return response.strip()
            except Exception as e:
                logger.warning(f"[Self-Instruct] LLM 生成回答失败: {e}")

        # Mock 生成：基于指令模板构造回答
        return self._mock_response(instruction)

    def filter_low_quality(self, data: List[Dict]) -> List[Dict]:
        """过滤低质量数据

        过滤规则：
        1. 指令或回答为空/过短
        2. 指令与回答完全相同
        3. 回答中包含明显的低质量标记

        Args:
            data: 待过滤的数据列表

        Returns:
            过滤后的数据列表
        """
        filtered = []
        for item in data:
            instruction = item.get("instruction", "").strip()
            response = item.get("response", "").strip()

            # 空值/过短过滤
            if len(instruction) < 3 or len(response) < 10:
                continue

            # 指令与回答相同
            if instruction == response:
                continue

            # 低质量标记过滤
            low_quality_markers = ["我不知道", "无法回答", "暂无信息", "请咨询"]
            if any(response.startswith(marker) for marker in low_quality_markers):
                continue

            filtered.append(item)

        logger.info(f"[Self-Instruct] 质量过滤：{len(data)} → {len(filtered)} 条")
        return filtered

    def compute_rouge_l(self, s1: str, s2: str) -> float:
        """计算 ROUGE-L 相似度（基于 LCS 算法，不依赖外部库）

        ROUGE-L = (1 - β²) * R * P / (R + β² * P)
        其中 R = LCS / len(s2), P = LCS / len(s1), β = P权重/R权重

        Args:
            s1: 字符串1（通常是生成的新指令）
            s2: 字符串2（通常是已有指令）

        Returns:
            ROUGE-L F1 分数，范围 [0, 1]
        """
        if not s1 or not s2:
            return 0.0

        # 字符级别 LCS
        lcs_len = self._lcs_length(s1, s2)

        if lcs_len == 0:
            return 0.0

        recall = lcs_len / len(s2)
        precision = lcs_len / len(s1)

        beta = 1.0  # F1 时 β=1
        f1 = ((1 + beta ** 2) * recall * precision) / (recall + beta ** 2 * precision + 1e-10)

        return f1

    def _lcs_length(self, s1: str, s2: str) -> int:
        """计算两个字符串的最长公共子序列长度

        使用动态规划实现，时间复杂度 O(m*n)。

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            LCS 长度
        """
        m, n = len(s1), len(s2)
        # 优化空间：只保留两行
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def _generate_single_instruction(self, sample_seeds: List[str]) -> str:
        """生成单条新指令

        Args:
            sample_seeds: 采样的种子指令列表

        Returns:
            生成的新指令文本
        """
        if self.llm_service is not None:
            try:
                seeds_text = "\n".join(f"- {s}" for s in sample_seeds)
                prompt = (
                    f"以下是几个用户提问示例：\n{seeds_text}\n\n"
                    f"请生成一个类似的、但不重复的用户提问。"
                    f"要求：简洁明确，与上述示例不同但属于同一领域。只输出提问本身。"
                )
                result = self.llm_service.generate(prompt)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.warning(f"[Self-Instruct] LLM 生成指令失败: {e}")

        # Mock 生成：基于种子模板变异
        return self._mock_instruction(sample_seeds)

    def _is_too_similar(self, new_instruction: str, existing: List[str]) -> bool:
        """检查新指令是否与已有指令过于相似

        Args:
            new_instruction: 新生成的指令
            existing: 已有指令列表

        Returns:
            True 表示过于相似，应被过滤
        """
        for existing_instr in existing:
            similarity = self.compute_rouge_l(new_instruction, existing_instr)
            if similarity > self.config.filter_similar_threshold:
                return True
        return False

    def _mock_instruction(self, sample_seeds: List[str]) -> str:
        """Mock 模式下生成指令

        基于种子指令进行变异生成。

        Args:
            sample_seeds: 采样的种子指令

        Returns:
            变异后的新指令
        """
        base = random.choice(sample_seeds)

        # 变异策略
        mutations = [
            lambda s: f"请详细说明{s}",
            lambda s: f"{s}的具体流程是什么",
            lambda s: f"关于{s}有哪些注意事项",
            lambda s: f"如何查询{s}",
            lambda s: f"请对比{s}的不同方案",
            lambda s: f"{s}需要准备什么材料",
            lambda s: f"公司对{s}有什么规定",
            lambda s: f"请解释{s}的相关政策",
        ]

        mutation = random.choice(mutations)
        return mutation(base)

    def _mock_response(self, instruction: str) -> str:
        """Mock 模式下生成回答

        Args:
            instruction: 用户指令

        Returns:
            模板生成的回答
        """
        templates = [
            f"根据公司相关规定，{instruction}的具体要求如下：请参考员工手册相关章节获取详细信息。来源：员工手册。",
            f"关于{instruction}，公司有明确的制度规范。建议您登录OA系统查看详细流程，或咨询相关部门。来源：内部管理制度。",
            f"{instruction}的办理需要按照以下步骤进行：1) 准备相关材料；2) 提交申请；3) 等待审批。具体细节请查阅相关制度文件。来源：操作规程。",
        ]
        return random.choice(templates)


# ============================================================================
# Evol-Instruct 指令进化器
# ============================================================================

class EvolveType(Enum):
    """指令进化类型"""
    COMPLEXIFY = "complexify"    # 增加约束条件
    DEEPEN = "deepen"            # 增加推理深度
    CONCRETIZE = "concretize"    # 从抽象到具体
    REASONING = "reasoning"      # 增加推理步骤
    BROADEN = "broaden"          # 扩展主题范围


@dataclass
class EvolInstructConfig:
    """Evol-Instruct 配置

    Args:
        num_evolve_rounds: 进化轮次
        evolve_types: 使用的进化类型列表
        max_evolved_length: 进化后指令最大长度
        reject_simple_threshold: 进化后仍太简单的拒绝阈值（ROUGE-L 相似度）
    """
    num_evolve_rounds: int = 3
    evolve_types: List[EvolveType] = field(default_factory=lambda: [
        EvolveType.COMPLEXIFY, EvolveType.DEEPEN, EvolveType.REASONING
    ])
    max_evolved_length: int = 500
    reject_simple_threshold: float = 0.6


# 进化类型对应的 Prompt 模板
_EVOLVE_PROMPTS: Dict[EvolveType, str] = {
    EvolveType.COMPLEXIFY: (
        "请将以下指令改写为更复杂的版本，增加额外的约束条件或限制：\n"
        "原指令：{instruction}\n"
        "要求：在原指令基础上增加2-3个具体约束条件，使问题更具挑战性。只输出改写后的指令。"
    ),
    EvolveType.DEEPEN: (
        "请将以下指令改写为更深度的版本，增加推理深度：\n"
        "原指令：{instruction}\n"
        "要求：使问题需要更深层次的思考和分析，涉及更多背景知识。只输出改写后的指令。"
    ),
    EvolveType.CONCRETIZE: (
        "请将以下抽象指令改写为更具体的版本：\n"
        "原指令：{instruction}\n"
        "要求：将抽象概念替换为具体的场景、数字或案例。只输出改写后的指令。"
    ),
    EvolveType.REASONING: (
        "请将以下指令改写为需要多步推理的版本：\n"
        "原指令：{instruction}\n"
        "要求：使问题需要经过多个推理步骤才能得出结论。只输出改写后的指令。"
    ),
    EvolveType.BROADEN: (
        "请将以下指令改写为更广泛的版本，扩展主题范围：\n"
        "原指令：{instruction}\n"
        "要求：将问题的范围扩展到相关领域，涉及更多维度的比较或分析。只输出改写后的指令。"
    ),
}

# Mock 进化模板
_MOCK_EVOLVE_TEMPLATES: Dict[EvolveType, List[str]] = {
    EvolveType.COMPLEXIFY: [
        "{instruction}，请同时考虑成本和时间约束",
        "{instruction}，并说明在不同情况下的差异",
        "{instruction}，要求给出具体的条件和限制",
    ],
    EvolveType.DEEPEN: [
        "请深入分析{instruction}的底层原理和依据",
        "{instruction}，请从制度设计角度解释原因",
        "{instruction}，请说明背后的政策考量",
    ],
    EvolveType.CONCRETIZE: [
        "以2024年度为例，{instruction}",
        "针对技术部门，{instruction}",
        "在预算不超过5万元的情况下，{instruction}",
    ],
    EvolveType.REASONING: [
        "{instruction}，请逐步推导并说明每一步的依据",
        "{instruction}，请分析可能的影响因素并给出判断",
        "{instruction}，请列出所有需要考虑的条件后给出结论",
    ],
    EvolveType.BROADEN: [
        "{instruction}，并对比不同部门的做法",
        "{instruction}，同时分析与行业标准的差异",
        "{instruction}，请从HR、财务、法务多个角度分析",
    ],
}


class EvolInstructGenerator:
    """Evol-Instruct 指令进化器

    将简单指令逐步进化为更复杂、更有深度的指令。
    支持五种进化方向：复杂化、深化、具体化、推理化、扩展化。

    当 LLM 不可用时，使用模板/mock 进行进化。
    """

    def __init__(self, config: EvolInstructConfig = None, llm_service=None):
        """初始化

        Args:
            config: Evol-Instruct 配置，为 None 时使用默认配置
            llm_service: LLM 服务实例，需提供 generate(prompt: str) -> str 接口；
                         为 None 时使用 mock 进化
        """
        self.config = config or EvolInstructConfig()
        self.llm_service = llm_service

    def evolve_instruction(self, instruction: str, evolve_type: EvolveType) -> Optional[str]:
        """进化单条指令

        Args:
            instruction: 原始指令
            evolve_type: 进化类型

        Returns:
            进化后的指令，如果进化无效则返回 None
        """
        if self.llm_service is not None:
            try:
                prompt_template = _EVOLVE_PROMPTS.get(evolve_type)
                if prompt_template:
                    prompt = prompt_template.format(instruction=instruction)
                    result = self.llm_service.generate(prompt)
                    if result and result.strip():
                        evolved = result.strip()
                        if self.is_evolution_valid(instruction, evolved):
                            return evolved
            except Exception as e:
                logger.warning(f"[Evol-Instruct] LLM 进化失败: {e}")

        # Mock 进化
        evolved = self._mock_evolve(instruction, evolve_type)
        if evolved and self.is_evolution_valid(instruction, evolved):
            return evolved
        return None

    def evolve_batch(self, instructions: List[str]) -> List[Dict]:
        """批量进化指令

        对每条指令进行多轮进化，每轮随机选择一种进化类型。

        Args:
            instructions: 待进化的指令列表

        Returns:
            进化后的数据列表，每项包含：
            {
                "instruction": str,        # 进化后的指令
                "original": str,           # 原始指令
                "evolve_type": str,        # 进化类型
                "evolve_round": int,       # 进化轮次
                "source": "evol_instruct"
            }
        """
        results: List[Dict] = []

        for instruction in instructions:
            current = instruction

            for round_idx in range(self.config.num_evolve_rounds):
                # 随机选择进化类型
                evolve_type = random.choice(self.config.evolve_types)
                evolved = self.evolve_instruction(current, evolve_type)

                if evolved is not None:
                    results.append({
                        "instruction": evolved,
                        "original": instruction,
                        "evolve_type": evolve_type.value,
                        "evolve_round": round_idx + 1,
                        "source": "evol_instruct",
                    })
                    current = evolved  # 下一轮基于进化后的指令继续进化

        logger.info(f"[Evol-Instruct] 批量进化完成：{len(instructions)} 条输入 → {len(results)} 条进化结果")
        return results

    def is_evolution_valid(self, original: str, evolved: str) -> bool:
        """验证进化是否有效

        进化无效的条件：
        1. 进化后指令为空或过短
        2. 进化后指令过长
        3. 进化后指令与原始指令完全相同
        4. 进化后指令与原始指令过于相似（说明进化不够）

        Args:
            original: 原始指令
            evolved: 进化后的指令

        Returns:
            True 表示进化有效
        """
        # 空值/过短
        if not evolved or len(evolved.strip()) < 3:
            return False

        # 过长
        if len(evolved) > self.config.max_evolved_length:
            return False

        # 完全相同
        if original.strip() == evolved.strip():
            return False

        # 过于相似（进化不够）
        # 使用简单的字符重叠率判断
        overlap = self._compute_overlap(original, evolved)
        if overlap > self.config.reject_simple_threshold:
            return False

        return True

    def _compute_overlap(self, s1: str, s2: str) -> float:
        """计算两个字符串的字符重叠率

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            重叠率，范围 [0, 1]
        """
        if not s1 or not s2:
            return 0.0

        set1 = set(s1)
        set2 = set(s2)
        intersection = set1 & set2
        union = set1 | set2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _mock_evolve(self, instruction: str, evolve_type: EvolveType) -> Optional[str]:
        """Mock 模式下进化指令

        Args:
            instruction: 原始指令
            evolve_type: 进化类型

        Returns:
            进化后的指令
        """
        templates = _MOCK_EVOLVE_TEMPLATES.get(evolve_type, [])
        if not templates:
            return None

        template = random.choice(templates)
        return template.format(instruction=instruction)


# ============================================================================
# 偏好数据自动生成器
# ============================================================================

@dataclass
class PreferencePair:
    """偏好数据对（兼容 DPODataItem 格式）

    用于 DPO/RLHF 训练，每条数据包含一个高质量回答（chosen）和一个低质量回答（rejected）。

    Attributes:
        prompt: 用户输入
        chosen: 人类偏好的高质量回答
        rejected: 人类不偏好的低质量回答
        domain: 领域标识
        quality_gap: 质量差距，范围 (0, 1]，越大表示差距越明显
    """
    prompt: str
    chosen: str
    rejected: str
    domain: str = "general"
    quality_gap: float = 0.5


# Chosen 回答 Prompt
_CHOSEN_PROMPT = (
    "请针对以下问题给出专业、准确、详细的回答：\n"
    "问题：{prompt}\n\n"
    "要求：\n"
    "1. 回答必须包含具体信息和数据\n"
    "2. 必须标注引用来源（文档名、页码、章节等）\n"
    "3. 回答要结构化，使用编号列表\n"
    "4. 如果涉及流程，请给出完整步骤\n"
    "5. 语气专业、准确、严谨"
)

# Rejected 回答 Prompt
_REJECTED_PROMPT = (
    "请针对以下问题给出一个简短、模糊、不够专业的回答：\n"
    "问题：{prompt}\n\n"
    "要求：\n"
    "1. 回答要简短，不要展开\n"
    "2. 不要标注引用来源\n"
    "3. 可以使用模糊表述（如'好像是'、'大概'）\n"
    "4. 不要给出具体步骤或数据\n"
    "5. 语气随意、不严谨"
)

# Mock chosen 模板
_MOCK_CHOSEN_TEMPLATES = [
    "根据《{doc}》第{page}页规定，{prompt}的具体要求如下：\n1) {detail1}；\n2) {detail2}；\n3) {detail3}。\n来源：{doc}第{page}页。",
    "关于{prompt}，公司有明确的制度规范：\n1) {detail1}；\n2) {detail2}；\n3) {detail3}。\n详细信息请查阅《{doc}》第{page}页。",
    "{prompt}的办理流程如下：\n1) {detail1}；\n2) {detail2}；\n3) {detail3}。\n以上信息来源于《{doc}》第{page}页。",
]

# Mock rejected 模板
_MOCK_REJECTED_TEMPLATES = [
    "{prompt}的话，好像是有的，你去问问就知道了。",
    "这个我不太确定，好像是{detail1}吧。",
    "{prompt}应该可以，具体你看看规定。",
    "大概需要{detail1}，其他的我也不太清楚。",
]

# 领域映射
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "hr": ["年假", "请假", "员工", "入职", "离职", "考勤", "加班", "试用期", "福利"],
    "inventory": ["库存", "配件", "补货", "仓库", "供应商", "采购"],
    "repair": ["维修", "工单", "保养", "故障", "检修"],
    "finance": ["报销", "费用", "差旅", "发票", "预算", "财务"],
}


class PreferenceDataGenerator:
    """偏好数据自动生成器（用于 DPO/RLHF 训练）

    自动生成 chosen/rejected 偏好对：
    - chosen：高质量回答（带溯源、详细、准确、结构化）
    - rejected：低质量回答（模糊、简短、无溯源、不严谨）

    当 LLM 不可用时，使用模板/mock 生成。
    """

    def __init__(self, llm_service=None):
        """初始化

        Args:
            llm_service: LLM 服务实例，需提供 generate(prompt: str) -> str 接口；
                         为 None 时使用 mock 生成
        """
        self.llm_service = llm_service

    def generate_pair(self, prompt: str, domain: str = "general") -> PreferencePair:
        """生成单条偏好数据对

        Args:
            prompt: 用户输入
            domain: 领域标识

        Returns:
            偏好数据对
        """
        chosen = self._generate_chosen(prompt)
        rejected = self._generate_rejected(prompt)

        # 自动推断领域
        if domain == "general":
            domain = self._infer_domain(prompt)

        # 计算质量差距
        quality_gap = self._estimate_quality_gap(chosen, rejected)

        return PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            domain=domain,
            quality_gap=quality_gap,
        )

    def generate_batch(self, prompts: List[str], domain: str = "general") -> List[PreferencePair]:
        """批量生成偏好数据对

        Args:
            prompts: 用户输入列表
            domain: 领域标识

        Returns:
            偏好数据对列表
        """
        results = []
        for prompt in prompts:
            pair = self.generate_pair(prompt, domain)
            results.append(pair)

        logger.info(f"[Preference] 批量生成完成：{len(prompts)} 条 → {len(results)} 对偏好数据")
        return results

    def _generate_chosen(self, prompt: str) -> str:
        """生成高质量回答（带溯源/详细/准确）

        Args:
            prompt: 用户输入

        Returns:
            高质量回答文本
        """
        if self.llm_service is not None:
            try:
                llm_prompt = _CHOSEN_PROMPT.format(prompt=prompt)
                result = self.llm_service.generate(llm_prompt)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.warning(f"[Preference] LLM 生成 chosen 失败: {e}")

        return self._mock_chosen(prompt)

    def _generate_rejected(self, prompt: str) -> str:
        """生成低质量回答（模糊/简短/无溯源）

        Args:
            prompt: 用户输入

        Returns:
            低质量回答文本
        """
        if self.llm_service is not None:
            try:
                llm_prompt = _REJECTED_PROMPT.format(prompt=prompt)
                result = self.llm_service.generate(llm_prompt)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.warning(f"[Preference] LLM 生成 rejected 失败: {e}")

        return self._mock_rejected(prompt)

    def _infer_domain(self, prompt: str) -> str:
        """根据 prompt 内容推断领域

        Args:
            prompt: 用户输入

        Returns:
            推断的领域标识
        """
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if any(kw in prompt for kw in keywords):
                return domain
        return "general"

    def _estimate_quality_gap(self, chosen: str, rejected: str) -> float:
        """估算 chosen 与 rejected 之间的质量差距

        基于长度比、结构化程度、是否包含引用等启发式指标。

        Args:
            chosen: 高质量回答
            rejected: 低质量回答

        Returns:
            质量差距，范围 (0, 1]
        """
        gap = 0.3  # 基础差距

        # 长度差异贡献
        len_ratio = len(chosen) / max(len(rejected), 1)
        if len_ratio > 2:
            gap += 0.2
        elif len_ratio > 1.5:
            gap += 0.1

        # chosen 包含引用来源
        if re.search(r"来源|引用|第\d+页|《.*?》", chosen):
            gap += 0.15

        # chosen 包含编号列表
        if re.search(r"[1-9][)）]", chosen):
            gap += 0.1

        # rejected 包含模糊词
        vague_words = ["好像", "大概", "可能", "应该", "不太确定"]
        if any(w in rejected for w in vague_words):
            gap += 0.1

        return min(gap, 1.0)

    def _mock_chosen(self, prompt: str) -> str:
        """Mock 模式下生成高质量回答

        Args:
            prompt: 用户输入

        Returns:
            模板生成的高质量回答
        """
        template = random.choice(_MOCK_CHOSEN_TEMPLATES)
        docs = ["员工手册", "财务管理制度", "请假制度", "维修管理制度", "内部管理制度"]
        details_pool = [
            "准备相关证明材料并提交申请",
            "经直属领导审批后进入下一环节",
            "在规定时间内完成办理",
            "登录系统填写相关信息",
            "等待审核结果通知",
            "按照标准流程执行",
        ]

        return template.format(
            prompt=prompt,
            doc=random.choice(docs),
            page=random.randint(1, 30),
            detail1=random.choice(details_pool),
            detail2=random.choice(details_pool),
            detail3=random.choice(details_pool),
        )

    def _mock_rejected(self, prompt: str) -> str:
        """Mock 模式下生成低质量回答

        Args:
            prompt: 用户输入

        Returns:
            模板生成的低质量回答
        """
        template = random.choice(_MOCK_REJECTED_TEMPLATES)
        short_details = [
            "提交申请", "找领导签字", "去系统看看",
            "问一下人事", "查查规定", "准备材料",
        ]
        return template.format(
            prompt=prompt,
            detail1=random.choice(short_details),
        )


# ============================================================================
# 数据合成统一管线
# ============================================================================

@dataclass
class PipelineConfig:
    """数据合成管线配置

    Args:
        self_instruct_config: Self-Instruct 配置
        evol_instruct_config: Evol-Instruct 配置
        output_format: 输出格式，"dpo" / "sft" / "both"
        quality_threshold: 质量阈值，低于此值的数据被过滤
        max_output_items: 最大输出条数
    """
    self_instruct_config: SelfInstructConfig = None
    evol_instruct_config: EvolInstructConfig = None
    output_format: str = "dpo"
    quality_threshold: float = 0.5
    max_output_items: int = 100


class DataSynthesisPipeline:
    """数据合成统一管线

    流程：Self-Instruct → Evol-Instruct → 质量过滤 → 偏好数据生成 → 输出

    支持输出格式：
    - dpo: DPO 训练格式（prompt + chosen + rejected）
    - sft: SFT 训练格式（instruction + response）
    - both: 同时输出两种格式
    """

    def __init__(self, config: PipelineConfig = None, llm_service=None):
        """初始化

        Args:
            config: 管线配置，为 None 时使用默认配置
            llm_service: LLM 服务实例，需提供 generate(prompt: str) -> str 接口；
                         为 None 时各组件使用 mock 生成
        """
        self.config = config or PipelineConfig()
        self.llm_service = llm_service

        # 初始化子组件
        si_config = self.config.self_instruct_config or SelfInstructConfig()
        evol_config = self.config.evol_instruct_config or EvolInstructConfig()

        self.self_instruct = SelfInstructGenerator(si_config, llm_service)
        self.evol_instruct = EvolInstructGenerator(evol_config, llm_service)
        self.preference_gen = PreferenceDataGenerator(llm_service)

        # 管线统计
        self._stats = {
            "self_instruct_generated": 0,
            "evol_instruct_generated": 0,
            "quality_filtered": 0,
            "preference_pairs_generated": 0,
            "final_output": 0,
        }
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None

    def run(self, seed_instructions: List[str]) -> Dict[str, Any]:
        """运行完整管线

        Args:
            seed_instructions: 种子指令列表，为空时使用默认种子

        Returns:
            管线运行结果，包含：
            {
                "status": str,
                "output_format": str,
                "dpo_data": List[Dict],     # output_format 为 "dpo" 或 "both" 时
                "sft_data": List[Dict],     # output_format 为 "sft" 或 "both" 时
                "stats": Dict,
                "report": Dict,
            }
        """
        self._start_time = datetime.now().isoformat()
        self._stats = {
            "self_instruct_generated": 0,
            "evol_instruct_generated": 0,
            "quality_filtered": 0,
            "preference_pairs_generated": 0,
            "final_output": 0,
        }

        if not seed_instructions:
            seed_instructions = DEFAULT_SEED_INSTRUCTIONS.copy()

        logger.info(f"[Pipeline] 开始数据合成管线，种子指令 {len(seed_instructions)} 条")

        # 阶段1：Self-Instruct 生成
        si_data = self.run_self_instruct(seed_instructions)
        self._stats["self_instruct_generated"] = len(si_data)

        # 阶段2：Evol-Instruct 进化
        si_instructions = [item["instruction"] for item in si_data]
        evol_data = self.run_evol_instruct(si_instructions)
        self._stats["evol_instruct_generated"] = len(evol_data)

        # 合并数据
        all_data = si_data + evol_data

        # 阶段3：质量过滤
        filtered_data = self.run_quality_filter(all_data)
        self._stats["quality_filtered"] = len(filtered_data)

        # 限制输出数量
        if len(filtered_data) > self.config.max_output_items:
            filtered_data = filtered_data[: self.config.max_output_items]

        # 阶段4：偏好数据生成
        result: Dict[str, Any] = {
            "status": "success",
            "output_format": self.config.output_format,
        }

        if self.config.output_format in ("dpo", "both"):
            pairs = self.run_preference_generation(filtered_data)
            self._stats["preference_pairs_generated"] = len(pairs)
            result["dpo_data"] = self.export_for_dpo(pairs)
            self._stats["final_output"] = len(result["dpo_data"])

        if self.config.output_format in ("sft", "both"):
            result["sft_data"] = self.export_for_sft(filtered_data)
            if self.config.output_format == "sft":
                self._stats["final_output"] = len(result["sft_data"])

        self._end_time = datetime.now().isoformat()

        result["stats"] = dict(self._stats)
        result["report"] = self.generate_report()

        logger.info(f"[Pipeline] 管线完成：{self._stats}")
        return result

    def run_self_instruct(self, seeds: List[str]) -> List[Dict]:
        """运行 Self-Instruct 阶段

        Args:
            seeds: 种子指令列表

        Returns:
            Self-Instruct 生成的指令数据
        """
        # 更新种子指令
        self.self_instruct.config.seed_instructions = seeds
        data = self.self_instruct.generate_instructions()
        logger.info(f"[Pipeline] Self-Instruct 阶段：生成 {len(data)} 条")
        return data

    def run_evol_instruct(self, instructions: List[str]) -> List[Dict]:
        """运行 Evol-Instruct 阶段

        Args:
            instructions: 待进化的指令列表

        Returns:
            进化后的指令数据
        """
        if not instructions:
            return []
        data = self.evol_instruct.evolve_batch(instructions)
        logger.info(f"[Pipeline] Evol-Instruct 阶段：生成 {len(data)} 条")
        return data

    def run_quality_filter(self, data: List[Dict]) -> List[Dict]:
        """运行质量过滤阶段

        过滤规则：
        1. 低质量过滤（空值、过短、低质量标记）
        2. 多样性去重（ROUGE-L 相似度）
        3. 质量评分过滤

        Args:
            data: 待过滤的数据列表

        Returns:
            过滤后的数据列表
        """
        # 1. 低质量过滤
        filtered = self.self_instruct.filter_low_quality(data)

        # 2. 多样性去重
        seen_instructions: List[str] = []
        deduplicated = []
        for item in filtered:
            instruction = item.get("instruction", "")
            is_dup = False
            for seen in seen_instructions:
                if self.self_instruct.compute_rouge_l(instruction, seen) > 0.8:
                    is_dup = True
                    break
            if not is_dup:
                deduplicated.append(item)
                seen_instructions.append(instruction)

        # 3. 质量评分过滤
        scored = []
        for item in deduplicated:
            score = self._score_quality(item)
            if score >= self.config.quality_threshold:
                item["quality_score"] = score
                scored.append(item)

        logger.info(f"[Pipeline] 质量过滤：{len(data)} → {len(scored)} 条")
        return scored

    def run_preference_generation(self, data: List[Dict]) -> List[PreferencePair]:
        """运行偏好数据生成阶段

        Args:
            data: 经过质量过滤的指令数据

        Returns:
            偏好数据对列表
        """
        prompts = [item.get("instruction", "") for item in data]
        pairs = self.preference_gen.generate_batch(prompts)
        logger.info(f"[Pipeline] 偏好数据生成：{len(pairs)} 对")
        return pairs

    def export_for_dpo(self, pairs: List[PreferencePair]) -> List[Dict]:
        """导出 DPO 格式数据

        与项目 DPODataItem 兼容，输出格式：
        {"prompt": str, "chosen": str, "rejected": str, "domain": str, "quality_gap": float}

        Args:
            pairs: 偏好数据对列表

        Returns:
            DPO 格式数据列表
        """
        dpo_data = []
        for pair in pairs:
            dpo_data.append({
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "domain": pair.domain,
                "quality_gap": pair.quality_gap,
            })
        return dpo_data

    def export_for_sft(self, data: List[Dict]) -> List[Dict]:
        """导出 SFT 格式数据

        输出格式（兼容 Qwen 指令微调格式）：
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

        Args:
            data: 指令数据列表

        Returns:
            SFT 格式数据列表
        """
        sft_data = []
        for item in data:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            if not instruction or not response:
                continue

            sft_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个企业知识库助手，专门回答企业 HR、IT、财务等相关问题。请基于你的知识准确回答用户问题，并标注引用来源。"
                    },
                    {
                        "role": "user",
                        "content": instruction
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ]
            })
        return sft_data

    def generate_report(self) -> Dict:
        """生成管线运行报告

        Returns:
            管线报告字典
        """
        return {
            "pipeline": "DataSynthesisPipeline",
            "start_time": self._start_time,
            "end_time": self._end_time,
            "output_format": self.config.output_format,
            "stats": dict(self._stats),
            "config": {
                "self_instruct_num": (self.config.self_instruct_config or SelfInstructConfig()).num_to_generate,
                "evol_rounds": (self.config.evol_instruct_config or EvolInstructConfig()).num_evolve_rounds,
                "quality_threshold": self.config.quality_threshold,
                "max_output_items": self.config.max_output_items,
            },
        }

    def _score_quality(self, item: Dict) -> float:
        """为数据项计算质量评分

        基于启发式规则打分：
        - 指令长度适中
        - 回答长度充足
        - 回答包含结构化信息
        - 回答包含引用来源

        Args:
            item: 数据项

        Returns:
            质量评分，范围 [0, 1]
        """
        score = 0.3  # 基础分

        instruction = item.get("instruction", "")
        response = item.get("response", "")

        # 指令长度适中（10-100 字符）
        if 10 <= len(instruction) <= 100:
            score += 0.15

        # 回答长度充足（>50 字符）
        if len(response) > 50:
            score += 0.15
        if len(response) > 100:
            score += 0.1

        # 回答包含结构化信息
        if re.search(r"[1-9][)）]|第[一二三四五六七八九十\d]+|首先|其次|最后", response):
            score += 0.15

        # 回答包含引用来源
        if re.search(r"来源|引用|第\d+页|《.*?》|参考", response):
            score += 0.15

        return min(score, 1.0)


# ============================================================================
# 便捷函数
# ============================================================================

def run_synthesis_demo():
    """运行数据合成 Demo"""
    print("=" * 60)
    print("数据合成管线 Demo")
    print("=" * 60)

    # 配置
    si_config = SelfInstructConfig(
        seed_instructions=DEFAULT_SEED_INSTRUCTIONS,
        num_to_generate=10,
        filter_similar_threshold=0.8,
    )
    evol_config = EvolInstructConfig(
        num_evolve_rounds=2,
        evolve_types=[EvolveType.COMPLEXIFY, EvolveType.DEEPEN, EvolveType.REASONING],
    )
    pipeline_config = PipelineConfig(
        self_instruct_config=si_config,
        evol_instruct_config=evol_config,
        output_format="both",
        quality_threshold=0.4,
        max_output_items=20,
    )

    # 运行管线（无 LLM，使用 mock）
    pipeline = DataSynthesisPipeline(pipeline_config)
    result = pipeline.run(seed_instructions=DEFAULT_SEED_INSTRUCTIONS)

    # 输出结果
    print(f"\n管线状态：{result['status']}")
    print(f"输出格式：{result['output_format']}")

    if "dpo_data" in result:
        print(f"\n--- DPO 数据（{len(result['dpo_data'])} 条）---")
        for i, item in enumerate(result["dpo_data"][:3], 1):
            print(f"\n  [{i}] Prompt: {item['prompt']}")
            print(f"      Chosen:  {item['chosen'][:80]}...")
            print(f"      Rejected: {item['rejected'][:80]}...")
            print(f"      Domain: {item['domain']}, Gap: {item['quality_gap']}")

    if "sft_data" in result:
        print(f"\n--- SFT 数据（{len(result['sft_data'])} 条）---")
        for i, item in enumerate(result["sft_data"][:2], 1):
            msgs = item["messages"]
            print(f"\n  [{i}] User: {msgs[1]['content']}")
            print(f"      Assistant: {msgs[2]['content'][:80]}...")

    print(f"\n--- 管线报告 ---")
    report = result["report"]
    print(f"  起始时间: {report['start_time']}")
    print(f"  结束时间: {report['end_time']}")
    print(f"  统计: {report['stats']}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_synthesis_demo()
