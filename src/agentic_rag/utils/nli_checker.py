"""
NLI (Natural Language Inference) 一致性校验模块

面试核心考点：
1. DeBERTa NLI模型：判断生成回答与检索文档的逻辑蕴含关系
2. 三分类：entailment(蕴含)/contradiction(矛盾)/neutral(中立)
3. 用途：校验RAG回答是否忠实于原文，防止幻觉、编造事实
4. 降级策略：模型不可用时回退规则校验

工业实践：
- 生成后校验：answer vs context → NLI判断
- 矛盾检测：contradiction → 标记为幻觉，触发重生成或拦截
- 中立判断：neutral → 可能是合理推断，标记但不过滤
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class NLIVerdict(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class NLIConsistencyChecker:
    """NLI一致性校验器"""

    def __init__(self, use_deberta: bool = True):
        self._model = None
        self._tokenizer = None
        self._use_deberta = use_deberta
        self._model_loaded = False

        if use_deberta:
            self._try_load_model()

    def _try_load_model(self):
        """尝试加载DeBERTa NLI模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "cross-encoder/nli-deberta-v3-large"
            logger.info(f"[NLI] 尝试加载模型：{model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            self._model_loaded = True

            logger.info("[NLI] DeBERTa NLI模型加载成功")
        except ImportError:
            logger.warning("[NLI] transformers未安装，回退规则校验")
            self._use_deberta = False
        except Exception as e:
            logger.warning(f"[NLI] DeBERTa模型加载失败：{e}，回退规则校验")
            self._use_deberta = False

    def check_consistency(
        self,
        answer: str,
        contexts: List[str],
        threshold: float = 0.5
    ) -> Dict:
        """
        校验生成回答与检索文档的一致性

        Args:
            answer: 生成的回答
            contexts: 检索到的文档片段列表
            threshold: 矛盾判定阈值

        Returns:
            {
                'verdict': NLIVerdict,
                'consistency_score': float,
                'contradiction_details': List[Dict],
                'is_hallucination': bool,
                'suggestion': str
            }
        """
        if not answer or not contexts:
            return {
                'verdict': NLIVerdict.NEUTRAL,
                'consistency_score': 0.5,
                'contradiction_details': [],
                'is_hallucination': False,
                'suggestion': '缺少回答或上下文，无法校验'
            }

        if self._use_deberta and self._model_loaded:
            return self._model_based_check(answer, contexts, threshold)
        else:
            return self._rule_based_check(answer, contexts, threshold)

    def _model_based_check(
        self,
        answer: str,
        contexts: List[str],
        threshold: float
    ) -> Dict:
        """基于DeBERTa NLI模型的一致性校验"""
        import torch

        context_text = " ".join(contexts[:3])

        inputs = self._tokenizer(
            context_text, answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_list = probs[0].tolist()

        label_map = {0: NLIVerdict.CONTRADICTION, 1: NLIVerdict.NEUTRAL, 2: NLIVerdict.ENTAILMENT}
        max_idx = max(range(len(probs_list)), key=lambda i: probs_list[i])
        verdict = label_map.get(max_idx, NLIVerdict.NEUTRAL)

        contradiction_prob = probs_list[0]
        entailment_prob = probs_list[2]
        consistency_score = entailment_prob

        is_hallucination = contradiction_prob > threshold

        contradiction_details = []
        if is_hallucination:
            contradiction_details.append({
                "type": "nli_contradiction",
                "contradiction_prob": round(contradiction_prob, 3),
                "entailment_prob": round(entailment_prob, 3),
            })

        suggestion = ""
        if is_hallucination:
            suggestion = "检测到幻觉：回答与文档矛盾，建议重生成或标注不确定"
        elif verdict == NLIVerdict.NEUTRAL:
            suggestion = "回答包含文档未明确提及的信息，建议标注为推断"
        else:
            suggestion = "回答与文档一致"

        logger.info(
            f"[NLI] verdict={verdict.value}, consistency={consistency_score:.3f}, "
            f"hallucination={is_hallucination}"
        )

        return {
            'verdict': verdict,
            'consistency_score': round(consistency_score, 3),
            'contradiction_details': contradiction_details,
            'is_hallucination': is_hallucination,
            'suggestion': suggestion
        }

    def _rule_based_check(
        self,
        answer: str,
        contexts: List[str],
        threshold: float
    ) -> Dict:
        """规则校验（降级方案）：基于关键词重叠和否定词检测"""
        context_text = " ".join(contexts)
        context_chars = set(context_text)
        answer_chars = set(answer)

        if not context_chars or not answer_chars:
            consistency_score = 0.5
        else:
            overlap = context_chars & answer_chars
            consistency_score = len(overlap) / len(answer_chars)

        negation_words = ["不是", "没有", "并非", "不可能", "绝不", "从不", "无法"]
        contradiction_indicators = []
        for neg in negation_words:
            if neg in answer and neg not in context_text:
                for kw in answer:
                    if kw in context_text and neg + kw in answer:
                        contradiction_indicators.append(neg)

        is_hallucination = len(contradiction_indicators) >= 2 or consistency_score < 0.2

        if is_hallucination:
            verdict = NLIVerdict.CONTRADICTION
        elif consistency_score >= 0.5:
            verdict = NLIVerdict.ENTAILMENT
        else:
            verdict = NLIVerdict.NEUTRAL

        contradiction_details = []
        if contradiction_indicators:
            contradiction_details.append({
                "type": "rule_contradiction",
                "indicators": contradiction_indicators,
                "consistency_score": round(consistency_score, 3),
            })

        suggestion = ""
        if is_hallucination:
            suggestion = "规则校验检测到潜在幻觉，建议人工复核"
        elif verdict == NLIVerdict.NEUTRAL:
            suggestion = "回答包含文档未明确提及的信息"
        else:
            suggestion = "回答与文档基本一致（规则校验）"

        logger.info(
            f"[NLI-规则] verdict={verdict.value}, consistency={consistency_score:.3f}, "
            f"hallucination={is_hallucination}"
        )

        return {
            'verdict': verdict,
            'consistency_score': round(consistency_score, 3),
            'contradiction_details': contradiction_details,
            'is_hallucination': is_hallucination,
            'suggestion': suggestion
        }


nli_checker = NLIConsistencyChecker(use_deberta=False)
