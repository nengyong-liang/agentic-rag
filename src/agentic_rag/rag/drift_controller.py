"""
语义漂移控制器
校验扩展结果与原查询的一致性，防止查询扩展导致语义偏移

多维度检查：
1. 字符重叠率：简单快速，适合字面相近的扩展
2. Bigram Jaccard相似度：比单字符重叠更精确，捕捉局部结构相似
3. 同义词词典信任：识别有效同义词扩展（如"社保"→"五险一金"）
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.5
EXPANSION_CONSISTENCY_THRESHOLD = 0.3


class DriftController:
    """语义漂移控制器：校验扩展结果与原查询的一致性"""

    def __init__(self):
        pass

    def filter_by_threshold(self, results: List[Dict], min_score: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """基于分数阈值过滤低质量检索结果"""
        filtered = []
        for result in results:
            vector_score = result.get("vector_score", 0)
            es_score = result.get("es_score", 0)
            combined = vector_score * 0.7 + (es_score or 0) * 0.3

            if combined >= min_score or vector_score >= min_score:
                result["combined_score"] = combined
                filtered.append(result)

        if len(filtered) < len(results):
            logger.info(f"[漂移控制] 阈值过滤：{len(results)} → {len(filtered)} 条（阈值={min_score}）")

        return filtered if filtered else results[:3]

    def validate_expansion(self, original_query: str, expanded_queries: List[str]) -> List[str]:
        """校验扩展查询与原查询的一致性，过滤漂移严重的扩展"""
        if len(expanded_queries) <= 1:
            return expanded_queries

        validated = [original_query]
        for eq in expanded_queries[1:]:
            if self._check_consistency(original_query, eq):
                validated.append(eq)
            else:
                logger.info(f"[漂移控制] 过滤漂移扩展：{eq}（与原查询不一致）")

        return validated

    def _check_consistency(self, original: str, expanded: str) -> bool:
        """多维度检查扩展查询与原查询的一致性"""
        if expanded == original:
            return True

        keyword_preserved = self._keyword_preserved(original, expanded) # 检查核心关键词是否在扩展中保留
        if keyword_preserved:
            return True

        if original in expanded:
            return True

        char_overlap = self._char_overlap_ratio(original, expanded) # 字符重叠率
        length_ratio = len(expanded) / max(len(original), 1) # 长度比
        if char_overlap >= EXPANSION_CONSISTENCY_THRESHOLD and length_ratio <= 2.5: # 字符重叠率和长度比都符合阈值
            if len(original) <= 3:
                pass
            else:
                return True

        bigram_sim = self._bigram_jaccard(original, expanded) # 字符二元组Jaccard相似度
        if bigram_sim >= 0.2:
            return True

        return False

    def _char_overlap_ratio(self, original: str, expanded: str) -> float:
        """字符重叠率"""
        original_chars = set(original)
        expanded_chars = set(expanded)
        if not original_chars:
            return 0.0
        overlap = original_chars & expanded_chars
        return len(overlap) / len(original_chars)

    def _bigram_jaccard(self, s1: str, s2: str) -> float:
        """字符二元组Jaccard相似度（比单字符重叠更精确）"""
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else set(s)

        bg1 = get_bigrams(s1)
        bg2 = get_bigrams(s2)
        if not bg1 or not bg2:
            return 0.0
        intersection = bg1 & bg2
        union = bg1 | bg2
        return len(intersection) / len(union)

    def _keyword_preserved(self, original: str, expanded: str) -> bool:
        """检查原查询中的核心关键词是否在扩展中保留，或扩展是否为同义词词典中的有效扩展"""
        synonym_groups = [
            {"退货", "退款", "退单", "售后申请", "退换货"},
            {"报销", "费用报销", "报销申请", "费用申请"},
            {"入职", "新员工入职", "入职办理", "入职手续", "新入职"},
            {"离职", "辞职", "离职手续", "离职办理", "退职"},
            {"年假", "带薪年假", "年休假", "法定年假"},
            {"社保", "社会保险", "五险", "五险一金"},
            {"公积金", "住房公积金", "公积金缴纳"},
            {"工资", "薪资", "薪酬", "工资发放", "工资条"},
            {"考勤", "打卡", "出勤", "考勤记录", "签到"},
            {"加班", "加班申请", "加班费", "加班审批"},
            {"请假", "假期申请", "休假", "请假审批", "调休"},
            {"合同", "劳动合同", "用工合同", "聘用合同"},
            {"绩效", "绩效考核", "KPI", "绩效评估", "业绩考核"},
            {"培训", "员工培训", "培训申请", "技能培训"},
            {"调岗", "岗位调整", "岗位调动", "内部调动"},
            {"库存", "库存查询", "库存数量", "库存余量", "存货"},
            {"配件", "零部件", "零件", "备件"},
            {"维修", "维修工单", "维修申请", "修理"},
            {"工单", "维修工单", "服务工单", "工单生成"},
        ]

        for group in synonym_groups:
            orig_in_group = any(kw in original for kw in group)
            exp_in_group = any(kw in expanded for kw in group)
            if orig_in_group and exp_in_group:
                return True

        core_keywords = [
            "退货", "退款", "报销", "入职", "离职", "年假", "社保",
            "公积金", "工资", "考勤", "加班", "请假", "合同",
            "绩效", "培训", "调岗", "库存", "配件", "维修", "工单",
            "五险一金", "带薪", "休假", "薪资", "薪酬",
        ]

        orig_keywords = [kw for kw in core_keywords if kw in original]
        if not orig_keywords:
            return False

        preserved = sum(1 for kw in orig_keywords if kw in expanded)
        return preserved >= len(orig_keywords) * 0.5


drift_controller = DriftController()
