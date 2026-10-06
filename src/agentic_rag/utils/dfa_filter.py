"""
DFA (Deterministic Finite Automaton) 敏感词过滤器
基于字典树实现 O(n) 时间复杂度的敏感词检测与过滤

面试核心考点：
1. DFA原理：有限状态自动机，每个字符走状态流转，走到终点即命中
2. 时间复杂度：O(n)，n为文本长度，与敏感词数量无关
3. 空间复杂度：O(m*k)，m为敏感词数，k为平均长度
4. 优势：万级词库只扫一遍文本，比逐词匹配快数十倍
"""

import os
import logging
from typing import List, Dict, Set, Tuple, Optional

logger = logging.getLogger(__name__)


class DFATrie:
    """DFA字典树（确定性有限状态自动机）"""

    END_FLAG = "\x00"

    def __init__(self):
        self._root: Dict = {}

    def add_word(self, word: str) -> None:
        """添加敏感词到字典树"""
        if not word:
            return
        node = self._root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.END_FLAG] = True

    def add_words(self, words: List[str]) -> None:
        """批量添加敏感词"""
        for word in words:
            self.add_word(word)

    def contains(self, text: str) -> bool:
        """检查文本是否包含敏感词"""
        result = self.search(text)
        return len(result) > 0

    def search(self, text: str) -> List[Dict]:
        """
        搜索文本中所有敏感词

        Returns:
            List[Dict]: 每个元素包含 word, start, end
        """
        results = []
        if not text:
            return results

        i = 0
        while i < len(text):
            node = self._root
            j = i
            last_match_end = -1
            last_match_word = ""

            while j < len(text) and text[j] in node:
                node = node[text[j]]
                j += 1
                if self.END_FLAG in node:
                    last_match_end = j
                    last_match_word = text[i:j]

            if last_match_end > 0:
                results.append({
                    "word": last_match_word,
                    "start": i,
                    "end": last_match_end
                })
                i = last_match_end
            else:
                i += 1

        return results

    def filter(self, text: str, replacement: str = "*") -> Tuple[str, List[Dict]]:
        """
        过滤文本中的敏感词

        Args:
            text: 原始文本
            replacement: 替换字符

        Returns:
            (filtered_text, matched_words)
        """
        if not text:
            return text, []

        matches = self.search(text)
        if not matches:
            return text, []

        text_list = list(text)
        for match in matches:
            start = match["start"]
            end = match["end"]
            for k in range(start, end):
                text_list[k] = replacement

        filtered = "".join(text_list)
        return filtered, matches

    def get_root(self) -> Dict:
        return self._root


class DFASensitiveFilter:
    """DFA敏感词过滤器（工业级）"""

    DEFAULT_SENSITIVE_WORDS = {
        "政治": [
            "反动", "颠覆", "分裂国家", "煽动颠覆",
            "邪教", "法轮功", "台独", "藏独", "疆独",
        ],
        "违法": [
            "赌博", "洗钱", "贩毒", "走私", "诈骗",
            "非法集资", "传销", "偷税漏税", "行贿受贿",
        ],
        "恶意": [
            "攻击", "辱骂", "恐吓", "威胁", "骚扰",
            "人肉搜索", "网络暴力", "恶意举报",
        ],
        "色情": [
            "色情", "裸体", "淫秽",
        ],
        "歧视": [
            "种族歧视", "性别歧视", "地域歧视",
        ],
    }

    def __init__(self, sensitive_words: Dict[str, List[str]] = None):
        self._trie = DFATrie()
        self._word_categories: Dict[str, str] = {}
        self._all_words: Set[str] = set()

        words_dict = sensitive_words or self.DEFAULT_SENSITIVE_WORDS
        self._load_words(words_dict)
        logger.info(f"DFA敏感词过滤器初始化完成，加载 {len(self._all_words)} 个敏感词")

    def _load_words(self, words_dict: Dict[str, List[str]]) -> None:
        """加载敏感词到DFA字典树"""
        for category, words in words_dict.items():
            for word in words:
                word = word.strip()
                if word and word not in self._all_words:
                    self._trie.add_word(word)
                    self._word_categories[word] = category
                    self._all_words.add(word)

    def add_word(self, word: str, category: str = "其他") -> None:
        """动态添加敏感词"""
        word = word.strip()
        if word and word not in self._all_words:
            self._trie.add_word(word)
            self._word_categories[word] = category
            self._all_words.add(word)
            logger.info(f"添加敏感词: {word} (分类: {category})")

    def remove_word(self, word: str) -> None:
        """移除敏感词（标记移除，重建字典树）"""
        if word in self._all_words:
            self._all_words.discard(word)
            self._word_categories.pop(word, None)
            self._rebuild_trie()
            logger.info(f"移除敏感词: {word}")

    def _rebuild_trie(self) -> None:
        """重建字典树"""
        self._trie = DFATrie()
        for word in self._all_words:
            self._trie.add_word(word)

    def check(self, text: str) -> Dict:
        """
        检查文本是否包含敏感词

        Returns:
            {
                'is_sensitive': bool,
                'sensitive_words': List[str],
                'categories': List[str],
                'matches': List[Dict],
                'suggestion': str
            }
        """
        if not text:
            return {
                'is_sensitive': False,
                'sensitive_words': [],
                'categories': [],
                'matches': [],
                'suggestion': '文本为空'
            }

        matches = self._trie.search(text)
        sensitive_words = list(set(m["word"] for m in matches))
        categories = list(set(
            self._word_categories.get(m["word"], "未知")
            for m in matches
        ))

        is_sensitive = len(matches) > 0

        return {
            'is_sensitive': is_sensitive,
            'sensitive_words': sensitive_words,
            'categories': categories,
            'matches': matches,
            'suggestion': f'文本包含敏感内容({",".join(categories)})，请修改' if is_sensitive else '文本正常'
        }

    def filter(self, text: str, replacement: str = "*") -> Tuple[str, Dict]:
        """
        过滤文本中的敏感词

        Returns:
            (filtered_text, check_result)
        """
        if not text:
            return text, self.check(text)

        filtered_text, matches = self._trie.filter(text, replacement)
        check_result = self.check(text)

        if matches:
            logger.info(f"DFA过滤：发现 {len(matches)} 处敏感词")

        return filtered_text, check_result

    def get_all_words(self) -> Dict[str, List[str]]:
        """获取所有敏感词（按分类）"""
        result: Dict[str, List[str]] = {}
        for word, category in self._word_categories.items():
            if category not in result:
                result[category] = []
            result[category].append(word)
        return result

    def get_word_count(self) -> int:
        """获取敏感词总数"""
        return len(self._all_words)


if os.environ.get('PYTEST_COLLECT_ONLY') != '1':
    dfa_sensitive_filter = DFASensitiveFilter()
else:
    dfa_sensitive_filter = None
