
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau

from src.evaluation.dice_engine import SimplifiedDICEEvaluator, SimplifiedDICEConfig
from src.utils.ragas_impl import RagasEvaluator, RagasConfig

class DICEValidationEvaluator:
    """DICE验证评估器 - 用于评估DICE本身的准确性"""
    
    def __init__(self, config: SimplifiedDICEConfig, tournament_result_file: str = None):
        self.config = config
        self.logger = logging.getLogger("DICEValidation")
        self.dice_evaluator = SimplifiedDICEEvaluator(config)
        self.tournament_result_file = tournament_result_file
        self.tournament_results = None
        
        # 设置日志
        self._setup_logger()
        
        # 如果提供了tournament结果文件，则加载它
        if self.tournament_result_file and Path(self.tournament_result_file).exists():
            self._load_tournament_results()
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _load_tournament_results(self):
        """加载tournament结果文件"""
        try:
            self.logger.info(f"开始加载tournament结果文件: {self.tournament_result_file}")
            with open(self.tournament_result_file, 'r', encoding='utf-8') as f:
                self.tournament_results = json.load(f)
            self.logger.info(f"成功加载tournament结果文件，包含 {len(self.tournament_results.get('swiss_results', {}).get('match_records', []))} 个对决记录")
        except Exception as e:
            self.logger.error(f"加载tournament结果文件失败: {e}")
            self.tournament_results = None
    
    def _find_tournament_match(self, system_a: str, system_b: str, question: str) -> Dict[str, Any]:
        """在tournament结果中查找匹配的对决"""
        if not self.tournament_results:
            return None
        
        # 查找匹配的系统对
        match_records = self.tournament_results.get('swiss_results', {}).get('match_records', [])
        
        for match in match_records:
            match_system_a = match.get('system_a', '')
            match_system_b = match.get('system_b', '')
            
            # 检查系统对是否匹配（考虑顺序）
            if ((match_system_a == system_a and match_system_b == system_b) or 
                (match_system_a == system_b and match_system_b == system_a)):
                
                # 在comparison结果中查找匹配的问题
                comparison = match.get('comparison', {})
                question_results = comparison.get('question_results', [])
                
                for q_result in question_results:
                    if q_result.get('question', '') == question:
                        return q_result
        
        return None
    
    def sample_evaluation_pairs(self, qacg_files: List[str], num_samples: int = 200, 
                               random_seed: int = 42) -> List[Dict[str, Any]]:
        """采样评估对"""
        import random
        random.seed(random_seed)
        
        all_pairs = []
        for file_path in qacg_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_pairs.extend(data)
        
        if len(all_pairs) < num_samples:
            self.logger.warning(f"可用数据对数量({len(all_pairs)})少于请求的采样数量({num_samples})")
            return all_pairs
        
        return random.sample(all_pairs, num_samples)
    
    def run_dice_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """运行DICE评估"""
        results = []
        
        for i, pair in enumerate(evaluation_pairs):
            try:
                # 从QACG格式中提取问答对
                qa_a = pair.get('qa_a', {})
                qa_b = pair.get('qa_b', {})
                
                question = qa_a.get('question', '')
                system_a = pair.get('system_a', '')
                system_b = pair.get('system_b', '')
                
                # 首先尝试从tournament结果中查找匹配
                tournament_match = self._find_tournament_match(system_a, system_b, question)
                
                if tournament_match:
                    # 使用tournament中的已有结果
                    self.logger.info(f"使用tournament结果: {system_a} vs {system_b} - {question[:50]}...")
                    
                    passage_judgment = tournament_match.get('passage_judgment', {})
                    score_a = passage_judgment.get('prob_a', 0.0)
                    score_b = passage_judgment.get('prob_b', 0.0)
                    dice_score = score_a - score_b
                    
                    result = {
                        'index': i,
                        'question': question,
                        'system_a': system_a,
                        'system_b': system_b,
                        'answer_a': qa_a.get('rag_answer', ''),
                        'answer_b': qa_b.get('rag_answer', ''),
                        'context_a': qa_a.get('context', []),
                        'context_b': qa_b.get('context', []),
                        'dice_score': dice_score,
                        'dice_explanation': passage_judgment.get('reason', ''),
                        'human_annotation': pair.get('human_annotation', ''),
                        'prob_a': score_a,
                        'prob_b': score_b,
                        'win_type': passage_judgment.get('win_type', 'Unknown'),
                        'source': 'tournament'  # 标记来源
                    }
                else:
                    # 没有找到tournament结果，进行新的推理
                    self.logger.info(f"未找到tournament结果，进行新推理: {system_a} vs {system_b} - {question[:50]}...")
                    
                    # 构建问答对格式
                    target_qa_a = {
                        'answer': qa_a.get('rag_answer', ''),
                        'context': qa_a.get('context', [])
                    }
                    
                    target_qa_b = {
                        'answer': qa_b.get('rag_answer', ''),
                        'context': qa_b.get('context', [])
                    }
                    
                    # 使用DICE的pairwise judge进行评估
                    judgment = self.dice_evaluator.pairwise_judge.judge_pair(
                        question=question,
                        qa_a=target_qa_a,
                        qa_b=target_qa_b,
                        granularity="passage"  # 使用passage粒度进行评估
                    )
                    
                    # 从判决结果中提取分数
                    passage_judgment = judgment.get('passage_judgment', {})
                    score_a = passage_judgment.get('prob_a', 0.0)
                    score_b = passage_judgment.get('prob_b', 0.0)
                    
                    # 计算相对分数（系统A相对于系统B的优势）
                    dice_score = score_a - score_b
                    
                    result = {
                        'index': i,
                        'question': question,
                        'system_a': system_a,
                        'system_b': system_b,
                        'answer_a': qa_a.get('rag_answer', ''),
                        'answer_b': qa_b.get('rag_answer', ''),
                        'context_a': qa_a.get('context', []),
                        'context_b': qa_b.get('context', []),
                        'dice_score': dice_score,
                        'dice_explanation': passage_judgment.get('reason', ''),
                        'human_annotation': pair.get('human_annotation', ''),
                        'prob_a': score_a,
                        'prob_b': score_b,
                        'win_type': passage_judgment.get('win_type', 'Unknown'),
                        'source': 'new_inference'  # 标记来源
                    }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已完成 {i + 1}/{len(evaluation_pairs)} 个评估")
                    
            except Exception as e:
                self.logger.error(f"评估第{i}个样本时出错: {e}")
                # 添加一个默认结果
                result = {
                    'index': i,
                    'question': pair.get('qa_a', {}).get('question', ''),
                    'system_a': pair.get('system_a', ''),
                    'system_b': pair.get('system_b', ''),
                    'answer_a': pair.get('qa_a', {}).get('rag_answer', ''),
                    'answer_b': pair.get('qa_b', {}).get('rag_answer', ''),
                    'context_a': pair.get('qa_a', {}).get('context', []),
                    'context_b': pair.get('qa_b', {}).get('context', []),
                    'dice_score': 0.0,
                    'dice_explanation': f'评估出错: {str(e)}',
                    'human_annotation': pair.get('human_annotation', ''),
                    'prob_a': 0.0,
                    'prob_b': 0.0,
                    'win_type': 'Error',
                    'source': 'error'
                }
                results.append(result)
                continue
        
        return results
    
    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """加载人工标注"""
        annotations = {}
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'index' in item and 'human_annotation' in item:
                        annotations[item['index']] = item['human_annotation']
        except Exception as e:
            self.logger.error(f"加载人工标注文件失败: {e}")
        return annotations
    
    def calculate_agreement(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, float]:
        """计算一致性指标"""
        dice_scores = []
        human_scores = []
        
        for result in results:
            if result['index'] in gold_labels:
                dice_scores.append(result['dice_score'])
                # 将人工标注转换为数值分数
                human_annotation = gold_labels[result['index']]
                if human_annotation.lower() in ['a', 'system_a', 'good', 'correct', 'accurate']:
                    human_scores.append(1.0)  # 系统A更好
                elif human_annotation.lower() in ['b', 'system_b', 'bad', 'incorrect', 'inaccurate']:
                    human_scores.append(-1.0)  # 系统B更好
                else:
                    human_scores.append(0.0)  # 平局或中性
        
        if len(dice_scores) == 0:
            return {'correlation': 0.0, 'kappa': 0.0}
        
        # 计算皮尔逊相关系数
        correlation = np.corrcoef(dice_scores, human_scores)[0, 1] if len(dice_scores) > 1 else 0.0
        
        # 计算Cohen's Kappa (将分数转换为二分类)
        dice_binary = [1 if score > 0 else 0 for score in dice_scores]  # 正数表示A更好
        human_binary = [1 if score > 0 else 0 for score in human_scores]  # 正数表示A更好
        kappa = cohen_kappa_score(dice_binary, human_binary) if len(dice_scores) > 1 else 0.0
        
        return {
            'correlation': correlation,
            'kappa': kappa,
            'sample_size': len(dice_scores)
        }
    
    def calculate_elo_correlation(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, float]:
        """计算ELO相关性"""
        # 这里可以实现ELO评分系统的相关性计算
        # 暂时返回基本的相关性指标
        return self.calculate_agreement(results, gold_labels)
    
    def analyze_disagreement_cases(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """分析不一致案例"""
        disagreement_cases = []
        
        for result in results:
            if result['index'] in gold_labels:
                dice_score = result['dice_score']
                human_annotation = gold_labels[result['index']]
                
                # 判断是否不一致
                dice_a_better = dice_score > 0  # DICE认为系统A更好
                human_a_better = human_annotation.lower() in ['a', 'system_a', 'good', 'correct', 'accurate']
                
                if dice_a_better != human_a_better:
                    disagreement_cases.append({
                        'index': result['index'],
                        'question': result['question'],
                        'system_a': result.get('system_a', ''),
                        'system_b': result.get('system_b', ''),
                        'answer_a': result.get('answer_a', ''),
                        'answer_b': result.get('answer_b', ''),
                        'dice_score': dice_score,
                        'human_annotation': human_annotation,
                        'disagreement_type': 'dice_a_better_human_b_better' if dice_a_better else 'dice_b_better_human_a_better'
                    })
        
        return disagreement_cases
    
    def print_disagreement_analysis(self, disagreement_cases: List[Dict[str, Any]]) -> None:
        """打印不一致分析"""
        if not disagreement_cases:
            self.logger.info("没有发现不一致案例")
            return
        
        self.logger.info(f"发现 {len(disagreement_cases)} 个不一致案例:")
        
        for case in disagreement_cases[:5]:  # 只显示前5个
            self.logger.info(f"案例 {case['index']}: DICE分数={case['dice_score']:.3f}, 人工标注={case['human_annotation']}")
            self.logger.info(f"问题: {case['question'][:100]}...")
    
    def generate_validation_report(self, results: List[Dict[str, Any]], gold_labels: Dict[int, str]) -> Dict[str, Any]:
        """生成验证报告"""
        agreement_metrics = self.calculate_agreement(results, gold_labels)
        disagreement_cases = self.analyze_disagreement_cases(results, gold_labels)
        
        report = {
            'total_samples': len(results),
            'annotated_samples': len([r for r in results if r['index'] in gold_labels]),
            'agreement_metrics': agreement_metrics,
            'disagreement_count': len(disagreement_cases),
            'disagreement_rate': len(disagreement_cases) / len(results) if results else 0.0,
            'dice_scores_summary': {
                'mean': np.mean([r['dice_score'] for r in results]) if results else 0.0,
                'std': np.std([r['dice_score'] for r in results]) if results else 0.0,
                'min': min([r['dice_score'] for r in results]) if results else 0.0,
                'max': max([r['dice_score'] for r in results]) if results else 0.0
            }
        }
        
        return report


class RagasValidationEvaluator:
    """RAGAS验证评估器"""
    
    def __init__(self, config: RagasConfig):
        self.config = config
        self.logger = logging.getLogger("RagasValidation")
        self.ragas_evaluator = RagasEvaluator(config)
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def run_ragas_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用RAGAS评估所有采样的对比对"""
        self.logger.info(f"开始RAGAS评估 {len(evaluation_pairs)} 对样本")
        
        print(f"\n🚀 RAGAS批量评估开始")
        print(f"📊 总共需要评估: {len(evaluation_pairs)} 对样本")
        print("🔔 每次评估会显示详细的判断过程和结果")
        print("=" * 120)
        
        ragas_results = []
        for i, pair in enumerate(evaluation_pairs):
            print(f"\n⏳ 进度: {i+1}/{len(evaluation_pairs)} ({(i+1)/len(evaluation_pairs)*100:.1f}%)")
            print(f"🔍 评估对 #{i+1}: {pair['system_a']} vs {pair['system_b']}")
            
            qa_a = pair["qa_a"]
            qa_b = pair["qa_b"]
            
            result = self.ragas_evaluator._pairwise_comparison(
                [qa_a], [qa_b], 
                pair["system_a"], pair["system_b"],
                max_questions=1
            )
            
            if result["question_results"]:
                question_result = result["question_results"][0]
                passage_judgment = question_result.get("passage_judgment", {})
                ragas_details = question_result.get("ragas_details", {})
                
                # 显示本次评估的最终结果
                judgment = passage_judgment.get("label", "Tie")
                score = passage_judgment.get("score", 0.5)
                reason = passage_judgment.get("reason", "")
                
                judgment_icon = "🏆" if judgment != "Tie" else "⚖️"
                print(f"\n✅ 评估对 #{i+1} 完成:")
                print(f"   {judgment_icon} 结果: {judgment}")
                print(f"   📊 置信度: {score:.4f}")
                print(f"   📝 理由: {reason}")
                
                ragas_result = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": judgment,
                    "dice_score": score,
                    "dice_reason": reason,
                    "dice_margin_score": passage_judgment.get("margin_score", 0.0),
                    "combined_delta": ragas_details.get("composite_a", 0) - ragas_details.get("composite_b", 0),
                    "ragas_scores_a": ragas_details.get("scores_a", {}),
                    "ragas_scores_b": ragas_details.get("scores_b", {}),
                    "original_pair": pair
                }
            else:
                print(f"\n❌ 评估对 #{i+1} 失败:")
                print(f"   ⚠️ RAGAS评估过程中出现错误")
                
                ragas_result = {
                    "pair_id": i,
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": "Tie",
                    "dice_score": 0.5,
                    "dice_reason": "RAGAS评估失败",
                    "dice_margin_score": 0.0,
                    "combined_delta": 0.0,
                    "ragas_scores_a": {},
                    "ragas_scores_b": {},
                    "original_pair": pair
                }
            
            ragas_results.append(ragas_result)
            print("═" * 120)
        
        # 显示批量评估统计
        print(f"\n🎊 RAGAS批量评估完成！")
        print(f"📊 评估统计:")
        
        # 统计结果
        judgments = [r["dice_judgment"] for r in ragas_results]
        a_wins = judgments.count("A wins")
        b_wins = judgments.count("B wins")
        ties = judgments.count("Tie")
        
        print(f"   🏆 A wins: {a_wins} 次 ({a_wins/len(ragas_results)*100:.1f}%)")
        print(f"   🏆 B wins: {b_wins} 次 ({b_wins/len(ragas_results)*100:.1f}%)")
        print(f"   ⚖️ Tie: {ties} 次 ({ties/len(ragas_results)*100:.1f}%)")
        print("=" * 120)
        
        return ragas_results
    
    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """代理到具体评估器的标注加载方法 - 这里需要重新实现，因为不能依赖DICEValidationEvaluator的实例"""
        # 由于RagasValidationEvaluator不包含load_human_annotations逻辑（在DICEValidationEvaluator中），
        # 我们这里为了接口统一，简单实例化一个DICEValidationEvaluator来调用
        # 注意：这可能会导致不必要的初始化开销，但在验证脚本中是可以接受的
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.load_human_annotations(annotation_file)
    
    def calculate_agreement(self, results, gold_labels):
        """代理一致性计算"""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.calculate_agreement(results, gold_labels)
    
    def calculate_elo_correlation(self, results, gold_labels):
        """代理Elo相关性计算"""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.calculate_elo_correlation(results, gold_labels)
    
    def analyze_disagreement_cases(self, results, gold_labels):
        """代理分歧分析"""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.analyze_disagreement_cases(results, gold_labels)
    
    def print_disagreement_analysis(self, disagreement_cases):
        """代理分歧打印"""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.print_disagreement_analysis(disagreement_cases)
    
    def generate_validation_report(self, agreement_metrics, correlation_metrics, results, gold_labels, output_file):
        """代理报告生成"""
        from src.evaluation.dice_engine import SimplifiedDICEConfig
        temp_dice_evaluator = DICEValidationEvaluator(SimplifiedDICEConfig())
        return temp_dice_evaluator.generate_validation_report(agreement_metrics, correlation_metrics, results, gold_labels, output_file)


class UnifiedValidationEvaluator:
    """统一验证评估器 - 支持DICE和RAGAS两种评估方法"""
    
    def __init__(self, evaluation_method: str = "dice", dice_config: SimplifiedDICEConfig = None, 
                 ragas_config: RagasConfig = None, tournament_result_file: str = None):
        self.evaluation_method = evaluation_method.lower()
        self.logger = logging.getLogger("UnifiedValidation")
        
        # 根据评估方法初始化相应的评估器
        if self.evaluation_method == "dice":
            if dice_config is None:
                raise ValueError("使用DICE方法时必须提供dice_config")
            self.evaluator = DICEValidationEvaluator(dice_config, tournament_result_file)
        elif self.evaluation_method == "ragas":
            if ragas_config is None:
                raise ValueError("使用RAGAS方法时必须提供ragas_config")
            self.evaluator = RagasValidationEvaluator(ragas_config)
        else:
            raise ValueError(f"不支持的评估方法: {evaluation_method}")
        
        # 设置日志
        self._setup_logger()
        
        self.logger.info(f"初始化统一验证评估器，使用方法: {self.evaluation_method.upper()}")
    
    def _setup_logger(self):
        """设置日志"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _derive_dice_label(self, result: Dict[str, Any]) -> str:
        """统一推断DICE标签的逻辑，避免分值尺度误判导致统计错误。"""
        explicit_label = result.get("dice_judgment")
        if explicit_label in {"A wins", "B wins", "Tie"}:
            return explicit_label
        
        score = result.get("dice_score")
        if isinstance(score, (int, float)):
            # 若是[0,1]分尺度，则以0.5为中性阈值，加入轻微缓冲
            if 0.0 <= score <= 1.0:
                if score > 0.55:
                    return "A wins"
                if score < 0.45:
                    return "B wins"
                return "Tie"
            # 否则视为对称分制（如[-1,1]），以0为中性阈值，加入轻微缓冲
            if score > 0.1:
                return "A wins"
            if score < -0.1:
                return "B wins"
            return "Tie"
        
        # 回退：若有prob_a/prob_b可比较
        prob_a = result.get("prob_a")
        prob_b = result.get("prob_b")
        if isinstance(prob_a, (int, float)) and isinstance(prob_b, (int, float)):
            delta = prob_a - prob_b
            if delta > 0.05:
                return "A wins"
            if delta < -0.05:
                return "B wins"
            return "Tie"
        
        return "Tie"
        
    def sample_evaluation_pairs(self, qacg_files: List[str], num_samples: int = 200, 
                               random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        从70题中随机抽取200对(q, cA, aA, cB, aB)用于人工标注
        
        Args:
            qacg_files: QACG文件路径列表
            num_samples: 采样数量
            random_seed: 随机种子
            
        Returns:
            采样的评估对列表
        """
        self.logger.info(f"开始采样 {num_samples} 对评估样本")
        import random
        random.seed(random_seed)
        
        # 加载所有系统数据
        all_systems_data = {}
        for file_path in qacg_files:
            system_name = Path(file_path).stem.replace("qacg_", "")
            with open(file_path, 'r', encoding='utf-8') as f:
                all_systems_data[system_name] = json.load(f)
        
        systems = list(all_systems_data.keys())
        if len(systems) < 2:
            raise ValueError(f"需要至少2个系统，实际获得{len(systems)}个")
        
        self.logger.info(f"加载了 {len(systems)} 个系统: {systems}")
        
        # 确定数据长度（使用最短的系统数据长度）
        min_length = min(len(data) for data in all_systems_data.values())
        self.logger.info(f"每个系统有 {min_length} 题数据")
        
        # 生成所有可能的系统对和题目组合
        all_combinations = []
        for i, system_a in enumerate(systems):
            for j, system_b in enumerate(systems):
                if i < j:  # 避免重复对比
                    for q_idx in range(min_length):
                        qa_a = all_systems_data[system_a][q_idx]
                        qa_b = all_systems_data[system_b][q_idx]
                        
                        # 确保两个系统回答的是同一个问题
                        if qa_a["question"] == qa_b["question"]:
                            combination = {
                                "question_idx": q_idx,
                                "system_a": system_a,
                                "system_b": system_b,
                                "qa_a": qa_a,
                                "qa_b": qa_b,
                                "question": qa_a["question"],
                                "answer_a": qa_a.get("rag_answer", ""),
                                "answer_b": qa_b.get("rag_answer", ""),
                                "expected_answer": qa_a.get("expected_answer", ""),
                                "context_a": qa_a.get("context", []),
                                "context_b": qa_b.get("context", []),
                                "groundtruth": qa_a.get("groundtruth", qa_a.get("expected_answer", ""))
                            }
                            all_combinations.append(combination)
        
        self.logger.info(f"总共有 {len(all_combinations)} 个可能的组合")
        
        # 随机采样
        if len(all_combinations) < num_samples:
            self.logger.warning(f"可用组合数 ({len(all_combinations)}) 少于需求样本数 ({num_samples})")
            sampled_pairs = all_combinations
        else:
            sampled_pairs = random.sample(all_combinations, num_samples)
        
        self.logger.info(f"成功采样 {len(sampled_pairs)} 对评估样本")
        return sampled_pairs
    
    def run_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """运行相应的评估方法"""
        if self.evaluation_method == "dice":
            return self.run_dice_evaluation(evaluation_pairs)
        elif self.evaluation_method == "ragas":
            return self.evaluator.run_ragas_evaluation(evaluation_pairs)
    
    def load_human_annotations(self, annotation_file: str) -> Dict[int, str]:
        """代理到具体评估器的标注加载方法"""
        return self.evaluator.load_human_annotations(annotation_file)
    
    def calculate_agreement(self, results: List[Dict[str, Any]], 
                          gold_labels: Dict[int, str]) -> Dict[str, float]:
        """代理到具体评估器的一致性计算方法"""
        return self.evaluator.calculate_agreement(results, gold_labels)
    
    def calculate_elo_correlation(self, results: List[Dict[str, Any]], 
                                gold_labels: Dict[int, str]) -> Dict[str, float]:
        """代理到具体评估器的Elo相关性计算方法"""
        return self.evaluator.calculate_elo_correlation(results, gold_labels)
    
    def analyze_disagreement_cases(self, results: List[Dict[str, Any]], 
                                  gold_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """代理到具体评估器的分歧分析方法"""
        return self.evaluator.analyze_disagreement_cases(results, gold_labels)
    
    def print_disagreement_analysis(self, disagreement_cases: List[Dict[str, Any]]):
        """代理到具体评估器的分歧打印方法"""
        return self.evaluator.print_disagreement_analysis(disagreement_cases)
    
    def generate_validation_report(self, agreement_metrics: Dict[str, Any], 
                                 correlation_metrics: Dict[str, Any],
                                 results: List[Dict[str, Any]],
                                 gold_labels: Dict[int, str],
                                 output_file: str):
        """代理到具体评估器的报告生成方法"""
        return self.evaluator.generate_validation_report(
            agreement_metrics, correlation_metrics, results, gold_labels, output_file
        )
    
    def run_dice_evaluation(self, evaluation_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用DICE评估所有采样的对比对
        
        Args:
            evaluation_pairs: 评估对列表
            
        Returns:
            DICE评估结果列表
        """
        self.logger.info(f"开始DICE评估 {len(evaluation_pairs)} 对样本")
        
        dice_results = []
        for i, pair in enumerate(evaluation_pairs):
            self.logger.info(f"评估第 {i+1}/{len(evaluation_pairs)} 对")
            
            # 使用DICE进行评估
            qa_a = pair["qa_a"]
            qa_b = pair["qa_b"]
            
            # 使用DICE评估器的_pairwise_comparison方法
            result = self.evaluator.dice_evaluator._pairwise_comparison(
                [qa_a], [qa_b], 
                pair["system_a"], pair["system_b"],
                max_questions=1
            )
            
            # 提取关键信息
            if result["question_results"]:
                question_result = result["question_results"][0]
                passage_judgment = question_result.get("passage_judgment", {})
                
                dice_result = {
                    "pair_id": i,  # 使用索引作为pair_id，与标注模板保持一致
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": passage_judgment.get("label", "Tie"),
                    "dice_score": passage_judgment.get("score", 0.5),
                    "dice_reason": passage_judgment.get("reason", ""),
                    "dice_margin_score": passage_judgment.get("margin_score", 0.0),
                    "combined_delta": question_result.get("elo_delta", 0.0),
                    "original_pair": pair
                }
            else:
                # 备用结果
                dice_result = {
                    "pair_id": i,  # 使用索引作为pair_id，与标注模板保持一致
                    "question": pair["question"],
                    "system_a": pair["system_a"],
                    "system_b": pair["system_b"],
                    "dice_judgment": "Tie",
                    "dice_score": 0.5,
                    "dice_reason": "评估失败",
                    "dice_margin_score": 0.0,
                    "combined_delta": 0.0,
                    "original_pair": pair
                }
            
            dice_results.append(dice_result)
        
        return dice_results
    
    def _create_annotation_template(self, annotation_file: str):
        """创建人工标注模板文件"""
        self.logger.info(f"创建标注模板: {annotation_file}")
        
        template = {
            "instructions": "请3位专家独立完成标注，每个pair_id对应一个评估对，请为每位专家在expert_votes中填入 'A wins'、'B wins' 或 'Tie'",
            "annotation_guide": {
                "A wins": "系统A明显优于系统B",
                "B wins": "系统B明显优于系统A", 
                "Tie": "两个系统表现相当，难以区分优劣"
            },
            "annotations": [
                {
                    "pair_id": 0,
                    "question": "示例问题",
                    "system_a": "system_a_name",
                    "answer_a": "系统A的回答",
                    "system_b": "system_b_name", 
                    "answer_b": "系统B的回答",
                    "expert_votes": ["A wins", "B wins", "A wins"]  # 3位专家的投票
                }
            ]
        }
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
    
    def _generate_conclusion(self, agreement_metrics: Dict[str, Any], 
                           correlation_metrics: Dict[str, Any]) -> str:
        """生成结论"""
        kappa = agreement_metrics["kappa"]
        tau = correlation_metrics["kendall_tau"]
        
        # 检查是否为2系统的特殊情况
        num_systems = len(correlation_metrics.get("dice_ranking", []))
        if num_systems == 2:
            if tau == -1.0:
                conclusion = "📊 2系统验证：DICE与人工排序完全相反（τ=-1.0）。"
                if kappa >= 0.6:
                    conclusion += f"但κ值({kappa:.3f})表明总体一致性尚可，可能存在系统偏好差异。"
                else:
                    conclusion += f"且κ值({kappa:.3f})较低，建议检查判决逻辑或增加更多系统进行验证。"
                return conclusion
            elif tau == 1.0:
                return f"✅ 2系统验证：DICE与人工排序完全一致（τ=1.0），κ值={kappa:.3f}。"
        
        # 标准的多系统评估
        if kappa >= 0.85 and tau >= 0.9:
            return "✅ DICE系统验证通过！κ值和Kendall-τ均达标，系统可信度高，可用于后续评估。"
        elif kappa >= 0.85:
            return "⚠️ DICE系统部分通过。κ值达标但排序相关性不足，建议检查Elo计算逻辑。"
        elif tau >= 0.9:
            return "⚠️ DICE系统部分通过。排序相关性达标但一致性不足，建议检查判决逻辑。"
        else:
            return "❌ DICE系统验证失败。κ值和Kendall-τ均未达标，需要重新调整评估策略。"
    
    def _print_validation_summary(self, report: Dict[str, Any]):
        """打印验证摘要"""
        summary = report["validation_summary"]
        
        print("\n" + "="*60)
        print("🔬 DICE系统验证结果")
        print("="*60)
        print(f"κ 值 (目标≥0.85): {summary['kappa_score']:.3f}")
        print(f"准确率: {summary['accuracy']:.3f}")
        print(f"Kendall-τ (目标≥0.9): {summary['kendall_tau']:.3f}")
        print(f"验证状态: {'✅ 通过' if summary['validation_passed'] else '❌ 未通过'}")
        print("\n" + report["conclusion"])
        print("="*60)

