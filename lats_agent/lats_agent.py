# lats_agent/lats_agent.py
"""
Language Agent Tree Search (LATS) による修復オーケストレーター。

既存のRecoveryAgent(線形ループ)との違い:
  - 線形ループ: エラーを見て→候補リストの先頭から1つ試す→繰り返す
  - LATS:       複数の修復経路を木として展開し、UCBで有望な枝を選択。
                LLMが価値推定・振り返り・優先順位付けを担う。

1AONのような複合失敗(MISSING_ATOM + HETERO_CHAIN_TYPE_MISMATCH)に対して
複数の修復経路を同時に探索し、最も短い成功経路を見つけることが目的。
"""

import json
import os
import shutil
import tempfile
import time
from typing import Optional

from recovery_agent.diagnosis import diagnose_error, extract_fatal_error
from recovery_agent.observation import ObservationModule
from recovery_agent.repair import get_repair_candidates
from recovery_agent.utils import run_with_timeout
from lats_agent.mcts_node import MCTSNode
from lats_agent.llm_evaluator import (
    estimate_value,
    generate_reflection,
    prioritize_actions,
)


class LATSRecoveryAgent:
    def __init__(self, config: dict):
        self.obs_module = ObservationModule(
            force_field=config["gromacs"]["force_field"],
            water_model=config["gromacs"]["water_model"],
        )
        self.max_iterations = config.get("lats", {}).get("max_iterations", 20)
        self.max_depth = config.get("lats", {}).get("max_depth", 5)
        self.exploration_constant = config.get("lats", {}).get("exploration_constant", 1.4)
        self.repair_timeout = config.get("lats", {}).get("repair_timeout_sec", 300)
        self.gmx_timeout = config.get("lats", {}).get("gmx_timeout_sec", 120)
        self.log_dir = config["agent"]["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        self.use_llm = config.get("lats", {}).get("use_llm", True)

    # ------------------------------------------------------------------
    # メインエントリポイント
    # ------------------------------------------------------------------

    def run(self, initial_pdb: str) -> dict:
        work_dir = tempfile.mkdtemp(prefix=f"lats_{os.path.basename(initial_pdb)}_")
        print(f"\n{'='*60}")
        print(f"LATS Recovery: {initial_pdb}")
        print(f"work_dir: {work_dir}")
        print(f"{'='*60}")

        # ルートノードを作成(最初のgmx実行)
        root = self._make_root(initial_pdb, work_dir)
        if root.is_success:
            print(">> Already successful, no repair needed.")
            return self._finalize(root, work_dir, initial_pdb)

        best_success_node: Optional[MCTSNode] = None

        for iteration in range(self.max_iterations):
            print(f"\n--- LATS Iteration {iteration + 1}/{self.max_iterations} ---")

            # 1. Selection: UCBで有望なノードを選ぶ
            node = self._select(root)
            print(f"  Selected: {node}")

            # 2. Expansion: 未試行の修復操作を1つ展開
            child = self._expand(node, work_dir)
            if child is None:
                print("  No expansion possible from this node.")
                continue

            # 3. Simulation: 修復を実行してgmxを試す
            self._simulate(child)
            print(f"  After repair: success={child.is_success}, "
                  f"diagnosis={diagnose_error(child.fatal_error_text or '') if child.fatal_error_text else 'N/A'}")

            # 成功ならベスト候補として記録
            if child.is_success:
                print(f"  *** SUCCESS found at depth {child.depth}! ***")
                print(f"      Path: {child.action_path()}")
                if best_success_node is None or child.depth < best_success_node.depth:
                    best_success_node = child

            # 4. LLM Value Estimation(use_llm=Trueの場合のみ)
            reward = self._compute_reward(child)
            if self.use_llm and not child.is_success and child.fatal_error_text:
                llm_score = self._llm_value(child)
                # 決定論的報酬とLLM推定値を混合(0.7:0.3)
                reward = 0.7 * reward + 0.3 * llm_score
                child.llm_value_estimate = llm_score
                print(f"  LLM value estimate: {llm_score:.3f}, blended reward: {reward:.3f}")

            # 5. Reflection: 失敗した場合にLLMで振り返りを生成
            if self.use_llm and not child.is_success and child.fatal_error_text:
                last_op = child.repair_history[-1] if child.repair_history else "none"
                child.reflection = generate_reflection(
                    child.fatal_error_text, child.repair_history, last_op
                )
                print(f"  Reflection: {child.reflection[:100]}...")

            # 6. Backpropagation: 報酬を祖先ノードに伝播
            self._backpropagate(child, reward)

            # 早期終了: 成功ノードが見つかれば十分な探索が終わったと判断
            if best_success_node is not None and iteration >= self.max_iterations // 2:
                print(f"\n  Early stopping: success found and half iterations done.")
                break

        result = self._finalize(best_success_node or root, work_dir, initial_pdb)
        self._cleanup_work_dir(work_dir, keep=not result["success"])
        return result

    # ------------------------------------------------------------------
    # MCTS 4ステップ
    # ------------------------------------------------------------------

    def _make_root(self, initial_pdb: str, work_dir: str) -> MCTSNode:
        """ルートノード: 最初のgmx実行を行い、状態を初期化する"""
        t0 = time.time()
        obs = self.obs_module.run_pdb2gmx(
            initial_pdb, work_dir=work_dir, timeout=self.gmx_timeout
        )
        duration = time.time() - t0
        fatal = extract_fatal_error(obs["stderr"])

        root = MCTSNode(
            pdb_path=initial_pdb,
            repair_history=[],
            fatal_error_text=fatal,
            extra_flags=None,
            depth=0,
            is_terminal=obs["success"],
            is_success=obs["success"],
            gmx_duration_sec=duration,
        )
        root.visit_count = 1

        # 未試行アクションを設定
        if not obs["success"]:
            root.untried_actions = self._get_actions(root)

        return root

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Selection: 葉ノードに到達するまでUCBで子を選び続ける。
        未展開ノードがあればそこで止まる。
        """
        node = root
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node  # まだ展開していない候補がある
            if not node.children:
                return node  # 子がない(候補が尽きた)
            node = node.best_child(self.exploration_constant)
        return node

    def _expand(self, node: MCTSNode, work_dir: str) -> Optional[MCTSNode]:
        """
        Expansion: 未試行の修復操作を1つ選んで子ノードを作る。
        LLMが有効な場合、固定順序ではなくLLMが優先順位を決める。
        """
        if not node.untried_actions or node.is_terminal:
            return None

        # LLMによる動的な優先順位付け
        reflection_ctx = node.reflection or ""
        if self.use_llm and node.fatal_error_text:
            action_names = [fn.__name__ for fn in node.untried_actions]
            ordered_names = prioritize_actions(
                node.fatal_error_text,
                node.repair_history,
                action_names,
                reflection_context=reflection_ctx,
            )
            # 名前→関数オブジェクトのマッピングを復元
            name_to_fn = {fn.__name__: fn for fn in node.untried_actions}
            fn = name_to_fn.get(ordered_names[0]) if ordered_names else node.untried_actions[0]
        else:
            fn = node.untried_actions[0]

        # untried_actionsから除去(展開済みとして記録)
        node.untried_actions = [f for f in node.untried_actions if f != fn]

        # 子ノードを作成(まだgmxは実行しない。simulateで行う)
        child = MCTSNode(
            pdb_path=node.pdb_path,   # 修復後のパスはsimulateで更新
            repair_history=node.repair_history + [fn.__name__],
            fatal_error_text=node.fatal_error_text,  # simulateで更新
            extra_flags=node.extra_flags,
            depth=node.depth + 1,
            parent=node,
        )
        child._repair_fn = fn  # simulate時に使う
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> None:
        """
        Simulation: 修復を実行してgmxを試す。結果でノードの状態を更新する。
        ここが「環境との実際のインタラクション」に相当する。
        """
        fn = getattr(node, "_repair_fn", None)
        if fn is None:
            return

        work_dir = os.path.dirname(
            node.parent.pdb_path
        ) if node.parent and node.parent.pdb_path != node.pdb_path else \
            tempfile.mkdtemp(prefix="lats_sim_")

        # 修復実行(タイムアウト付き)
        print(f"  Executing: {fn.__name__} on {os.path.basename(node.parent.pdb_path)}")
        t0 = time.time()
        result = run_with_timeout(
            fn,
            args=(node.parent.pdb_path, node.depth, work_dir),
            timeout_sec=self.repair_timeout,
        )

        if result.get("status") == "repair_timeout":
            print(f"  TIMEOUT after {self.repair_timeout}s")
            node.is_terminal = True
            node.fatal_error_text = f"REPAIR_TIMEOUT: {fn.__name__} exceeded {self.repair_timeout}s"
            node.gmx_duration_sec = time.time() - t0
            node.untried_actions = []
            return

        # PDBパスとフラグを更新
        if result.get("new_pdb_path"):
            node.pdb_path = result["new_pdb_path"]
        node.extra_flags = result.get("extra_flags")
        node.structure_altered = result.get("structure_altered", False)

        # gmxを試す
        obs = self.obs_module.run_pdb2gmx(
            node.pdb_path,
            work_dir=work_dir,
            additional_flags=node.extra_flags,
            timeout=self.gmx_timeout,
        )
        node.gmx_duration_sec = time.time() - t0
        node.fatal_error_text = extract_fatal_error(obs["stderr"])
        node.is_success = obs["success"]

        if obs["success"]:
            node.is_terminal = True
            node.untried_actions = []
        elif node.depth >= self.max_depth:
            node.is_terminal = True
            node.untried_actions = []
        else:
            # 次に展開できる候補を設定(深さを掘り続けられる)
            node.untried_actions = self._get_actions(node)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation: 報酬を根まで伝播する。
        各祖先ノードのvisit_countとtotal_valueを更新する。
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent

    # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------

    def _get_actions(self, node: MCTSNode) -> list:
        """
        このノードから試せる修復関数のリスト。
        既に repair_history に含まれるものは除外する。
        """
        if not node.fatal_error_text:
            return []
        category = diagnose_error(node.fatal_error_text)
        candidates = get_repair_candidates(category)
        return [fn for fn in candidates if fn.__name__ not in node.repair_history]

    def _compute_reward(self, node: MCTSNode) -> float:
        """
        決定論的な報酬計算。LLMスコアとブレンドする前の純粋な観測ベースの報酬。

        success:  1.0
        timeout:  0.0
        failure:  0.1 * (1 - depth/max_depth)  ← 浅い失敗ほど少しマシ
        structure_altered: -0.2のペナルティ(生物学的情報の損失)
        """
        if node.is_success:
            base = 1.0
        elif "REPAIR_TIMEOUT" in (node.fatal_error_text or ""):
            base = 0.0
        else:
            base = 0.1 * max(0.0, 1.0 - node.depth / self.max_depth)

        # 構造変更ペナルティ(strip_hetero_cofactorsなど)
        if node.structure_altered:
            base = max(0.0, base - 0.2)

        return base

    def _llm_value(self, node: MCTSNode) -> float:
        """LLMに現在状態の価値を推定させる"""
        remaining = [fn.__name__ for fn in node.untried_actions]
        return estimate_value(
            node.fatal_error_text or "",
            node.repair_history,
            remaining,
        )

    def _finalize(self, node: MCTSNode, work_dir: str, initial_pdb: str) -> dict:
        """最終結果をまとめてJSONLに保存する"""
        result = {
            "initial_pdb": initial_pdb,
            "success": node.is_success,
            "final_pdb": node.pdb_path if node.is_success else None,
            "repair_path": node.action_path(),
            "depth": node.depth,
            "structure_altered": node.structure_altered,
            "work_dir": work_dir,
        }

        # 木全体の統計
        all_nodes = self._collect_all_nodes(node)
        result["tree_stats"] = {
            "total_nodes": len(all_nodes),
            "success_nodes": sum(1 for n in all_nodes if n.is_success),
            "timeout_nodes": sum(1 for n in all_nodes if "REPAIR_TIMEOUT" in (n.fatal_error_text or "")),
            "max_depth_reached": max((n.depth for n in all_nodes), default=0),
        }

        # ログ保存
        filename = os.path.basename(initial_pdb).replace(".pdb", "_lats.jsonl")
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            for n in all_nodes:
                entry = {
                    "depth": n.depth,
                    "path": n.action_path(),
                    "visit_count": n.visit_count,
                    "q_value": round(n.q_value, 4),
                    "llm_value_estimate": n.llm_value_estimate,
                    "is_success": n.is_success,
                    "is_terminal": n.is_terminal,
                    "structure_altered": n.structure_altered,
                    "fatal_error_text": n.fatal_error_text,
                    "reflection": n.reflection,
                    "gmx_duration_sec": round(n.gmx_duration_sec, 2),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"\nResult: {result}")
        print(f"Log: {filepath}")
        return result

    def _collect_all_nodes(self, root: MCTSNode) -> list:
        """木を幅優先でたどって全ノードを返す"""
        result, queue = [], [root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result

    def _cleanup_work_dir(self, work_dir: str, keep: bool = False) -> None:
        """失敗ケースはデバッグのため残す。成功ケースは削除する。"""
        if keep:
            print(f"  Keeping work_dir for debug: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"  Cleaned up work_dir: {work_dir}")
