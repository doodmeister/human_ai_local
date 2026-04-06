from __future__ import annotations

import abc
import inspect
import math
import random
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING, cast
import json
import os

# Initialize logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .outcome import Outcome


# =========================
# Events (simple in-proc bus)
# =========================

class EventBus:
    """Minimal pub/sub for auditability & metrics; now with failure tracking and timing."""
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._failed_events: List[Tuple[str, Dict[str, Any], Exception]] = []
        self._publish_counts: Dict[str, int] = {}
        self._handler_latency_ms: List[float] = []  # rolling sample

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        self._publish_counts[topic] = self._publish_counts.get(topic, 0) + 1
        handlers = list(self._subs.get(topic, []))
        for h in handlers:
            start = monotonic()
            try:
                h(payload)
            except Exception as e:  # capture failure
                self._failed_events.append((topic, payload, e))
                # Emit an internal failure event (not recursive on same topic)
                if topic != "executive.event_failure":
                    fail_payload = {"topic": topic, "error": str(e)}
                    for fh in self._subs.get("executive.event_failure", []):
                        try:
                            fh(fail_payload)
                        except Exception:
                            pass
            finally:
                dt_ms = (monotonic() - start) * 1000.0
                self._handler_latency_ms.append(dt_ms)
                if len(self._handler_latency_ms) > 200:
                    # keep memory bounded
                    self._handler_latency_ms = self._handler_latency_ms[-200:]

    def failed_events(self) -> List[Tuple[str, Dict[str, Any], Exception]]:
        return list(self._failed_events)

    def publish_counts(self) -> Dict[str, int]:
        return dict(self._publish_counts)

    def avg_handler_latency_ms(self) -> float:
        if not self._handler_latency_ms:
            return 0.0
        return sum(self._handler_latency_ms) / len(self._handler_latency_ms)


# =========================
# Core Data Models
# =========================

class ExecutiveMode(Enum):
    FOCUSED = auto()
    MULTI_TASK = auto()
    EXPLORATION = auto()
    REFLECTION = auto()
    RECOVERY = auto()

@dataclass(frozen=True)
class Goal:
    id: str
    description: str
    priority: float = 0.5          # 0..1
    deadline_ts: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Task:
    id: str
    goal_id: str
    description: str
    complexity: float = 0.3         # heuristic 0..1
    urgency: float = 0.3            # heuristic 0..1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Option:
    """A candidate action/plan the DecisionEngine can score."""
    id: str
    task: Task
    prompt: str
    context_snippets: List[str]
    est_cost_tokens: int = 0
    est_latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Decision:
    option_id: str
    score: float
    rationale: str
    policy: str

@dataclass(frozen=True)
class ActionResult:
    ok: bool
    content: str
    tokens_used: int = 0
    latency_ms: int = 0
    new_memories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================
# Service Interfaces (Ports)
# =========================

class AttentionService(Protocol):
    async def allocate(self, *, query:str, candidates: List[str], capacity:int = 5) -> List[str]:
        """Return a subset (<=capacity) of candidates in priority order."""
        ...

class MemoryService(Protocol):
    async def stm_add(self, text: str, tags: Optional[List[str]] = None) -> None: ...
    async def stm_recent(self, k: int = 10) -> List[str]: ...
    async def ltm_similar(self, query: str, k: int = 8) -> List[Tuple[str, float]]: ...
    async def ltm_add(self, text: str, tags: Optional[List[str]] = None) -> str: ...
    async def consolidate(self, items: List[str]) -> str: ...

class LLMService(Protocol):
    async def complete(self, prompt: str, context: List[str], max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        """Return (text, metadata), where metadata may contain token/latency info."""
        ...

class ActuatorService(Protocol):
    async def act(self, action_text: str) -> Dict[str, Any]:
        """For tool-use or external actions. Return metadata of execution."""
        ...

class Clock(Protocol):
    def now(self) -> float: ...

class RealClock:
    def now(self) -> float:
        return monotonic()


_WORD_RE = re.compile(r"\b\w+\b")


def _lexical_tokens(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.lower()) if token}


def _normalized_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# =========================
# Decision Engine (pluggable)
# =========================

class DecisionEngine(abc.ABC):
    @abc.abstractmethod
    async def choose(self, options: List[Option]) -> Decision:
        ...

class WeightedDecisionEngine(DecisionEngine):
    """Deterministic scorer with transparent rationale."""
    def __init__(self, w_relevance:float=0.5, w_cost:float=0.2, w_latency:float=0.1, w_urgency:float=0.2, rng: Optional[random.Random]=None):
        self.w_rel = w_relevance
        self.w_cost = w_cost
        self.w_lat = w_latency
        self.w_urg = w_urgency
        self.rng = rng or random.Random(7)

    async def choose(self, options: List[Option]) -> Decision:
        if not options:
            raise ValueError("No options to choose from.")
        # Pull features from metadata (fallbacks allow graceful degradation)
        best: Tuple[Option, float] = (options[0], -1e9)
        rationales: Dict[str, str] = {}
        for opt in options:
            rel = float(opt.metadata.get("relevance", 0.5))   # 0..1
            urg = float(opt.metadata.get("urgency", 0.5))     # 0..1 (from Task)
            cost = float(opt.est_cost_tokens or opt.metadata.get("cost_tokens", 200)) # lower is better
            lat  = float(opt.est_latency_ms or opt.metadata.get("latency_ms", 500))   # lower is better

            # Normalize inversely for cost/latency
            cost_term = 1.0 - min(cost / 2000.0, 1.0)
            lat_term  = 1.0 - min(lat / 5000.0, 1.0)

            score = (
                self.w_rel * rel +
                self.w_urg * urg +
                self.w_cost * cost_term +
                self.w_lat * lat_term
            )
            # Tie-break with tiny noise for stability
            score += self.rng.random() * 1e-6

            rationales[opt.id] = (
                f"rel={rel:.2f}, urg={urg:.2f}, cost_term={cost_term:.2f}, lat_term={lat_term:.2f}"
            )
            if score > best[1]:
                best = (opt, score)

        chosen = best[0]
        return Decision(
            option_id=chosen.id,
            score=best[1],
            rationale=rationales[chosen.id],
            policy="weighted_v1",
        )


# Optional: simple UCB policy selector for exploration vs exploitation
class BanditPolicySelector:
    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.values: Dict[str, float] = {}
        self.t = 1

    def update(self, policy: str, reward: float) -> None:
        self.counts[policy] = self.counts.get(policy, 0) + 1
        n = self.counts[policy]
        v = self.values.get(policy, 0.0)
        self.values[policy] = v + (reward - v) / n
        self.t += 1

    def ucb(self, policy: str) -> float:
        n = self.counts.get(policy, 0) + 1e-9
        return self.values.get(policy, 0.0) + math.sqrt(2 * math.log(max(self.t, 2)) / n)


# =========================
# Planner (minimal, goal→tasks)
# =========================

class Planner:
    """Very simple planner: turn a Goal into 1..N tasks.
    Replace with a true hierarchical or LLM-backed planner later.
    """
    def expand(self, goal: Goal) -> List[Task]:
        # Heuristic: one task if low complexity/priority; two if higher
        base = Task(
            id=f"task::{goal.id}::0",
            goal_id=goal.id,
            description=goal.description,
            complexity=min(0.7, 0.2 + goal.priority * 0.6),
            urgency=min(1.0, 0.3 + goal.priority * 0.6),
        )
        tasks = [base]
        if base.complexity + base.urgency > 1.0:
            tasks.append(Task(
                id=f"task::{goal.id}::1",
                goal_id=goal.id,
                description=f"Refine: {goal.description}",
                complexity=0.4,
                urgency=base.urgency,
            ))
        return tasks


# =========================
# Executive Controller
# =========================

class ExecutiveController:
    """Runs a cognitive turn and manages mode transitions."""
    def __init__(
        self,
        memory: MemoryService,
        attention: AttentionService,
        llm: LLMService,
        actuator: ActuatorService,
        decision_engine: DecisionEngine | None = None,
        planner: Planner | None = None,
        clock: Clock | None = None,
        event_bus: EventBus | None = None,
        stm_context_k: int = 8,
        attention_capacity: int = 5,
        max_tokens: int = 512,
    ):
        self.memory = memory
        self.attention = attention
        self.llm = llm
        self.actuator = actuator
        self.decision_engine = decision_engine or WeightedDecisionEngine()
        self.planner = planner or Planner()
        self.clock = clock or RealClock()
        self.events = event_bus or EventBus()
        self.mode = ExecutiveMode.FOCUSED

        # adaptive knobs
        self.stm_context_k = stm_context_k
        self.attention_capacity = attention_capacity
        self.max_tokens = max_tokens
        self.bandit = BanditPolicySelector()
        self._salience_threshold = 0.55  # adaptive consolidation threshold
        try:
            from .salience import SalienceScorer
            self._salience = SalienceScorer()
        except Exception:
            self._salience = None
        self._outcome_adapter: Any | None = None
        # Load persisted adaptive state if available
        self._state_path = os.environ.get("EXEC_ADAPTIVE_STATE", "executive_state.json")
        self._load_adaptive_state()

    # --------- Goal Selection ---------

    def _default_goal(self, user_input: str) -> Goal:
        desc = user_input.strip() or "Respond to user"
        return Goal(
            id=f"goal::turn::{int(self.clock.now() * 1000)}",
            description=desc,
            priority=0.5,
        )

    def select_goal(self, user_input: str, goals: Optional[List[Goal]] = None) -> Goal:
        """Select a single goal to pursue this turn.

        Other modules may propose candidate goals, but only the executive selects.
        """
        candidates = goals or []
        if not candidates:
            return self._default_goal(user_input)

        # Normalized lexical match + priority bias.
        tokens = _lexical_tokens(user_input)
        best = candidates[0]
        best_score = -1e9
        for g in candidates:
            g_tokens = _lexical_tokens(g.description)
            overlap_score = _normalized_overlap_score(tokens, g_tokens)
            score = float(g.priority) + (0.1 * overlap_score)
            if score > best_score:
                best_score = score
                best = g
        return best

    def _build_option_metadata(self, task: Task, user_input: str, context_snippets: List[str]) -> Dict[str, Any]:
        user_tokens = _lexical_tokens(user_input)
        task_tokens = _lexical_tokens(task.description)
        overlap_score = _normalized_overlap_score(user_tokens, task_tokens)
        context_load = min(1.0, len(context_snippets) / max(self.stm_context_k + self.attention_capacity, 1))
        complexity = min(max(float(task.complexity), 0.0), 1.0)
        relevance = min(1.0, 0.15 + (0.65 * overlap_score) + (0.20 * min(max(float(task.urgency), 0.0), 1.0)))
        latency_ms = int(250 + (850 * complexity) + (150 * context_load))
        cost_tokens = int(128 + (384 * complexity) + (64 * context_load))
        return {
            "relevance": relevance,
            "urgency": task.urgency,
            "latency_ms": latency_ms,
            "cost_tokens": cost_tokens,
        }

    def _decision_engine_accepts_mode(self) -> bool:
        choose = getattr(self.decision_engine, "choose", None)
        if not callable(choose):
            return False
        try:
            params = list(inspect.signature(choose).parameters.values())
        except (TypeError, ValueError):
            return False
        positional_params = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if any(param.name == "mode" for param in params):
            return True
        return len(positional_params) >= 2

    async def _choose_decision(self, options: List[Option]) -> Decision:
        choose = getattr(self.decision_engine, "choose", None)
        if not callable(choose):
            raise RuntimeError("Decision engine lacks choose method")
        choose_async = cast(Callable[..., Awaitable[Decision]], choose)
        if self._decision_engine_accepts_mode():
            return await choose_async(options, self.mode)
        return await choose_async(options)

    def _resolve_chosen_option(self, options: List[Option], option_id: str) -> Option:
        chosen = next((option for option in options if option.id == option_id), None)
        if chosen is None:
            raise ValueError(f"Decision engine returned unknown option_id: {option_id}")
        return chosen

    def _turn_ok(self, text: str, llm_metadata: Dict[str, Any], act_meta: Dict[str, Any]) -> bool:
        if not text.strip():
            return False
        for metadata in (llm_metadata, act_meta):
            if not isinstance(metadata, dict):
                continue
            if metadata.get("error") or metadata.get("failed"):
                return False
            if "ok" in metadata and metadata.get("ok") is False:
                return False
        return True

    def _get_outcome_adapter(self):
        if self._outcome_adapter is not None:
            return self._outcome_adapter
        from .outcome import OutcomeAdapter

        self._outcome_adapter = OutcomeAdapter()
        return self._outcome_adapter

    def _reward_for_result(self, result: ActionResult) -> float:
        if not result.ok:
            return 0.0
        return 1.0 + min(1.0, len(result.content) / 500.0)

    def _finalize_turn(
        self,
        *,
        result: ActionResult,
        decision: Decision,
        candidates: List[str],
        attended: List[str],
    ) -> ActionResult:
        self.events.publish("executive.turn_finished", {
            "latency_ms": result.latency_ms,
            "tokens": result.tokens_used,
            "mode": self.mode.name,
            "new_memories": len(result.new_memories),
            "ok": result.ok,
            "error": result.metadata.get("error"),
        })
        try:
            utilization = len(attended) / max(len(candidates), 1)
            self.events.publish(
                "executive.memory_utilization",
                {"utilization": utilization, "attended": len(attended), "candidates": len(candidates)},
            )
        except Exception:
            pass
        self.bandit.update(decision.policy, self._reward_for_result(result))
        return result

    def _build_failed_result(
        self,
        *,
        t0: float,
        decision: Decision,
        text: str,
        tokens_used: int,
        act_meta: Dict[str, Any],
        error: Exception,
        stage: str,
    ) -> ActionResult:
        return ActionResult(
            ok=False,
            content=text,
            tokens_used=tokens_used,
            latency_ms=int((self.clock.now() - t0) * 1000),
            new_memories=[],
            metadata={
                "actuator": act_meta,
                "mode": self.mode.name,
                "decision": decision,
                "error": str(error),
                "error_stage": stage,
                "failed": True,
            },
        )

    # --------- Mode State Machine ---------

    def _transition(self, signal: str, fatigue: float = 0.0, load: float = 0.0) -> None:
        """Simple, explicit transition table."""
        prev = self.mode
        if self.mode == ExecutiveMode.FOCUSED:
            if signal == "overload" or fatigue > 0.7:
                self.mode = ExecutiveMode.RECOVERY
            elif signal == "reflect":
                self.mode = ExecutiveMode.REFLECTION
            elif signal == "multi":
                self.mode = ExecutiveMode.MULTI_TASK
            elif signal == "stuck":
                self.mode = ExecutiveMode.EXPLORATION

        elif self.mode == ExecutiveMode.MULTI_TASK:
            if signal == "focus":
                self.mode = ExecutiveMode.FOCUSED
            elif signal == "reflect":
                self.mode = ExecutiveMode.REFLECTION
            elif fatigue > 0.7:
                self.mode = ExecutiveMode.RECOVERY

        elif self.mode == ExecutiveMode.EXPLORATION:
            if signal in ("found", "focus"):
                self.mode = ExecutiveMode.FOCUSED
            elif signal == "reflect":
                self.mode = ExecutiveMode.REFLECTION
            elif fatigue > 0.7:
                self.mode = ExecutiveMode.RECOVERY

        elif self.mode == ExecutiveMode.REFLECTION:
            if signal in ("done", "focus"):
                self.mode = ExecutiveMode.FOCUSED

        elif self.mode == ExecutiveMode.RECOVERY:
            if fatigue < 0.4 and load < 0.5:
                self.mode = ExecutiveMode.FOCUSED

        if prev != self.mode:
            self.events.publish("executive.mode_changed", {
                "from": prev.name, "to": self.mode.name, "signal": signal, "fatigue": fatigue, "load": load
            })

    # --------- One Cognitive Turn ---------

    async def run_turn(
        self,
        user_input: str,
        high_level_goal: Goal | None = None,
        candidate_goals: Optional[List[Goal]] = None,
    ) -> ActionResult:
        t0 = self.clock.now()
        goal = high_level_goal or self.select_goal(user_input, candidate_goals)
        self.events.publish("executive.turn_started", {
            "mode": self.mode.name, "goal": goal.description, "input": user_input
        })

        # 1) Perceive & update STM
        await self.memory.stm_add(f"USER: {user_input}", tags=["input"])
        stm = await self.memory.stm_recent(self.stm_context_k)

        # 2) Retrieve candidates from LTM (+ attention allocation)
        similar = await self.memory.ltm_similar(user_input, k=12)
        candidates = [s for (s, _score) in similar]
        attended = await self.attention.allocate(query=user_input, candidates=candidates, capacity=self.attention_capacity)

        # 3) Expand goal → tasks → options
        tasks = self.planner.expand(goal)
        options: List[Option] = []
        for i, task in enumerate(tasks):
            prompt = f"{task.description}\nUser said: {user_input}\n"
            context_snippets = stm + attended
            meta = self._build_option_metadata(task, user_input, context_snippets)
            options.append(Option(
                id=f"opt::{task.id}::{i}",
                task=task,
                prompt=prompt,
                context_snippets=context_snippets,
                est_cost_tokens=int(meta["cost_tokens"]),
                est_latency_ms=int(meta["latency_ms"]),
                metadata=meta,
            ))

        # 4) Decide (policy selection could switch engines; here we log UCB)
        decision = await self._choose_decision(options)
        self.events.publish("executive.decision", {
            "policy": decision.policy,
            "option_id": decision.option_id,
            "score": decision.score,
            "rationale": decision.rationale,
        })

        chosen = self._resolve_chosen_option(options, decision.option_id)

        # 5) Act (LLM completion and/or tool action)
        try:
            text, md = await self.llm.complete(chosen.prompt, chosen.context_snippets, max_tokens=self.max_tokens)
        except Exception as e:
            logger.warning("Executive turn failed during LLM completion: %s", e)
            failed = self._build_failed_result(
                t0=t0,
                decision=decision,
                text="",
                tokens_used=0,
                act_meta={},
                error=e,
                stage="llm",
            )
            return self._finalize_turn(result=failed, decision=decision, candidates=candidates, attended=attended)

        await self.memory.stm_add(f"ASSISTANT: {text}", tags=["output"])

        # Optional: route through Actuator for tool-use if requested
        act_meta: Dict[str, Any] = {}
        if chosen.metadata.get("tool_action"):
            try:
                act_meta = await self.actuator.act(chosen.metadata["tool_action"])
            except Exception as e:
                logger.warning("Executive turn failed during actuator action: %s", e)
                failed = self._build_failed_result(
                    t0=t0,
                    decision=decision,
                    text=text,
                    tokens_used=int(md.get("tokens_used", chosen.est_cost_tokens)) if isinstance(md, dict) else chosen.est_cost_tokens,
                    act_meta={},
                    error=e,
                    stage="actuator",
                )
                return self._finalize_turn(result=failed, decision=decision, candidates=candidates, attended=attended)

        # 6) Consolidate (store salient stuff in LTM)
        new_ids: List[str] = []
        consolidate_ok = False
        if self._salience:
            try:
                sal_score = self._salience.score(text, stm)
                consolidate_ok = sal_score >= self._salience_threshold
            except Exception:
                consolidate_ok = False
        else:
            consolidate_ok = len(text) > 120
        if consolidate_ok:
            new_id = await self.memory.ltm_add(text, tags=["response", "summary"])
            new_ids.append(new_id)

        # 7) Mode transition heuristic
        fatigue = float(md.get("fatigue", 0.3)) if isinstance(md, dict) else 0.3
        load = min(1.0, (chosen.est_cost_tokens / 2048.0) + (chosen.est_latency_ms / 5000.0))
        signal = "focus"
        if load > 0.9:
            signal = "overload"
        elif decision.score < 0.3 and self.mode == ExecutiveMode.EXPLORATION:
            signal = "reflect"
        elif decision.score < 0.3:
            signal = "stuck"
        self._transition(signal=signal, fatigue=fatigue, load=load)

        dt_ms = int((self.clock.now() - t0) * 1000)
        ok = self._turn_ok(text, md if isinstance(md, dict) else {}, act_meta)
        res = ActionResult(
            ok=ok,
            content=text,
            tokens_used=int(md.get("tokens_used", chosen.est_cost_tokens)) if isinstance(md, dict) else chosen.est_cost_tokens,
            latency_ms=dt_ms,
            new_memories=new_ids,
            metadata={"actuator": act_meta, "mode": self.mode.name, "decision": decision}
        )
        return self._finalize_turn(result=res, decision=decision, candidates=candidates, attended=attended)

    # --------- Outcome adaptation ---------

    async def update_from_outcome(self, outcome: "Outcome") -> None:  # type: ignore[name-defined]
        """Adjust internal knobs based on observed outcome.

        - Increase exploration randomness when repeated low rewards.
        - Tighten/loosen salience threshold based on memory utility.
        """
        try:
            adapter = self._get_outcome_adapter()
        except Exception as e:
            logger.warning("Failed to initialize OutcomeAdapter: %s", e)
            return
        reward = adapter.compute_reward(outcome)
        # adjust salience threshold (higher reward -> slightly lower threshold to capture more)
        if reward > 1.2:
            self._salience_threshold = max(0.45, self._salience_threshold - 0.02)
        elif reward < 0.6:
            self._salience_threshold = min(0.75, self._salience_threshold + 0.03)
        # update bandit with refined reward
        self.bandit.update(outcome.decision.policy, reward)
        # adapt contextual decision engine exploration scale if supported
        if hasattr(self.decision_engine, "adapt"):
            try:
                self.decision_engine.adapt(reward)  # type: ignore[misc]
            except Exception:
                pass
        self._persist_adaptive_state()

    # --------- Adaptive state persistence ---------
    def _load_adaptive_state(self) -> None:
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._salience_threshold = float(data.get("salience_threshold", self._salience_threshold))
                if hasattr(self.decision_engine, "exploration_scale"):
                    current_scale = float(getattr(self.decision_engine, "exploration_scale", 0.05))
                    exploration_scale = float(data.get("exploration_scale", current_scale))
                    setattr(self.decision_engine, "exploration_scale", exploration_scale)
        except Exception as e:
            logger.warning("Failed to load executive adaptive state from %s: %s", self._state_path, e)

    def _persist_adaptive_state(self) -> None:
        try:
            data = {
                "salience_threshold": self._salience_threshold,
            }
            if hasattr(self.decision_engine, "exploration_scale"):
                data["exploration_scale"] = getattr(self.decision_engine, "exploration_scale")
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed to persist executive adaptive state to %s: %s", self._state_path, e)


# =========================
# Wiring Example (use/adapt)
# =========================

# You already have implementations for Memory (STM/LTM with ChromaDB),
# Attention (top-k allocator), and LLM. Make adapter classes that conform
# to the Protocols above. Then wire like this inside your app startup:
#
# memory: MemoryService = YourMemoryAdapter(...)
# attention: AttentionService = YourAttentionAdapter(...)
# llm: LLMService = YourLLMAdapter(...)
# actuator: ActuatorService = YourActuatorAdapter(...)
#
# bus = EventBus()
# bus.subscribe("executive.decision", lambda e: print("[DECISION]", e))
# bus.subscribe("executive.mode_changed", lambda e: print("[MODE]", e))
# bus.subscribe("executive.turn_finished", lambda e: print("[DONE]", e))
#
# controller = ExecutiveController(
#     memory=memory,
#     attention=attention,
#     llm=llm,
#     actuator=actuator,
#     event_bus=bus,
# )
#
# goal = Goal(id="g1", description="Answer user helpfully with memory-aware context", priority=0.7)
# result = await controller.run_turn(user_input="Summarize my last session and next steps.", high_level_goal=goal)
# print(result.content)
