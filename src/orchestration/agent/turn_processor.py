from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid

from src.memory.schema import canonical_item_to_prompt_memory_payload, normalize_memory_search_results

from .. import CognitiveStep, CognitiveTick
from ..chat.emotion_salience import estimate_salience_and_valence


logger = logging.getLogger(__name__)


class CognitiveTurnProcessor:
    def __init__(
        self,
        *,
        get_session: Callable[[], Any],
        get_session_id: Callable[[], str],
        get_cognitive_layers: Callable[[], Any],
        get_sensory_interface: Callable[[], Any],
        get_memory: Callable[[], Any],
        get_attention: Callable[[], Any],
        get_llm_session: Callable[[], Any],
        get_conversation_context: Callable[[], List[Dict[str, Any]]],
        get_neural_integration: Callable[[], Any],
        get_current_fatigue: Callable[[], float],
        get_turn_counter: Callable[[], int],
        increment_turn_counter: Callable[[], None],
        set_current_fatigue: Callable[[float], None],
        set_attention_focus: Callable[[List[Any]], None],
    ) -> None:
        self._get_session = get_session
        self._get_session_id = get_session_id
        self._get_cognitive_layers = get_cognitive_layers
        self._get_sensory_interface = get_sensory_interface
        self._get_memory = get_memory
        self._get_attention = get_attention
        self._get_llm_session = get_llm_session
        self._get_conversation_context = get_conversation_context
        self._get_neural_integration = get_neural_integration
        self._get_current_fatigue = get_current_fatigue
        self._get_turn_counter = get_turn_counter
        self._increment_turn_counter = increment_turn_counter
        self._set_current_fatigue = set_current_fatigue
        self._set_attention_focus = set_attention_focus

    async def process_input(
        self,
        input_data: str,
        input_type: str = "text",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            logger.debug("Processing %s input: %s...", input_type, input_data[:100])

            tick = CognitiveTick(owner="cognitive_agent", kind="input_turn")
            tick.state["session_id"] = self._get_session_id()
            tick.state["input_type"] = input_type
            if context is not None:
                tick.state["context"] = context

            tick.assert_step(CognitiveStep.PERCEIVE)
            processed_input = await self.process_sensory_input(input_data, input_type)
            tick.state["processed_input"] = processed_input

            salience, valence = estimate_salience_and_valence(input_data)
            tick.state["message_salience"] = salience
            tick.state["message_valence"] = valence
            try:
                self._get_cognitive_layers().process_turn(
                    session=self._get_session(),
                    tick=tick,
                    message=input_data,
                    salience=salience,
                    valence=valence,
                    global_turn_counter=self._get_turn_counter(),
                )
                response_policy = tick.state.get("response_policy")
                if response_policy is not None:
                    if hasattr(response_policy, "to_dict"):
                        processed_input["response_policy"] = response_policy.to_dict()
                    else:
                        processed_input["response_policy"] = response_policy
            except Exception as exc:
                logger.debug("Cognitive-layer processing skipped: %s", exc)

            tick.advance(CognitiveStep.PERCEIVE)

            tick.assert_step(CognitiveStep.UPDATE_STM)
            tick.state["raw_input"] = input_data
            tick.advance(CognitiveStep.UPDATE_STM)

            tick.assert_step(CognitiveStep.RETRIEVE)
            memory_context = await self.retrieve_memory_context(processed_input)
            tick.state["memory_context"] = memory_context
            tick.advance(CognitiveStep.RETRIEVE)

            tick.assert_step(CognitiveStep.DECIDE)
            attention_scores = await self.calculate_attention_allocation(processed_input, memory_context)
            tick.mark_decided({"attention_scores": attention_scores})
            tick.state["attention_scores"] = attention_scores
            tick.advance(CognitiveStep.DECIDE)

            tick.assert_step(CognitiveStep.ACT)
            response = await self.generate_response(processed_input, memory_context, attention_scores)
            tick.state["response"] = response
            tick.advance(CognitiveStep.ACT)

            tick.assert_step(CognitiveStep.REFLECT)
            self.update_cognitive_state(attention_scores)
            tick.advance(CognitiveStep.REFLECT)

            tick.assert_step(CognitiveStep.CONSOLIDATE)
            await self.consolidate_memory(
                input_data,
                response,
                attention_scores,
                recalled_memories=memory_context,
                response_policy=processed_input.get("response_policy"),
            )
            tick.finish()
            return response
        except Exception as exc:
            logger.error("Error in cognitive processing: %s", exc)
            return "I encountered an error while processing your request. Please try again."

    async def process_sensory_input(self, input_data: str, input_type: str) -> Dict[str, Any]:
        try:
            processed_sensory_data = self._get_sensory_interface().process_user_input(input_data)
            return {
                "raw_input": input_data,
                "type": input_type,
                "processed_at": datetime.now(),
                "entropy_score": processed_sensory_data.entropy_score,
                "salience_score": processed_sensory_data.salience_score,
                "relevance_score": processed_sensory_data.relevance_score,
                "embedding": processed_sensory_data.embedding,
                "filtered": processed_sensory_data.filtered,
                "processing_metadata": processed_sensory_data.processing_metadata,
            }
        except Exception as exc:
            logger.error("Error in sensory processing: %s", exc)
            return {
                "raw_input": input_data,
                "type": input_type,
                "processed_at": datetime.now(),
                "entropy_score": 0.5,
                "salience_score": 0.5,
                "relevance_score": 0.5,
                "embedding": None,
                "filtered": False,
                "processing_metadata": {"error": str(exc)},
            }

    async def retrieve_memory_context(self, processed_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            conversation_context = self._get_conversation_context()
            if conversation_context:
                recent_interactions = [
                    f"User: {turn['user_input']}\nAI: {turn['ai_response']}"
                    for turn in conversation_context[-2:]
                ]
                proactive_query = "\n".join(recent_interactions)
                proactive_query += f"\nUser: {processed_input['raw_input']}"
            else:
                proactive_query = processed_input["raw_input"]

            logger.debug("Proactive memory search with query: '%s...'", proactive_query[:200])
            memories = self._get_memory().search_memories(query=proactive_query, max_results=5)

            context_memories = [
                canonical_item_to_prompt_memory_payload(item)
                for item in normalize_memory_search_results(memories)
                if item.content
            ]

            try:
                semantic_results = self._get_memory().semantic.search(query=proactive_query)
                context_memories.extend(
                    canonical_item_to_prompt_memory_payload(item)
                    for item in normalize_memory_search_results(
                        [(fact, 0.8, "Semantic") for fact in list(semantic_results)[:5]]
                    )
                    if item.content
                )
            except Exception as exc:
                logger.debug("Semantic memory search unavailable: %s", exc)

            return context_memories
        except Exception as exc:
            logger.error("Error retrieving memory context: %s", exc)
            return []

    async def calculate_attention_allocation(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        relevance = processed_input.get("relevance_score", 0.5)
        novelty = processed_input.get("entropy_score", 0.6)
        emotional_salience = processed_input.get("salience_score", 0.0)

        if memory_context:
            avg_memory_relevance = sum(mem["relevance"] for mem in memory_context) / len(memory_context)
            relevance = min(1.0, relevance + (avg_memory_relevance * 0.2))

        base_salience = (relevance * 0.6) + (emotional_salience * 0.4)

        priority = 0.7 if processed_input.get("type") == "text" else 0.5
        effort_required = 0.7 if len(processed_input.get("raw_input", "")) > 100 else 0.5

        attention = self._get_attention()
        attention_result = attention.allocate_attention(
            stimulus_id=f"input_{datetime.now().strftime('%H%M%S_%f')}",
            content=processed_input["raw_input"],
            salience=base_salience,
            novelty=novelty,
            priority=priority,
            effort_required=effort_required,
        )

        enhanced_attention = await self.enhance_attention_with_neural(
            processed_input,
            attention_result,
            base_salience,
            novelty,
        )
        attention.update_attention_state()

        return {
            "overall_attention": enhanced_attention.get("attention_score", 0.5),
            "relevance": relevance,
            "novelty": enhanced_attention.get("neural_novelty", novelty),
            "emotional_salience": emotional_salience,
            "allocated": enhanced_attention.get("allocated", False),
            "cognitive_load": enhanced_attention.get("current_load", 0.0),
            "fatigue_level": enhanced_attention.get("fatigue_level", 0.0),
            "items_in_focus": enhanced_attention.get("items_in_focus", 0),
            "neural_enhanced": enhanced_attention.get("neural_enhanced", False),
            "neural_enhancement": enhanced_attention.get("neural_enhancement", 0.0),
        }

    async def generate_response(
        self,
        processed_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        attention_scores: Dict[str, float],
    ) -> str:
        response_policy = processed_input.get("response_policy")
        return await self._get_llm_session().generate_response(
            processed_input=processed_input,
            memory_context=memory_context,
            response_policy=response_policy,
        )

    async def consolidate_memory(
        self,
        input_data: str,
        response: str,
        attention_scores: Dict[str, float],
        recalled_memories: Optional[List[Dict[str, Any]]] = None,
        response_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            interaction_content = f"User: {input_data}\nAI: {response}"
            self._get_memory().store_memory(
                memory_id=str(uuid.uuid4()),
                content=interaction_content,
                importance=attention_scores.get("overall_salience", 0.5),
            )

            interaction_record = {
                "user_input": input_data,
                "ai_response": response,
                "timestamp": datetime.now(),
            }
            conversation_context = self._get_conversation_context()
            conversation_context.append(interaction_record)
            if len(conversation_context) > 10:
                conversation_context.pop(0)

            if recalled_memories and not str(response).startswith("[ERROR]"):
                try:
                    self._get_memory().reconsolidate_recalled_memories(
                        recalled_memories,
                        outcome="reinforce",
                        response_policy=response_policy,
                        note="turn_response",
                    )
                except Exception as exc:
                    logger.debug("Reconsolidation skipped: %s", exc)

            self._increment_turn_counter()

            logger.debug("Interaction consolidated into memory and conversation context updated.")
        except Exception as exc:
            logger.error("Error in memory consolidation: %s", exc)

    def update_cognitive_state(self, attention_scores: Dict[str, float]) -> None:
        attention = self._get_attention()
        self._set_current_fatigue(attention.current_fatigue)
        self._set_attention_focus(attention.get_attention_focus())

        logger.debug(
            "Updated cognitive state - Fatigue: %.3f, Cognitive Load: %.3f, Items in Focus: %s",
            self._get_current_fatigue(),
            attention_scores.get("cognitive_load", 0.0),
            attention_scores.get("items_in_focus", 0),
        )

    async def enhance_attention_with_neural(
        self,
        processed_input: Dict[str, Any],
        attention_result: Dict[str, Any],
        base_salience: float,
        novelty: float,
    ) -> Dict[str, Any]:
        neural_integration = self._get_neural_integration()
        if not neural_integration:
            return attention_result

        try:
            if "embedding" not in processed_input:
                return attention_result

            embedding = processed_input["embedding"]

            import numpy as np
            import torch

            if isinstance(embedding, np.ndarray):
                embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0)
            else:
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

            attention_scores = torch.tensor([base_salience], dtype=torch.float32)
            neural_result = await neural_integration.process_attention_update(
                embedding_tensor,
                attention_scores,
                salience_scores=torch.tensor([novelty], dtype=torch.float32),
            )

            if "error" not in neural_result:
                novelty_scores = neural_result.get("novelty_scores", torch.tensor([novelty]))
                processing_quality = neural_result.get("processing_quality", 1.0)

                if len(novelty_scores) > 0:
                    enhanced_novelty = float(novelty_scores[0])
                    neural_enhancement = min(0.2, enhanced_novelty * 0.1)
                    enhanced_attention_score = min(
                        1.0,
                        attention_result.get("attention_score", 0.5) + neural_enhancement,
                    )
                    attention_result.update(
                        {
                            "attention_score": enhanced_attention_score,
                            "neural_enhancement": neural_enhancement,
                            "neural_novelty": enhanced_novelty,
                            "neural_processing_quality": processing_quality,
                            "neural_enhanced": True,
                        }
                    )
                    logger.debug(
                        "Neural attention enhancement: +%.3f (novelty: %.3f)",
                        neural_enhancement,
                        enhanced_novelty,
                    )

            return attention_result
        except Exception as exc:
            logger.warning("Neural attention enhancement error: %s", exc)
            return attention_result