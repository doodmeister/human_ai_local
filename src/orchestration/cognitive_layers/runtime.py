from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class ChatCognitiveLayerRuntime:
    def __init__(self) -> None:
        self._drive_state = None
        self._drive_processor = None
        self._drive_last_turn_time: Optional[float] = None
        self._drive_impact_history: list = []

        self._felt_sense_generator = None
        self._felt_sense_history = None
        self._mood_labeler = None
        self._current_mood = None
        self._recent_valences: list = []

        self._relational_field = None
        self._relational_processor = None

        self._pattern_field = None
        self._pattern_detector = None
        self._pattern_turn_counter = 0

        self._self_model = None
        self._self_model_builder = None
        self._self_model_turn_counter = 0

        self._narrative = None
        self._narrative_constructor = None
        self._narrative_turn_counter = 0

    def get_drive_system(self):
        if self._drive_state is None:
            from src.cognition.drives import DriveConfig, DriveProcessor, DriveState
            from src.core.config import get_global_config

            cfg = get_global_config()
            drive_cfg = cfg.drives if cfg.drives is not None else DriveConfig()
            self._drive_processor = DriveProcessor(config=drive_cfg)
            self._drive_state = DriveState()
            self._drive_last_turn_time = time.time()
            logger.info("Drive system initialized: %s", self._drive_state.summary())
        return self._drive_state, self._drive_processor

    def get_drive_state(self) -> Optional[Dict[str, Any]]:
        if self._drive_state is None:
            return None
        return self._drive_state.to_dict()

    def get_felt_sense_system(self):
        if self._felt_sense_generator is None:
            from src.cognition.felt_sense import FeltSenseConfig, FeltSenseGenerator, FeltSenseHistory, MoodLabeler
            from src.core.config import get_global_config

            cfg = get_global_config()
            fs_cfg = cfg.felt_sense if cfg.felt_sense is not None else FeltSenseConfig()
            self._felt_sense_generator = FeltSenseGenerator(config=fs_cfg)
            self._felt_sense_history = FeltSenseHistory(max_size=fs_cfg.history_size)
            self._mood_labeler = MoodLabeler(config=fs_cfg)
            logger.info("Felt-sense system initialized")
        return self._felt_sense_generator, self._felt_sense_history, self._mood_labeler

    def get_mood_state(self) -> Optional[Dict[str, Any]]:
        if self._current_mood is None:
            return None
        return self._current_mood.to_dict()

    def get_relational_system(self):
        if self._relational_field is None:
            from src.cognition.relational import RelationalConfig, RelationalField, RelationalProcessor
            from src.core.config import get_global_config

            cfg = get_global_config()
            rel_cfg = cfg.relational if cfg.relational is not None else RelationalConfig()
            self._relational_field = RelationalField()
            self._relational_processor = RelationalProcessor(config=rel_cfg)
            logger.info("Relational field initialized")
        return self._relational_field, self._relational_processor

    def get_relational_state(self) -> Optional[Dict[str, Any]]:
        if self._relational_field is None:
            return None
        return self._relational_field.to_dict()

    def get_pattern_system(self):
        if self._pattern_field is None:
            from src.cognition.patterns import PatternConfig, PatternDetector, PatternField
            from src.core.config import get_global_config

            cfg = get_global_config()
            pat_cfg = cfg.patterns if cfg.patterns is not None else PatternConfig()
            self._pattern_field = PatternField(max_patterns=pat_cfg.max_patterns)
            self._pattern_detector = PatternDetector(config=pat_cfg)
            logger.info("Pattern system initialized")
        return self._pattern_field, self._pattern_detector

    def get_pattern_state(self) -> Optional[Dict[str, Any]]:
        if self._pattern_field is None:
            return None
        return self._pattern_field.to_dict()

    def get_self_model_system(self):
        if self._self_model_builder is None:
            from src.cognition.selfmodel import SelfModelBuilder, SelfModelConfig
            from src.core.config import get_global_config

            cfg = get_global_config()
            sm_cfg = cfg.selfmodel if cfg.selfmodel is not None else SelfModelConfig()
            self._self_model_builder = SelfModelBuilder(config=sm_cfg)
            logger.info("Self-model system initialized")
        return self._self_model_builder

    def get_self_model_state(self) -> Optional[Dict[str, Any]]:
        if self._self_model is None:
            return None
        data = self._self_model.to_dict()
        data.pop("_blind_spots", None)
        return data

    def get_narrative_system(self):
        if self._narrative_constructor is None:
            from src.cognition.narrative import NarrativeConfig, NarrativeConstructor
            from src.core.config import get_global_config

            cfg = get_global_config()
            narr_cfg = cfg.narrative if cfg.narrative is not None else NarrativeConfig()
            self._narrative_constructor = NarrativeConstructor(config=narr_cfg)
            logger.info("Narrative system initialized")
        return self._narrative_constructor

    def get_narrative_state(self) -> Optional[Dict[str, Any]]:
        if self._narrative is None:
            return None
        return self._narrative.to_dict()

    def process_turn(
        self,
        session: Any,
        tick: Any,
        message: str,
        salience: float,
        valence: float,
        global_turn_counter: int,
    ) -> None:
        try:
            drive_state, drive_processor = self.get_drive_system()
            now_ts = time.time()
            elapsed_min = (now_ts - self._drive_last_turn_time) / 60.0 if self._drive_last_turn_time else 0.0
            self._drive_last_turn_time = now_ts
            drive_state, drive_impact = drive_processor.process_turn(
                drive_state,
                message,
                salience=salience,
                valence=valence,
                elapsed_minutes=elapsed_min,
            )
            drive_conflicts = drive_processor.detect_conflicts(drive_state)
            tick.state["drive_state"] = drive_state
            tick.state["drive_impact"] = drive_impact
            tick.state["drive_conflicts"] = drive_conflicts
            setattr(session, "_drive_state_snapshot", drive_state)
            self._drive_impact_history.append(drive_impact)
            if len(self._drive_impact_history) > 100:
                self._drive_impact_history = self._drive_impact_history[-50:]
            if global_turn_counter % 20 == 0 and global_turn_counter > 0:
                drive_processor.adapt_baselines(drive_state)
                drive_processor.adapt_sensitivities(drive_state, self._drive_impact_history[-20:])
        except Exception as exc:
            logger.debug("Drive processing skipped: %s", exc)

        try:
            self._recent_valences.append(valence)
            if len(self._recent_valences) > 10:
                self._recent_valences = self._recent_valences[-10:]

            drive_state_for_felt = tick.state.get("drive_state") or self._drive_state
            if drive_state_for_felt is not None:
                felt_sense_generator, felt_sense_history, mood_labeler = self.get_felt_sense_system()
                felt = felt_sense_generator.generate(drive_state_for_felt, recent_valences=self._recent_valences)
                felt_sense_history.update(felt)
                mood = mood_labeler.label_mood(felt)
                self._current_mood = mood

                tick.state["felt_sense"] = felt
                tick.state["mood"] = mood
                tick.state["felt_sense_trend"] = felt_sense_history.trend()
                setattr(session, "_felt_sense_snapshot", felt)
                setattr(session, "_mood_snapshot", mood)
        except Exception as exc:
            logger.debug("Felt-sense processing skipped: %s", exc)

        try:
            relational_field, relational_processor = self.get_relational_system()
            person_id = session.session_id or "default"
            relational_field.set_interlocutor(person_id)
            drive_impact_obj = tick.state.get("drive_impact")
            relational_model = relational_processor.process_turn(
                relational_field,
                person_id,
                message,
                valence=valence,
                salience=salience,
                drive_impact=drive_impact_obj,
            )
            tick.state["relational_model"] = relational_model
            setattr(session, "_relational_model_snapshot", relational_model)
        except Exception as exc:
            logger.debug("Relational processing skipped: %s", exc)

        try:
            self._pattern_turn_counter += 1
            pattern_field, pattern_detector = self.get_pattern_system()
            if self._pattern_turn_counter % pattern_detector.config.detection_interval == 0:
                pattern_detector.detect_patterns(
                    pattern_field,
                    drive_state=tick.state.get("drive_state"),
                    drive_impacts=self._drive_impact_history[-20:] or None,
                    felt_sense_history=self._felt_sense_history,
                    relational_field=self._relational_field,
                    conflicts=tick.state.get("drive_conflicts") or None,
                )
            tick.state["pattern_field"] = pattern_field
            setattr(session, "_pattern_field_snapshot", pattern_field)
        except Exception as exc:
            logger.debug("Pattern detection skipped: %s", exc)

        try:
            self._self_model_turn_counter += 1
            self_model_builder = self.get_self_model_system()
            if self._self_model_turn_counter % self_model_builder.config.update_interval == 0:
                self._self_model = self_model_builder.build_self_model(
                    pattern_field=self._pattern_field,
                    drive_state=tick.state.get("drive_state"),
                    felt_sense=tick.state.get("felt_sense"),
                    mood=tick.state.get("mood"),
                    existing_self_model=self._self_model,
                )
            if self._self_model is not None:
                tick.state["self_model"] = self._self_model
                setattr(session, "_self_model_snapshot", self._self_model)
        except Exception as exc:
            logger.debug("Self-model update skipped: %s", exc)

        try:
            self._narrative_turn_counter += 1
            narrative_constructor = self.get_narrative_system()
            should_update, trigger = narrative_constructor.should_update(
                self_model=self._self_model,
                previous_narrative=self._narrative,
                turn_counter=self._narrative_turn_counter,
            )
            if should_update:
                self._narrative = narrative_constructor.construct_narrative(
                    self_model=self._self_model,
                    pattern_field=self._pattern_field,
                    drive_state=tick.state.get("drive_state"),
                    relational_field=self._relational_field,
                    mood=tick.state.get("mood"),
                    previous_narrative=self._narrative,
                    trigger=trigger,
                )
            if self._narrative is not None:
                tick.state["narrative"] = self._narrative
                setattr(session, "_narrative_snapshot", self._narrative)
        except Exception as exc:
            logger.debug("Narrative construction skipped: %s", exc)