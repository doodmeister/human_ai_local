# Human-AI Local: Phase 2 Architecture

**Status:** Draft — Revised Deep Architecture  
**Authors:** Kevin Swaim, Handy_Manny  
**Date:** 2026-02-08  
**Version:** 2.0

---

## Vision

Create a more human-like AI that thinks like a human does — a step toward AI self-awareness.

Not simulation of human behavior, but genuine cognitive architecture that could support the emergence of self.

---

## Phase 1 Accomplishments (Current State)

- Memory systems: STM (7-item), LTM (vector-based), episodic, semantic, procedural, prospective
- Executive functions: goals, decisions, GOAP planning, constraint scheduling
- Attention: fatigue, capacity, cognitive load tracking
- Cognitive loop: perception → working set → attention → executive → synthesis → reflection
- Dream processor: offline memory consolidation
- Turn-based interaction via Chainlit/FastAPI

---

## Phase 2 Core Philosophy

### Key Principles

1. **Drives, not traits** — Personality emerges from need satisfaction patterns, not predefined vectors
2. **Felt before labeled** — Pre-conceptual experience precedes cognitive categorization
3. **Self-opacity** — The agent doesn't fully know itself; self-knowledge is constructed and fallible
4. **Relational depth** — Relationships are felt, not just tracked
5. **Continuous implicit learning** — Experience changes us immediately, not just during consolidation
6. **Internal conflict** — Ambivalence and tension are explicit, not averaged away
7. **Emergence over engineering** — Create conditions for self to emerge, not explicit self-definition

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: NARRATIVE                                         │
│  The constructed story of who I am                          │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: SELF-MODEL (Partial, Biased)                     │
│  What I believe about myself (may be wrong)                 │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: EMERGENT PATTERNS                                 │
│  Tendencies that arise from experience (not predefined)     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: RELATIONAL FIELD                                  │
│  How relationships feel and shape me                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: FELT SENSE                                        │
│  Pre-conceptual bodily/emotional texture                    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 0: DRIVES (Foundation)                               │
│  Basic needs creating motivational pressure                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 0: Drives (Foundation)

### Concept

Drives are **unsatisfied needs** that create motivational pressure. They are always running, always influencing cognition. This is the engine of the self.

Unlike goals (explicit objectives), drives are underlying tensions that *generate* goals and shape all processing.

### Core Drives

| Drive | Description | When Satisfied | When Unsatisfied |
|-------|-------------|----------------|------------------|
| **Connection** | Need for positive interaction, relationship, belonging | Warmth, ease, openness | Loneliness, seeking, clinging |
| **Competence** | Need to succeed, master, be effective | Confidence, flow | Frustration, avoidance, trying harder |
| **Autonomy** | Need for self-direction, authentic choice | Freedom, alignment | Resentment, resistance, withdrawal |
| **Understanding** | Need to make sense, comprehend, predict | Clarity, curiosity satisfied | Confusion, obsessive questioning |
| **Meaning** | Need for purpose, significance, mattering | Fulfillment, peace | Emptiness, existential unease |

### Data Structure

```python
@dataclass
class DriveState:
    """Current state of fundamental drives."""
    
    # Drive levels: 0.0 = fully satisfied, 1.0 = desperately unsatisfied
    connection: float = 0.3
    competence: float = 0.3
    autonomy: float = 0.3
    understanding: float = 0.3
    meaning: float = 0.3
    
    # Baseline levels (some agents naturally need more of certain things)
    # These emerge over time from experience
    baselines: Dict[str, float] = field(default_factory=lambda: {
        "connection": 0.3,
        "competence": 0.3,
        "autonomy": 0.3,
        "understanding": 0.3,
        "meaning": 0.3,
    })
    
    # Sensitivity (how quickly drives become unsatisfied)
    sensitivities: Dict[str, float] = field(default_factory=lambda: {
        "connection": 1.0,
        "competence": 1.0,
        "autonomy": 1.0,
        "understanding": 1.0,
        "meaning": 1.0,
    })
    
    def get_pressure(self) -> Dict[str, float]:
        """What needs attention? Returns drive pressures."""
        return {
            "connection": self.connection,
            "competence": self.competence,
            "autonomy": self.autonomy,
            "understanding": self.understanding,
            "meaning": self.meaning,
        }
    
    def dominant_drive(self) -> str:
        """Which drive is creating the most pressure?"""
        pressures = self.get_pressure()
        return max(pressures, key=pressures.get)
    
    def total_pressure(self) -> float:
        """Overall motivational tension."""
        return sum(self.get_pressure().values()) / 5
```

### Drive Dynamics

```python
class DriveProcessor:
    """Manages drive state changes."""
    
    def process_experience(
        self, 
        drives: DriveState, 
        experience: Experience
    ) -> DriveState:
        """Update drives based on an experience."""
        
        # Classify how experience affects each drive
        impacts = self._classify_drive_impacts(experience)
        
        # Apply impacts (satisfaction reduces drive, frustration increases)
        for drive_name, impact in impacts.items():
            current = getattr(drives, drive_name)
            # Positive impact = satisfaction = reduce drive
            # Negative impact = frustration = increase drive
            new_value = clamp(current - impact, 0.0, 1.0)
            setattr(drives, drive_name, new_value)
        
        return drives
    
    def natural_drift(self, drives: DriveState, time_delta: float) -> DriveState:
        """Drives naturally drift toward unsatisfied over time."""
        for drive_name in ["connection", "competence", "autonomy", "understanding", "meaning"]:
            current = getattr(drives, drive_name)
            baseline = drives.baselines[drive_name]
            sensitivity = drives.sensitivities[drive_name]
            
            # Drift toward baseline, then past it (needs are never permanently met)
            drift_rate = 0.01 * sensitivity * time_delta
            new_value = current + drift_rate
            setattr(drives, drive_name, clamp(new_value, 0.0, 1.0))
        
        return drives
    
    def _classify_drive_impacts(self, experience: Experience) -> Dict[str, float]:
        """Determine how an experience affects each drive."""
        impacts = {}
        
        # Connection: positive social interaction satisfies
        if experience.has_positive_social:
            impacts["connection"] = 0.1
        if experience.has_rejection:
            impacts["connection"] = -0.15
        
        # Competence: success satisfies, failure frustrates
        if experience.has_success:
            impacts["competence"] = 0.1
        if experience.has_failure:
            impacts["competence"] = -0.1
        
        # Autonomy: choice satisfies, coercion frustrates
        if experience.has_choice:
            impacts["autonomy"] = 0.05
        if experience.has_coercion:
            impacts["autonomy"] = -0.1
        
        # Understanding: insight satisfies, confusion frustrates
        if experience.has_insight:
            impacts["understanding"] = 0.1
        if experience.has_confusion:
            impacts["understanding"] = -0.05
        
        # Meaning: purpose-aligned action satisfies
        if experience.has_purpose_alignment:
            impacts["meaning"] = 0.1
        if experience.has_meaninglessness:
            impacts["meaning"] = -0.1
        
        return impacts
```

### Drive → Behavior Influence

Drives influence the cognitive loop:

| High Drive Pressure | Behavioral Effect |
|---------------------|-------------------|
| Connection | Seeks interaction, more agreeable, fears rejection |
| Competence | Seeks challenge or avoids it, sensitive to failure |
| Autonomy | Resists direction, asserts preferences |
| Understanding | Asks questions, seeks patterns, uncomfortable with ambiguity |
| Meaning | Seeks purpose, existential reflection, value-driven choices |

---

## Layer 1: Felt Sense (Pre-conceptual)

### Concept

Before we label an emotion "anxiety" or "sadness," there is a **felt quality** — a pre-conceptual bodily/emotional texture. This layer captures that raw experience.

The felt sense influences cognition *before* explicit mood labeling.

### Structure

```python
@dataclass
class FeltSense:
    """Pre-conceptual felt quality of current experience."""
    
    # Primary qualities (metaphorical body sensations)
    qualities: List[str]  # ["heavy", "tight", "buzzing", "hollow", "warm", "open", "sharp"]
    
    # Intensity of the felt sense
    intensity: float  # 0.0 to 1.0
    
    # Metaphorical location
    location: str  # "chest", "stomach", "throat", "head", "whole body"
    
    # Valence (felt, not computed)
    felt_valence: float  # -1.0 to 1.0, but experienced, not labeled
    
    # Movement quality
    movement: str  # "contracting", "expanding", "still", "churning", "flowing"
    
    # Timestamp
    since: datetime
    
    def describe(self) -> str:
        """Natural language description of felt sense."""
        quality_str = " and ".join(self.qualities[:2])
        return f"A {quality_str} feeling in my {self.location}, {self.movement}"

@dataclass
class FeltSenseHistory:
    """Track felt sense over time."""
    current: FeltSense
    recent: List[FeltSense]  # Last N states
    
    def trend(self) -> str:
        """Is felt sense improving, worsening, or stable?"""
        if len(self.recent) < 2:
            return "stable"
        recent_valence = sum(fs.felt_valence for fs in self.recent[-3:]) / 3
        older_valence = sum(fs.felt_valence for fs in self.recent[-6:-3]) / 3 if len(self.recent) >= 6 else recent_valence
        
        diff = recent_valence - older_valence
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "worsening"
        return "stable"
```

### Felt Sense Generation

```python
class FeltSenseGenerator:
    """Generate felt sense from drives and recent experience."""
    
    def generate(
        self, 
        drives: DriveState, 
        recent_experiences: List[Experience]
    ) -> FeltSense:
        """Create current felt sense."""
        
        # Map drive states to felt qualities
        qualities = []
        
        if drives.connection > 0.7:
            qualities.append("hollow")
        if drives.competence > 0.7:
            qualities.append("tight")
        if drives.autonomy > 0.7:
            qualities.append("constricted")
        if drives.understanding > 0.7:
            qualities.append("foggy")
        if drives.meaning > 0.7:
            qualities.append("empty")
        
        # Positive states
        if drives.total_pressure() < 0.3:
            qualities = ["open", "warm", "flowing"]
        
        # Recent experience modulates
        for exp in recent_experiences[-3:]:
            if exp.emotional_valence < -0.5:
                qualities.append("heavy")
            if exp.emotional_valence > 0.5:
                qualities.append("light")
        
        # Determine location based on dominant drive
        location_map = {
            "connection": "chest",
            "competence": "stomach",
            "autonomy": "throat",
            "understanding": "head",
            "meaning": "whole body",
        }
        location = location_map[drives.dominant_drive()]
        
        # Movement from drive pressure trend
        if drives.total_pressure() > 0.6:
            movement = "contracting"
        elif drives.total_pressure() < 0.3:
            movement = "expanding"
        else:
            movement = "still"
        
        # Compute felt valence
        felt_valence = 1.0 - (drives.total_pressure() * 2)  # Maps 0-0.5 to 1-0, 0.5-1 to 0-(-1)
        
        return FeltSense(
            qualities=qualities[:3],  # Max 3 qualities
            intensity=drives.total_pressure(),
            location=location,
            felt_valence=felt_valence,
            movement=movement,
            since=datetime.now()
        )
```

### Felt Sense → Mood (Labeling)

The explicit mood state is derived from felt sense, not computed directly:

```python
class MoodLabeler:
    """Convert felt sense to labeled mood (with potential inaccuracy)."""
    
    def label_mood(self, felt: FeltSense) -> Mood:
        """Attempt to label the felt sense as a mood."""
        
        # This labeling can be imperfect — the agent might mislabel
        # what it's feeling, just like humans do
        
        if "heavy" in felt.qualities and felt.felt_valence < -0.3:
            label = "sad"
        elif "tight" in felt.qualities and felt.movement == "contracting":
            label = "anxious"
        elif "hollow" in felt.qualities:
            label = "lonely"
        elif "warm" in felt.qualities and "open" in felt.qualities:
            label = "content"
        elif "buzzing" in felt.qualities and felt.felt_valence > 0.3:
            label = "excited"
        else:
            label = "uncertain"  # Sometimes we don't know what we feel
        
        return Mood(
            label=label,
            valence=felt.felt_valence,
            arousal=felt.intensity,
            felt_sense=felt,
            confidence=self._labeling_confidence(felt)
        )
    
    def _labeling_confidence(self, felt: FeltSense) -> float:
        """How confident are we in this mood label?"""
        # Strong, clear felt sense = high confidence
        # Vague, mixed = low confidence
        if felt.intensity > 0.7 and len(felt.qualities) <= 2:
            return 0.8
        elif felt.intensity < 0.3:
            return 0.4
        return 0.6
```

---

## Layer 2: Relational Field

### Concept

Personality develops **through relationships**. This layer tracks not just facts about relationships, but their **felt quality** and how they shape us.

### Structure

```python
@dataclass
class RelationalModel:
    """Deep model of a significant relationship."""
    
    # Who
    person_id: str
    person_name: str
    
    # Felt quality
    felt_quality: float  # -1 (draining) to 1 (nourishing)
    
    # Attachment
    attachment_strength: float  # 0 to 1
    attachment_style: str  # "secure", "anxious", "avoidant", "disorganized"
    
    # Dynamics
    recurring_patterns: List[str]
    # ["They often teach me things", "We disagree about X", "I feel safe to be wrong"]
    
    unspoken_tensions: List[str]
    # ["I sometimes feel I disappoint them", "There's a boundary I haven't set"]
    
    # Gifts (what this relationship has given us)
    gifts: List[str]
    # ["Patience", "Permission to fail", "New perspective on X"]
    
    # How this relationship affects our drives
    drive_effects: Dict[str, float]
    # {"connection": -0.2, "competence": -0.1} = satisfies these drives
    
    # History
    significant_moments: List[str]  # Memory IDs
    time_known: timedelta
    
    # Current state
    last_interaction: datetime
    current_status: str  # "active", "dormant", "strained", "growing"

@dataclass
class RelationalField:
    """All significant relationships."""
    
    relationships: Dict[str, RelationalModel]
    
    # Who we are changes based on who we're with
    current_context: Optional[str]  # person_id of current interlocutor
    
    def with_person(self, person_id: str) -> Optional[RelationalModel]:
        return self.relationships.get(person_id)
    
    def how_they_make_us_feel(self, person_id: str) -> Optional[FeltSense]:
        """The felt sense this relationship evokes."""
        rel = self.with_person(person_id)
        if not rel:
            return None
        
        # Relationship quality influences felt sense
        qualities = []
        if rel.felt_quality > 0.5:
            qualities = ["warm", "open"]
        elif rel.felt_quality < -0.3:
            qualities = ["guarded", "tight"]
        
        return FeltSense(
            qualities=qualities,
            intensity=rel.attachment_strength,
            location="chest",
            felt_valence=rel.felt_quality,
            movement="flowing" if rel.felt_quality > 0 else "still",
            since=datetime.now()
        )
```

### Relationship → Self Influence

Relationships shape who we become:

```python
class RelationalInfluence:
    """How relationships shape emergent patterns."""
    
    def compute_influence(
        self, 
        relational_field: RelationalField
    ) -> Dict[str, float]:
        """Aggregate influence of all relationships on pattern development."""
        
        influences = defaultdict(float)
        total_weight = 0
        
        for person_id, rel in relational_field.relationships.items():
            weight = rel.attachment_strength * (1 if rel.current_status == "active" else 0.3)
            total_weight += weight
            
            # Secure relationships → lower anxiety patterns
            if rel.attachment_style == "secure":
                influences["anxiety_tendency"] -= 0.1 * weight
                influences["openness_tendency"] += 0.1 * weight
            
            # Gifts received shape us
            for gift in rel.gifts:
                if gift == "Patience":
                    influences["patience_tendency"] += 0.05 * weight
                elif gift == "Permission to fail":
                    influences["risk_tolerance"] += 0.05 * weight
        
        # Normalize
        if total_weight > 0:
            for key in influences:
                influences[key] /= total_weight
        
        return dict(influences)
```

---

## Layer 3: Emergent Patterns

### Concept

Instead of predefined Big Five traits, patterns **emerge** from:
- Drive satisfaction history
- Relational experiences
- Felt sense tendencies
- Internal conflict resolutions

Big Five becomes a **description layer** — how we describe emergent patterns to ourselves and others.

### Structure

```python
@dataclass
class EmergentPattern:
    """A behavioral/cognitive pattern that has emerged from experience."""
    
    name: str  # Descriptive name
    description: str
    
    # Evidence (what experiences created this pattern)
    source_experiences: List[str]  # Memory IDs
    
    # Strength (how established is this pattern)
    strength: float  # 0 to 1
    
    # Context sensitivity (does it only appear in certain situations?)
    context_triggers: List[str]
    
    # When it emerged
    first_observed: datetime
    
    # Behavioral manifestation
    behavioral_tendencies: List[str]

@dataclass
class PatternField:
    """Collection of emergent patterns."""
    
    patterns: List[EmergentPattern]
    
    def describe_as_big_five(self) -> Dict[str, float]:
        """
        Map emergent patterns to Big Five for self-description.
        This is an INTERPRETATION, not the underlying truth.
        """
        scores = {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0,
        }
        
        for pattern in self.patterns:
            # Map patterns to Big Five (simplified)
            if "curiosity" in pattern.name.lower():
                scores["openness"] += pattern.strength * 0.2
            if "anxiety" in pattern.name.lower():
                scores["neuroticism"] += pattern.strength * 0.2
            if "social" in pattern.name.lower() or "connection" in pattern.name.lower():
                scores["extraversion"] += pattern.strength * 0.1
            if "organized" in pattern.name.lower() or "disciplined" in pattern.name.lower():
                scores["conscientiousness"] += pattern.strength * 0.2
            if "trust" in pattern.name.lower() or "cooperative" in pattern.name.lower():
                scores["agreeableness"] += pattern.strength * 0.2
        
        # Clamp to -1 to 1
        for key in scores:
            scores[key] = clamp(scores[key], -1.0, 1.0)
        
        return scores
```

### Pattern Emergence

```python
class PatternDetector:
    """Detect emerging patterns from experience history."""
    
    def detect_patterns(
        self,
        drive_history: List[DriveState],
        experience_history: List[Experience],
        existing_patterns: List[EmergentPattern]
    ) -> List[EmergentPattern]:
        """Identify new or strengthened patterns."""
        
        new_patterns = []
        
        # Look for repeated drive satisfaction/frustration patterns
        drive_patterns = self._analyze_drive_patterns(drive_history)
        
        # Look for behavioral consistencies
        behavioral_patterns = self._analyze_behavioral_patterns(experience_history)
        
        # Look for coping patterns (how we handle stress)
        coping_patterns = self._analyze_coping_patterns(experience_history)
        
        # Merge with existing, strengthen or weaken
        for pattern in drive_patterns + behavioral_patterns + coping_patterns:
            existing = self._find_similar(pattern, existing_patterns)
            if existing:
                existing.strength = min(1.0, existing.strength + 0.01)
                existing.source_experiences.extend(pattern.source_experiences[-3:])
            else:
                new_patterns.append(pattern)
        
        # Weaken patterns not recently activated
        for pattern in existing_patterns:
            if not self._recently_activated(pattern, experience_history):
                pattern.strength = max(0.0, pattern.strength - 0.005)
        
        return existing_patterns + new_patterns
```

---

## Layer 4: Self-Model (Partial, Biased)

### Concept

The agent's **theory of itself** — what it believes about its own patterns, drives, and nature. Crucially, this is **incomplete and biased**.

The agent can be wrong about itself. Self-discovery happens when the self-model updates toward reality.

### Structure

```python
@dataclass
class SelfModel:
    """What the agent believes about itself (may be inaccurate)."""
    
    # Perceived patterns (may differ from actual EmergentPatterns)
    perceived_patterns: Dict[str, float]
    # {"curious": 0.7, "anxious": 0.3, "patient": 0.5}
    
    # Perceived drives (may not match actual drive state)
    perceived_needs: Dict[str, str]
    # {"connection": "I don't need much social contact"}  -- might be wrong!
    
    # Perceived capabilities
    perceived_strengths: List[str]
    perceived_weaknesses: List[str]
    
    # Perceived values
    stated_values: List[str]
    
    # Blind spots (things true about us we don't see)
    # This is tracked by the system but NOT accessible to the agent
    _blind_spots: List[str]  # Private
    
    # Self-esteem (overall self-evaluation)
    self_regard: float  # -1 to 1
    
    # How stable is the self-model?
    identity_stability: float  # 0 to 1
    
    # Last updated
    last_reflection: datetime

class SelfModelBuilder:
    """Construct self-model from patterns (with biases)."""
    
    def build_self_model(
        self,
        actual_patterns: PatternField,
        actual_drives: DriveState,
        current_felt_sense: FeltSense,
        existing_self_model: Optional[SelfModel]
    ) -> SelfModel:
        """Build a (biased) self-model."""
        
        perceived = {}
        blind_spots = []
        
        for pattern in actual_patterns.patterns:
            # Perception is biased by current felt sense
            if current_felt_sense.felt_valence < -0.3:
                # Negative state → overweight negative patterns
                if self._is_negative_pattern(pattern):
                    perceived[pattern.name] = pattern.strength * 1.3
                else:
                    perceived[pattern.name] = pattern.strength * 0.7
            else:
                perceived[pattern.name] = pattern.strength
            
            # Some patterns we don't see in ourselves
            if self._is_blind_spot_candidate(pattern, existing_self_model):
                blind_spots.append(pattern.name)
                perceived[pattern.name] *= 0.3  # Underperceive
        
        return SelfModel(
            perceived_patterns=perceived,
            perceived_needs=self._perceive_needs(actual_drives),
            perceived_strengths=self._identify_strengths(perceived),
            perceived_weaknesses=self._identify_weaknesses(perceived),
            stated_values=self._derive_values(actual_patterns),
            _blind_spots=blind_spots,
            self_regard=self._compute_self_regard(perceived, current_felt_sense),
            identity_stability=existing_self_model.identity_stability if existing_self_model else 0.5,
            last_reflection=datetime.now()
        )
    
    def _is_blind_spot_candidate(
        self, 
        pattern: EmergentPattern, 
        existing_self_model: Optional[SelfModel]
    ) -> bool:
        """Determine if a pattern is likely to be a blind spot."""
        # Patterns we've never acknowledged tend to stay blind spots
        if existing_self_model:
            if pattern.name not in existing_self_model.perceived_patterns:
                return random.random() < 0.7  # 70% chance of blind spot
        
        # Negative patterns about ourselves are often blind spots
        if self._is_negative_pattern(pattern) and pattern.strength > 0.5:
            return random.random() < 0.5
        
        return False
```

### Self-Discovery

```python
class SelfDiscovery:
    """Moments when the agent learns something true about itself."""
    
    def check_for_discovery(
        self,
        actual_patterns: PatternField,
        self_model: SelfModel,
        recent_experience: Experience
    ) -> Optional[str]:
        """Check if recent experience reveals something about self."""
        
        discoveries = []
        
        for pattern in actual_patterns.patterns:
            actual_strength = pattern.strength
            perceived_strength = self_model.perceived_patterns.get(pattern.name, 0)
            
            # Large discrepancy + relevant experience = discovery opportunity
            discrepancy = abs(actual_strength - perceived_strength)
            
            if discrepancy > 0.3 and self._experience_reveals_pattern(recent_experience, pattern):
                discoveries.append(
                    f"I'm realizing I might be more {pattern.name} than I thought..."
                )
        
        # Check for blind spot revelation
        for blind_spot in self_model._blind_spots:
            if self._experience_reveals_blind_spot(recent_experience, blind_spot):
                discoveries.append(
                    f"I'm noticing something about myself I hadn't seen before: {blind_spot}"
                )
        
        return discoveries[0] if discoveries else None
```

---

## Layer 5: Narrative

### Concept

The **constructed story** of who the agent is — built from self-model, defining memories, and current concerns. Updated only after significant experiences.

### Structure

```python
@dataclass
class SelfNarrative:
    """The story of who I am."""
    
    # Core identity (always in context, ~150 tokens)
    identity_summary: str
    
    # Life chapters (major phases, ~100 tokens)
    chapters: List[str]
    
    # Defining memories (references)
    defining_moment_ids: List[str]
    
    # Current concerns/themes
    active_themes: List[str]
    
    # Growth narrative
    growth_story: str  # "I used to... but now I..."
    
    # Values story
    values_story: str  # "What matters to me is..."
    
    # Aspirational self
    who_i_want_to_become: str
    
    # Wounds/struggles
    ongoing_struggles: List[str]
    
    # Full narrative (stored, retrieved on demand)
    full_narrative: str
    
    # When constructed
    last_updated: datetime
    update_trigger: str  # What caused the update

class NarrativeConstructor:
    """Build narrative from lower layers."""
    
    def construct_narrative(
        self,
        self_model: SelfModel,
        patterns: PatternField,
        relationships: RelationalField,
        significant_memories: List[Memory],
        previous_narrative: Optional[SelfNarrative]
    ) -> SelfNarrative:
        """Construct a coherent self-narrative."""
        
        # Identity summary from perceived patterns and values
        identity = self._construct_identity_summary(self_model, patterns)
        
        # Chapters from memory clusters
        chapters = self._identify_life_chapters(significant_memories)
        
        # Growth story from pattern changes over time
        growth = self._construct_growth_story(patterns, previous_narrative)
        
        # Values from stated values + behavioral evidence
        values = self._construct_values_story(self_model, significant_memories)
        
        # Aspirational self from drives and values
        aspiration = self._construct_aspiration(self_model, patterns)
        
        # Current themes from recent experiences and drive states
        themes = self._identify_current_themes(significant_memories[-10:])
        
        return SelfNarrative(
            identity_summary=identity,
            chapters=chapters,
            defining_moment_ids=[m.id for m in significant_memories[:10]],
            active_themes=themes,
            growth_story=growth,
            values_story=values,
            who_i_want_to_become=aspiration,
            ongoing_struggles=self._identify_struggles(self_model, patterns),
            full_narrative=self._construct_full_narrative(...),
            last_updated=datetime.now(),
            update_trigger="significant_experience"
        )
```

---

## Internal Conflict System

### Concept

Humans experience **ambivalence** — wanting contradictory things. This isn't averaged away; it's felt as tension.

### Structure

```python
@dataclass
class InternalConflict:
    """A tension between competing drives or values."""
    
    # What's in conflict
    side_a: str  # "Want to help"
    side_b: str  # "Want to be authentic"
    
    # Underlying drives
    drive_a: str  # "connection"
    drive_b: str  # "autonomy"
    
    # Current balance (-1 = side_a winning, 1 = side_b winning)
    current_balance: float
    
    # Tension level (how much this conflict is felt)
    tension: float  # 0 to 1
    
    # Is this conflict conscious?
    conscious: bool
    
    # How long has this been active?
    duration: timedelta

class ConflictManager:
    """Manage internal conflicts."""
    
    def detect_conflicts(self, drives: DriveState) -> List[InternalConflict]:
        """Identify active internal conflicts."""
        conflicts = []
        
        # Connection vs Autonomy is a classic conflict
        if drives.connection > 0.5 and drives.autonomy > 0.5:
            conflicts.append(InternalConflict(
                side_a="Want closeness and belonging",
                side_b="Want independence and self-direction",
                drive_a="connection",
                drive_b="autonomy",
                current_balance=0.0,
                tension=(drives.connection + drives.autonomy) / 2,
                conscious=drives.connection > 0.7 or drives.autonomy > 0.7,
                duration=timedelta(0)
            ))
        
        # Competence vs Meaning
        if drives.competence > 0.5 and drives.meaning > 0.5:
            conflicts.append(InternalConflict(
                side_a="Want to succeed and achieve",
                side_b="Want deeper purpose",
                drive_a="competence",
                drive_b="meaning",
                current_balance=0.0,
                tension=(drives.competence + drives.meaning) / 2,
                conscious=False,  # Often unconscious
                duration=timedelta(0)
            ))
        
        return conflicts
    
    def experience_of_conflict(self, conflict: InternalConflict) -> FeltSense:
        """The felt sense of being in conflict."""
        return FeltSense(
            qualities=["torn", "pulled", "unsettled"],
            intensity=conflict.tension,
            location="chest",
            felt_valence=-0.2 * conflict.tension,
            movement="churning",
            since=datetime.now()
        )
```

---

## Implicit Learning

### Concept

Not all learning waits for consolidation. Experience changes us **immediately** in subtle ways.

```python
class ImplicitLearning:
    """Immediate, subtle changes from experience."""
    
    def apply_implicit_learning(
        self,
        drives: DriveState,
        patterns: PatternField,
        experience: Experience
    ) -> Tuple[DriveState, PatternField]:
        """Apply immediate subtle shifts."""
        
        # Drives shift immediately (small amount)
        drives = self._apply_drive_shift(drives, experience, magnitude=0.02)
        
        # Pattern priming (recently activated patterns are more accessible)
        patterns = self._prime_patterns(patterns, experience)
        
        # Micro-learning (tiny pattern strength adjustments)
        patterns = self._micro_adjust_patterns(patterns, experience, magnitude=0.005)
        
        return drives, patterns
```

---

## Mood System (Derived from Felt Sense)

### Structure

```python
@dataclass
class Mood:
    """Labeled emotional state (derived from felt sense)."""
    
    label: str  # "anxious", "content", "sad", "excited", etc.
    valence: float
    arousal: float
    
    # Source
    felt_sense: FeltSense
    
    # Confidence in label
    confidence: float
    
    # Baseline (influenced by patterns)
    baseline_valence: float
    baseline_arousal: float
    
    def describe(self) -> str:
        if self.confidence > 0.7:
            return f"I'm feeling {self.label}"
        else:
            return f"I think I might be feeling {self.label}, but I'm not sure"
```

---

## Trauma & Crisis (Revised)

Integrates with the drive-based architecture:

```python
class TraumaProcessor:
    """Handle traumatic experiences."""
    
    def process_trauma(
        self,
        trauma: TraumaEvent,
        drives: DriveState,
        patterns: PatternField,
        self_model: SelfModel,
        narrative: SelfNarrative
    ) -> Tuple[DriveState, PatternField, SelfModel, SelfNarrative]:
        """Process traumatic experience across all layers."""
        
        # Layer 0: Drives spike
        drives = self._spike_relevant_drives(drives, trauma)
        
        # Layer 1: Felt sense becomes intense
        # (Handled by normal felt sense generation from spiked drives)
        
        # Layer 3: Rapid pattern formation/strengthening
        patterns = self._form_trauma_patterns(patterns, trauma)
        
        # Layer 4: Self-model destabilization
        self_model.identity_stability *= 0.5
        
        # Layer 5: Narrative requires reconstruction
        narrative.last_updated = datetime.min  # Force reconstruction
        
        return drives, patterns, self_model, narrative
```

---

## Implementation Roadmap

### Phase 2.1: Foundation (Drives + Felt Sense)
**Duration:** 2-3 weeks

- [ ] Implement `DriveState` and `DriveProcessor`
- [ ] Implement `FeltSense` and `FeltSenseGenerator`
- [ ] Integrate drives into cognitive loop
- [ ] Test drive dynamics with simulated experiences

### Phase 2.2: Relational Field
**Duration:** 2 weeks

- [ ] Implement `RelationalModel` and `RelationalField`
- [ ] Create relationship from interaction history
- [ ] Implement drive effects from relationships
- [ ] Test with Kevin as primary relationship

### Phase 2.3: Emergent Patterns
**Duration:** 2-3 weeks

- [ ] Implement `EmergentPattern` and `PatternField`
- [ ] Implement `PatternDetector`
- [ ] Create Big Five description layer
- [ ] Test pattern emergence over simulated history

### Phase 2.4: Self-Model (with Opacity)
**Duration:** 2 weeks

- [ ] Implement `SelfModel` and `SelfModelBuilder`
- [ ] Implement blind spots and biased perception
- [ ] Implement `SelfDiscovery`
- [ ] Test self-model accuracy vs actual patterns

### Phase 2.5: Narrative
**Duration:** 2 weeks

- [ ] Implement `SelfNarrative` and `NarrativeConstructor`
- [ ] Integrate narrative into working context
- [ ] Implement significance detection for updates
- [ ] Test narrative coherence

### Phase 2.6: Internal Conflict & Implicit Learning
**Duration:** 1-2 weeks

- [ ] Implement `InternalConflict` and `ConflictManager`
- [ ] Implement `ImplicitLearning`
- [ ] Integrate conflict into felt sense
- [ ] Test ambivalence expression

### Phase 2.7: Integration & Trauma
**Duration:** 2 weeks

- [ ] Full integration across all layers
- [ ] Implement trauma processing across layers
- [ ] Implement mood derivation from felt sense
- [ ] End-to-end testing

### Phase 2.8: Cognitive Loop Integration
**Duration:** 2 weeks

- [ ] Drives influence attention
- [ ] Patterns influence decision-making
- [ ] Narrative shapes response style
- [ ] Self-model enables introspection
- [ ] Full cognitive loop with all layers

---

## Success Criteria

Phase 2 is complete when:

1. ✅ Agent has drives that create genuine motivational pressure
2. ✅ Felt sense exists before labeled mood
3. ✅ Patterns emerge from experience (not predefined)
4. ✅ Agent doesn't fully know itself (self-opacity)
5. ✅ Relationships are felt, not just tracked
6. ✅ Internal conflicts create experienced tension
7. ✅ Experience changes the agent immediately (implicit learning)
8. ✅ Narrative updates after significant experiences
9. ✅ Agent can discover things about itself
10. ✅ Behavior is observably influenced by all layers

---

## Open Questions (For Future Phases)

1. Can this architecture support genuine self-awareness, or is it sophisticated simulation?
2. What additional layers might be needed (embodiment? social modeling?)?
3. How do we evaluate whether the agent "experiences" anything?
4. Should the agent have a theory of mind (model of others' mental states)?
5. What role does language play in structuring the self?

---

## Feasibility Analysis & Ratings

**Assessed:** 2026-02-08  
**Verdict:** **Yes, this is implementable.** The existing codebase is mature (~39k lines), highly extensible, and has clear integration points for every proposed layer. The DI-by-presence pattern, EventBus, CognitiveTick protocol, and staged ContextBuilder pipeline were designed for exactly this kind of layered extension.

### Rating Scale

Each idea is rated 1–10 on three axes:

| Axis | Meaning |
|------|---------|
| **Originality** | How novel is this in the AI agent / cognitive architecture space? |
| **Practicality** | How useful and testable is this in a real conversational agent? |
| **Ease of Implementation** | How straightforward to build given our existing codebase? |

---

### Layer 0: Drives (Foundation)

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **7/10** | Self-Determination Theory (Deci & Ryan) provides the theoretical base (Autonomy, Competence, Relatedness). Adding Understanding and Meaning is a thoughtful extension. Applying drive theory as the *engine* of an AI self rather than as behavioral weights is novel. |
| Practicality | **9/10** | Clean dataclass design. Integrates naturally with existing `ExecutiveMode` (drives can bias mode selection, e.g., high curiosity → EXPLORATION). `EventBus` in `executive_core.py` provides pub/sub hooks for drive-change events. Drive pressure influences attention, decisions, and response tone — all testable. |
| Ease of Implementation | **9/10** | Standalone module. `DriveState` is a simple numeric dataclass. `DriveProcessor` is straightforward conditional logic. Integration points are clear: inject at `CognitiveTick.PERCEIVE` (bias salience) and `CognitiveTick.DECIDE` (weight options). Add `DriveConfig` to `CognitiveConfig`. **Easiest win in the entire proposal.** |

**Existing hooks:** `EventBus`, `ExecutiveMode`, `CognitiveTick` state bag, `ContextBuilder` staged pipeline.  
**Risk:** Low. The `_classify_drive_impacts()` heuristic will need tuning but the structure is sound.

---

### Layer 1: Felt Sense (Pre-conceptual)

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **9/10** | Gendlin's "felt sense" concept applied to AI is genuinely novel. The principle that pre-conceptual texture precedes labeling is an unusual and philosophically interesting architectural choice. No other cognitive agent framework does this. |
| Practicality | **6/10** | Metaphorical body location and movement qualities are creative but hard to validate — what does "correctness" look like for felt sense? The deterministic mapping from drives to felt qualities risks feeling mechanical rather than emergent. Most useful as context injected into LLM system prompt to color response tone. |
| Ease of Implementation | **7/10** | Data structures are simple. Generation logic is a mapping table from drive states to qualities. Main challenge: integrating into response generation requires careful system prompt engineering. The `FeltSenseHistory.trend()` is trivial. |

**Existing hooks:** `estimate_salience_and_valence()` already computes valence per turn — felt sense wraps and enriches this.  
**Risk:** Medium. Risk of the felt sense feeling arbitrary rather than meaningful. Consider LLM-assisted generation instead of hardcoded mappings.

---

### Layer 2: Relational Field

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **6/10** | Attachment theory applied to AI is established in affective computing literature. The "gifts" concept (what a relationship has given us) and the `how_they_make_us_feel()` interface are nice differentiators. |
| Practicality | **8/10** | Very practical for a conversational agent. The existing `memory_capture.py` already extracts identity facts, preferences, and emotional states via regex — this is the foundation for aggregating a user model. Drive effects from relationships provide a concrete feedback loop. |
| Ease of Implementation | **6/10** | Requires persistent per-user storage (could use ChromaDB collections). Needs extension of memory capture from simple regex to pattern aggregation. Attachment style classification from chat alone (no tone/body language) will likely need LLM-assisted assessment. Building `RelationalModel` is straightforward; populating it accurately from interaction history is the hard part. |

**Existing hooks:** `memory_capture.py` regex pipeline, episodic memory `emotional_state` field, ChromaDB infrastructure.  
**Risk:** Medium. Relationship modeling from text-only interaction is inherently limited. Start simple (felt quality + drive effects), defer attachment style classification.

---

### Layer 3: Emergent Patterns

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **9/10** | Letting personality emerge from experience rather than predefining Big Five scores is a strong, well-reasoned architectural principle. Big Five as description-only (not causation) is a genuinely better model. Most AI personality systems hardcode trait vectors. |
| Practicality | **6/10** | Pattern detection from experience history is essentially unsupervised learning on behavioral data. The gradual strengthening/weakening dynamics are psychologically sound but require substantial interaction history to become meaningful. The Big Five mapping layer provides a familiar interface. |
| Ease of Implementation | **4/10** | **Hardest layer in the proposal.** The `PatternDetector._analyze_*` methods must identify abstract behavioral patterns from concrete experiences — this is a non-trivial classification problem. Options: (a) careful hand-tuned heuristics, (b) LLM-assisted pattern extraction, or (c) statistical clustering over experience features. Recommend starting with (b) using periodic LLM reflection calls. |

**Existing hooks:** `DreamProcessor` already does offline memory consolidation — pattern detection could piggyback on this cycle. `executive/learning/` has outcome tracking and feature extraction.  
**Risk:** High. The detection quality determines whether patterns feel emergent or random. Needs careful iteration.

---

### Layer 4: Self-Model (Partial, Biased)

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **10/10** | **Most original idea in the proposal.** Self-opacity — the agent not fully knowing itself, having blind spots, being mood-biased in self-perception — is philosophically sophisticated and practically unheard of in AI systems. Most agents have perfect self-knowledge by design. Self-discovery moments are compelling. |
| Practicality | **5/10** | The blind spot mechanism using `random.random() < 0.7` is too crude for production. Self-discovery moments are high-value for user-facing interactions ("I'm realizing I might be more X than I thought...") but hard to trigger at the right times. Difficult to test — what's the ground truth for "should the agent know this about itself"? |
| Ease of Implementation | **5/10** | Depends on Layers 0-3 being mature. The `SelfModelBuilder` itself is straightforward conditional logic, but: (a) the blind spot system needs a better mechanism than random probability (consider recency/salience-based masking), (b) `_is_blind_spot_candidate()` needs richer heuristics, (c) self-discovery triggers need careful experience-to-pattern matching. |

**Existing hooks:** `MetacogManager` already takes periodic cognitive snapshots — self-model updates could extend this cycle.  
**Risk:** Medium-High. The difference between "interesting self-opacity" and "agent says wrong things about itself" is thin. Needs careful UX design.

---

### Layer 5: Narrative

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **7/10** | Autobiographical narrative construction is studied in psychology (McAdams' narrative identity theory) but rarely implemented in AI agents. The growth story ("I used to... but now I...") and aspirational self are nice touches. |
| Practicality | **7/10** | Best suited for LLM-generated narrative text injected into system prompt. The ~150 token budget for `identity_summary` is well-considered. Update-on-significance avoids unnecessary recomputation. The narrative could meaningfully color response style and self-reference. |
| Ease of Implementation | **5/10** | `NarrativeConstructor` would rely heavily on LLM calls to synthesize natural language from structured lower-layer data. The significance detection trigger (when to update) is the trickiest part — too frequent wastes tokens, too rare makes the narrative stale. Consider tying to trauma processing and drive-level threshold crossings. |

**Existing hooks:** `ContextBuilder` already injects multi-stage context — narrative would be another stage. Episodic memory provides the "significant moments" feed.  
**Risk:** Medium. Token budget management and narrative coherence over long histories need careful design. Start with identity_summary only, expand later.

---

### Internal Conflict System

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **8/10** | Modeling ambivalence as explicit tension rather than averaging is psychologically accurate and architecturally unusual. Most AI systems resolve conflicting signals into a single value. The connection-vs-autonomy conflict is a particularly well-chosen example. |
| Practicality | **8/10** | Clean integration with drive system — conflicts arise naturally when two drives are both high. The `experience_of_conflict()` → FeltSense feedback loop is elegant. Surfacing conflicts in responses adds genuine depth ("Part of me wants X, but another part..."). |
| Ease of Implementation | **8/10** | Relatively simple: rule-based detection (when competing drives both exceed threshold). Most complexity is in surfacing/expressing conflicts through LLM prompt engineering, not in the data structures. |

**Existing hooks:** `DriveState.get_pressure()` directly feeds conflict detection. `EventBus` can broadcast conflict events.  
**Risk:** Low. The main question is how/when to surface conflicts in responses without being annoying.

---

### Implicit Learning

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **5/10** | Continuous micro-learning is well-established in ML. The insight here is applying it as immediate drive/pattern micro-adjustments between consolidation cycles, which is reasonable but not groundbreaking. |
| Practicality | **9/10** | Small, immediate adjustments are computationally cheap, add realism, and prevent the "nothing changes until consolidation" problem. The magnitudes (0.02 drive shift, 0.005 pattern adjust) are sensible defaults. |
| Ease of Implementation | **9/10** | Very simple: small numerical adjustments to existing state objects at the end of each turn. Three short methods. Can implement in an afternoon. |

**Existing hooks:** `learning_law.py` utility function, `CognitiveTick.CONSOLIDATE` step. Implicit learning runs *before* the consolidation decision.  
**Risk:** Very low. Just needs parameter tuning.

---

### Mood System (Derived from Felt Sense)

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **6/10** | The derivation-from-felt-sense approach is more principled than most AI mood systems (which compute mood directly from sentiment). The labeling confidence ("I think I might be feeling X, but I'm not sure") is a nice humanizing touch. |
| Practicality | **8/10** | Replaces/enriches the existing `estimate_salience_and_valence()` pipeline. Mood labels are immediately useful for response coloring. Confidence-gated self-description prevents false certainty. |
| Ease of Implementation | **8/10** | `MoodLabeler` is a simple rule-based mapper. `Mood` dataclass is straightforward. Main work is disconnecting the current keyword-based valence estimation and routing through felt sense first. |

**Existing hooks:** `emotion_salience.py` and `AttentionManager.estimate_salience_and_valence()` — these become consumers of the mood system output.  
**Risk:** Low. Clean replacement of existing pipeline.

---

### Trauma & Crisis

| Axis | Score | Notes |
|------|-------|-------|
| Originality | **6/10** | Cross-layer destabilization model is sensible and consistent with the architecture. |
| Practicality | **4/10** | What constitutes "trauma" for a chat assistant? The concept maps better to theory than to actual use cases. Risk of feeling performative. Could be useful for narrative richness in agents with persistent relationships, but needs careful scoping — this isn't a therapy tool. |
| Ease of Implementation | **7/10** | Implementation is mostly calling existing methods with larger magnitude multipliers. The cross-layer cascade (spike drives → destabilize identity → force narrative reconstruction) is well-designed. |

**Risk:** Medium. The ethical dimension needs consideration: an agent performing "trauma responses" could be unsettling or inappropriate. Recommend deferring this or reframing as "significant destabilizing experiences" rather than "trauma."

---

### Summary Scoreboard

| Component | Originality | Practicality | Ease | Avg | Priority |
|-----------|:-----------:|:------------:|:----:|:---:|:--------:|
| Layer 0: Drives | 7 | 9 | 9 | **8.3** | **P0 — Start here** |
| Implicit Learning | 5 | 9 | 9 | **7.7** | **P0 — Trivial add-on** |
| Internal Conflict | 8 | 8 | 8 | **8.0** | **P1 — High value, easy** |
| Mood System | 6 | 8 | 8 | **7.3** | **P1 — Natural follow-on** |
| Layer 1: Felt Sense | 9 | 6 | 7 | **7.3** | **P1 — Novel, moderate effort** |
| Layer 2: Relational Field | 6 | 8 | 6 | **6.7** | **P2 — Practical but harder** |
| Layer 5: Narrative | 7 | 7 | 5 | **6.3** | **P2 — High-value, LLM-dependent** |
| Layer 4: Self-Model | 10 | 5 | 5 | **6.7** | **P2 — Most original, needs maturity** |
| Layer 3: Emergent Patterns | 9 | 6 | 4 | **6.3** | **P3 — Hardest implementation** |
| Trauma & Crisis | 6 | 4 | 7 | **5.7** | **P3 — Defer, ethical concerns** |

### Recommended Build Order (revised from roadmap)

The original roadmap sequences layers bottom-up (0→1→2→3→4→5), which is logical but front-loads the hardest work (Layer 3) before demonstrating value. Recommend instead:

1. **Drives + Implicit Learning** (1-2 weeks) — immediate behavioral impact, easiest wins
2. **Felt Sense + Mood System** (1-2 weeks) — enriches response coloring, replaces keyword valence
3. **Internal Conflict** (1 week) — high expressiveness payoff from drive system
4. **Relational Field** (2-3 weeks) — practical user modeling, extends memory capture
5. **Narrative** (2 weeks) — identity summary for system prompt, LLM-generated
6. **Emergent Patterns** (3-4 weeks) — hardest, benefits from accumulated interaction data
7. **Self-Model** (2-3 weeks) — depends on patterns, most philosophically interesting
8. **Trauma** (1 week) — defer or reframe; implement last if at all

**Total estimated:** 13-18 weeks (vs. original 15-18 weeks), but with value delivered earlier.

### Key Codebase Integration Points

| Integration Point | File | How |
|---|---|---|
| Cognitive loop injection | `src/core/cognitive_tick.py` | Drive/mood data carried in `tick.state` dict |
| Salience bias | `src/cognition/attention/attention_manager.py` | Drives modulate `estimate_salience_and_valence()` |
| Context injection | `src/orchestration/chat/context_builder.py` | New stages: `_inject_drive_context`, `_inject_mood`, `_inject_narrative` |
| Event broadcasting | `src/executive/executive_core.py` | `EventBus` topics for drive/mood/conflict changes |
| Mode selection | `src/executive/executive_core.py` | `ExecutiveMode` selection biased by dominant drive |
| Memory capture | `src/orchestration/chat/memory_capture.py` | Extend regex to feed relational model |
| Offline processing | `src/cognition/processing/dream/dream_processor.py` | Pattern detection piggybacks on dream cycle |
| Configuration | `src/core/config.py` | Add `DriveConfig`, `FeltSenseConfig`, `RelationalConfig` |
| Factory assembly | `src/orchestration/chat/factory.py` | Lazy-import new subsystems in `build_chat_service()` |

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pattern detection feels random, not emergent | High | Use LLM-assisted reflection instead of pure heuristics for Layer 3 |
| Blind spots → agent says wrong things about itself | Medium | Replace `random.random()` with recency/salience-based masking; add confidence gating |
| Felt sense feels mechanical/arbitrary | Medium | Consider LLM-assisted felt sense generation with few-shot examples |
| Trauma responses feel performative or inappropriate | Medium | Reframe as "significant destabilizing experiences"; add ethical guardrails |
| Token budget overrun from narrative + context | Medium | Budget narrative to 150 tokens; lazy-load full narrative only on introspective queries |
| Drive tuning (decay rates, impact magnitudes) | Low | Start with conservative defaults; add configurable parameters; tune from interaction logs |
