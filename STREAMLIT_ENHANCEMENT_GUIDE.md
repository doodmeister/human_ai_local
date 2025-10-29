# Streamlit UI Enhancement Guide - George Interface

**Note:** The legacy dashboard described here has been superseded by the minimal chat interface in `scripts/george_streamlit_chat.py`. Use this guide only when working on the historical production view.  
**Goal:** Enhance the Streamlit interface to visualize advanced cognitive features  
**Prerequisites:** Backend integration completed (see INTEGRATION_GUIDE.md)  
**Time Required:** 4-8 hours  
**File:** `scripts/george_streamlit_production.py`

---

## Table of Contents

1. [Enhanced Memory Visualization](#1-enhanced-memory-visualization)
2. [Performance Dashboard](#2-performance-dashboard)
3. [Consolidation Metrics Display](#3-consolidation-metrics-display)
4. [Metacognitive Awareness Panel](#4-metacognitive-awareness-panel)
5. [Session Management Improvements](#5-session-management-improvements)
6. [Prospective Memory UI](#6-prospective-memory-ui)
7. [Provenance Tracking Visualization](#7-provenance-tracking-visualization)
8. [Cognitive State Timeline](#8-cognitive-state-timeline)
9. [Memory Graph Visualization](#9-memory-graph-visualization)
10. [Executive Dashboard Integration](#10-executive-dashboard-integration)

---

## 1. Enhanced Memory Visualization

### 1.1 Memory Capture Display with Reinforcement

**Location:** After line 455 in `render_enhanced_chat()`

**Add this enhanced display:**

```python
def render_captured_memories_detail(captured_list):
    """Display captured memories with rich metadata"""
    if not captured_list:
        return
    
    with st.expander("🧩 Captured Memories This Turn", expanded=True):
        for idx, cm in enumerate(captured_list):
            # Determine icon based on memory characteristics
            if cm.get('contradiction'):
                icon = "⚠️"
                color = "#ff6b6b"
            elif cm.get('reinforced'):
                icon = "✨"
                color = "#4ecdc4"
            elif cm.get('frequency', 1) >= 3:
                icon = "🔥"
                color = "#ffe66d"
            else:
                icon = "📝"
                color = "#95e1d3"
            
            freq = cm.get('frequency', 1)
            mem_type = cm.get('memory_type', 'unknown')
            content = cm.get('content', '')
            
            # Create expandable memory card
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: {color}20; padding: 10px; border-radius: 5px; border-left: 4px solid {color};'>
                        <b>{icon} [{mem_type}]</b> {content}<br>
                        <small>Frequency: {freq} | First seen: {cm.get('first_seen_ts', 'unknown')[:19]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("Details", key=f"mem_detail_{idx}"):
                        st.session_state[f'show_mem_{idx}'] = not st.session_state.get(f'show_mem_{idx}', False)
                
                # Show details if expanded
                if st.session_state.get(f'show_mem_{idx}', False):
                    with st.container():
                        st.json({
                            "Subject": cm.get('subject'),
                            "Predicate": cm.get('predicate'),
                            "Object": cm.get('object'),
                            "Raw Text": cm.get('raw_text'),
                            "Metadata": {
                                "frequency": freq,
                                "first_seen": cm.get('first_seen_ts'),
                                "last_seen": cm.get('last_seen_ts'),
                                "reinforced": cm.get('reinforced', False),
                                "contradiction": cm.get('contradiction', False),
                            }
                        })
                        
                        # Show contradiction details
                        if cm.get('contradiction'):
                            st.error("⚠️ Contradiction Detected!")
                            st.write("Previous values:")
                            for prior in cm.get('contradicted_prior', []):
                                st.write(f"  • {prior}")
                        
                        # Show reinforcement milestones
                        if cm.get('reinforced'):
                            st.success(f"✨ Memory reinforced at frequency {freq}!")
                            if freq >= 5:
                                st.info("🎯 High-frequency memory - promoted to semantic storage")

# Call in render_enhanced_chat() after response received
if "captured_memories" in response:
    render_captured_memories_detail(response.get("captured_memories", []))
```

### 1.2 Memory Type Filter and Search

**Location:** In sidebar, after line 271

**Add enhanced filtering:**

```python
def render_advanced_memory_filters():
    """Advanced memory filtering and search"""
    with st.sidebar:
        st.markdown("### 🔍 Memory Search & Filters")
        
        # Memory type filter with counts
        mem_types = {}
        sid = st.session_state.get('active_session_id', 'default')
        captured_list = st.session_state.per_session_captured_memory_list.get(sid, [])
        
        for cm in captured_list:
            mtype = cm.get('memory_type', 'unknown')
            mem_types[mtype] = mem_types.get(mtype, 0) + 1
        
        # Display type buttons with counts
        st.write("**Filter by Type:**")
        filter_types = st.multiselect(
            "Types",
            options=list(mem_types.keys()),
            default=list(mem_types.keys()),
            format_func=lambda x: f"{x} ({mem_types[x]})",
            key='mem_type_filter'
        )
        
        # Frequency filter
        min_freq = st.slider("Minimum Frequency", 1, 10, 1, key='min_freq_filter')
        
        # Time range filter
        time_filter = st.selectbox(
            "Time Range",
            ["All Time", "Last Hour", "Last 24 Hours", "Last Week"],
            key='time_range_filter'
        )
        
        # Search box with advanced options
        search_query = st.text_input(
            "Search Content",
            placeholder="Search in memories...",
            key='mem_search_query'
        )
        
        search_in = st.multiselect(
            "Search In",
            ["content", "subject", "object", "raw_text"],
            default=["content"],
            key='search_fields'
        )
        
        # Special filters
        col1, col2 = st.columns(2)
        with col1:
            show_reinforced = st.checkbox("Reinforced Only", key='filter_reinforced')
        with col2:
            show_contradictions = st.checkbox("Contradictions Only", key='filter_contradictions')
        
        # Apply filters button
        if st.button("Apply Filters", type="primary", key='apply_filters_btn'):
            st.session_state['filters_applied'] = True
            st.rerun()

# Add to main render function
render_advanced_memory_filters()
```

### 1.3 Memory Statistics Panel

**Location:** New function, call in sidebar

```python
def render_memory_statistics():
    """Display comprehensive memory statistics"""
    with st.sidebar:
        st.markdown("### 📊 Memory Statistics")
        
        sid = st.session_state.get('active_session_id', 'default')
        captured_list = st.session_state.per_session_captured_memory_list.get(sid, [])
        
        if not captured_list:
            st.caption("No memories captured yet")
            return
        
        # Calculate statistics
        total = len(captured_list)
        types = {}
        freq_dist = {}
        contradictions = 0
        reinforced = 0
        
        for cm in captured_list:
            # Type distribution
            mtype = cm.get('memory_type', 'unknown')
            types[mtype] = types.get(mtype, 0) + 1
            
            # Frequency distribution
            freq = cm.get('frequency', 1)
            freq_dist[freq] = freq_dist.get(freq, 0) + 1
            
            # Special counts
            if cm.get('contradiction'):
                contradictions += 1
            if cm.get('reinforced'):
                reinforced += 1
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Reinforced", reinforced)
        with col3:
            st.metric("Conflicts", contradictions, 
                     delta="⚠️" if contradictions > 0 else None)
        
        # Type distribution chart
        if types:
            st.write("**Type Distribution:**")
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(types.keys()),
                    values=list(types.values()),
                    hole=0.3,
                    marker_colors=['#4ecdc4', '#ffe66d', '#ff6b6b', '#95e1d3', '#a8dadc']
                )
            ])
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        # Frequency distribution
        if freq_dist:
            st.write("**Frequency Distribution:**")
            freq_labels = sorted(freq_dist.keys())
            freq_values = [freq_dist[k] for k in freq_labels]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=freq_labels,
                    y=freq_values,
                    marker_color='#4ecdc4'
                )
            ])
            fig.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Frequency",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

# Add to sidebar rendering
render_memory_statistics()
```

---

## 2. Performance Dashboard

### 2.1 Real-Time Performance Metrics

**Location:** New tab or expander in main area

```python
def render_performance_dashboard():
    """Display comprehensive performance metrics"""
    st.markdown("### ⚡ Performance Dashboard")
    
    # Fetch performance data
    perf = GeorgeAPI.get("/chat/performance", timeout=10)
    
    if "error" in perf:
        st.error("Unable to fetch performance metrics")
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latency = perf.get('latency_p95_ms', 0)
        target = perf.get('target_p95_ms', 2000)
        delta = latency - target if target else None
        st.metric(
            "P95 Latency",
            f"{latency:.0f}ms",
            delta=f"{delta:+.0f}ms" if delta else None,
            delta_color="inverse"
        )
    
    with col2:
        degraded = perf.get('performance_degraded', False)
        status_icon = "⚠️" if degraded else "✅"
        status_text = "Degraded" if degraded else "Healthy"
        st.metric("Status", f"{status_icon} {status_text}")
    
    with col3:
        throughput = perf.get('chat_turns_per_sec', 0)
        st.metric("Throughput", f"{throughput:.2f} turns/s")
    
    with col4:
        ema_latency = perf.get('ema_turn_latency_ms', 0)
        st.metric("Avg Latency (EMA)", f"{ema_latency:.0f}ms")
    
    # Latency chart
    st.write("**Latency Trend:**")
    
    # Get or initialize latency history
    if 'latency_history' not in st.session_state:
        st.session_state.latency_history = []
    
    # Add current latency
    import time
    st.session_state.latency_history.append({
        'timestamp': time.time(),
        'latency': latency,
        'target': target
    })
    
    # Keep last 50 data points
    if len(st.session_state.latency_history) > 50:
        st.session_state.latency_history = st.session_state.latency_history[-50:]
    
    # Create chart
    if st.session_state.latency_history:
        timestamps = [h['timestamp'] for h in st.session_state.latency_history]
        latencies = [h['latency'] for h in st.session_state.latency_history]
        targets = [h['target'] for h in st.session_state.latency_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(latencies))),
            y=latencies,
            mode='lines+markers',
            name='P95 Latency',
            line=dict(color='#4ecdc4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(targets))),
            y=targets,
            mode='lines',
            name='Target',
            line=dict(color='#ff6b6b', width=1, dash='dash')
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Turn",
            yaxis_title="Latency (ms)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance warnings
    if degraded:
        st.warning("⚠️ **Performance Degraded** - System is under heavy load or experiencing delays")
        st.info("Adaptive measures may be in effect (reduced context, tighter consolidation)")

# Add to main dashboard
with st.expander("⚡ Performance Metrics", expanded=False):
    render_performance_dashboard()
    if st.button("Refresh Metrics", key='refresh_perf'):
        st.rerun()
```

### 2.2 Consolidation Performance

**Location:** Adjacent to performance dashboard

```python
def render_consolidation_dashboard():
    """Display consolidation metrics and performance"""
    st.markdown("### 🔄 Consolidation Dashboard")
    
    status = GeorgeAPI.get("/chat/consolidation/status", timeout=10)
    
    if not status.get("active"):
        st.info("Consolidation system not active")
        return
    
    cons = status.get("consolidation", {})
    counters = cons.get("counters", {})
    
    # Main consolidation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stm_stored = counters.get("stm_store_total", 0)
        st.metric("STM Stored", stm_stored)
    
    with col2:
        ltm_promoted = counters.get("ltm_promotions_total", 0)
        st.metric("LTM Promoted", ltm_promoted)
    
    with col3:
        if stm_stored > 0:
            selectivity = (ltm_promoted / stm_stored) * 100
        else:
            selectivity = 0
        st.metric("Selectivity", f"{selectivity:.1f}%")
    
    with col4:
        promo_age = cons.get("promotion_age_p95_seconds", 0)
        st.metric("Promotion Age (P95)", f"{promo_age:.1f}s")
    
    # Promotion age distribution
    recent_ages = cons.get("recent_promotion_age_seconds", {})
    if recent_ages and recent_ages.get("values"):
        st.write("**Recent Promotion Ages:**")
        
        ages = recent_ages.get("values", [])
        avg_age = recent_ages.get("avg", 0)
        count = recent_ages.get("count", 0)
        
        fig = go.Figure(data=[
            go.Histogram(
                x=ages,
                nbinsx=20,
                marker_color='#4ecdc4'
            )
        ])
        fig.add_vline(
            x=avg_age,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {avg_age:.1f}s"
        )
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Age at Promotion (seconds)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Based on {count} recent promotions")
    
    # Consolidation alerts
    if cons.get("promotion_age_alert"):
        threshold = cons.get("promotion_age_alert_threshold", 0)
        st.warning(f"⚠️ Promotion age exceeds threshold ({threshold}s) - memories waiting too long in STM")
    
    # Recent consolidation events
    recent_events = status.get("recent_events", [])
    if recent_events:
        with st.expander("Recent Consolidation Events"):
            for event in recent_events[-10:]:
                timestamp = event.get("timestamp", "")
                status_val = event.get("status", "")
                detail = event.get("detail", "")
                st.write(f"**{timestamp[:19]}** - {status_val}: {detail}")

# Add to dashboard
with st.expander("🔄 Consolidation Metrics", expanded=False):
    render_consolidation_dashboard()
```

---

## 3. Metacognitive Awareness Panel

### 3.1 Metacognitive Status Display

**Location:** New expander in main area

```python
def render_metacog_dashboard():
    """Display metacognitive awareness and self-monitoring"""
    st.markdown("### 🧠 Metacognitive Awareness")
    
    metacog = GeorgeAPI.get("/chat/metacog/status", timeout=10)
    
    if not metacog.get("available"):
        st.info("Metacognitive snapshots not available yet")
        return
    
    snapshot = metacog.get("snapshot", {})
    history = metacog.get("history_tail", [])
    
    # Current snapshot metrics
    st.write("**Current Snapshot:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        turn_interval = snapshot.get("turn_interval", 0)
        st.metric("Snapshot Interval", f"{turn_interval} turns")
    
    with col2:
        stm_util = snapshot.get("stm_utilization", 0)
        util_pct = stm_util * 100
        st.metric(
            "STM Utilization",
            f"{util_pct:.0f}%",
            delta="⚠️" if util_pct >= 85 else None
        )
    
    with col3:
        perf_deg = snapshot.get("performance_degraded", False)
        st.metric("Performance", "⚠️ Degraded" if perf_deg else "✅ Normal")
    
    with col4:
        adapt_applied = snapshot.get("adaptive_retrieval_applied", False)
        st.metric("Adaptive Mode", "🔄 Active" if adapt_applied else "⏸️ Idle")
    
    # Adaptive behavior indicators
    if snapshot.get("adaptive_retrieval_applied") or snapshot.get("adaptive_consolidation_applied"):
        st.info("🔄 **Adaptive behaviors active:**")
        if snapshot.get("adaptive_retrieval_applied"):
            original_limit = snapshot.get("original_retrieval_limit", 0)
            reduced_limit = snapshot.get("reduced_retrieval_limit", 0)
            st.write(f"  • Retrieval limit reduced: {original_limit} → {reduced_limit}")
        if snapshot.get("adaptive_consolidation_applied"):
            st.write(f"  • Consolidation thresholds tightened")
    
    # Metacognitive history chart
    if history:
        st.write("**Metacognitive History:**")
        
        turns = [h.get("turn_number", 0) for h in history]
        stm_utils = [h.get("stm_utilization", 0) * 100 for h in history]
        perf_flags = [1 if h.get("performance_degraded") else 0 for h in history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=turns,
            y=stm_utils,
            mode='lines+markers',
            name='STM Utilization %',
            yaxis='y',
            line=dict(color='#4ecdc4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=turns,
            y=perf_flags,
            mode='markers',
            name='Performance Degraded',
            yaxis='y2',
            marker=dict(color='#ff6b6b', size=10, symbol='square')
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Turn Number",
            yaxis=dict(title="STM Utilization (%)", side="left"),
            yaxis2=dict(title="Degraded Flag", side="right", overlaying="y", range=[-0.1, 1.1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Advisory items
    advisory_count = snapshot.get("advisory_items_injected", 0)
    if advisory_count > 0:
        st.success(f"✨ {advisory_count} metacognitive advisory items provided to assist reasoning")

# Add to dashboard
with st.expander("🧠 Metacognitive Awareness", expanded=False):
    render_metacog_dashboard()
    if st.button("Refresh Metacog Status", key='refresh_metacog'):
        st.rerun()
```

---

## 4. Session Management Improvements

### 4.1 Enhanced Session Panel

**Location:** Replace current session management (lines 354-389)

```python
def render_enhanced_session_management():
    """Enhanced session management with metadata and controls"""
    st.markdown("### 🗂️ Session Management")
    
    # Initialize session data structure
    if 'sessions' not in st.session_state:
        st.session_state.sessions = [{
            'id': 'default',
            'label': 'Default Session',
            'created': datetime.now(),
            'message_count': 0,
            'last_activity': datetime.now()
        }]
    
    if 'active_session_id' not in st.session_state:
        st.session_state.active_session_id = 'default'
    
    # Session selector with metadata
    sessions = st.session_state.sessions
    active_id = st.session_state.active_session_id
    
    # Find active session
    active_idx = 0
    for i, s in enumerate(sessions):
        if s['id'] == active_id:
            active_idx = i
            break
    
    # Session display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create rich session labels
        session_labels = []
        for s in sessions:
            msg_count = s.get('message_count', 0)
            last_act = s.get('last_activity', datetime.now())
            time_ago = (datetime.now() - last_act).seconds // 60
            label = f"{s['label']} ({msg_count} msgs, {time_ago}m ago)"
            session_labels.append(label)
        
        chosen_label = st.selectbox(
            "Active Session",
            session_labels,
            index=active_idx,
            key='session_selector'
        )
        
        # Map back to session ID
        chosen_idx = session_labels.index(chosen_label)
        st.session_state.active_session_id = sessions[chosen_idx]['id']
    
    with col2:
        if st.button("📊 Stats", key='session_stats_btn'):
            st.session_state['show_session_stats'] = True
    
    # Session creation
    with st.expander("➕ Create New Session"):
        new_label = st.text_input("Session Name", key='new_session_name')
        col1, col2 = st.columns(2)
        
        with col1:
            session_type = st.selectbox(
                "Type",
                ["General", "Project", "Research", "Personal"],
                key='session_type'
            )
        
        with col2:
            if st.button("Create", type="primary", key='create_session'):
                if new_label:
                    import uuid as _uuid
                    new_id = _uuid.uuid4().hex[:12]
                    st.session_state.sessions.append({
                        'id': new_id,
                        'label': new_label,
                        'type': session_type,
                        'created': datetime.now(),
                        'message_count': 0,
                        'last_activity': datetime.now(),
                        'tags': []
                    })
                    st.session_state.per_session_history[new_id] = []
                    st.session_state.active_session_id = new_id
                    st.success(f"Created session: {new_label}")
                    st.rerun()
                else:
                    st.error("Please enter a session name")
    
    # Session management actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏷️ Tag Session", key='tag_session_btn'):
            st.session_state['show_tag_dialog'] = True
    
    with col2:
        if st.button("📤 Export", key='export_session_btn'):
            st.session_state['show_export_dialog'] = True
    
    with col3:
        if len(sessions) > 1:  # Don't allow deleting last session
            if st.button("🗑️ Delete", key='delete_session_btn', type="secondary"):
                if st.session_state.active_session_id != 'default':
                    # Remove session
                    sessions = [s for s in st.session_state.sessions 
                               if s['id'] != st.session_state.active_session_id]
                    st.session_state.sessions = sessions
                    st.session_state.active_session_id = sessions[0]['id']
                    st.success("Session deleted")
                    st.rerun()
                else:
                    st.error("Cannot delete default session")
    
    # Session stats modal
    if st.session_state.get('show_session_stats'):
        show_session_statistics()

def show_session_statistics():
    """Display detailed session statistics"""
    st.markdown("#### 📊 Session Statistics")
    
    active_id = st.session_state.active_session_id
    session = next((s for s in st.session_state.sessions if s['id'] == active_id), None)
    
    if not session:
        return
    
    history = st.session_state.per_session_history.get(active_id, [])
    captured = st.session_state.per_session_captured_memory_list.get(active_id, [])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Messages", len(history))
    with col2:
        st.metric("Memories", len(captured))
    with col3:
        duration = (datetime.now() - session.get('created', datetime.now())).seconds // 60
        st.metric("Duration", f"{duration}m")
    with col4:
        st.metric("Type", session.get('type', 'General'))
    
    # Message timeline
    if history:
        st.write("**Message Timeline:**")
        user_msgs = [h for h in history if h.get('role') == 'user']
        timestamps = [h.get('timestamp', datetime.now()) for h in user_msgs]
        
        # Create timeline chart
        fig = go.Figure(data=[
            go.Scatter(
                x=timestamps,
                y=list(range(len(timestamps))),
                mode='lines+markers',
                marker=dict(size=10, color='#4ecdc4'),
                line=dict(color='#4ecdc4', width=2)
            )
        ])
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Time",
            yaxis_title="Message Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Close", key='close_stats'):
        st.session_state['show_session_stats'] = False
        st.rerun()

# Replace existing session management call
render_enhanced_session_management()
```

---

## 5. Prospective Memory UI

### 5.1 Reminders Panel

**Location:** New sidebar section or main tab

```python
def render_prospective_memory_panel():
    """Display and manage prospective memories (reminders)"""
    st.markdown("### ⏰ Reminders & Intentions")
    
    # Fetch reminders
    reminders_resp = GeorgeAPI.get("/agent/reminders", timeout=10)
    
    if "error" in reminders_resp:
        st.error("Unable to fetch reminders")
        return
    
    reminders = reminders_resp.get("reminders", [])
    
    # Separate due and pending
    due_reminders = [r for r in reminders if r.get('due_in_seconds', 999) <= 0 and not r.get('triggered_ts')]
    pending_reminders = [r for r in reminders if r.get('due_in_seconds', 999) > 0]
    triggered_reminders = [r for r in reminders if r.get('triggered_ts')]
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Due Now", len(due_reminders), delta="🔔" if due_reminders else None)
    with col2:
        st.metric("Pending", len(pending_reminders))
    with col3:
        st.metric("Completed", len(triggered_reminders))
    
    # Due reminders (urgent)
    if due_reminders:
        st.error("🔔 **Reminders Due Now:**")
        for reminder in due_reminders:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{reminder.get('content')}**")
                    created = reminder.get('created_ts', 'unknown')[:19]
                    st.caption(f"Created: {created}")
                with col2:
                    if st.button("✅ Done", key=f"done_{reminder.get('id')}"):
                        # Mark as triggered
                        st.success("Marked as complete")
    
    # Pending reminders
    with st.expander(f"📅 Pending Reminders ({len(pending_reminders)})", expanded=True):
        if not pending_reminders:
            st.caption("No pending reminders")
        else:
            for reminder in pending_reminders:
                due_in = reminder.get('due_in_seconds', 0)
                
                # Format time remaining
                if due_in < 60:
                    time_str = f"{int(due_in)}s"
                elif due_in < 3600:
                    time_str = f"{int(due_in/60)}m"
                else:
                    time_str = f"{int(due_in/3600)}h {int((due_in%3600)/60)}m"
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{reminder.get('content')}**")
                with col2:
                    st.write(f"🕐 {time_str}")
                with col3:
                    if st.button("❌", key=f"cancel_{reminder.get('id')}"):
                        # Delete reminder
                        st.warning("Cancelled")
    
    # Create new reminder
    with st.expander("➕ Create Reminder"):
        reminder_text = st.text_input("What to remember?", key='new_reminder_text')
        
        col1, col2 = st.columns(2)
        with col1:
            time_value = st.number_input("Time", min_value=1, value=5, key='reminder_time_val')
        with col2:
            time_unit = st.selectbox("Unit", ["minutes", "hours", "days"], key='reminder_time_unit')
        
        # Convert to seconds
        unit_multiplier = {
            "minutes": 60,
            "hours": 3600,
            "days": 86400
        }
        due_in_seconds = time_value * unit_multiplier[time_unit]
        
        if st.button("Create Reminder", type="primary", key='create_reminder_btn'):
            if reminder_text:
                result = GeorgeAPI.post("/agent/reminders", {
                    "content": reminder_text,
                    "due_in_seconds": due_in_seconds
                })
                if "error" not in result:
                    st.success(f"✅ Reminder set for {time_value} {time_unit}")
                    st.rerun()
                else:
                    st.error(f"Failed to create reminder: {result.get('error')}")
            else:
                st.error("Please enter reminder text")
    
    # Completed reminders
    if triggered_reminders:
        with st.expander(f"✅ Completed ({len(triggered_reminders)})"):
            for reminder in triggered_reminders[-10:]:  # Show last 10
                triggered = reminder.get('triggered_ts', 'unknown')[:19]
                st.write(f"~~{reminder.get('content')}~~ (triggered: {triggered})")
            
            if st.button("🗑️ Clear Completed", key='clear_completed_btn'):
                result = GeorgeAPI.delete("/agent/reminders/triggered")
                if "error" not in result:
                    st.success("Cleared completed reminders")
                    st.rerun()

# Add to main area or sidebar
with st.expander("⏰ Reminders & Intentions", expanded=False):
    render_prospective_memory_panel()
```

---

## 6. Provenance Tracking Visualization

### 6.1 Memory Provenance Display

**Location:** In message display, show provenance for retrieved memories

```python
def render_memory_provenance(memory_context):
    """Display detailed provenance for retrieved memories"""
    if not memory_context:
        return
    
    st.markdown("#### 🔍 Memory Provenance")
    
    for idx, mem in enumerate(memory_context):
        with st.expander(f"Memory {idx+1}: {mem.get('content', '')[:50]}...", expanded=False):
            # Basic info
            st.write(f"**Source:** {mem.get('source_system', 'unknown')}")
            st.write(f"**ID:** {mem.get('source_id', 'unknown')}")
            st.write(f"**Reason:** {mem.get('reason', 'unknown')}")
            
            # Composite score
            composite = mem.get('composite', mem.get('similarity', 0))
            st.metric("Composite Score", f"{composite:.3f}")
            
            # Factor breakdown
            if 'provenance_details' in mem or 'factors' in mem:
                st.write("**Scoring Factors:**")
                
                factors = mem.get('factors', mem.get('provenance_details', {}).get('factors', []))
                if factors:
                    # Create dataframe for factors
                    import pandas as pd
                    df = pd.DataFrame(factors)
                    
                    # Display as table
                    st.dataframe(df[['factor', 'weight', 'value', 'contribution']], 
                                hide_index=True, use_container_width=True)
                    
                    # Factor contribution chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f['factor'] for f in factors],
                            y=[f['contribution'] for f in factors],
                            marker_color='#4ecdc4',
                            text=[f"{f['contribution']:.3f}" for f in factors],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis_title="Factor",
                        yaxis_title="Contribution",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Promotion flag
            if mem.get('promoted_from_stm'):
                st.success("✨ Promoted from STM to LTM")
            
            # Full metadata
            with st.expander("View Full Metadata"):
                st.json(mem)

# Add after displaying response in chat
if "memory_context" in response:
    with st.expander("🔍 Memory Provenance", expanded=False):
        render_memory_provenance(response.get("memory_context", []))
```

---

## 7. Cognitive State Timeline

### 7.1 Real-Time Cognitive State Tracking

**Location:** New visualization panel

```python
def render_cognitive_timeline():
    """Display cognitive state changes over time"""
    st.markdown("### 📈 Cognitive State Timeline")
    
    # Initialize timeline history
    if 'cognitive_timeline' not in st.session_state:
        st.session_state.cognitive_timeline = []
    
    # Fetch current state
    status = GeorgeAPI.get("/api/agent/status", timeout=15)
    
    if "error" not in status:
        cog_status = status.get("cognitive_status", {})
        attention = cog_status.get("attention_status", {})
        memory = cog_status.get("memory_status", {})
        
        # Record state
        state_record = {
            'timestamp': time.time(),
            'cognitive_load': attention.get('cognitive_load', 0),
            'fatigue': attention.get('current_fatigue', 0),
            'stm_count': memory.get('stm', {}).get('vector_db_count', 0),
            'attention_capacity': attention.get('available_capacity', 0),
            'mode': cog_status.get('cognitive_mode', 'UNKNOWN')
        }
        
        st.session_state.cognitive_timeline.append(state_record)
        
        # Keep last 100 records
        if len(st.session_state.cognitive_timeline) > 100:
            st.session_state.cognitive_timeline = st.session_state.cognitive_timeline[-100:]
    
    # Plot timeline
    timeline = st.session_state.cognitive_timeline
    if len(timeline) > 1:
        indices = list(range(len(timeline)))
        
        # Create multi-line chart
        fig = go.Figure()
        
        # Cognitive load
        fig.add_trace(go.Scatter(
            x=indices,
            y=[t['cognitive_load'] * 100 for t in timeline],
            mode='lines',
            name='Cognitive Load %',
            line=dict(color='#4ecdc4', width=2)
        ))
        
        # Fatigue
        fig.add_trace(go.Scatter(
            x=indices,
            y=[t['fatigue'] * 100 for t in timeline],
            mode='lines',
            name='Fatigue %',
            line=dict(color='#ff6b6b', width=2)
        ))
        
        # Attention capacity
        fig.add_trace(go.Scatter(
            x=indices,
            y=[t['attention_capacity'] * 100 for t in timeline],
            mode='lines',
            name='Attention Capacity %',
            line=dict(color='#ffe66d', width=2)
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Time",
            yaxis_title="Percentage",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # STM growth chart
        st.write("**STM Growth:**")
        fig2 = go.Figure(data=[
            go.Scatter(
                x=indices,
                y=[t['stm_count'] for t in timeline],
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='#95e1d3', width=2),
                marker=dict(size=4)
            )
        ])
        fig2.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Time",
            yaxis_title="Memories in STM",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Current state summary
        if timeline:
            latest = timeline[-1]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Load", f"{latest['cognitive_load']*100:.0f}%")
            with col2:
                st.metric("Current Fatigue", f"{latest['fatigue']*100:.0f}%")
            with col3:
                st.metric("Mode", latest['mode'])
    else:
        st.info("Collecting cognitive state data...")

# Add to main dashboard
with st.expander("📈 Cognitive State Timeline", expanded=False):
    render_cognitive_timeline()
    if st.button("Update Timeline", key='update_timeline'):
        st.rerun()
```

---

## 8. Memory Graph Visualization

### 8.1 Network Graph of Memory Relationships

**Location:** New tab for advanced visualization

```python
def render_memory_graph():
    """Visualize memory relationships as a network graph"""
    st.markdown("### 🕸️ Memory Relationship Graph")
    
    # Fetch memories from STM and LTM
    stm_data = GeorgeAPI.get("/api/agent/memory/list/stm")
    ltm_data = GeorgeAPI.get("/api/agent/memory/list/ltm")
    
    memories = []
    if "error" not in stm_data:
        memories.extend([{**m, 'source': 'STM'} for m in stm_data.get("memories", [])])
    if "error" not in ltm_data:
        memories.extend([{**m, 'source': 'LTM'} for m in ltm_data.get("memories", [])])
    
    if not memories:
        st.info("No memories to visualize")
        return
    
    # Build graph data
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    for mem in memories[:50]:  # Limit to 50 for performance
        mem_id = mem.get('id', '')
        content = mem.get('content', '')[:30]
        source = mem.get('source', 'Unknown')
        
        G.add_node(mem_id, 
                  label=content,
                  source=source,
                  full_content=mem.get('content', ''))
    
    # Add edges based on content similarity (simplified)
    # In production, use proper embeddings similarity
    for i, mem1 in enumerate(memories[:50]):
        for mem2 in memories[i+1:50]:
            # Simple word overlap heuristic
            words1 = set(mem1.get('content', '').lower().split())
            words2 = set(mem2.get('content', '').lower().split())
            overlap = len(words1 & words2)
            
            if overlap >= 2:  # At least 2 words in common
                G.add_edge(mem1.get('id'), mem2.get('id'), weight=overlap)
    
    # Create interactive visualization using plotly
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Extract positions
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        node_text.append(f"{node_data['label']}<br>Source: {node_data['source']}")
        node_color.append('#4ecdc4' if node_data['source'] == 'STM' else '#ff6b6b')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=15,
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)'
                   ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔵 **STM** - Short-term memories")
    with col2:
        st.markdown("🔴 **LTM** - Long-term memories")
    
    st.caption(f"Showing {len(G.nodes())} memories with {len(G.edges())} relationships")

# Add as new tab or expander
with st.expander("🕸️ Memory Graph", expanded=False):
    render_memory_graph()
    if st.button("Refresh Graph", key='refresh_graph'):
        st.rerun()
```

---

## 9. Chat Analytics Dashboard

### 9.1 Comprehensive Chat Analytics

**Location:** New tab or section

```python
def render_chat_analytics():
    """Display comprehensive chat analytics"""
    st.markdown("### 📊 Chat Analytics Dashboard")
    
    sid = st.session_state.get('active_session_id', 'default')
    history = st.session_state.per_session_history.get(sid, [])
    captured = st.session_state.per_session_captured_memory_list.get(sid, [])
    
    if not history:
        st.info("No conversation data yet")
        return
    
    # Calculate analytics
    user_msgs = [h for h in history if h.get('role') == 'user']
    george_msgs = [h for h in history if h.get('role') == 'george']
    
    total_turns = len(user_msgs)
    avg_user_length = sum(len(m.get('content', '')) for m in user_msgs) / max(1, len(user_msgs))
    avg_george_length = sum(len(m.get('content', '')) for m in george_msgs) / max(1, len(george_msgs))
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Turns", total_turns)
    with col2:
        st.metric("Memories Captured", len(captured))
    with col3:
        st.metric("Avg User Length", f"{avg_user_length:.0f} chars")
    with col4:
        st.metric("Avg Response Length", f"{avg_george_length:.0f} chars")
    
    # Message length distribution
    st.write("**Message Length Distribution:**")
    user_lengths = [len(m.get('content', '')) for m in user_msgs]
    george_lengths = [len(m.get('content', '')) for m in george_msgs]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=user_lengths,
        name='User',
        marker_color='#4ecdc4',
        opacity=0.7
    ))
    fig.add_trace(go.Histogram(
        x=george_lengths,
        name='George',
        marker_color='#ff6b6b',
        opacity=0.7
    ))
    fig.update_layout(
        barmode='overlay',
        height=250,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Message Length (characters)",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory capture over time
    if captured:
        st.write("**Memory Capture Rate:**")
        
        # Group by type
        types_over_time = {}
        for cm in captured:
            mtype = cm.get('memory_type', 'unknown')
            if mtype not in types_over_time:
                types_over_time[mtype] = []
            types_over_time[mtype].append(cm)
        
        # Create stacked area chart
        fig = go.Figure()
        for mtype, mems in types_over_time.items():
            indices = list(range(len(mems)))
            counts = list(range(1, len(mems) + 1))
            fig.add_trace(go.Scatter(
                x=indices,
                y=counts,
                mode='lines',
                name=mtype,
                stackgroup='one',
                fill='tonexty'
            ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Memory Index",
            yaxis_title="Cumulative Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top memory types
    if captured:
        st.write("**Memory Type Distribution:**")
        type_counts = {}
        for cm in captured:
            mtype = cm.get('memory_type', 'unknown')
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color='#4ecdc4',
                text=list(type_counts.values()),
                textposition='auto'
            )
        ])
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Memory Type",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Conversation patterns
    st.write("**Conversation Patterns:**")
    if len(history) >= 4:
        # Simple pattern analysis
        patterns = {
            'Question-Answer': 0,
            'Statement-Acknowledgment': 0,
            'Command-Confirmation': 0
        }
        
        for i in range(0, len(history)-1, 2):
            if i+1 < len(history):
                user_msg = history[i].get('content', '').lower()
                if '?' in user_msg:
                    patterns['Question-Answer'] += 1
                elif any(cmd in user_msg for cmd in ['please', 'can you', 'could you']):
                    patterns['Command-Confirmation'] += 1
                else:
                    patterns['Statement-Acknowledgment'] += 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(patterns.keys()),
                values=list(patterns.values()),
                hole=0.4,
                marker_colors=['#4ecdc4', '#ffe66d', '#ff6b6b']
            )
        ])
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# Add to main dashboard
with st.expander("📊 Chat Analytics", expanded=False):
    render_chat_analytics()
```

---

## 10. Executive Dashboard Integration

### 10.1 Executive Function Visualization

**Location:** New section for executive features

```python
def render_executive_dashboard():
    """Display executive function status and metrics"""
    st.markdown("### 🎯 Executive Functions")
    
    exec_status = GeorgeAPI.get("/api/executive/status", timeout=15)
    
    if "error" in exec_status or exec_status.get("status") == "inactive":
        st.info("Executive functions not active")
        return
    
    # Executive metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Goals", len(exec_status.get("active_goals", [])))
    with col2:
        st.metric("Active Tasks", len(exec_status.get("active_tasks", [])))
    with col3:
        st.metric("Decisions Made", exec_status.get("decision_count", 0))
    
    # Goals overview
    goals = exec_status.get("active_goals", [])
    if goals:
        st.write("**Active Goals:**")
        for goal in goals:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"🎯 **{goal.get('description', 'Unknown goal')}**")
                    priority = goal.get('priority', 0)
                    st.progress(priority, text=f"Priority: {priority:.0%}")
                with col2:
                    status = goal.get('status', 'active')
                    st.write(f"Status: {status}")
    
    # Recent decisions
    if 'executive_data' in st.session_state and st.session_state.executive_data:
        recent_decision = st.session_state.executive_data.get('recent_decision')
        if recent_decision:
            with st.expander("🧠 Recent Executive Decision"):
                st.write(f"**Context:** {recent_decision.get('context', 'Unknown')}")
                st.write(f"**Selected:** {recent_decision.get('selected_option', 'Unknown')}")
                st.write(f"**Confidence:** {recent_decision.get('confidence', 0):.0%}")

# Add to main dashboard
with st.expander("🎯 Executive Functions", expanded=False):
    render_executive_dashboard()
```

---

## Implementation Priority

### Phase 1: Essential Enhancements (2-3 hours)
1. ✅ Enhanced memory capture display (Section 1.1)
2. ✅ Performance dashboard (Section 2.1)
3. ✅ Session management improvements (Section 4.1)
4. ✅ Prospective memory panel (Section 5.1)

### Phase 2: Advanced Features (2-3 hours)
5. ✅ Consolidation dashboard (Section 2.2)
6. ✅ Metacognitive awareness (Section 3.1)
7. ✅ Memory provenance (Section 6.1)
8. ✅ Memory statistics (Section 1.3)

### Phase 3: Polish & Analytics (2-3 hours)
9. ✅ Cognitive timeline (Section 7.1)
10. ✅ Chat analytics (Section 9.1)
11. ✅ Memory graph (Section 8.1)
12. ✅ Executive dashboard (Section 10.1)

---

## Testing Checklist

After implementing each section:

- [ ] UI renders without errors
- [ ] API calls succeed
- [ ] Data displays correctly
- [ ] Interactive elements work
- [ ] Charts/graphs render properly
- [ ] Mobile responsive (basic)
- [ ] Performance acceptable
- [ ] Error handling works

---

## Next Steps

1. **Implement Phase 1** - Core enhancements
2. **Test thoroughly** - Verify all features work
3. **Gather feedback** - Use the interface, note issues
4. **Implement Phase 2** - Advanced features
5. **Polish UI** - Improve styling, transitions
6. **Optimize performance** - Reduce API calls, cache data
7. **Add Phase 3** - Analytics and visualizations

---

**Estimated Total Time:** 6-10 hours for complete implementation  
**Difficulty:** Moderate (requires Streamlit + Plotly knowledge)  
**Impact:** High - Significantly improves user experience and cognitive visibility

---

**Tips:**
- Test each section individually before moving to the next
- Use `st.rerun()` sparingly to avoid performance issues
- Cache expensive computations with `@st.cache_data`
- Use `st.session_state` for persistent data across reruns
- Keep API timeout values reasonable (10-15 seconds)
- Add error handling for all API calls
- Use expanders to reduce initial visual clutter
- Make charts responsive with `use_container_width=True`

**Remember:** The backend integration (INTEGRATION_GUIDE.md) must be completed first for these UI enhancements to display real data!
