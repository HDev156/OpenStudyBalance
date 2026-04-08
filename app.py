import streamlit as st
from env import OpenStudyBalanceEnv
from models import Action
from graders import compute_detailed_scores, get_performance_summary

# Set page config for polished look
st.set_page_config(page_title="OpenStudyBalance", layout="wide", initial_sidebar_state="expanded")

TASKS = {
    "Balanced Assignment Week": "balanced_assignment_week",
    "Deadline Collision Week": "deadline_collision_week",
    "Burnout Prevention During Mid-Sems": "burnout_prevention_mid_sems",
}

TASK_DESCRIPTIONS = {
    "Balanced Assignment Week": "A relatively manageable week with a few assignments and enough study time. Tests basic prioritization and scheduling.",
    "Deadline Collision Week": "A more constrained week with overlapping deadlines and limited time. Tests prioritization under pressure.",
    "Burnout Prevention During Mid-Sems": "A high-stress academic scenario involving exams, assignments, and limited recovery time. Tests long-horizon planning and burnout-aware decision making.",
}


def init_session():
    """Initialize all session state variables."""
    if "env" not in st.session_state:
        st.session_state.env = OpenStudyBalanceEnv()
    if "task_key" not in st.session_state:
        st.session_state.task_key = TASKS["Balanced Assignment Week"]
    if "obs" not in st.session_state:
        st.session_state.obs = None
    if "reward" not in st.session_state:
        st.session_state.reward = None
    if "done" not in st.session_state:
        st.session_state.done = False
    if "info" not in st.session_state:
        st.session_state.info = {}
    if "message" not in st.session_state:
        st.session_state.message = "Ready to reset the environment."


def reset_environment(task_key: str):
    """Reset the environment with the given task."""
    st.session_state.task_key = task_key
    st.session_state.env.reset(task_key)
    st.session_state.obs = st.session_state.env.get_observation()
    st.session_state.reward = None
    st.session_state.done = False
    st.session_state.info = {}
    st.session_state.message = f"Environment reset to {task_key}."


def render_task_table(tasks):
    """Convert tasks to a formatted table data structure."""
    if not tasks:
        return []
    
    # Sort by priority (ascending) then by deadline (ascending)
    sorted_tasks = sorted(tasks, key=lambda t: (t.priority, t.deadline_days))
    
    return [
        {
            "Task": t.name,
            "Hours Needed": f"{t.hours_needed:.1f}",
            "Deadline (Days)": t.deadline_days,
            "Priority": t.priority,
            "Scheduled": f"{t.scheduled_hours:.1f}",
        }
        for t in sorted_tasks
    ]


def compute_planning_progress(tasks):
    """Compute total and scheduled hours."""
    total_needed = sum(t.hours_needed for t in tasks)
    total_scheduled = sum(t.scheduled_hours for t in tasks)
    return total_scheduled, total_needed, (total_scheduled / total_needed * 100 if total_needed > 0 else 0)


def get_performance_label(score):
    """Return a human-friendly performance label based on score."""
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.50:
        return "Fair"
    else:
        return "Needs Improvement"


def render_header():
    """Render the polished header section."""
    st.markdown("# 🎓 OpenStudyBalance")
    st.markdown("### Burnout-aware Academic Planning Environment")
    
    st.markdown(
        """
        **OpenStudyBalance** is an OpenEnv-compatible benchmark that evaluates how well an AI agent 
        can plan student workloads under deadlines, limited study time, stress, and burnout constraints.
        
        This demo showcases an interactive environment where you can experiment with planning strategies 
        and observe how different decisions impact overall performance.
        """
    )
    
    with st.expander("📖 How This Works", expanded=False):
        st.markdown("""
        1. **Choose a Scenario** – Select from Easy, Medium, or Hard difficulty levels
        2. **Reset Environment** – Initialize the environment with your chosen scenario
        3. **Observe State** – Review pending tasks, stress level, available study time, etc.
        4. **Take Actions** – Schedule tasks, add breaks, or finalize your plan
        5. **Get Feedback** – See real-time rewards and state changes
        6. **Evaluate Performance** – Get a final score when you complete the plan
        """)


def render_sidebar():
    """Render the sidebar with scenario selection and info."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        task_choice = st.selectbox("Select a Scenario", list(TASKS.keys()))
        
        if st.button("🔄 Reset Environment", use_container_width=True):
            reset_environment(TASKS[task_choice])
            st.rerun()
        
        st.divider()
        st.subheader("Scenario Info")
        st.info(TASK_DESCRIPTIONS[task_choice])
        
        return task_choice


def render_metrics(obs, env):
    """Render metrics cards for current state."""
    st.subheader("📊 Current Environment State")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Stress Level", f"{obs.stress_level:.2f}", delta=None)
    with col2:
        st.metric("Sleep Hours", f"{obs.sleep_hours:.1f}h", delta=None)
    with col3:
        st.metric("Hours/Day Available", f"{obs.available_hours_per_day:.1f}h", delta=None)
    with col4:
        st.metric("Pending Tasks", len(obs.pending_tasks), delta=None)
    with col5:
        burnout_risk = env.state().burnout_risk
        category = env.get_burnout_category()
        st.metric("Burnout Risk", f"{burnout_risk:.2f} ({category})", delta=None)
    
    st.divider()


def render_environment_status(obs):
    """Render current environment status in an info box."""
    if len(obs.pending_tasks) == 0:
        status_msg = "✅ All tasks are scheduled! Consider finalizing the plan."
    else:
        stress_desc = "high" if obs.stress_level > 0.7 else "moderate" if obs.stress_level > 0.3 else "low"
        status_msg = f"The student currently has {len(obs.pending_tasks)} pending academic tasks with {stress_desc} stress levels."
    
    st.info(status_msg)


def render_risk_warnings(obs, env):
    """Render risk warnings if any."""
    warnings = []
    
    # High-priority tasks close to deadline
    urgent_tasks = [t for t in obs.pending_tasks if t.priority <= 2 and t.deadline_days <= 2]
    if urgent_tasks:
        warnings.append(f"⚠️ {len(urgent_tasks)} high-priority task(s) due soon!")
    
    # Overloaded days
    state = env.state()
    overloaded_days = [day for day, hours in state.daily_scheduled.items() if hours > state.available_hours_per_day]
    if overloaded_days:
        warnings.append(f"⚠️ Day(s) {', '.join(map(str, overloaded_days))} are overloaded!")
    
    # High burnout risk
    if state.burnout_risk > 0.7:
        warnings.append("⚠️ Burnout risk is critical! Consider adding breaks.")
    
    if warnings:
        for warning in warnings:
            st.warning(warning)


def render_tasks_table(obs):
    """Render pending tasks in a clean table."""
    st.subheader("📋 Pending Tasks")
    
    if obs.pending_tasks:
        task_df = render_task_table(obs.pending_tasks)
        st.dataframe(task_df, use_container_width=True, hide_index=True)
    else:
        st.success("🎉 No pending tasks!")


def render_progress_bar(obs):
    """Render planning progress bar."""
    scheduled, needed, percent = compute_planning_progress(obs.pending_tasks)
    st.subheader("📈 Planning Progress")
    st.progress(min(percent / 100, 1.0))
    st.caption(f"{scheduled:.1f}h / {needed:.1f}h scheduled ({percent:.1f}%)")


def render_unexpected_events(env):
    """Render unexpected events section."""
    st.subheader("🌪️ Unexpected Events")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if env.last_event:
            st.info(f"**Latest Event:** {env.last_event['name']}\n\n{env.last_event['description']}")
        else:
            st.info("No unexpected events have occurred yet.")
    
    with col2:
        if st.button("💥 Inject Disruption", use_container_width=True):
            env.inject_disruption()
            st.rerun()
    
    st.caption("Events can occur automatically every 5 steps or be triggered manually.")


def render_action_section(obs):
    """Render the planning controls section."""
    st.subheader("🎯 Planning Controls")
    
    action_type = st.selectbox(
        "Choose an action",
        ["schedule_task", "add_break", "finalize_plan"],
        format_func=lambda x: {
            "schedule_task": "📅 Schedule Task",
            "add_break": "☕ Add Break",
            "finalize_plan": "✅ Finalize Plan"
        }[x]
    )
    
    # Show relevant inputs based on action type
    col1, col2, col3 = st.columns(3)
    
    task_name = None
    day = 1
    hours = 1.0
    
    if action_type == "schedule_task":
        with col1:
            pending_names = [t.name for t in obs.pending_tasks]
            if pending_names:
                task_name = st.selectbox("Task", pending_names)
            else:
                st.warning("No pending tasks available!")
                return None
        with col2:
            day = st.selectbox("Day", list(range(1, 8)))
        with col3:
            hours = st.number_input(
                "Hours",
                min_value=0.0,
                max_value=obs.available_hours_per_day,
                value=min(1.0, obs.available_hours_per_day),
                step=0.5,
                format="%.1f",
            )
    elif action_type == "add_break":
        with col1:
            day = st.selectbox("Day for Break", list(range(1, 8)))
        st.caption("Adding a break reduces stress by 5%")
    else:  # finalize_plan
        st.caption("Finalizing will compute a score based on your plan and end the episode.")
    
    st.divider()
    
    # Demo tip
    st.caption("💡 **Demo Tip:** Try scheduling urgent tasks first and avoid overloading a single day.")
    
    if st.button("Execute Action", use_container_width=True, type="primary"):
        if st.session_state.done:
            st.warning("Episode finished! Reset to try again.")
            return None
        else:
            action = Action(action_type=action_type, task_name=task_name, day=day, hours=hours)
            return action
    
    return None


def render_reward_feedback(reward):
    """Render color-coded reward feedback with explanations."""
    st.subheader("💬 Latest Environment Feedback")
    
    value = reward.value
    reason = reward.reason
    
    # Enhanced explanations
    if "Scheduled" in reason and value > 0:
        explanation = "This action helped reduce workload and potentially improved deadline adherence."
    elif "Overload" in reason:
        explanation = "This created imbalance in the schedule, increasing stress and burnout risk."
    elif "break" in reason.lower():
        explanation = "This action helped manage stress and reduce burnout risk."
    elif "finalized" in reason.lower():
        explanation = "The planning session is complete. Check the final evaluation below."
    else:
        explanation = "This action had minimal impact on the overall plan."
    
    if value > 0.05:
        st.success(f"✅ **Reward: +{value:.2f}**\n\n{reason}\n\n{explanation}")
    elif value < -0.05:
        st.error(f"❌ **Reward: {value:.2f}**\n\n{reason}\n\n{explanation}")
    else:
        st.info(f"ℹ️ **Reward: {value:.2f}**\n\n{reason}\n\n{explanation}")


def render_episode_status(done, steps):
    """Render episode status card."""
    st.subheader("📍 Episode Status")
    
    col1, col2 = st.columns(2)
    with col1:
        status_text = "🏁 Complete" if done else "⏳ In Progress"
        st.metric("Status", status_text)
    with col2:
        st.metric("Steps", steps, delta=None)


def render_final_evaluation(done, env_state):
    """Render final evaluation with multi-objective breakdown."""
    if done:
        st.divider()
        st.subheader("🏆 Final Evaluation Breakdown")
        
        scores = compute_detailed_scores(env_state)
        summary = get_performance_summary(scores)
        label = get_performance_label(scores["overall"])
        
        # Overall score
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Score", f"{scores['overall']:.2f}/1.00")
        with col2:
            st.metric("Performance", label)
        
        st.markdown("### Sub-Score Breakdown")
        sub_cols = st.columns(4)
        with sub_cols[0]:
            st.metric("Deadline Adherence", f"{scores['deadline_adherence']:.2f}")
        with sub_cols[1]:
            st.metric("Workload Balance", f"{scores['workload_balance']:.2f}")
        with sub_cols[2]:
            st.metric("Burnout Safety", f"{scores['burnout_safety']:.2f}")
        with sub_cols[3]:
            st.metric("Priority Handling", f"{scores['priority_handling']:.2f}")
        
        st.info(summary)


def main():
    """Main application entry point."""
    init_session()
    
    # Polished header
    render_header()
    st.divider()
    
    # Sidebar
    task_choice = render_sidebar()
    
    # Main content
    if st.session_state.obs is None:
        st.warning("👈 Select a scenario and click **Reset Environment** in the sidebar to begin.")
        return
    
    obs = st.session_state.obs
    env = st.session_state.env
    
    # Metrics
    render_metrics(obs, env)
    
    # Environment status
    render_environment_status(obs)
    
    # Risk warnings
    render_risk_warnings(obs, env)
    
    # Tasks table
    render_tasks_table(obs)
    
    # Progress bar
    render_progress_bar(obs)
    
    # Unexpected events
    render_unexpected_events(env)
    
    st.divider()
    
    # Action section
    action = render_action_section(obs)
    
    # Execute action if one was created
    if action is not None:
        obs, reward, done, info = st.session_state.env.step(action)
        st.session_state.obs = obs
        st.session_state.reward = reward
        st.session_state.done = done
        st.session_state.info = info
        st.rerun()
    
    st.divider()
    
    # Reward feedback
    if st.session_state.reward is not None:
        render_reward_feedback(st.session_state.reward)
    
    st.divider()
    
    # Episode status
    render_episode_status(st.session_state.done, st.session_state.info.get('steps', 0))
    
    # Final evaluation
    if st.session_state.done:
        render_final_evaluation(st.session_state.done, st.session_state.env.state())


if __name__ == "__main__":
    main()
