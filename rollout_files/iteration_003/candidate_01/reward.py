def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)

    # Use dt-normalized forward velocity for consistent scaling across timesteps.
    dt = jnp.asarray(info.get('dt', 0.05))
    dt = jnp.clip(dt, 1e-3, 0.2)
    dx = next_obs[..., 0] - obs[..., 0]
    v = dx / dt

    # Preserve rectified forward progress with explicit backward penalty (bounded).
    v_fwd = jnp.clip(v, 0.0, 10.0)
    v_back = jnp.clip(-v, 0.0, 10.0)
    vel_reward = 0.22 * v_fwd - 0.35 * v_back  # bounded in [-3.5, 2.2]

    # Stability shaping: keep some always-on posture reward + extra when moving forward.
    q = next_obs[..., 1:9]
    posture = jnp.exp(-1.2 * jnp.mean(jnp.square(q)))  # (0, 1]
    forward_gate = jax.nn.sigmoid(0.8 * (v - 0.5))     # soft gate, not hard off
    posture_reward = 0.10 * posture + 0.18 * posture * forward_gate  # in (0, 0.28]

    # Height shaping if available (fallback to 0 if not provided).
    height = info.get('torso_height', info.get('height', None))
    if height is None:
        height_reward = jnp.zeros_like(dx)
        height_pen = jnp.zeros_like(dx)
    else:
        height = jnp.asarray(height)
        target_h = jnp.asarray(info.get('target_height', 0.5))
        h_err = height - target_h
        height_reward = 0.06 * jnp.exp(-6.0 * jnp.square(h_err))  # in (0, 0.06]
        height_pen = -0.06 * jnp.tanh(3.0 * jnp.abs(h_err))       # in [-0.06, 0]

    # Joint speed penalty, mildly stronger and speed-dependent (capped).
    qd = next_obs[..., 9:18]
    qd2 = jnp.mean(jnp.square(qd))
    speed_scale = 1.0 + 0.25 * jnp.clip(v_fwd, 0.0, 6.0)  # in [1, 2.5]
    speed_pen = -0.10 * speed_scale * jnp.tanh(0.45 * qd2)  # in [-0.25, 0]

    # Energy efficiency: slightly stronger, still bounded to avoid freezing.
    a_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action)))
    ctrl_pen = -0.045 * jnp.tanh(1.6 * a_l1)  # in [-0.045, 0]

    # Smoothness: use prev_action if provided, else no smoothness penalty (avoid inconsistent gradients).
    prev_a = info.get('prev_action', None)
    if prev_a is None:
        smooth_pen = jnp.zeros_like(dx)
    else:
        prev_a = jnp.asarray(prev_a)
        da2 = jnp.mean(jnp.square(action - prev_a))
        smooth_scale = 1.0 + 0.20 * jnp.clip(v_fwd, 0.0, 6.0)  # in [1, 2.2]
        smooth_pen = -0.025 * smooth_scale * jnp.tanh(3.0 * da2)  # in [-0.055, 0]

    # Keep-alive bonus and termination penalty (bounded).
    done = jnp.asarray(info.get('done', 0.0))
    alive = 0.05 * (1.0 - done)
    fall_pen = -0.5 * done

    total_reward = (
        vel_reward
        + posture_reward
        + height_reward
        + height_pen
        + speed_pen
        + ctrl_pen
        + smooth_pen
        + alive
        + fall_pen
    )

    reward_terms = {
        "vel_dt_norm_rectified_backpen": vel_reward,
        "posture_always_on_plus_fwd": posture_reward,
        "height_reward": height_reward,
        "height_pen": height_pen,
        "joint_speed_pen_speed_scaled": speed_pen,
        "ctrl_l1_pen_capped": ctrl_pen,
        "smooth_delta2_pen_speed_scaled": smooth_pen,
        "alive": alive,
        "fall_pen": fall_pen,
    }
    return total_reward, reward_terms