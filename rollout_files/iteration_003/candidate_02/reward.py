def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)

    # dt-normalized velocity (with safe default).
    dt = jnp.asarray(info.get('dt', 0.05))
    dt = jnp.clip(dt, 1e-3, 0.2)
    dx = next_obs[..., 0] - obs[..., 0]
    v = dx / dt

    # Forward progress with explicit backward penalty; slightly reduced forward weight to rebalance efficiency.
    v_fwd = jnp.clip(v, 0.0, 9.0)
    v_back = jnp.clip(-v, 0.0, 9.0)
    vel_reward = 0.20 * v_fwd - 0.40 * v_back  # bounded in [-3.6, 1.8]

    # Posture shaping: blend (not gate) so stability gradients exist even when not moving well.
    q = next_obs[..., 1:9]
    posture = jnp.exp(-1.0 * jnp.mean(jnp.square(q)))  # (0, 1]
    blend = 0.35 + 0.65 * jax.nn.sigmoid(0.9 * (v - 0.3))  # in (0.35, 1)
    posture_reward = 0.26 * posture * blend  # in (0, 0.26]

    # Optional torso pitch penalty if provided (fallback to 0).
    pitch = info.get('torso_pitch', info.get('pitch', None))
    if pitch is None:
        pitch_pen = jnp.zeros_like(dx)
    else:
        pitch = jnp.asarray(pitch)
        pitch_pen = -0.10 * jnp.tanh(2.5 * jnp.abs(pitch))  # in [-0.10, 0]

    # Optional height penalty if provided (fallback to 0).
    height = info.get('torso_height', info.get('height', None))
    if height is None:
        height_pen = jnp.zeros_like(dx)
    else:
        height = jnp.asarray(height)
        target_h = jnp.asarray(info.get('target_height', 0.5))
        h_err = height - target_h
        height_pen = -0.08 * jnp.tanh(3.0 * jnp.abs(h_err))  # in [-0.08, 0]

    # Joint speed penalty: stronger and increases with forward speed (capped).
    qd = next_obs[..., 9:18]
    qd2 = jnp.mean(jnp.square(qd))
    speed_scale = 1.0 + 0.30 * jnp.clip(v_fwd, 0.0, 5.0)  # in [1, 2.5]
    speed_pen = -0.11 * speed_scale * jnp.tanh(0.40 * qd2)  # in [-0.275, 0]

    # Energy efficiency: slightly stronger, bounded.
    a_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action)))
    ctrl_pen = -0.055 * jnp.tanh(1.4 * a_l1)  # in [-0.055, 0]

    # Smoothness: robust to missing prev_action by treating it as zero-change (no penalty).
    prev_a = info.get('prev_action', None)
    if prev_a is None:
        smooth_pen = jnp.zeros_like(dx)
    else:
        prev_a = jnp.asarray(prev_a)
        da_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action - prev_a)))
        smooth_scale = 1.0 + 0.25 * jnp.clip(v_fwd, 0.0, 5.0)  # in [1, 2.25]
        smooth_pen = -0.030 * smooth_scale * jnp.tanh(2.2 * da_l1)  # in [-0.0675, 0]

    # Small alive bonus and bounded termination penalty.
    done = jnp.asarray(info.get('done', 0.0))
    alive = 0.05 * (1.0 - done)
    fall_pen = -0.5 * done

    total_reward = (
        vel_reward
        + posture_reward
        + pitch_pen
        + height_pen
        + speed_pen
        + ctrl_pen
        + smooth_pen
        + alive
        + fall_pen
    )

    reward_terms = {
        "vel_dt_norm_rectified_backpen": vel_reward,
        "posture_blended": posture_reward,
        "pitch_pen": pitch_pen,
        "height_pen": height_pen,
        "joint_speed_pen_speed_scaled": speed_pen,
        "ctrl_l1_pen_capped": ctrl_pen,
        "smooth_l1_pen_speed_scaled": smooth_pen,
        "alive": alive,
        "fall_pen": fall_pen,
    }
    return total_reward, reward_terms