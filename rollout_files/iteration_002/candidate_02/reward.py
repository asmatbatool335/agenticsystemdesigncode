def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)

    # Prefer reliable velocity signal if provided by env/info; otherwise fall back to dx.
    dx = next_obs[..., 0] - obs[..., 0]
    v_info = info.get('x_velocity', None)
    v = jnp.asarray(v_info) if v_info is not None else dx

    # Softplus forward reward (no reward for backward), plus explicit backward penalty.
    v_clip = jnp.clip(v, -1.0, 2.0)
    fwd = jax.nn.softplus(3.0 * v_clip) / 3.0  # ~max(v,0) but smooth, bounded by clip
    back = jax.nn.softplus(-3.0 * v_clip) / 3.0
    vel_reward = 1.6 * fwd - 2.4 * back  # bounded given v_clip

    # Upright shaping: smaller and conditional on forward motion to avoid "stable backward" optimum.
    q = next_obs[..., 1:9]
    upright = jnp.exp(-1.2 * jnp.mean(jnp.square(q)))
    forward_gate = jax.nn.sigmoid(4.0 * v_clip)  # ~0 when moving backward
    upright_reward = 0.22 * upright * forward_gate

    # Joint speed penalty: capped with tanh for boundedness.
    qd = next_obs[..., 9:18]
    qd2 = jnp.mean(jnp.square(qd))
    speed_pen = -0.06 * jnp.tanh(0.6 * qd2)

    # Control penalty: keep moderate and bounded.
    a_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action)))
    ctrl_pen = -0.025 * jnp.tanh(2.0 * a_l1)

    # Smoothness penalty: keep moderate and bounded.
    prev_a = info.get('prev_action', jnp.zeros_like(action))
    prev_a = jnp.asarray(prev_a)
    da_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action - prev_a)))
    smooth_pen = -0.018 * jnp.tanh(2.5 * da_l1)

    # Small alive bonus and bounded termination penalty.
    done = info.get('done', 0.0)
    done = jnp.asarray(done)
    alive = 0.06 * (1.0 - done)
    fall_pen = -0.6 * done

    total_reward = vel_reward + upright_reward + speed_pen + ctrl_pen + smooth_pen + alive + fall_pen
    reward_terms = {
        "vel_softplus_backpen": vel_reward,
        "upright_forward_gated": upright_reward,
        "joint_speed_pen_capped": speed_pen,
        "ctrl_l1_pen_capped": ctrl_pen,
        "smooth_l1_pen_capped": smooth_pen,
        "alive": alive,
        "fall_pen": fall_pen,
    }
    return total_reward, reward_terms