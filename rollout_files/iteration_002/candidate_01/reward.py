def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)

    # Forward progress: keep dx-based term but make backward motion strongly undesirable.
    dx = next_obs[..., 0] - obs[..., 0]
    dx_fwd = jnp.clip(dx, 0.0, 1.5)
    dx_back = jnp.clip(-dx, 0.0, 1.5)
    vel_reward = 1.4 * dx_fwd - 2.0 * dx_back  # bounded in [-3.0, 2.1]

    # Upright shaping: reduce magnitude and gate by forward progress so it can't dominate when moving backward.
    q = next_obs[..., 1:9]
    upright = jnp.exp(-1.5 * jnp.mean(jnp.square(q)))  # (0, 1]
    gate = jax.nn.sigmoid(6.0 * (dx - 0.02))  # ~0 if not making forward progress
    upright_reward = 0.25 * upright * gate

    # Penalize excessive joint speeds, but cap to keep per-step reward bounded.
    qd = next_obs[..., 9:18]
    qd2 = jnp.mean(jnp.square(qd))
    speed_pen = -0.08 * jnp.tanh(0.5 * qd2)  # in [-0.08, 0]

    # Energy efficiency: L1-like penalty, softly capped.
    a_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action)))
    ctrl_pen = -0.03 * jnp.tanh(1.5 * a_l1)  # in [-0.03, 0]

    # Smoothness: penalize action changes, softly capped.
    prev_a = info.get('prev_action', jnp.zeros_like(action))
    prev_a = jnp.asarray(prev_a)
    da_l1 = jnp.mean(jnp.sqrt(1e-6 + jnp.square(action - prev_a)))
    smooth_pen = -0.02 * jnp.tanh(2.0 * da_l1)  # in [-0.02, 0]

    # Keep-alive bonus (bounded) and termination penalty (bounded).
    done = info.get('done', 0.0)
    done = jnp.asarray(done)
    alive = 0.05 * (1.0 - done)
    fall_pen = -0.5 * done

    total_reward = vel_reward + upright_reward + speed_pen + ctrl_pen + smooth_pen + alive + fall_pen
    reward_terms = {
        "vel_rectified_backpen": vel_reward,
        "upright_gated": upright_reward,
        "joint_speed_pen_capped": speed_pen,
        "ctrl_l1_pen_capped": ctrl_pen,
        "smooth_l1_pen_capped": smooth_pen,
        "alive": alive,
        "fall_pen": fall_pen,
    }
    return total_reward, reward_terms