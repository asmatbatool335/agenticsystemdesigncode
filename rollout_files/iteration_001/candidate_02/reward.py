def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)

    # Forward velocity estimate from x delta; use linear reward but clipped for stability
    dx = next_obs[..., 0] - obs[..., 0]
    vel_clip = jnp.clip(dx, -0.5, 1.5)
    vel_reward = 1.2 * vel_clip

    # Stability via "uprightness" proxy: keep joint angles near nominal (0)
    q = next_obs[..., 1:9]
    upright = jnp.exp(-1.5 * jnp.mean(jnp.square(q)))  # in (0, 1]
    upright_reward = 0.6 * upright

    # Penalize excessive joint speeds (helps prevent flailing at high speed)
    qd = next_obs[..., 9:18]
    speed_pen = -0.03 * jnp.mean(jnp.square(qd))

    # Energy efficiency: use L1-like penalty (smooth near 0) to avoid freezing
    ctrl_pen = -0.02 * jnp.mean(jnp.sqrt(1e-6 + jnp.square(action)))

    # Encourage smooth actions if prev_action exists
    prev_a = info.get('prev_action', jnp.zeros_like(action))
    prev_a = jnp.asarray(prev_a)
    smooth_pen = -0.015 * jnp.mean(jnp.sqrt(1e-6 + jnp.square(action - prev_a)))

    # Fall/termination penalty (bounded)
    done = info.get('done', 0.0)
    done = jnp.asarray(done)
    fall_pen = -1.0 * done

    total_reward = vel_reward + upright_reward + speed_pen + ctrl_pen + smooth_pen + fall_pen
    reward_terms = {
        "vel_clip_dx": vel_reward,
        "upright_exp": upright_reward,
        "joint_speed_pen": speed_pen,
        "ctrl_l1_pen": ctrl_pen,
        "smooth_l1_pen": smooth_pen,
        "fall_pen": fall_pen,
    }
    return total_reward, reward_terms