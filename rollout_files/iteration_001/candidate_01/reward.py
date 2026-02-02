def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)

    # Forward progress: use x-position delta (robust across obs layouts)
    x0 = obs[..., 0]
    x1 = next_obs[..., 0]
    dx = x1 - x0

    # Soft-saturate speed reward to keep bounded and avoid over-prioritizing speed
    vel_reward = 1.6 * jnp.tanh(2.0 * dx)

    # Upright/pose stability: penalize large joint angles (exclude root x at index 0)
    q = next_obs[..., 1:9]
    pose_pen = -0.25 * jnp.mean(jnp.square(q))

    # Angular velocity stability: penalize large joint angular velocities
    qd = next_obs[..., 9:18]
    angvel_pen = -0.05 * jnp.mean(jnp.square(qd))

    # Control smoothness/energy: moderate action L2 penalty
    ctrl_pen = -0.03 * jnp.mean(jnp.square(action))

    # Smoothness across time: penalize action changes if previous action is available
    prev_a = info.get('prev_action', jnp.zeros_like(action))
    prev_a = jnp.asarray(prev_a)
    da = action - prev_a
    smooth_pen = -0.02 * jnp.mean(jnp.square(da))

    # Keep alive bonus to discourage falling/termination without dominating
    done = info.get('done', 0.0)
    done = jnp.asarray(done)
    alive = 0.2 * (1.0 - done)

    total_reward = vel_reward + pose_pen + angvel_pen + ctrl_pen + smooth_pen + alive
    reward_terms = {
        "vel_tanh_dx": vel_reward,
        "pose_pen": pose_pen,
        "angvel_pen": angvel_pen,
        "ctrl_pen": ctrl_pen,
        "smooth_pen": smooth_pen,
        "alive": alive,
    }
    return total_reward, reward_terms