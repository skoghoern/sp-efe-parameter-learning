"""Log-space message passing primitives on the POMDP chain.

Everything both planners share lives here: the log-space guards, the forward and
backward sweeps over the dynamics chain, and the geometric channel damping that
makes the channel fixed point converge.

The factor graph is a chain of dynamics factors with an observation factor
hanging off every state:

    p(x0)--x0--[dyn0]--x1--[dyn1]--x2-- ... --[dyn_{T-1}]--x_T--goal
            |           |           |                        |
         [obs0]_k    [obs1]_k    [obs2]_k               [obsT]_k   (k=1..n_fov)

A sweep is one ``forward_pass`` followed by one ``backward_pass``; the backward
pass also returns the action marginal q(u_t). Planners call these in a
``lax.scan``, damping their channels after each sweep -- see ``sp_efe.py``.

All internal computation is in log-space; probability-space tensors are accepted
at the boundary.

``log_*`` arrays may hold ``LOG_ZERO`` for impossible entries. Mask with
``is_live`` before any arithmetic other than adding a finite term (especially
−log r) and before every expectation (``sp_efe.weighted_log``). In float32,
LOG_ZERO + x stays LOG_ZERO, so a masked entry stays masked through the sums.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

# Structural zeros are carried as a large negative number rather than -inf so
# that differences and sums stay finite under JAX transformations.
LOG_ZERO = -1e12


def safe_log(x):
    """Log that maps 0 → LOG_ZERO instead of -inf."""
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-30)), LOG_ZERO)


def is_live(log_x):
    """Mask of entries that are not structural zeros (the LOG_ZERO sentinel)."""
    return log_x > LOG_ZERO / 2


def log_normalize(log_x, axis):
    """log_x − logsumexp(log_x) over ``axis``; a normalized log-conditional."""
    return jax.nn.log_softmax(log_x, axis=axis)


# =============================================================================
# Channel damping
# =============================================================================


def damp_log_channel(log_old, log_new, damping, cond_axis):
    """Geometric damping of log-space channels.

    damped = (1-α)*log_old + α*log_new

    damping=1.0 -> new channels (no damping)
    damping=0.5 -> geometric mean in probability space
    """
    damped = (1.0 - damping) * log_old + damping * log_new

    # Structural zeros: only zero if BOTH old and new are LOG_ZERO
    valid = is_live(log_old) | is_live(log_new)
    damped = jnp.where(valid, damped, LOG_ZERO)

    # Renormalize conditional for numerical stability
    normalizer = logsumexp(damped, axis=cond_axis, keepdims=True)
    return jnp.where(valid, damped - normalizer, LOG_ZERO)


def forward_pass(log_reduced_per_t, log_q_x0, log_action_prior, log_obs_to_x, horizon,
                 log_prev_fwd=None, msg_damping=1.0):
    """
    Forward pass with obs->x message injection (log-space).

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_q_x0: (n_states,) log initial state belief
        log_action_prior: (n_actions,) log prior over actions
        log_obs_to_x: (T+1, n_states) log messages from obs factors to x
        horizon: T
        log_prev_fwd: (T+1, n_states) previous iteration's fwd messages (optional)
        msg_damping: float, inertial damping weight (1.0 = no damping)

    Returns:
        log_fwd_msgs: (T+1, n_states) log forward messages
    """
    n_states = log_q_x0.shape[0]
    log_fwd = jnp.zeros((horizon + 1, n_states))
    log_fwd = log_fwd.at[0].set(log_q_x0)

    def body_fn(t, log_fwd):
        log_x_msg = log_fwd[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_x_msg[None, :, None]
                 + log_action_prior[None, None, :])
        log_q_next = logsumexp(terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        # Inertial damping against previous iteration's message
        if log_prev_fwd is not None:
            log_q_next = (1.0 - msg_damping) * log_prev_fwd[t + 1] + msg_damping * log_q_next
            log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def backward_pass(log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                  log_obs_to_x, horizon, log_prev_bwd=None, msg_damping=1.0):
    """
    Backward pass with obs->x message injection (log-space).

    Returns:
        log_bwd_msgs: (T+1, n_states) log backward messages
        q_u: (T, n_actions) action marginals (probability space)
    """
    n_states = log_goal.shape[0]
    n_actions = log_action_prior.shape[0]
    log_bwd = jnp.zeros((horizon + 1, n_states))
    log_bwd = log_bwd.at[horizon].set(log_goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        log_bwd, q_u = carry
        log_bwd_t1 = log_bwd[t + 1] + log_obs_to_x[t + 1]

        # Action marginal
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_bwd_t1[:, None, None]
                 + log_fwd_t[None, :, None])
        log_msg_to_u = logsumexp(terms, axis=(0, 1))
        q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t (no obs_to_x[t])
        terms_bwd = (log_reduced_per_t[t]
                     + log_bwd_t1[:, None, None]
                     + log_action_prior[None, None, :])
        log_bwd_t = logsumexp(terms_bwd, axis=(0, 2))
        log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        # Inertial damping against previous iteration's message
        if log_prev_bwd is not None:
            log_bwd_t = (1.0 - msg_damping) * log_prev_bwd[t] + msg_damping * log_bwd_t
            log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        log_bwd = log_bwd.at[t].set(log_bwd_t)

        return (log_bwd, q_u), None

    (log_bwd, q_u), _ = lax.scan(
        body_fn, (log_bwd, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return log_bwd, q_u
