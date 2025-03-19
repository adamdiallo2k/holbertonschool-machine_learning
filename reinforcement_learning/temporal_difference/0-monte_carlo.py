def monte_carlo(env, V, policy, episodes=5000, gamma=0.9):
    """
    Monte Carlo Every-Visit : on moyenne tous les retours obtenus
    """
    nS = len(V)                   # nb d’états
    returns_sum = np.zeros(nS)
    returns_count = np.zeros(nS)

    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        done = False
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            done = (terminated or truncated)

        # On calcule G en sens inverse et on met à jour
        G = 0
        for t in range(len(episode)-1, -1, -1):
            s_t, r_t = episode[t]
            G = r_t + gamma * G
            returns_sum[s_t] += G
            returns_count[s_t] += 1

    # Moyenne
    for s in range(nS):
        if returns_count[s] != 0:
            V[s] = returns_sum[s] / returns_count[s]

    return V
