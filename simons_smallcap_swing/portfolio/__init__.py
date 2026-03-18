"""
portfolio — Institutional portfolio construction pipeline.

    sizing.py       Score → raw weights (rank_symmetric, vol/liq adjust, gross/net normalise)
    neutralize.py   Factor/group neutralisation via constrained projection
    optimizer.py    QP: maximise α - λ_r risk - λ_c cost under constraints
    constraints.py  Project weights to feasible set (bounds, gross, net, turnover, liquidity)
    rebalance.py    Target → executable orders with budget, rounding, repair
    capacity.py     AUM capacity estimation (participation, DTL, impact cost)

Pipeline:  signal → sizing → neutralize → optimizer/constraints → rebalance → execution
"""
