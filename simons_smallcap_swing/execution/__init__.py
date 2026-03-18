"""
execution — Institutional execution simulation and cost accounting.

    cost_model.py        Fill → fee/spread/slippage/impact/borrow, reconciled 4 levels
    impact_model.py      Power-law market impact by liquidity bucket
    fills_simulator.py   Orders → fills with shared daily capacity, carry, expiry
    turnover_control.py  Turnover budgeting: bands, tiers, throttle, stagger
    borrow_execution.py  Short admission: eligibility, capacity, recall, borrow cost
"""
